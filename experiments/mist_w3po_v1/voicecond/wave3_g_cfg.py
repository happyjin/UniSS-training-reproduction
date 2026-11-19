"""Robust wave 3, group G (reports/mist_w3po_v1/robust_wave3_single_prereg.zh-CN.md, G): negative-condition guidance
of the T1c Talker inside si_infer's block protocol, on the single-speaker LA2 base.

si_infer (experiments.uniss_qwen3_omni_v1.evaluation.si_infer) runs UNCHANGED (front end, blocks from the S1 Thinker,
speed tokens, prompts, seeds per block = --seed + 1000 n_it + r, guard settings, decode, timeline, records), except:

  * voice: BiCodecTokenizer.encode_wav_to_tokens(<out>/tmp/<id>_voice.wav) answers the Idea 1 C tokens of that very
    prompt (denoising speaker encoder voicecond_idea1/m2/ctrl/encoder.pt, u = phi_C(y), as idea1_tokens --mode ctrl).
    These g+ tokens are the Talker condition AND the BiCodec decoder's tokens (as in wave 1 P1b).
  * talker: vllm.LLM is replaced by an HF transformers (bf16) generator with the same interface. Per request (one
    block) and step, raw logits l = l(g+) + w [l(g+) - l(g-)] (the g- row is fed the same sampled tokens), then the
    request's own SIGuard (make_si_guard with the request's extra_args, exactly as the vLLM logits processor), then
    temperature 0.7 / top-p 0.8 (vLLM rule) and a draw with uniforms from a generator seeded by the request's
    SamplingParams.seed only -> common random numbers between the arms. Stops at <|wait_read|> / <|end_semantic|>
    (kept in token_ids, as vLLM's stop_token_ids), max_tokens 1500. w = 0 computes the g+ row only (l = l(g+)).
  * g- : C tokens of the sentence's 3 s pure-background pause before speech (wave-1 manifest `gap`) for the noisy
    condition; for the clean twin its own pause, i.e. 3 s of digital silence (C gives finite tokens on it).

vLLM pair decoding was not used: SIGuard is a per-request AdapterLogitsProcessor (it never sees the partner row), and
the two rows would have to be scheduled in the same step every step and fed the partner's sampled token.

    <venvs/vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.wave3_g_cfg --w 1 --rows 64 \\
        <si_infer arguments: --model .. --direction .. --window 1 --output .. --text-policy file --blocks-file ..
         --speed-policy block --items-file .. [--limit n]>
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
MANIFEST = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5/data/manifest.jsonl"
CENC = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"
GLOBAL_0, START_GLOBAL, END_GLOBAL = 151665, 180387, 180391
TEMP, TOP_P = 0.7, 0.8


def log(s):
    print(f"{time.strftime('%F %T')} [wave3_g] {s}", flush=True)


def top_p_sample(logits: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """temperature + vLLM top-p rule + inverse-CDF draw with given uniforms (as wave1_talker); logits (J, V) fp32"""
    probs = torch.softmax(logits / TEMP, -1)
    ps, idx = probs.sort(-1, descending=False)
    cs = ps.cumsum(-1)
    drop = cs <= (1 - TOP_P)
    drop[:, -1] = False
    ps = ps.masked_fill(drop, 0.0)
    ps = ps / ps.sum(-1, keepdim=True)
    ps, idx = ps.flip(-1), idx.flip(-1)
    c = ps.cumsum(-1)
    k = torch.searchsorted(c, u[:, None].to(c.dtype) * c[:, -1:]).clamp(max=ps.shape[-1] - 1)
    return idx.gather(-1, k)[:, 0]


class Voices:
    """C tokens for g+ (the prompt si_infer wrote) and g- (the sentence's pause), in item order"""

    def __init__(self, device: str):
        from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
        from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
        self.P = load_encoder(BiCodecPhi(device), str(CENC))
        self.gap = {r["id"]: r["gap"] for r in map(json.loads, open(MANIFEST, encoding="utf-8"))}
        self.order, self.pos, self.neg, self.rows = [], {}, {}, []

    def c_tokens(self, y: np.ndarray) -> list[int]:
        return [int(v) for v in self.P.tokens(self.P.phi_wave(np.asarray(y, dtype=np.float64)))]

    def encode(self, wav_path) -> torch.Tensor:
        import soundfile as sf
        p = Path(str(wav_path))
        assert p.stem.endswith("_voice"), p
        sid = p.stem[: -len("_voice")]
        cond, cv = sid.split("__", 1)
        assert cond in ("a_clean", "a_noisy"), f"G runs on set (a) only: {sid}"
        y, sr = sf.read(str(p), dtype="float64")
        assert sr == 16000
        pos = self.c_tokens(y)
        if cond == "a_noisy":
            g, sr = sf.read(str(REPO / self.gap[cv]), dtype="float64")
            assert sr == 16000 and len(g) == 48000, (cv, len(g))
            neg_src = self.gap[cv]
        else:
            g, neg_src = np.zeros(48000), "digital_silence_3s"
        neg = self.c_tokens(g)
        assert sid not in self.pos, f"voice asked twice for {sid}"
        self.order.append(sid)
        self.pos[sid], self.neg[sid] = pos, neg
        self.rows.append({"id": sid, "pos": pos, "neg": neg, "neg_src": neg_src,
                          "prompt_sha1": hashlib.sha1(p.read_bytes()).hexdigest()})
        return torch.tensor([pos], dtype=torch.long)


class HFCfgLLM:
    """drop-in for vllm.LLM(...).generate(reqs, params) in si_infer's original block loop (n = 1 per request)"""

    def __init__(self, voices: Voices, w: float, rows: int, base_seed: int, model: str, **_):
        from transformers import AutoModelForCausalLM
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.dev = torch.device("cuda:0")
        self.m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(self.dev).eval()
        self.V, self.w, self.rows, self.base = voices, float(w), rows, base_seed
        self.stats = {"requests": 0, "steps": 0, "sec": 0.0}

    def _neg_seq(self, seq: list[int], seed: int) -> list[int]:
        n_it = (seed - self.base) // 1000
        sid = self.V.order[n_it]
        assert seq[3] == START_GLOBAL and seq[36] == END_GLOBAL
        assert [t - GLOBAL_0 for t in seq[4:36]] == self.V.pos[sid], f"request {seed} is not item {sid}"
        return seq[:4] + [GLOBAL_0 + g for g in self.V.neg[sid]] + seq[36:]

    @torch.no_grad()
    def generate(self, reqs, params, use_tqdm=False):
        from experiments.uniss_qwen3_omni_v1.evaluation.si_guard import make_si_guard
        t0 = time.time()
        out = [None] * len(reqs)
        nrow = 2 if self.w != 0 else 1
        per = max(1, self.rows // nrow)
        order = sorted(range(len(reqs)), key=lambda i: len(reqs[i]["prompt_token_ids"]))
        for b0 in range(0, len(order), per):
            J = order[b0:b0 + per]
            nj = len(J)
            pos = [list(reqs[i]["prompt_token_ids"]) for i in J]
            for i in J:
                p = params[i]
                assert p.n == 1 and abs(p.temperature - TEMP) < 1e-9 and abs(p.top_p - TOP_P) < 1e-9 and p.max_tokens == 1500
            seqs = pos + ([self._neg_seq(s, params[i].seed) for s, i in zip(pos, J)] if nrow == 2 else [])
            R, Lm = len(seqs), max(len(s) for s in seqs)
            ids = torch.zeros(R, Lm, dtype=torch.long)
            mask = torch.zeros(R, Lm, dtype=torch.long)
            for n, s in enumerate(seqs):                       # left padding
                ids[n, Lm - len(s):] = torch.tensor(s)
                mask[n, Lm - len(s):] = 1
            ids, mask = ids.to(self.dev), mask.to(self.dev)
            posid = (mask.cumsum(1) - 1).clamp(min=0)
            guards = [make_si_guard(**params[i].extra_args["si_guard"]) for i in J]
            stops = [set(params[i].stop_token_ids) for i in J]
            gens = [torch.Generator().manual_seed(int(params[i].seed)) for i in J]
            outs = [[] for _ in J]
            done = [False] * nj
            o = self.m(input_ids=ids, attention_mask=mask, position_ids=posid, use_cache=True, logits_to_keep=1)
            for step in range(1500):
                lg = o.logits[:, -1].float()
                l = (lg[:nj] + self.w * (lg[:nj] - lg[nj:])) if nrow == 2 else lg.clone()
                for k in range(nj):
                    if not done[k]:
                        l[k] = guards[k](outs[k], l[k])
                u = torch.tensor([float(torch.rand(1, generator=g)) for g in gens], device=self.dev)
                tok = top_p_sample(l, u).tolist()
                for k in range(nj):
                    if done[k]:
                        continue
                    t = int(tok[k])
                    outs[k].append(t)
                    if t in stops[k] or len(outs[k]) >= 1500:
                        done[k] = True
                self.stats["steps"] += 1
                if all(done):
                    break
                nxt = torch.tensor([outs[k][-1] for k in range(nj)] * nrow, device=self.dev)[:, None]
                mask = torch.cat([mask, torch.ones(R, 1, dtype=mask.dtype, device=self.dev)], 1)
                posid = posid[:, -1:] + 1
                o = self.m(input_ids=nxt, attention_mask=mask, past_key_values=o.past_key_values, position_ids=posid,
                           use_cache=True)
            for k, i in enumerate(J):
                out[i] = SimpleNamespace(outputs=[SimpleNamespace(token_ids=outs[k], cumulative_logprob=None)])
            del o
        self.stats["requests"] += len(reqs)
        self.stats["sec"] += time.time() - t0
        log(f"generate: {len(reqs)} requests, {time.time() - t0:.0f}s (w={self.w})")
        return out


def main() -> None:
    argv = sys.argv[1:]

    def pop(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    w = float(pop("--w"))
    rows = int(pop("--rows", "64"))
    seed = int(pop("--seed", "20261021"))
    backend = pop("--backend", "hf")   # vllm: only the C voice hook (the w = 0 backend check's si_infer reference)
    assert backend in ("hf", "vllm") and (backend == "hf" or w == 0)
    argv += ["--seed", str(seed)]
    import vllm
    from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
    voices = Voices("cuda:0")
    BiCodecTokenizer.encode_wav_to_tokens = lambda self, wav_path: voices.encode(wav_path)
    if backend == "hf":
        vllm.LLM = lambda **kw: HFCfgLLM(voices, w, rows, seed, **kw)
    log(f"backend={backend} w={w} rows={rows} seed={seed} args={' '.join(argv)}")
    sys.argv = [sys.argv[0]] + argv
    from experiments.uniss_qwen3_omni_v1.evaluation import si_infer
    si_infer.main()
    out = Path(argv[argv.index("--output") + 1])
    shard = argv[argv.index("--shard") + 1] if "--shard" in argv else "0"
    with open(out / f"voices_{shard}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in voices.rows)
    log(f"done: {len(voices.rows)} items")


if __name__ == "__main__":
    main()
