"""si_infer with the K5 soft voice condition (reports/mist_w3po_v1/robust_k5_softvoice_prereg.zh-CN.md section 3).
experiments.uniss_qwen3_omni_v1.evaluation.si_infer is run unchanged (same flags, item order, seeds); three hooks:

  * always: BiCodecTokenizer.encode_wav_to_tokens -> lookup in the K5 table (k5_tokens.py rows {key, id, sha1, ids, b}),
    keyed by the prompt file stem with sha1 check (as idea1_si_infer_vo.py), returning the V1 hard ids; the call order
    gives item index n_it.
  * --k5-lm 1: vllm.LLM is replaced by a subclass with enable_prompt_embeds=True whose generate() turns every
    {"prompt_token_ids": seq} into {"prompt_embeds": E}: E = embed_tokens[seq] (bf16) with positions 4..35 (the 32
    GLOBAL tokens between START_GLOBAL and END_GLOBAL) replaced by the expectation of the global-token embeddings under
    the barycentric FSQ distribution of b (k5_soft.corners; fp32 sum, then bf16). The request's item is n_it =
    (seed - base_seed) // 1000 (si_infer's per-request seed = seed + 1000 n_it + r); its 32 ids are checked against the
    table (mismatch raises).
  * --k5-dec 1: BiCodecTokenizer.decode_tokens_to_audio uses d = k5_soft.d_cont(b) (detokenize without the round)
    instead of speaker_encoder.detokenize(ids); the item is found by its 32 ids (ties broken by its generated codes).
  * --k5-sanity 1: b is replaced by the hard ids' level centres (soft embedding / d-vector reduce to the hard ones).

    python -m experiments.mist_w3po_v1.voicecond.k5_si_infer_soft --k5-table <tokens_k5_<d>.jsonl> --k5-lm 0|1
        --k5-dec 0|1 [--k5-sanity 1] <si_infer args...>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def pop_opt(argv, name, default=None):
    if name not in argv:
        return default
    i = argv.index(name)
    v = argv[i + 1]
    del argv[i:i + 2]
    return v


def main() -> None:
    argv = sys.argv[1:]
    table_path = pop_opt(argv, "--k5-table")
    use_lm = int(pop_opt(argv, "--k5-lm", "0"))
    use_dec = int(pop_opt(argv, "--k5-dec", "0"))
    sanity = int(pop_opt(argv, "--k5-sanity", "0"))
    assert table_path, "--k5-table required"
    base_seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 20261021
    model_dir = argv[argv.index("--model") + 1]
    from experiments.mist_w3po_v1.voicecond.k5_soft import centres, corners, d_cont
    table, bs = {}, {}
    for line in open(table_path, encoding="utf-8"):
        r = json.loads(line)
        assert len(r["ids"]) == 32 and all(0 <= int(v) < 4096 for v in r["ids"]), r["key"]
        assert r["key"] not in table, f"duplicate key {r['key']}"
        table[r["key"]] = r
        bs[r["key"]] = centres(r["ids"]) if sanity else np.asarray(r["b"], dtype=np.float64)
    import torch
    from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import SEMANTIC_0, SEMANTIC_LAST
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import END_GLOBAL, GLOBAL_0, START_GLOBAL
    order: list[str] = []
    codes_acc: dict[int, dict[int, list[int]]] = {}
    stats = {"encode": 0, "reqs": 0, "embeds": 0, "decodes": 0, "decode_ties": 0, "sanity_checks": 0}

    def encode_k5(self, wav_path):
        key = Path(str(wav_path)).stem
        r = table.get(key)
        if r is None:
            raise RuntimeError(f"k5: no tokens for {wav_path}")
        sha = hashlib.sha1(Path(str(wav_path)).read_bytes()).hexdigest()
        if sha != r["sha1"]:
            raise RuntimeError(f"k5: prompt {wav_path} sha1 {sha} != {r['sha1']}")
        order.append(key)
        stats["encode"] += 1
        return torch.tensor([[int(v) for v in r["ids"]]], dtype=torch.long)

    BiCodecTokenizer.encode_wav_to_tokens = encode_k5

    if use_lm or use_dec:
        import vllm
        Base = vllm.LLM
        emb_box, soft_cache = {}, {}

        def soft_of(key):
            if key not in soft_cache:
                if "E" not in emb_box:
                    from safetensors import safe_open
                    with safe_open(str(Path(model_dir) / "model.safetensors"), framework="pt") as fh:
                        emb_box["E"] = fh.get_tensor("model.embed_tokens.weight")      # (V, H) bf16, CPU
                E = emb_box["E"]
                idx, w = corners(bs[key])
                rows = E[torch.as_tensor(GLOBAL_0 + idx.reshape(-1))].float().reshape(32, 64, -1)
                s = (rows * torch.as_tensor(w, dtype=torch.float32)[..., None]).sum(1).to(E.dtype)
                if sanity:
                    hard = E[torch.as_tensor([GLOBAL_0 + int(v) for v in table[key]["ids"]])]
                    assert torch.equal(s, hard), f"k5 sanity: soft embedding != hard embedding for {key}"
                    stats["sanity_checks"] += 1
                soft_cache[key] = s
            return soft_cache[key]

        class K5LLM(Base):
            def __init__(self, *a, **kw):
                if use_lm:
                    kw["enable_prompt_embeds"] = True
                super().__init__(*a, **kw)

            def generate(self, prompts, sampling_params=None, *a, **kw):
                ps = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
                where, new = [], []
                for req, p in zip(prompts, ps):
                    assert p.n == 1, "k5: n must be 1"
                    n_it, r = divmod(int(p.seed) - base_seed, 1000)
                    key = order[n_it]
                    seq = req["prompt_token_ids"]
                    if not (seq[3] == START_GLOBAL and seq[36] == END_GLOBAL
                            and [t - GLOBAL_0 for t in seq[4:36]] == table[key]["ids"]):
                        raise RuntimeError(f"k5: request seed {p.seed} -> item {n_it} {key}: global ids differ")
                    where.append((n_it, r))
                    if use_lm:
                        s = soft_of(key)                      # also loads embed_tokens on first use
                        x = emb_box["E"][torch.as_tensor(seq)].clone()
                        x[4:36] = s
                        new.append({"prompt_embeds": x})
                        stats["embeds"] += 1
                    else:
                        new.append(req)
                    stats["reqs"] += 1
                res = super().generate(new, sampling_params, *a, **kw)
                for (n_it, r), o in zip(where, res):
                    codes_acc.setdefault(n_it, {})[r] = [t - SEMANTIC_0 for t in o.outputs[0].token_ids
                                                         if SEMANTIC_0 <= t <= SEMANTIC_LAST]
                return res

        vllm.LLM = K5LLM

    if use_dec:
        orig_dec = BiCodecTokenizer.decode_tokens_to_audio

        @torch.no_grad()
        def decode_k5(self, tokens):
            g = [int(v) for v in tokens[:32].tolist()]
            sem = tokens[32:]
            cands = [n for n, k in enumerate(order) if table[k]["ids"] == g]
            if len({bs[order[n]].tobytes() for n in cands}) > 1:
                seml = [int(v) for v in sem.tolist()]
                cands = [n for n in cands if [c for r in sorted(codes_acc.get(n, {})) for c in codes_acc[n][r]] == seml]
                stats["decode_ties"] += 1
            if not cands or len({bs[order[n]].tobytes() for n in cands}) != 1:
                raise RuntimeError(f"k5: decode cannot identify the item ({len(cands)} candidates)")
            key = order[cands[0]]
            m = self.model
            d = d_cont(m.speaker_encoder, bs[key])
            z_q = m.quantizer.detokenize(sem.unsqueeze(0).to(self.device))
            x = m.prenet(z_q, d)
            x = x + d.unsqueeze(-1)
            wav = m.decoder(x).detach().squeeze().cpu().numpy()
            if sanity and stats["decodes"] < 5:
                # the gate is on the decoder condition (exact); the waveform differs at ~1e-4 even between two hard
                # decodes (non-deterministic GPU kernels), so it is printed with that floor, not asserted
                dh = m.speaker_encoder.detokenize(torch.as_tensor(g, dtype=torch.long, device=d.device)[None, None])
                dd = float((d - dh).abs().max())
                assert dd <= 1e-6, f"k5 sanity: d-vector differs from detokenize by {dd}"
                ref1, ref2 = np.asarray(orig_dec(self, tokens)), np.asarray(orig_dec(self, tokens))
                print(f"[k5 sanity] decode {key}: d-vector max abs diff {dd:.3g}; waveform max abs diff to hard decode "
                      f"{float(np.abs(ref1 - wav).max()):.3g} (hard vs hard {float(np.abs(ref1 - ref2).max()):.3g})", flush=True)
            stats["decodes"] += 1
            return wav

        BiCodecTokenizer.decode_tokens_to_audio = decode_k5

    print(f"[k5] {len(table)} entries from {table_path}; lm={use_lm} dec={use_dec} sanity={sanity}", flush=True)
    sys.argv = [sys.argv[0]] + argv
    from experiments.uniss_qwen3_omni_v1.evaluation import si_infer
    si_infer.main()
    print(f"[k5] stats {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
