"""Robust wave 3, group T (reports/mist_w3po_v1/robust_wave3_single_prereg.zh-CN.md, incl. change 1): the wave-1 Thinker
prechecks P2a, K2 (P2b), K3, K7, K8 re-run on the single-speaker base's Thinker -- the ORIGINAL Qwen3-Omni-30B-A3B-Instruct
(pretrained_models/Qwen3-Omni-30B-A3B-Instruct, no adapter), sentence level: each sentence is read ONCE, whole, with
si_thinker_stream's base prompt (E1 instruction, inference_instruction()[tgt], chat template with the audio) and greedy
decoding (max 160 new tokens, stop at <|im_end|> 151645); the translation is normalised as si_thinker_stream joins units
(zh: whitespace dropped, en: words joined by one space). HF transformers, bf16; every arm runs in the same backend.

Data, constants and statistics of wave 1 are reused unchanged (eval_outputs/mist_w3po_v1/wave1_K2K3K5/): the CoVoST 2
validation manifest (300 sentences per direction: clean / noisy / bgonly / gap), the Idea 2 lambda (data/lambda.jsonl,
incl. the HDemucs vocals path), K8's sigma2_a (data/k8cal.json), K3's training-silence log-mel mean (data/k3_silence.npz)
and K7's calibration statistics (k7_stats.pt). The base Thinker's audio encoder is bitwise identical to wave 1's
B Thinker (525 / 525 audio_tower tensors equal; same preprocessor_config) and the feature extractor is the same, so
these encoder / log-mel statistics are the base model's own.

Arms (one setting each; change 1: the K8 reference arms pure-HDemucs and alpha = 0.5 are cancelled, no lin unit):
  base   clean / noisy (P2a = text BLEU clean - noisy)
  p2b    K2 background-contrastive decoding: rows [main; bg] in one batch, every step
         l = l_main + w (l_main - l_bg) over the tokens with p_main >= 0.1 max p_main, w = min(3, (1 - lam) / lam),
         lam = Idea 2 lambda of that condition's sentence; the bg row is forced to the main row's token (same prefix).
         noisy: bg = the sentence's equal-length background-only clip; clean: bg = equal-length digital silence
         (the clean twin's background), lam of the clean sentence
  k3     delta = mean(log-mel of the 3 s gap, frames 0..294 of [gap; x]) - training-silence mean, subtracted from every
         valid input frame; clean: the gap is 3 s of digital silence (as K7's clean condition and the clean twin)
  k7     online soft subspace Wiener on audio-encoder layers 16-31 (wave 1's k7_projectors, identical maths);
         clean: 3 s digital-silence gap
  k8     x = a* HDemucs(x) + (1 - a*) x, a* = s2n / (s2n + s2a), s2n = (1 - lam) / lam
The clean conditions of k3 / p2b (not in wave 1) are added because the wave-3 gate asks every arm for clean input
not to lose more than 0.2 BLEU; they are the same method applied to clean input, not another setting.

Workers pull (arm, cond, dir, shard) units from one queue (O_EXCL claim files); outputs .partial -> renamed, never
rewritten. Output root eval_outputs/mist_w3po_v1/wave3_combos/T/.

    CUDA_VISIBLE_DEVICES=<card> python -I experiments/mist_w3po_v1/voicecond/wave3_T_thinker.py --worker k [--max-gib 66]
    CUDA_VISIBLE_DEVICES=<card> python -I .../wave3_T_thinker.py --worker 0 --smoke 4      (every arm, 4 en2zh sentences)
    python -I .../wave3_T_thinker.py --summary                                            (CPU: BLEU, CIs, gates)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from experiments.mist_w3po_v1.voicecond import wave1_thinker as W1   # noqa: E402  (constants, k7 projector maths)

QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
W1OUT = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5"
DATA = W1OUT / "data"
OUT = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/T"
RUNS, CLAIMS = OUT / "runs", OUT / "claims"
DIRS = ("en2zh", "zh2en")
N_SHARD = 8
THINKER_EOS = 151645
MAX_NEW = 160
SR = 16000
ARMS = [("base", "clean"), ("base", "noisy"), ("p2b", "noisy"), ("k3", "noisy"), ("k7", "noisy"), ("k8", "noisy"),
        ("p2b", "clean"), ("k3", "clean"), ("k7", "clean"), ("k8", "clean")]
TGT = {"en2zh": "cmn", "zh2en": "eng"}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} {msg}", flush=True)


def norm_text(t: str, d: str) -> str:
    return "".join(c for c in t if not c.isspace()) if d == "en2zh" else " ".join(t.split())


# ----------------------------------------------------------------------------------------------------------- engine
class BaseThinker:
    """The original Qwen3-Omni Thinker (HF) with the K3 feature offset, the K7 layer projector (forward hooks on the
    upper encoder layers) and the K2 contrastive logits processor."""

    def __init__(self, max_gib: int):
        from transformers import Qwen3OmniMoeProcessor
        from experiments.mist_w3po_v1.core.engine import load_thinker
        from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
        self.thinker = load_thinker(str(QWEN3), max_gib)
        off = [nm for nm, p in self.thinker.named_parameters() if p.device.type != "cuda"]
        assert not off, f"{len(off)} thinker parameters not on the GPU, e.g. {off[:3]}"
        self.dev = torch.device("cuda")
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(str(QWEN3))
        self.proc.tokenizer.padding_side = "left"
        self.tok = self.proc.tokenizer
        ins = inference_instruction()
        self.head = {}
        for d in DIRS:
            conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                                 {"type": "text", "text": ins[TGT[d]]}]}]
            self.head[d] = self.proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
        self._ctx = None
        self._cap = None
        enc = self.thinker.audio_tower
        for li in W1.UPPER:
            enc.layers[li].register_forward_hook(self._hook(li))

    def _hook(self, li):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if self._cap is not None:
                self._cap[li].append(h.detach().float())
            if self._ctx is None:
                return None
            lens, Ps = self._ctx
            assert sum(lens) == h.shape[0], (lens, h.shape)
            hn, pos = h.clone(), 0
            for n, P in zip(lens, Ps):
                if P is not None:
                    x = h[pos: pos + n].float()
                    hn[pos: pos + n] = (x - x @ P[li]).to(h.dtype)
                pos += n
            return (hn,) + tuple(output[1:]) if isinstance(output, tuple) else hn
        return fn

    # wave1_thinker.k7_projectors / k3_delta expect eng.encode(...) and eng.pr.proc.feature_extractor
    @property
    def pr(self):
        class _P:
            proc = self.proc
        return _P

    @torch.no_grad()
    def encode(self, wavs, feat_fn=None, Ps=None, capture=False):
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths
        f = self.proc.feature_extractor(wavs, sampling_rate=SR, padding=True, truncation=False,
                                        return_attention_mask=True, return_tensors="pt")
        feats, am = f["input_features"], f["attention_mask"]
        n_audio = _get_feat_extract_output_lengths(am.sum(1)).tolist()
        if capture:
            self._cap = {li: [] for li in W1.UPPER}
        try:
            a = self.thinker.get_audio_features(feats.to(self.dev, torch.bfloat16), am.to(self.dev)).last_hidden_state
        finally:
            cap, self._cap = self._cap, None
        return a, n_audio, feats, am, cap

    @torch.no_grad()
    def generate(self, d: str, wavs: list[np.ndarray], deltas=None, Ps=None, cd=None) -> list[dict]:
        """one whole-sentence read per wav; deltas: per-row log-mel offsets (k3); Ps: per-row projectors (k7);
        cd: (bg_wavs, ws) -> rows [main; bg], contrastive logits (k2)"""
        from transformers import LogitsProcessor, LogitsProcessorList
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths
        B = len(wavs)
        rows = [w if len(w) >= 1600 else np.pad(w, (0, 1600 - len(w))) for w in wavs]
        if cd is not None:
            rows += [w if len(w) >= 1600 else np.pad(w, (0, 1600 - len(w))) for w in cd[0]]
        inp = self.proc(text=[self.head[d]] * len(rows), audio=rows, return_tensors="pt", padding=True)
        feats, fam = inp["input_features"], inp["feature_attention_mask"]
        if deltas is not None:
            for j in range(B):
                n = int(fam[j].sum())
                feats[j, :, :n] -= torch.as_tensor(deltas[j], dtype=feats.dtype)[:, None]
        inputs = {k: v.to(self.dev) for k, v in inp.items()}
        inputs["input_features"] = feats.to(self.dev, torch.bfloat16)
        lp = None
        stats = {"steps": 0, "changed": 0}
        if cd is not None:
            wv = torch.tensor(cd[1], device=self.dev, dtype=torch.float32)[:, None]

            class CD(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    lm, lb = scores[:B].float(), scores[B:].float()
                    pm = torch.softmax(lm, -1)
                    ok = pm >= W1.PLAUS * pm.max(-1, keepdim=True).values
                    comb = torch.where(ok, lm + wv * (lm - lb), torch.full_like(lm, -float("inf")))
                    nxt = comb.argmax(-1)
                    stats["steps"] += B
                    stats["changed"] += int((nxt != lm.argmax(-1)).sum())
                    out = torch.full_like(scores, -float("inf"))
                    ar = torch.arange(B, device=scores.device)
                    out[ar, nxt] = 0.0
                    out[ar + B, nxt] = 0.0
                    return out
            lp = LogitsProcessorList([CD()])
        if Ps is not None:
            lens = _get_feat_extract_output_lengths(fam.sum(1)).tolist()
            self._ctx = (lens, list(Ps) + [None] * (len(rows) - len(Ps)))
        try:
            out = self.thinker.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=THINKER_EOS,
                                        pad_token_id=self.tok.pad_token_id, logits_processor=lp)
        finally:
            self._ctx = None
        gen = out[:, inputs["input_ids"].shape[1]:].tolist()
        res = []
        for j in range(B):
            ids = gen[j]
            if THINKER_EOS in ids:
                ids = ids[: ids.index(THINKER_EOS)]
            ids = [i for i in ids if i != self.tok.pad_token_id]
            res.append({"ids": ids, "raw": self.tok.decode(ids, skip_special_tokens=True),
                        "hit_max": len(gen[j]) >= MAX_NEW and THINKER_EOS not in gen[j]})
        if cd is not None:
            for r in res:
                r["cd_batch"] = {k: v for k, v in stats.items()}
        return res


# ------------------------------------------------------------------------------------------------------ units
def manifest():
    return W1.manifest()


def units():
    out = []
    for k in range(N_SHARD):
        for d in DIRS:
            for a, c in ARMS:
                out.append((a, c, d, k))
    return out


def uname(u) -> str:
    return f"{u[0]}_{u[1]}_{u[2]}/s{u[3]}"


def out_path(u, root: Path = RUNS) -> Path:
    return root / f"{u[0]}_{u[1]}_{u[2]}" / f"s{u[3]}.jsonl"


def run_unit(eng: BaseThinker, u, M, LAM, cache: dict, batch: int, sents=None) -> list[dict]:
    a, c, d, k = u
    if sents is None:
        sents = [x for i, x in enumerate(M[d]) if i % N_SHARD == k]
    base_w = {x["id"]: W1.wav(x[c]) for x in sents}
    extra = {}
    if a == "k8":
        s2a = json.load(open(DATA / "k8cal.json"))["sigma2_a"]
        for x in sents:
            lr = LAM[(x["id"], c)]
            vh = W1.wav(lr["vhat"])[: len(base_w[x["id"]])]
            s2n = (1 - lr["lam"]) / lr["lam"]
            al = s2n / (s2n + s2a)
            base_w[x["id"]] = (al * vh + (1 - al) * base_w[x["id"]]).astype(np.float32)
            extra[x["id"]] = {"alpha": round(al, 5), "lam": lr["lam"]}
    res = []
    for i in range(0, len(sents), batch):
        part = sents[i: i + batch]
        wavs = [base_w[x["id"]] for x in part]
        if a in ("base", "k8"):
            outs = eng.generate(d, wavs)
        elif a == "k3":
            mu = np.load(DATA / "k3_silence.npz")["mean"]
            dl = []
            for x, w in zip(part, wavs):
                gap = W1.wav(x["gap"]) if c == "noisy" else np.zeros(int(3.0 * SR), np.float32)
                dl.append(W1.k3_delta(eng, gap, w, mu))
                extra[x["id"]] = {"delta_l2": round(float(np.linalg.norm(dl[-1])), 4)}
            outs = eng.generate(d, wavs, deltas=dl)
        elif a == "k7":
            if "k7" not in cache:
                cache["k7"] = torch.load(W1OUT / "k7_stats.pt", map_location="cpu")
            Ps = []
            for x in part:
                gap = W1.wav(x["gap"]) if c == "noisy" else np.zeros(int(3.0 * SR), np.float32)
                P, inf = W1.k7_projectors(eng, cache["k7"], gap)
                Ps.append(P)
                extra[x["id"]] = {"k7": {"n_gap": inf["n_gap"], "r_mean": round(float(np.mean(list(inf["r"].values()))), 2),
                                         "r95_mean": round(float(np.mean(list(inf["r95"].values()))), 2)}}
            outs = eng.generate(d, wavs, Ps=Ps)
            del Ps
        elif a == "p2b":
            bgs, ws = [], []
            for x, w in zip(part, wavs):
                bgs.append(W1.wav(x["bgonly"])[: len(w)] if c == "noisy" else np.zeros(len(w), np.float32))
                if len(bgs[-1]) < len(w):
                    bgs[-1] = np.pad(bgs[-1], (0, len(w) - len(bgs[-1])))
                lam = LAM[(x["id"], c)]["lam"]
                ws.append(min(W1.W_CAP, (1 - lam) / max(lam, 1e-6)))
                extra[x["id"]] = {"lam": lam, "w": round(ws[-1], 5)}
            outs = eng.generate(d, wavs, cd=(bgs, ws))
        else:
            raise ValueError(a)
        for x, o in zip(part, outs):
            res.append({"id": x["id"], "dir": d, "arm": a, "cond": c, "translation": norm_text(o["raw"], d),
                        "raw": o["raw"], "n_tok": len(o["ids"]), "hit_max": o["hit_max"],
                        **({"cd_batch": o["cd_batch"]} if "cd_batch" in o else {}), **extra.get(x["id"], {})})
    return res


# --------------------------------------------------------------------------------------------------------- worker
def claim(u, wid: int) -> bool:
    p = CLAIMS / (uname(u).replace("/", "__") + ".claim")
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"worker {wid} pid {os.getpid()} host {socket.gethostname()} {time.strftime('%F %T')}\n".encode())
    os.close(fd)
    return True


def worker(a) -> None:
    for p in (RUNS, CLAIMS):
        p.mkdir(parents=True, exist_ok=True)
    M, LAM = manifest(), W1.lambdas()
    eng = BaseThinker(a.max_gib)
    log(f"worker {a.worker} ready on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"{torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB")
    if a.smoke:
        sm = OUT / f"smoke_{os.getpid()}"
        sm.mkdir(parents=True)
        for arm, c in ARMS:
            t0 = time.time()
            res = run_unit(eng, (arm, c, "en2zh", 0), M, LAM, {}, a.batch, [dict(x) for x in M["en2zh"][: a.smoke]])
            with open(sm / f"{arm}_{c}_en2zh.jsonl", "x", encoding="utf-8") as fh:
                for r in res:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"smoke {arm} {c}: {len(res)} rows in {time.time() - t0:.0f}s; peak "
                f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB; first: {res[0]['translation'][:80]}")
        return
    cache, n_done = {}, 0
    for u in units():
        if not claim(u, a.worker):
            continue
        op = out_path(u)
        if op.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {op}")
        op.parent.mkdir(parents=True, exist_ok=True)
        tmp = op.with_name(op.name + ".partial")
        if tmp.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {tmp}")
        t0 = time.time()
        log(f"start {uname(u)}")
        try:
            res = run_unit(eng, u, M, LAM, cache, a.batch)
            with open(tmp, "x", encoding="utf-8") as fh:
                for r in res:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception:   # noqa: BLE001  the unit stays claimed (no output); FAILED marker for the record
            import traceback
            tb = traceback.format_exc()
            log(f"FAILED {uname(u)}:\n{tb}")
            with open(CLAIMS / (uname(u).replace("/", "__") + f".FAILED.{os.getpid()}"), "x") as fh:
                fh.write(tb)
            torch.cuda.empty_cache()
            continue
        os.rename(tmp, op)
        n_done += 1
        log(f"done {uname(u)} in {time.time() - t0:.0f}s")
    log(f"worker {a.worker}: queue empty after {n_done} units")


# -------------------------------------------------------------------------------------------------------- summary
def summary() -> None:
    from sacrebleu.metrics import BLEU, CHRF
    M = manifest()
    H = {}
    for u in units():
        p = out_path(u)
        if not p.exists():
            raise SystemExit(f"missing {p}")
        for r in map(json.loads, open(p, encoding="utf-8")):
            H[(r["arm"], r["cond"], r["dir"], r["id"])] = r
    rng = np.random.default_rng(0)
    NB = 2000
    S = {"n_boot": NB, "dirs": {}}

    def stats(d, hyp, ref):
        """per-sentence BLEU sufficient statistics (sacrebleu corpus BLEU is sum-decomposable)"""
        bl = BLEU(tokenize="zh" if d == "en2zh" else "13a")
        st = np.array([bl._extract_corpus_statistics([h], [[r]])[0] for h, r in zip(hyp, ref)], dtype=np.float64)
        return bl, st

    def bleu_of(bl, st):
        return bl._compute_score_from_stats(list(st.sum(0))).score

    for d in DIRS:
        ids = [x["id"] for x in M[d]]
        ref = [x["ref"] for x in M[d]]
        sts, bls = {}, {}
        D = {}
        for a, c in ARMS:
            hyp = [H[(a, c, d, i)]["translation"] for i in ids]
            bl, st = stats(d, hyp, ref)
            sts[(a, c)], bls[(a, c)] = st, bl
            D[f"{a}_{c}"] = {"bleu": round(bleu_of(bl, st), 2),
                             "chrf": round(CHRF().corpus_score(hyp, [ref]).score, 2),
                             "hit_max": int(sum(H[(a, c, d, i)]["hit_max"] for i in ids))}
        idx = rng.integers(0, len(ids), size=(NB, len(ids)))

        def diff(x, y):   # BLEU(x) - BLEU(y), paired bootstrap over sentences
            bx, by = bls[x], bls[y]
            pt = bleu_of(bx, sts[x]) - bleu_of(by, sts[y])
            bs = np.array([bleu_of(bx, sts[x][j]) - bleu_of(by, sts[y][j]) for j in idx])
            return {"delta": round(pt, 2), "ci": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]}
        D["p2a_loss"] = diff(("base", "clean"), ("base", "noisy"))
        for a in ("p2b", "k3", "k7", "k8"):
            for c in ("noisy", "clean"):
                D[f"{a}_{c}_vs_base"] = diff((a, c), ("base", c))
        S["dirs"][d] = D
    # prereg change 2 (written after S1, before any T number): symmetric by direction. zh2en (the base loses there):
    # noisy delta CI lower > 0 and recovery >= 40 % of T's OWN zh2en P2a loss (same backend, same protocol; a loss
    # <= 0 leaves nothing to recover -> the condition fails); recovery against S1's 3.22 listed for reference.
    # en2zh: noisy >= -0.2; clean >= -0.2 both directions. The original (wave-1 direction) gates are listed too.
    lz = S["dirs"]["zh2en"]["p2a_loss"]["delta"]
    G2 = {}
    for a in ("p2b", "k3", "k7", "k8"):
        e = S["dirs"]["zh2en"][f"{a}_noisy_vs_base"]
        g = {"zh2en_noisy_ci_lower_gt0": e["ci"][0] > 0,
             "zh2en_recovery": round(e["delta"] / lz, 3) if lz > 0 else None,
             "zh2en_recovery_vs_S1_3.22": round(e["delta"] / 3.22, 3),
             "en2zh_noisy_ge_-0.2": S["dirs"]["en2zh"][f"{a}_noisy_vs_base"]["delta"] >= -0.2,
             "en2zh_clean_ge_-0.2": S["dirs"]["en2zh"][f"{a}_clean_vs_base"]["delta"] >= -0.2,
             "zh2en_clean_ge_-0.2": S["dirs"]["zh2en"][f"{a}_clean_vs_base"]["delta"] >= -0.2}
        g["zh2en_recovery_ge_0.4"] = lz > 0 and g["zh2en_recovery"] >= 0.4
        g["pass"] = all(v for k, v in g.items() if isinstance(v, bool))
        G2[a] = g
    S["gates_change2"] = G2
    loss = S["dirs"]["en2zh"]["p2a_loss"]["delta"]
    G = {}
    for a in ("p2b", "k3", "k7", "k8"):
        e = S["dirs"]["en2zh"][f"{a}_noisy_vs_base"]
        g = {"en2zh_noisy_ci_lower_gt0": e["ci"][0] > 0,
             "zh2en_noisy_ge_-0.2": S["dirs"]["zh2en"][f"{a}_noisy_vs_base"]["delta"] >= -0.2,
             "en2zh_clean_ge_-0.2": S["dirs"]["en2zh"][f"{a}_clean_vs_base"]["delta"] >= -0.2,
             "zh2en_clean_ge_-0.2": S["dirs"]["zh2en"][f"{a}_clean_vs_base"]["delta"] >= -0.2}
        if loss >= 0.5:
            g["recovery"] = round(e["delta"] / loss, 3)
            g["recovery_ge_0.4"] = g["recovery"] >= 0.4
        g["pass"] = all(v for k, v in g.items() if isinstance(v, bool))
        G[a] = g
    S["gates_original"] = G
    S["p2a_en2zh_loss"] = loss
    S["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    op = OUT / "summary.json"
    if op.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {op}")
    with open(op, "x", encoding="utf-8") as fh:
        json.dump(S, fh, indent=1, ensure_ascii=False)
    print(json.dumps(S, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--max-gib", type=int, default=66)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    else:
        assert a.worker >= 0
        worker(a)


if __name__ == "__main__":
    main()
