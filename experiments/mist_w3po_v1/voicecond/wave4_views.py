"""Robust wave 4, diagnosis #4 (reports/mist_w3po_v1/robust_wave4_diag_prereg.zh-CN.md): flip rate of the S1 LA2
committed tokens under 8 FIXED audio views, and how well it separates wrong from right commits (AUROC).

Model: the base Thinker (pretrained_models/Qwen3-Omni-30B-A3B-Instruct, no adapter), HF bf16, loaded and prompted as
wave3_T_thinker.BaseThinker (si_thinker_stream's base prompt). Forward only, one setting.

Objects: every NON-final block of the S1 LA2 commits (eval_outputs/mist_w3po_v1/wave2_S1/thinker_<dir>/blocks_<dir>_w1.jsonl)
of the CoVoST 2 pairs (ids a_{clean,noisy}__<cv id>). The commit moment is rebuilt as si_thinker_stream built the read:
audio[: write_ms], prompt = head + join(units committed before the block) (zh: characters, no spaces; en: words joined by
a space), and the block's text is teacher-forced after it ((" " if en and something was committed) + join(block units)),
its token ids appended to the prompt's ids (the prompt is tokenised alone, as vLLM did). At every block token position
the argmax is taken for 9 rows: the original prefix (backend check against vLLM: agreement of HF argmax with the
committed token) and the 8 views fixed in the prereg, applied to the truncated prefix:
  shift+20  20 ms of zeros prepended          shift-20  first 20 ms cut
  speed0.95 resample_poly(x, 20, 19)          speed1.05 resample_poly(x, 20, 21)
  gain+3    x * 10^(3/20)                     gain-3    x * 10^(-3/20)
  lowshelf+3@300   RBJ low shelf +3 dB, f0 300 Hz, S 1      highshelf-3@3k  RBJ high shelf -3 dB, f0 3 kHz, S 1
Block flip rate = share of the block's tokens whose argmax differs from the committed token under at least one view.
Labels (CPU, --summary): units by wave3_d_cpu.toks; the sentence's non-final committed units are aligned to the T
group's whole-sentence translation of the same condition (wave3_combos/T/runs/base_<cond>_<dir>) by difflib's matching
blocks (the LCS of wave3_d_cpu); a block is "wrong" when more than half of its units are outside the match; blocks
without units are excluded (counted). AUROC of the flip rate (wrong = positive) per direction x condition; main cell
noisy zh2en, gate >= 0.65. A diagnosis, never a commit rule.

Workers claim (dir, cond, shard) units with O_EXCL claim files; outputs .partial -> renamed, never rewritten.
    CUDA_VISIBLE_DEVICES=<card> python -I experiments/mist_w3po_v1/voicecond/wave4_views.py --worker k [--max-gib 66]
    CUDA_VISIBLE_DEVICES=<card> python -I .../wave4_views.py --worker 0 --smoke 3     (3 en2zh noisy sentences, to smoke/)
    python -I .../wave4_views.py --summary                                            (CPU)
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
TRUNS = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/T/runs"
OUT = REPO / "eval_outputs/mist_w3po_v1/wave4_diag/views"
RUNS, CLAIMS = OUT / "runs", OUT / "claims"
DIRS = ("zh2en", "en2zh")
CONDS = ("noisy", "clean")
N_SHARD = 8
SR = 16000
VIEWS = ("shift+20", "shift-20", "speed0.95", "speed1.05", "gain+3", "gain-3", "lowshelf+3@300", "highshelf-3@3k")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} {msg}", flush=True)


def units(text: str, zh: bool) -> list[str]:          # si_thinker_stream.units
    return [c for c in text if not c.isspace()] if zh else text.split()


def join(us: list[str], zh: bool) -> str:             # si_thinker_stream.join
    return "".join(us) if zh else " ".join(us)


def shelf(kind: str, gain_db: float, f0: float):
    """RBJ audio-EQ-cookbook shelving biquad, S = 1."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    c, al = np.cos(w0), np.sin(w0) / 2 * np.sqrt(2.0)
    sA = 2 * np.sqrt(A) * al
    if kind == "low":
        b = [A * ((A + 1) - (A - 1) * c + sA), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sA)]
        a = [(A + 1) + (A - 1) * c + sA, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sA]
    else:
        b = [A * ((A + 1) + (A - 1) * c + sA), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sA)]
        a = [(A + 1) - (A - 1) * c + sA, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sA]
    return np.array(b) / a[0], np.array(a) / a[0]


def views(x: np.ndarray) -> list[np.ndarray]:
    from scipy.signal import lfilter, resample_poly
    g = 10 ** (3 / 20)
    bl, al = shelf("low", 3.0, 300.0)
    bh, ah = shelf("high", -3.0, 3000.0)
    out = [np.concatenate([np.zeros(320, np.float32), x]), x[320:],
           resample_poly(x, 20, 19), resample_poly(x, 20, 21), x * g, x / g,
           lfilter(bl, al, x), lfilter(bh, ah, x)]
    return [np.asarray(v, np.float32) for v in out]


def blocks_of(d: str, cond: str) -> list[dict]:
    """The S1 non-final blocks of set (a) in condition cond, in item order, with the text before them."""
    zh = d.endswith("zh")
    items = {r["id"]: r for r in map(json.loads, open(S1 / f"data/items_{d}.jsonl", encoding="utf-8"))}
    out = []
    for r in map(json.loads, open(S1 / f"thinker_{d}/blocks_{d}_w1.jsonl", encoding="utf-8")):
        if not r["id"].startswith(f"a_{cond}__"):
            continue
        committed = []
        for bi, b in enumerate(r["blocks"]):
            new = units(b["text"], zh)
            if not b.get("final"):
                tgt = ("" if zh or not committed else " ") + join(new, zh)
                out.append({"id": r["id"], "cv": r["id"].split("__", 1)[1], "bi": bi, "write_ms": b["write_ms"],
                            "prefix": join(committed, zh), "target": tgt, "text": b["text"],
                            "audio": items[r["id"]]["audio"]})
            committed += new
    return out


# ------------------------------------------------------------------------------------------------------ GPU
class Engine:
    def __init__(self, max_gib: int):
        import torch
        from experiments.mist_w3po_v1.voicecond.wave3_T_thinker import BaseThinker
        self.torch = torch
        self.bt = BaseThinker(max_gib)

    def block(self, d: str, blk: dict, wav: np.ndarray) -> dict:
        torch, bt = self.torch, self.bt
        x = wav[: int(blk["write_ms"]) * 16].astype(np.float32)
        rows = [x] + views(x)
        rows = [w if len(w) >= 1600 else np.pad(w, (0, 1600 - len(w))) for w in rows]
        inp = bt.proc(text=[bt.head[d] + blk["prefix"]] * len(rows), audio=rows, return_tensors="pt", padding=True)
        tgt = bt.tok(blk["target"], add_special_tokens=False)["input_ids"]
        T = len(tgt)
        if T == 0:
            return {"n_tok": 0}
        tg = torch.tensor(tgt)[None].repeat(len(rows), 1)
        ids = torch.cat([inp["input_ids"], tg], 1)
        am = torch.cat([inp["attention_mask"], torch.ones_like(tg)], 1)
        with torch.no_grad():
            out = bt.thinker(input_ids=ids.to(bt.dev), attention_mask=am.to(bt.dev),
                             input_features=inp["input_features"].to(bt.dev, torch.bfloat16),
                             feature_attention_mask=inp["feature_attention_mask"].to(bt.dev))
        lg = out.logits[:, -T - 1:-1, :]
        am_ = lg.argmax(-1).cpu().numpy()                     # [9, T]
        ref = np.array(tgt)
        diff = am_ != ref[None]                               # [9, T]
        return {"n_tok": T, "orig_agree": int((~diff[0]).sum()), "view_flip": diff[1:].sum(1).astype(int).tolist(),
                "any_flip": int(diff[1:].any(0).sum())}


def wav_cache():
    import soundfile as sf
    cache = {}

    def get(p):
        if p not in cache:
            x, sr = sf.read(str(REPO / p), dtype="float32")
            assert sr == SR, p
            cache[p] = x if x.ndim == 1 else x.mean(1)
        return cache[p]
    return get


def run(eng: Engine, d: str, blks: list[dict], get) -> list[dict]:
    res = []
    for b in blks:
        r = eng.block(d, b, get(b["audio"]))
        res.append({k: b[k] for k in ("id", "cv", "bi", "write_ms", "text")} | r)
    return res


def worker(a) -> None:
    eng = Engine(a.max_gib)
    get = wav_cache()
    log(f"worker {a.worker} ready on {socket.gethostname()} cuda {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if a.smoke:
        d, cond = "en2zh", "noisy"
        blks = blocks_of(d, cond)
        ids = list(dict.fromkeys(b["id"] for b in blks))[: a.smoke]
        res = run(eng, d, [b for b in blks if b["id"] in ids], get)
        p = OUT / "smoke" / f"smoke_{os.getpid()}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for r in res:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tok = sum(r["n_tok"] for r in res)
        log(f"smoke: {len(res)} blocks, {tok} tokens, orig agreement {sum(r.get('orig_agree', 0) for r in res) / max(1, tok):.3f}, "
            f"any-view flip {sum(r.get('any_flip', 0) for r in res) / max(1, tok):.3f} -> {p}")
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    CLAIMS.mkdir(parents=True, exist_ok=True)
    for d in DIRS:
        for cond in CONDS:
            allb = None
            for k in range(N_SHARD):
                name = f"{d}_{cond}_s{k}"
                fin = RUNS / f"{name}.jsonl"
                if fin.exists():
                    continue
                try:
                    fd = os.open(CLAIMS / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, f"{a.worker} {os.getpid()}\n".encode())
                    os.close(fd)
                except FileExistsError:
                    continue
                if allb is None:
                    allb = blocks_of(d, cond)
                    sids = list(dict.fromkeys(b["id"] for b in allb))
                part = set(sids[k::N_SHARD])
                t0 = time.time()
                res = run(eng, d, [b for b in allb if b["id"] in part], get)
                tmp = RUNS / f"{name}.jsonl.partial"
                with open(tmp, "w", encoding="utf-8") as fh:
                    for r in res:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                os.rename(tmp, fin)
                log(f"{name}: {len(res)} blocks in {time.time() - t0:.0f}s")
    log("worker done")


# ------------------------------------------------------------------------------------------------------ CPU
def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    from scipy.stats import rankdata
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summary(a) -> None:
    from experiments.mist_w3po_v1.voicecond.wave3_d_cpu import toks
    outp = OUT / "summary.json"
    if outp.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {outp}")
    res = {"views": list(VIEWS), "gate": "AUROC(noisy zh2en) >= 0.65"}
    for d in DIRS:
        lang = "zh" if d.endswith("zh") else "en"
        for cond in CONDS:
            rows = [json.loads(l) for p in sorted(glob.glob(str(RUNS / f"{d}_{cond}_s*.jsonl"))) for l in open(p, encoding="utf-8")]
            assert len(glob.glob(str(RUNS / f"{d}_{cond}_s*.jsonl"))) == N_SHARD, (d, cond)
            whole = {}
            for p in glob.glob(str(TRUNS / f"base_{cond}_{d}/s*.jsonl")):
                for r in map(json.loads, open(p, encoding="utf-8")):
                    whole[r["id"]] = r["translation"]
            by = {}
            for r in rows:
                by.setdefault(r["cv"], []).append(r)
            lab, fr, sid, excl, notok = [], [], [], 0, 0
            ntok = agree = 0
            vflip = np.zeros(len(VIEWS))
            vrate = [[] for _ in VIEWS]
            for si, (cv, bl) in enumerate(sorted(by.items())):
                bl.sort(key=lambda r: r["bi"])
                us = [toks(b["text"], lang) for b in bl]
                flat = [u for x in us for u in x]
                w = toks(whole[cv], lang)
                matched = np.zeros(len(flat), bool)
                for m in difflib.SequenceMatcher(None, flat, w, autojunk=False).get_matching_blocks():
                    matched[m.a: m.a + m.size] = True
                pos = 0
                for b, u in zip(bl, us):
                    seg = matched[pos: pos + len(u)]
                    pos += len(u)
                    if b["n_tok"] == 0:
                        notok += 1
                        continue
                    ntok += b["n_tok"]
                    agree += b["orig_agree"]
                    vflip += np.array(b["view_flip"])
                    for j, v in enumerate(b["view_flip"]):
                        vrate[j].append(v / b["n_tok"])
                    if not len(u):
                        excl += 1
                        continue
                    lab.append((~seg).mean() > 0.5)
                    fr.append(b["any_flip"] / b["n_tok"])
                    sid.append(si)
            lab, fr, sid = np.array(lab), np.array(fr), np.array(sid)
            A = auroc(fr, lab)
            rng = np.random.default_rng(0)
            us_ = np.unique(sid)
            grp = {s: np.where(sid == s)[0] for s in us_}
            bs = []
            for _ in range(2000):
                ix = np.concatenate([grp[s] for s in rng.choice(us_, len(us_))])
                bs.append(auroc(fr[ix], lab[ix]))
            bs = np.array(bs)
            bs = bs[~np.isnan(bs)]
            res[f"{d}_{cond}"] = {
                "n_sentences": len(by), "n_blocks": int(len(lab)), "n_wrong": int(lab.sum()), "n_right": int((~lab).sum()),
                "excluded_no_units": excl, "excluded_no_tokens": notok,
                "auroc": A, "auroc_ci95_sentence_bootstrap": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "flip_rate_mean": {"wrong": float(fr[lab].mean()) if lab.any() else None,
                                   "right": float(fr[~lab].mean()) if (~lab).any() else None},
                "backend_check_hf_vs_vllm_token_agreement": agree / max(1, ntok),
                "per_view_token_flip_rate": {v: float(vflip[j] / max(1, ntok)) for j, v in enumerate(VIEWS)},
                "per_view_block_mean_flip_rate": {v: float(np.mean(vrate[j])) if vrate[j] else None for j, v in enumerate(VIEWS)},
                "n_tokens": int(ntok)}
            log(f"{d} {cond}: AUROC {A:.3f} blocks {len(lab)} (wrong {int(lab.sum())}) agree {agree / max(1, ntok):.3f}")
    res["gate_pass"] = bool(res["zh2en_noisy"]["auroc"] >= 0.65)
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"gate (noisy zh2en AUROC >= 0.65): {res['gate_pass']} -> {outp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--max-gib", type=int, default=66)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(a)
    else:
        assert a.worker >= 0
        worker(a)


if __name__ == "__main__":
    main()
