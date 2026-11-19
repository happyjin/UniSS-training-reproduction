"""Wave 3 LA2 commit diagnosis, D-CPU (reports/mist_w3po_v1/robust_wave3_la2diag_prereg.zh-CN.md).

Inputs (all existing, nothing rerun): the S1 LA2 commits and drafts (wave2_S1/thinker_<dir>/blocks_<dir>_w1.jsonl), the T
group's whole-sentence base-Thinker translations (wave3_combos/T/runs/base_<cond>_<dir>/s*.jsonl, HF, greedy), the
wave-1 references (wave1_K2K3K5/data/manifest.jsonl) and the S1 meta (wave2_S1/data/meta_<dir>.jsonl).
Per direction x condition (clean / noisy), set (a) CoVoST pairs: corpus BLEU of LA2 and of the whole-sentence read (both through evaluation.text_metrics.normalize_for_bleu, as S1 scores), the
streaming extra loss, commit regret (share of committed tokens before the final block outside the LCS with the
whole-sentence translation of the same audio), first write, number of commits, committed share before the source ends,
draft jitter (mean normalised token edit distance of consecutive drafts); noisy - clean with a paired bootstrap (2000,
seed 0) over sentences; Spearman between the sentence chrF drop and the regret increase. Set (b): timing and jitter only
(no whole-sentence read there), all turns and turns without crosstalk.

    <uniss-eval python> experiments/mist_w3po_v1/voicecond/wave3_d_cpu.py --out eval_outputs/mist_w3po_v1/wave3_combos/D/cpu.json
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import re
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evaluation.text_metrics import normalize_for_bleu  # noqa: E402  (S1 scoring convention)

REPO = Path(__file__).resolve().parents[3]
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
T = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/T/runs"
MAN = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5/data/manifest.jsonl"
TGT = {"en2zh": "zh", "zh2en": "en"}
B = 2000


def toks(s: str, lang: str) -> list:
    if lang == "zh":
        return [c for c in s if not c.isspace() and not re.match(r"[，。、！？；：,.!?;:\"'“”‘’《》()（）\-]", c)]
    return re.findall(r"[a-z0-9']+", s.lower())


def lcs(a: list, b: list) -> int:
    return sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())


def edit_norm(a: list, b: list) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def commit_stats(r: dict, dur_ms: float, lang: str) -> dict:
    bl = r["blocks"]
    nonfinal = "".join(b["text"] for b in bl if not b.get("final"))
    all_t = toks(r["translation"], lang)
    before = sum(len(toks(b["text"], lang)) for b in bl if b["write_ms"] < dur_ms - 1)
    dr = [toks(d["draft"], lang) for d in r.get("drafts", [])]
    jit = [edit_norm(dr[i], dr[i + 1]) for i in range(len(dr) - 1)]
    return {"first_ms": bl[0]["write_ms"] if bl else np.nan, "n_commits": len(bl), "nonfinal": nonfinal,
            "pre_end_share": before / max(1, len(all_t)), "jitter": float(np.mean(jit)) if jit else np.nan}


def boot_diff(x: np.ndarray, y: np.ndarray, rng) -> list:
    """mean(y - x) and its 95 % CI over paired resamples (nan-aware)."""
    d = y - x
    d = d[~np.isnan(d)]
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    man = {r["id"]: r for r in map(json.loads, open(MAN, encoding="utf-8"))}
    res = {}
    for d in ("zh2en", "en2zh"):
        lang = TGT[d]
        bleu = BLEU(tokenize="zh" if lang == "zh" else "13a")
        chrf = CHRF()
        la2 = {r["id"]: r for r in map(json.loads, open(S1 / f"thinker_{d}/blocks_{d}_w1.jsonl", encoding="utf-8"))}
        meta = {r["id"]: r for r in map(json.loads, open(S1 / f"data/meta_{d}.jsonl", encoding="utf-8"))}
        off = {}
        for cond in ("clean", "noisy"):
            for p in glob.glob(str(T / f"base_{cond}_{d}/s*.jsonl")):
                for r in map(json.loads, open(p, encoding="utf-8")):
                    off[(cond, r["id"])] = r["translation"]
        ids = sorted(i for i in man if man[i]["dir"] == d and all((c, i) in off for c in ("clean", "noisy"))
                     and all(f"a_{c}__{i}" in la2 for c in ("clean", "noisy")))
        rng = np.random.default_rng(0)
        R = {"n": len(ids)}
        NL = "cmn" if lang == "zh" else "eng"
        nz = lambda x: normalize_for_bleu(x, NL)   # noqa: E731
        refs = [nz(man[i]["ref"]) for i in ids]
        per = {}
        for cond in ("clean", "noisy"):
            hl = [nz(la2[f"a_{cond}__{i}"]["translation"]) for i in ids]
            ho = [nz(off[(cond, i)]) for i in ids]
            R[f"bleu_la2_{cond}"] = bleu.corpus_score(hl, [refs]).score
            R[f"bleu_whole_{cond}"] = bleu.corpus_score(ho, [refs]).score
            st = [commit_stats(la2[f"a_{cond}__{i}"], meta[f"a_{cond}__{i}"]["dur_ms"], lang) for i in ids]
            regret = []
            for s, o in zip(st, ho):
                c, w = toks(s["nonfinal"], lang), toks(o, lang)
                regret.append(np.nan if not c else 1.0 - lcs(c, w) / len(c))
            per[cond] = {"chrf": np.array([chrf.sentence_score(h, [r]).score for h, r in zip(hl, refs)]),
                         "regret": np.array(regret), **{k: np.array([s[k] for s in st], float)
                                                        for k in ("first_ms", "n_commits", "pre_end_share", "jitter")}}
        R["loss_la2"] = R["bleu_la2_clean"] - R["bleu_la2_noisy"]
        R["loss_whole"] = R["bleu_whole_clean"] - R["bleu_whole_noisy"]
        R["streaming_extra_loss"] = R["loss_la2"] - R["loss_whole"]
        # paired bootstrap of the corpus-BLEU losses and of their difference
        hl = {c: [nz(la2[f"a_{c}__{i}"]["translation"]) for i in ids] for c in ("clean", "noisy")}
        ho = {c: [nz(off[(c, i)]) for i in ids] for c in ("clean", "noisy")}
        st_ = {k: [bleu.corpus_score([h], [[r]]) for h, r in zip(v, refs)] for k, v in
               (("lc", hl["clean"]), ("ln", hl["noisy"]), ("oc", ho["clean"]), ("on", ho["noisy"]))}
        stats = {k: np.array([[s.sys_len, s.ref_len] + s.counts + s.totals for s in v], float) for k, v in st_.items()}

        def cb(S):   # corpus BLEU from summed sentence statistics
            sys_len, ref_len, c, t = S[0], S[1], S[2:6], S[6:10]
            if np.any(t == 0) or np.any(c == 0):
                return 0.0
            bp = 1.0 if sys_len > ref_len else np.exp(1 - ref_len / max(sys_len, 1))
            return 100 * bp * np.exp(np.mean(np.log(c / t)))
        idx = rng.integers(0, len(ids), (B, len(ids)))
        bl = np.array([[cb(stats[k][j].sum(0)) for k in ("lc", "ln", "oc", "on")] for j in idx])
        la2_loss, wh_loss = bl[:, 0] - bl[:, 1], bl[:, 2] - bl[:, 3]
        for name, v in (("loss_la2_ci", la2_loss), ("loss_whole_ci", wh_loss), ("streaming_extra_loss_ci", la2_loss - wh_loss)):
            R[name] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for k in ("regret", "first_ms", "n_commits", "pre_end_share", "jitter", "chrf"):
            R[k] = {"clean": float(np.nanmean(per["clean"][k])), "noisy": float(np.nanmean(per["noisy"][k])),
                    "noisy_minus_clean": boot_diff(per["clean"][k], per["noisy"][k], rng)}
        drop = per["clean"]["chrf"] - per["noisy"]["chrf"]
        dreg = per["noisy"]["regret"] - per["clean"]["regret"]
        ok = ~np.isnan(dreg)
        rho, p = spearmanr(drop[ok], dreg[ok])
        R["spearman_chrf_drop_vs_regret_increase"] = {"rho": float(rho), "p": float(p), "n": int(ok.sum())}
        # set (b): timing and jitter only
        Rb = {}
        for sub, keep in (("all", lambda m: True), ("no_crosstalk", lambda m: not m["ovl"])):
            by = {}
            for m in meta.values():
                if m["set"] == "b":
                    by.setdefault(m["pair"], {})["clean" if m["cond"] == "clean" else "noisy"] = m
            pairs = sorted(p for p, v in by.items() if len(v) == 2 and keep(v["noisy"]) and keep(v["clean"]))
            vals = {c: [commit_stats(la2[by[p][c]["id"]], by[p][c]["dur_ms"], lang) for p in pairs] for c in ("clean", "noisy")}
            Rb[sub] = {"n": len(pairs)}
            for k in ("first_ms", "n_commits", "pre_end_share", "jitter"):
                x = np.array([s[k] for s in vals["clean"]], float)
                y = np.array([s[k] for s in vals["noisy"]], float)
                Rb[sub][k] = {"clean": float(np.nanmean(x)), "noisy": float(np.nanmean(y)), "noisy_minus_clean": boot_diff(x, y, rng)}
        res[d] = {"a": R, "b": Rb}
        print(d, json.dumps({k: R[k] for k in ("n", "loss_la2", "loss_la2_ci", "loss_whole", "loss_whole_ci",
                                               "streaming_extra_loss", "streaming_extra_loss_ci")}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
