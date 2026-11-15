"""Paired bootstrap of cpBLEU_sess differences between systems, both directions (paper v3).

The session-level slot matching of each system is fixed on the full test set (as gstar_score computes it); each
window then contributes its matched (hypothesis, reference) pairs. BLEU sufficient statistics are summed per window,
and 1,000 resamples draw either windows (with replacement) or whole meetings (cluster bootstrap, 16 meetings).
Reports the difference, its 95% interval and the one-sided p-value (share of resamples with difference <= 0).
Speech pairs ("sp:" prefix) score the ASR transcripts of each output speaker's synthesised speech instead of its text,
with the session matching still made on the text, as gstar_score --asr does. Zh->En uses the English references
("ref"), segments joined with spaces and 13a tokenisation, as gstar_score does.

    python scripts/bootstrap.py   (from the paper folder; writes generated/bootstrap.json)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systems import ASR_FILE, PAIRS, SPEECH_DIR, hyp_dir, windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match, pairs_of  # noqa: E402

B = 1000


def tok(dr: str) -> str:
    return "13a" if dr in ("zh2en", "ali") else "zh"


def window_stats(dr: str, d: Path, win: Path, asr_file: Path | None = None) -> dict:
    """window id -> (meeting, summed BLEU statistics of its session-matched pairs)."""
    from sacrebleu.metrics import BLEU

    bleu = BLEU(tokenize=tok(dr))
    sep = " " if dr in ("zh2en", "ali") else ""
    W = [json.loads(l) for l in open(win, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(d / "hyp.jsonl", encoding="utf-8"))}
    W = [w for w in W if w["id"] in H]
    hyp_w, ref_w = {}, {}
    hm, rm = defaultdict(lambda: defaultdict(str)), defaultdict(lambda: defaultdict(str))
    for w in W:
        h, r = defaultdict(str), defaultdict(str)
        for s in H[w["id"]]["segments"]:
            h[s["spk"]] += sep + s["text"]
        for u in sorted(w["utts"], key=lambda u: u["begin_s"]):
            if not u["hesitation"]:
                r[u["speaker_id"]] += sep + u.get("ref", u["zh"]).strip()
        hyp_w[w["id"]], ref_w[w["id"]] = h, r
        for k, t in h.items():
            hm[w["meeting"]][k] += t
        for k, t in r.items():
            rm[w["meeting"]][k] += t
    smap = {m: match(dict(hm[m]), dict(rm[m])) for m in rm}
    asr = ({r["id"]: r.get("asr_text", "") for r in map(json.loads, open(asr_file, encoding="utf-8"))}
           if asr_file else None)
    out = {}
    for w in W:
        h = hyp_w[w["id"]]
        if asr is not None:
            h = {k: asr.get(f"{w['id']}__S{k}", "") for k in h}
        ps = pairs_of(h, ref_w[w["id"]], smap[w["meeting"]])
        st = bleu._extract_corpus_statistics([a for a, _ in ps], [[b for _, b in ps]])
        out[w["id"]] = (w["meeting"], np.sum(np.array(st, dtype=np.float64), 0) if st else None)
    return out


def score(dr: str, stats: np.ndarray) -> float:
    from sacrebleu.metrics import BLEU

    return BLEU(tokenize=tok(dr))._compute_score_from_stats(list(stats)).score


def stats_of(key: str):
    """'en2zh:O2' (text) or 'sp:en2zh:O2' (speech) -> window statistics, or None when the outputs are missing."""
    speech = key.startswith("sp:")
    dr, name = key.removeprefix("sp:").split(":")
    d = hyp_dir(dr, name)
    if d is None or not (d / "hyp.jsonl").exists():
        return None
    asr = None
    if speech:
        sp = SPEECH_DIR(dr, name)
        asr = ASR_FILE(sp) if sp else None
        if asr is None or not asr.exists():
            return None
    return window_stats(dr, d, windows(dr, name), asr)


def main() -> None:
    cache, res = {}, {}
    rng = np.random.default_rng(20261028)
    for a, b in PAIRS:
        for k in (a, b):
            if k not in cache:
                cache[k] = stats_of(k)
        if cache[a] is None or cache[b] is None:
            print("skip", a, b)
            continue
        dr = a.removeprefix("sp:").split(":")[0]
        Sa, Sb = cache[a], cache[b]
        ids = sorted(set(Sa) & set(Sb))
        dim = next(s for _, s in Sa.values() if s is not None).shape[0]
        Xa = np.stack([Sa[i][1] if Sa[i][1] is not None else np.zeros(dim) for i in ids])
        Xb = np.stack([Sb[i][1] if Sb[i][1] is not None else np.zeros(dim) for i in ids])
        meet = np.array([Sa[i][0] for i in ids])
        full = score(dr, Xa.sum(0)) - score(dr, Xb.sum(0))
        r = {"a": a, "b": b, "diff": round(full, 2), "windows": len(ids)}
        for mode in ("window", "meeting"):
            ds = []
            if mode == "window":
                for _ in range(B):
                    j = rng.integers(0, len(ids), len(ids))
                    ds.append(score(dr, Xa[j].sum(0)) - score(dr, Xb[j].sum(0)))
            else:
                ms = np.unique(meet)
                idx = {m: np.flatnonzero(meet == m) for m in ms}
                for _ in range(B):
                    j = np.concatenate([idx[m] for m in rng.choice(ms, len(ms))])
                    ds.append(score(dr, Xa[j].sum(0)) - score(dr, Xb[j].sum(0)))
            ds = np.array(ds)
            r[mode] = {"ci95": [round(float(np.percentile(ds, 2.5)), 2), round(float(np.percentile(ds, 97.5)), 2)],
                       "p": round(float((ds <= 0).mean()), 4)}
        res[f"{a}_vs_{b}"] = r
        print(json.dumps(r))
    out = Path(__file__).resolve().parents[1] / "generated" / "bootstrap.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
