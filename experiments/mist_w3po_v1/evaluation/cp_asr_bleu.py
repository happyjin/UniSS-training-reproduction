"""cpASR-BLEU / cpBLEU for whole meetings (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §2).

Meeting-level concatenated minimum-permutation BLEU, the translation analogue of CHiME-6 cpWER:
  1. per meeting and output speaker label, the hypothesis segments (meeting clock, time order) are concatenated;
  2. per reference speaker, the scored reference utterances (non-hesitation, with a reference: the tcpBLEU universe of
     v11_metrics.score_ext) are concatenated in time order; a hypothesis segment is dropped, as in score_ext, when the
     annotated utterance it overlaps most (2 s collar) is not scored (a hesitation, or speech the reference windows
     dropped), so both sides cover the same universe;
  3. output labels (``unk`` excluded) are matched one-to-one to reference speakers by chrF (gstar_score.match,
     Hungarian), and the pairs are built by gstar_score.pairs_of (unmatched output -> empty reference, unmatched
     reference -> empty hypothesis): exactly the paper's cpBLEU construction, at meeting level;
  4. corpus BLEU (sacreBLEU, 13a for English, zh for Chinese) over all pairs of all meetings.
Per-meeting sufficient statistics are kept for paired tests (seeds pooled per meeting, as for tcpBLEU).

    python -m experiments.mist_w3po_v1.evaluation.cp_asr_bleu --set ami --run <decode run> [--run <seed-2 run>]
        [--hyp hyp_asr.jsonl] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import v11_metrics as V
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match, pairs_of

REPO = V.REPO


def refs_by_meeting(dr: str, ref_windows: list[dict], meetings: set, ids: set | None = None) -> dict:
    """meeting -> speaker -> concatenated reference (time order), the scored universe of score_ext; ids collects the
    scored record ids."""
    sep = " " if dr != "en2zh" else ""
    scored = {}
    for w in ref_windows:
        if w["meeting"] not in meetings:
            continue
        for u in w["utts"]:
            if not u["hesitation"] and u.get("ref", u["zh"]).strip():
                scored[u["record_id"]] = (w["meeting"], u["speaker_id"], w["start_s"] + u["begin_s"],
                                          u.get("ref", u["zh"]).strip())
    if ids is not None:
        ids.update(scored)
    out = defaultdict(lambda: defaultdict(list))
    for m, spk, a, t in scored.values():
        out[m][spk].append((a, t))
    return {m: {k: sep.join(t for _, t in sorted(v)) for k, v in d.items()} for m, d in out.items()}


def hyps_by_meeting(dr: str, W: list[dict], H: dict, U: dict, ids: set, collar: float = 2.0) -> dict:
    sep = " " if dr != "en2zh" else ""
    out = defaultdict(lambda: defaultdict(list))
    for w in W:
        h = H.get(w["id"])
        if h is None:
            continue
        for s in h["segments"]:
            t = s["text"].strip()
            if not t:
                continue
            a = w["start_s"] + s["start_s"]
            b = w["start_s"] + max(s["end_s"], s["start_s"] + 0.08)
            best, bu = 0.0, None
            for u in U[w["meeting"]]:
                ov = min(b + collar, u[3]) - max(a - collar, u[2])
                if ov > best:
                    best, bu = ov, u
            if bu is not None and bu[0] not in ids:
                continue      # on unscored speech: ignored, as in score_ext
            out[w["meeting"]][s["spk"]].append((a, t))
    return {m: {k: sep.join(t for _, t in sorted(v)) for k, v in d.items()} for m, d in out.items()}


def cp_stats(s: str, run: str, hyp_name: str = "hyp_asr.jsonl") -> dict:
    """meeting -> BLEU sufficient statistics (np.array) of one run."""
    from sacrebleu.metrics import BLEU

    dr = V.DR[s]
    bleu = BLEU(tokenize="13a" if dr != "en2zh" else "zh")
    H = {h["id"]: h for h in map(json.loads, open(REPO / V.DEC / run / hyp_name, encoding="utf-8"))}
    W = [w for w in map(json.loads, open(REPO / V.WIN[s], encoding="utf-8")) if w["id"] in H]
    meetings = {w["meeting"] for w in W}
    ids: set = set()
    R = refs_by_meeting(dr, list(map(json.loads, open(REPO / V.REF[s], encoding="utf-8"))), meetings, ids)
    Hm = hyps_by_meeting(dr, W, H, V.universe(dr, meetings), ids)
    out = {}
    for m in sorted(meetings):
        hyp = {k: v for k, v in Hm.get(m, {}).items()}
        ref = R.get(m, {})
        mp = match({k: v for k, v in hyp.items() if k != "unk"}, ref)
        pr = pairs_of(hyp, ref, mp)
        st = bleu._extract_corpus_statistics([x for x, _ in pr], [[y for _, y in pr]]) if pr else None
        out[m] = np.sum(np.array(st, dtype=np.float64), 0) if st else None
    return out


def corpus(stats_list: list[dict], eng: bool) -> float:
    from experiments.mist_w3po_v1.evaluation.v11_summarize import bleu_of
    tot = sum(v for st in stats_list for v in st.values() if v is not None)
    return round(bleu_of(tot, eng), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True, choices=sorted(V.DR))
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--hyp", default="hyp_asr.jsonl")
    ap.add_argument("--out")
    a = ap.parse_args()
    sts = [cp_stats(a.set, r, a.hyp) for r in a.run]
    res = {"set": a.set, "runs": a.run, "hyp": a.hyp, "cpBLEU": corpus(sts, V.DR[a.set] != "en2zh"),
           "per_meeting": [{m: (v.tolist() if v is not None else None) for m, v in st.items()} for st in sts]}
    if a.out:
        p = Path(a.out)
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "per_meeting"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
