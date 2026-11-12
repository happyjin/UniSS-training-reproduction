"""Final test sensitivity check (coordinator 2026-11-12 ~22:20 Prague): the windows whose block / segment pairing the
read-level merge lost (merge_pairlost_check.py outputs) are removed from EVERY system of a set, and the text and speech
tcpBLEU are recomputed on that common subset. Removal = (1) the hypotheses of those online windows are dropped and
(2) every reference utterance overlapping such a window's time span gets an empty reference (the scorer's don't-care,
as the DiariST references without a human translation). The scorer is score_run's (overlap_score.score, online, the
set's reference windows / universe). Writes <out>/<name>/score_text.json and score_asr.json (when the source has
hyp_asr.jsonl) for each given decode dir, plus <out>/windows.jsonl, <out>/ref_windows.jsonl, <out>/excluded.json;
gate_2seed then runs on the <out>/<name> dirs. Refuses an existing --out. CPU.

    python -m experiments.mist_w3po_v1.evaluation.final_test_excl --set ami|ali --lost <merge_pairlost json> [...] \\
        --out <new dir> --names <decode dir name> [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
DEC = Path("eval_outputs/mist_w3po_v1/decode")
SETS = {"ami": ("eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl",
                "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl", "zh2en"),
        "ali": ("eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_ali.jsonl",
                "data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl", "ali")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--lost", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    a = ap.parse_args()
    import overlap_score as O
    from meeting_score import universe
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    win, ref, sset = SETS[a.set]
    ex = sorted({x["id"] for f in a.lost for x in json.load(open(f))["lost"]})
    W = [json.loads(l) for l in open(win, encoding="utf-8")]
    span = {}
    for w in W:
        if w["id"] in ex:
            span.setdefault(w["meeting"], []).append((w["start_s"], w["start_s"] + w["dur_ms"] / 1000))
    Wk = [w for w in W if w["id"] not in ex]
    RW, blanked = [], 0
    for rw in map(json.loads, open(ref, encoding="utf-8")):
        for u in rw["utts"]:
            a0, b0 = rw["start_s"] + u["begin_s"], rw["start_s"] + u["end_s"]
            if any(a0 < e and b0 > s for s, e in span.get(rw["meeting"], [])):
                u["ref"] = ""
                blanked += 1
        RW.append(rw)
    out.mkdir(parents=True)
    with open(out / "windows.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(w) + "\n" for w in Wk)
    with open(out / "ref_windows.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in RW)
    with open(out / "excluded.json", "x", encoding="utf-8") as fh:
        json.dump({"set": a.set, "excluded_windows": ex, "n": len(ex), "ref_utterances_blanked": blanked,
                   "lost_files": a.lost}, fh, indent=1)
    U = universe(sset, {w["meeting"] for w in Wk})
    for name in a.names:
        d = out / name
        d.mkdir()
        for kind, hp in (("text", DEC / name / "hyp.jsonl"), ("asr", DEC / name / "hyp_asr.jsonl")):
            if not hp.exists():
                continue
            H = {h["id"]: h for h in map(json.loads, open(hp, encoding="utf-8")) if h["id"] not in ex}
            r = O.score(sset, [w for w in Wk if w["id"] in H], H, RW, U, online=True)
            r["hyp"], r["windows"], r["excluded"] = str(hp), str(out / "windows.jsonl"), len(ex)
            with open(d / f"score_{kind}.json", "x", encoding="utf-8") as fh:
                json.dump(r, fh, indent=1, ensure_ascii=False)
            print(name, kind, round(r["tcpBLEU_all"], 2), round(r["tcpBLEU_ov"], 2), round(r["tcpBLEU_1spk"], 2), flush=True)


if __name__ == "__main__":
    main()
