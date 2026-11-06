"""W3PO: merge a decode's shards and score it with the v8 overlap-aware whole-meeting scorer
(reports/gstar_si_v8/overlap_score.py, online=True: commits clamped to the windows' detection times).

  <run>/hyp_*.jsonl (except hyp_asr*) -> <run>/hyp.jsonl (once every window of --windows is present; never rewritten)
  -> <run>/score_text.json   ([--asr <hyp_asr.jsonl>] -> <run>/score_asr.json, the speech tcpBLEU)
Prints the headline numbers. Refuses to overwrite a score file.

    python -m experiments.mist_w3po_v1.evaluation.score_run --run <dir> --windows <online windows> \
        --ref-windows data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl [--set zh2en|ali]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))

HEAD = ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk", "CR", "collapse", "duplicate_rate", "omitted_ov", "omitted_1spk",
        "inserted_ov", "inserted_1spk", "pre_end_share")


def merge(run: Path, windows: list[dict]) -> Path:
    out = run / "hyp.jsonl"
    if out.exists():
        return out
    rec = {}
    for f in sorted(run.glob("hyp_*.jsonl")):
        if f.name.startswith("hyp_asr"):
            continue
        for l in open(f, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            rec[r["id"]] = r
    miss = [w["id"] for w in windows if w["id"] not in rec]
    if miss:
        raise SystemExit(f"{run}: {len(miss)} of {len(windows)} windows missing, e.g. {miss[:3]}")
    with open(out, "x", encoding="utf-8") as fh:
        for w in windows:
            fh.write(json.dumps(rec[w["id"]], ensure_ascii=False) + "\n")
    return out


def headline(r: dict) -> dict:
    h = {k: r.get(k) for k in HEAD}
    h["lag_ov_p50"] = (r.get("lag_ov") or {}).get("p50")
    h["lag_1spk_p50"] = (r.get("lag_1spk") or {}).get("p50")
    h["short_le0.5"] = {k: v["recall"] for k, v in r.get("short_recall", {}).items() if k.startswith("le0.5")}
    return h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--ref-windows", required=True)
    ap.add_argument("--set", default="zh2en", choices=["zh2en", "ali", "en2zh"])
    ap.add_argument("--asr", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import overlap_score as O
    from meeting_score import universe

    run = Path(args.run)
    W = [json.loads(l) for l in open(args.windows, encoding="utf-8")]
    hyp = merge(run, W)
    RW = [json.loads(l) for l in open(args.ref_windows, encoding="utf-8")]
    U = universe(args.set, {w["meeting"] for w in W})
    jobs = [("text", hyp)] + ([("asr", Path(args.asr))] if args.asr else [])
    for kind, hp in jobs:
        out = run / f"score_{kind}{args.tag}.json"
        if out.exists():
            print(f"{out} exists; kept")
            continue
        H = {h["id"]: h for h in map(json.loads, open(hp, encoding="utf-8"))}
        r = O.score(args.set, [w for w in W if w["id"] in H], H, RW, U, online=True)
        r["hyp"], r["windows"] = str(hp), args.windows
        with open(out, "x", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1, ensure_ascii=False)
        print(kind, json.dumps(headline(r), ensure_ascii=False))


if __name__ == "__main__":
    main()
