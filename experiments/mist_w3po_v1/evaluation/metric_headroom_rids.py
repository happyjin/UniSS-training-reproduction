"""Oracle tcpBLEU headroom of named utterance sets (CPU; reports/mist_w3po_v1/metrics_first_si_analysis.zh-CN.md §4).

For a decode and a jsonl of rows {"rid", ...} (e.g. the v2' S1' rows of the final-read causal P decode,
eval_outputs/mist_w3po_v1/crf/serial_v2p_precheck/s1/s1_rows_primary_finalpost.jsonl), fills every non-recalled
utterance of a row subset with its reference (metric_headroom.pairs_of fill oracle) and reports tcpBLEU all / ov / 1spk.
Subsets: all rows, chain == true, deferral == true, deferral and not chain, cls == MISLABEL, cls == ORDER_OMISSION.
Writes <out> (json; refuses to overwrite). Dev only.

    python -m experiments.mist_w3po_v1.evaluation.metric_headroom_rids --run <decode dir> --rows <jsonl> --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.evaluation import metric_headroom as M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from meeting_score import universe
    W = [json.loads(l) for l in open(M.REPO / M.WIN, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(Path(args.run) / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
    U = universe(M.DR, {w["meeting"] for w in W})
    sc = M.scored_utts(W, RW)
    As = M.assign_all(W, H, sc, U)
    base = M.corpus(As)
    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    subsets = {
        "all_rows": {r["rid"] for r in rows},
        "chain": {r["rid"] for r in rows if r.get("chain")},
        "deferral": {r["rid"] for r in rows if r.get("deferral")},
        "deferral_not_chain": {r["rid"] for r in rows if r.get("deferral") and not r.get("chain")},
        "mislabel": {r["rid"] for r in rows if r.get("cls") == "MISLABEL"},
        "order_omission": {r["rid"] for r in rows if r.get("cls") == "ORDER_OMISSION"},
    }
    res = {"run": args.run, "rows": args.rows, "base": base, "subsets": {}}
    for k, S in subsets.items():
        nfill = sum(1 for A in As.values() for rid in A["sc"] if rid in S and not A["rec"][rid])
        words = sum(len(A["sc"][rid]["ref"].split()) for A in As.values() for rid in A["sc"] if rid in S and not A["rec"][rid])
        r = M.corpus(As, fill=lambda rid, v, S=S: rid in S)
        res["subsets"][k] = {"n_rows": len(S), "n_filled": nfill, "words_filled": words,
                             "delta": {reg: round(r[reg]["bleu"] - base[reg]["bleu"], 2) for reg in r},
                             "oracle": {reg: r[reg]["bleu"] for reg in r}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: (v["n_filled"], v["words_filled"], v["delta"]) for k, v in res["subsets"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
