"""E10 dev gate (prereg reports/mist_w3po_v1/e10_liveinterpret_multispk_prereg.zh-CN.md §3), per dev set:
  1. collapsed windows <= 2 %           (run_e10 row["collapse"] = S5 collapse_ok criterion; E10 raw)
  2. tagged pieces >= 95 %              (pieces with a model speaker tag among pieces with text; E10 raw)
  3. tcpASR-BLEU(E10) >= tcpASR-BLEU(E10-0)                     (score_asr.json tcpBLEU_all)
  4. [text tcpBLEU - tcpASR-BLEU](E10) - [same](E10-0) <= 2     (score_text.json / score_asr.json)
Prints the table and writes <out>; exit code 0 = all gates pass, 3 = some gate fails, 2 = inputs missing.

    $PYE e10_multispk/dev_gate.py --sets devz,deve --out e10_multispk/output/eval/dev_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEC = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/mist_w3po_v1/decode")


def raw_stats(name):
    rows = [json.loads(l) for f in sorted((HERE / "output/eval" / name / "raw").glob("part_*.jsonl")) for l in open(f)]
    n = len(rows)
    col = sum(bool(r.get("collapse")) for r in rows)
    pieces = [s for r in rows for s in r.get("segs", []) if (s.get("text") or "").strip()]
    tagged = sum(isinstance(s.get("spk"), int) for s in pieces)
    return {"windows": n, "errors": sum(bool(r.get("error")) for r in rows),
            "collapse_rate": round(col / n, 4) if n else None,
            "missing_collapse_field": sum("collapse" not in r for r in rows),
            "tagged_piece_rate": round(tagged / len(pieces), 4) if pieces else None, "pieces": len(pieces)}


def bleu(name, kind):
    p = DEC / name / f"score_{kind}.json"
    return json.loads(p.read_text())["tcpBLEU_all"] if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="devz,deve")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, ok, missing = {}, True, False
    for s in a.sets.split(","):
        e, z = f"jdev_{s}_e10", f"jdev_{s}_e10_0"
        r = {"e10": raw_stats(e), "e10_0": raw_stats(z),
             "asr": {"e10": bleu(e, "asr"), "e10_0": bleu(z, "asr")},
             "text": {"e10": bleu(e, "text"), "e10_0": bleu(z, "text")}}
        g = {}
        st = r["e10"]
        g["1_collapse<=2%"] = st["collapse_rate"] is not None and st["missing_collapse_field"] == 0 \
            and st["collapse_rate"] <= 0.02
        g["2_tagged>=95%"] = st["tagged_piece_rate"] is not None and st["tagged_piece_rate"] >= 0.95
        if None in (r["asr"]["e10"], r["asr"]["e10_0"], r["text"]["e10"], r["text"]["e10_0"]):
            missing = True
            g["3_asr>=e10_0"] = g["4_gap_widening<=2"] = None
        else:
            g["3_asr>=e10_0"] = r["asr"]["e10"] >= r["asr"]["e10_0"]
            r["gap"] = {"e10": round(r["text"]["e10"] - r["asr"]["e10"], 2),
                        "e10_0": round(r["text"]["e10_0"] - r["asr"]["e10_0"], 2)}
            g["4_gap_widening<=2"] = r["gap"]["e10"] - r["gap"]["e10_0"] <= 2.0
        r["gates"] = g
        ok = ok and all(v is True for v in g.values())
        res[s] = r
        print(f"[{s}] collapse {st['collapse_rate']} tagged {st['tagged_piece_rate']} | tcpASR-BLEU E10 "
              f"{r['asr']['e10']} vs E10-0 {r['asr']['e10_0']} | text {r['text']['e10']} vs {r['text']['e10_0']} | "
              + " ".join(f"{k}={'PASS' if v else ('MISSING' if v is None else 'FAIL')}" for k, v in g.items()))
    res["all_pass"] = ok and not missing
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("ALL PASS" if res["all_pass"] else ("INPUTS MISSING" if missing else "GATE FAILED"))
    sys.exit(0 if res["all_pass"] else (2 if missing else 3))


if __name__ == "__main__":
    main()
