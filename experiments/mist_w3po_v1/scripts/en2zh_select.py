#!/usr/bin/env python
"""The fixed En->Zh checkpoint-selection rules (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §3, written
before any result). CPU; reads only dev5 outputs; writes <out> (json; refuses to overwrite) and prints it.

thinker  candidates = the SerialCRF checkpoints (steps 500 ... 3000); per checkpoint its dev5 text decode:
           Q  = text tcpBLEU-all (decode/<name>/score_text.json tcpBLEU_all)
           O  = text tcpBLEU-ov
           L  = text LAAL, current protocol (crf/t_eval/<name>/textlat_<name>.json unit.text_laal_online.all.mean)
         rule: feasible = {L <= L(last step) + 0.1 s}; pick the highest Q; ties (Q equal at 2 decimals) -> higher O
               (2 decimals), then lower L, then the later step. Missing / failed outputs of the LAST step: no selection
               (exit 2, ask the coordinator); of another step: that step is not a candidate (recorded).
         recorded only: tcpBLEU_1spk, text LAAL under the chunk protocol (textlat_chunk_<name>.json), held-out
               ce_label / crf_nll (heldout_<name>.json).
talker   candidates = the PL-EM talker checkpoints (iterations 400 / 800 / 1200); per checkpoint its dev5 speech proxy
         (one TTS seed, read-level input, the PL-EM c1b1s2 inference flags):
           Q  = speech ASR-BLEU all (decode/<name>/score_asr.json tcpBLEU_all)
           O  = speech ASR-BLEU ov
           L  = per-stream speech LAAL, current protocol (crf/t_eval/<name>/lat_<name>.json
                unit.speech_laal_online.all.mean)
         the same rule with the last iteration as the latency reference.

    python en2zh_select.py thinker|talker --cand <step> <name> [--cand ...] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEC = Path("eval_outputs/mist_w3po_v1/decode")
TE = Path("eval_outputs/mist_w3po_v1/crf/t_eval")
LAT_SLACK = 0.1


def pick(rows: list[dict]) -> dict:
    """rows: {step, Q, O, L} sorted by step; the last one is the latency reference. -> {chosen, feasible, ref_L}."""
    ref = rows[-1]["L"]
    feas = [r for r in rows if r["L"] <= ref + LAT_SLACK + 1e-12]
    best = max(feas, key=lambda r: (round(r["Q"], 2), round(r["O"], 2), -r["L"], r["step"]))
    return {"chosen": best["step"], "feasible": [r["step"] for r in feas], "ref_L": ref, "lat_cap": ref + LAT_SLACK}


def load(kind: str, step: int, name: str) -> dict:
    out = {"step": step, "name": name}
    if kind == "thinker":
        s = json.load(open(DEC / name / "score_text.json"))
        lat = json.load(open(TE / name / f"textlat_{name}.json"))
        out.update(Q=s["tcpBLEU_all"], O=s["tcpBLEU_ov"], one=s.get("tcpBLEU_1spk"),
                   L=lat["unit"]["text_laal_online"]["all"]["mean"])
        for tag, f in (("L_chunk", TE / name / f"textlat_chunk_{name}.json"), ("heldout", TE / name / f"heldout_{name}.json")):
            if f.exists():
                d = json.load(open(f))
                out[tag] = d["unit"]["text_laal_online"]["all"]["mean"] if tag == "L_chunk" else \
                    {k: d[k] for k in ("ce_label", "crf_nll", "q_label_mean", "windows")}
    else:
        s = json.load(open(DEC / name / "score_asr.json"))
        lat = json.load(open(TE / name / f"lat_{name}.json"))
        out.update(Q=s["tcpBLEU_all"], O=s["tcpBLEU_ov"], one=s.get("tcpBLEU_1spk"),
                   L=lat["unit"]["speech_laal_online"]["all"]["mean"])
        f = TE / name / f"lat_chunk_{name}.json"
        if f.exists():
            out["L_chunk"] = json.load(open(f))["unit"]["speech_laal_online"]["all"]["mean"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["thinker", "talker"])
    ap.add_argument("--cand", nargs=2, action="append", required=True, metavar=("STEP", "NAME"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    cands = sorted(((int(s), n) for s, n in a.cand), key=lambda x: x[0])
    rows, missing = [], []
    for step, name in cands:
        try:
            rows.append(load(a.kind, step, name))
        except (OSError, KeyError, ValueError) as e:
            missing.append({"step": step, "name": name, "error": repr(e)})
    if not rows or rows[-1]["step"] != cands[-1][0]:
        print(json.dumps({"error": "the last candidate has no complete dev5 outputs: no selection", "missing": missing}))
        sys.exit(2)
    res = {"kind": a.kind, "rule": f"max Q s.t. L <= L(last) + {LAT_SLACK}; ties (2 decimals) -> O, lower L, later",
           "rows": rows, "missing": missing, **pick(rows)}
    res["chosen_name"] = next(r["name"] for r in rows if r["step"] == res["chosen"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
