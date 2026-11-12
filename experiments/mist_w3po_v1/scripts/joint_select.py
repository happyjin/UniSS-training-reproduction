#!/usr/bin/env python
"""The pre-registered early-stop / selection rule of the joint bidirectional run (b) (reports/mist_w3po_v1/
joint_bidir_plan_prereg_v1.zh-CN.md §5, written before any result), the same for both stages. CPU; reads only dev5
outputs; prints one json line; writes --out (refuses to overwrite) only on a final decision.

Evaluation points s = 500, 1000, ... (one checkpoint each, both directions' dev5 text proxies). Per point, d in
{zh2en, en2zh}:
  Q_d  dev5 text tcpBLEU-all     decode/<name_d>/score_text.json tcpBLEU_all
  O_d  dev5 text tcpBLEU-ov      decode/<name_d>/score_text.json tcpBLEU_ov
  L_d  dev5 text LAAL (current protocol, the direction's own units)
                                 crf/t_eval/<name_d>/textlat_<name_d>.json unit.text_laal_online.all.mean
  H_d  (stage 2 only, tie-break) crf/t_eval/<name_d>/heldout_<name_d>.json ce_label
feasible  L_d(s) <= L_d^ref + 0.1 s for both d (ref: stage 1 the separate v8 models, stage 2 A' s3000 / (a)'s choice;
          --ref-zh2en / --ref-en2zh are their dev5 decode names, same files)
J(s)      (Q_zh2en(s) + Q_en2zh(s)) / 2, feasible points only
The points are read in step order and only as a contiguous prefix (500, 1000, ... up to the first point whose proxy is
still pending); a point whose proxy failed after its one same-configuration retry (status "failed", or "ok" with missing
outputs) is not a candidate and counts as "no improvement".
improvement  J(s) > best + eps, eps = 0.2; best = J at the last improvement (the first feasible point improves on -inf;
             a non-improving J never moves best -- the usual min-delta early stopping)
stop         s >= s_min = 3000 and the last 2 evaluation points (1000 steps) since the last improvement brought none
             (s - s_last_improvement >= 2 * 500; no improvement at all: s_last_improvement = 0)
ceiling      the prefix reaches 15000 without a stop: action "ceiling" (ask the coordinator; no selection)
selection    at a stop: among the points evaluated by then (every "ok" point, also beyond the prefix: they were all
             saved before the STOP) that are feasible, the highest round(J, 2); ties -> lower mean H (stage 2; a point
             without both H counts as +inf) -> higher round(mean O, 2) -> lower round(mean (L - L^ref), 3) -> later step.
             No feasible point: action "no_feasible" (JOINT_STOP, ask the coordinator).
actions      continue | stop | no_feasible | ceiling | wait_ref (a reference's outputs are not there yet: no decision)

    python joint_select.py --stage s1|s2 --ref-zh2en <name> --ref-en2zh <name> \\
        --cand <step> <zh2en name> <en2zh name> ok|failed|pending [--cand ...] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEC = Path("eval_outputs/mist_w3po_v1/decode")
TE = Path("eval_outputs/mist_w3po_v1/crf/t_eval")
EPS = 0.2
S_MIN = 3000
PATIENCE = 2
INTERVAL = 500
CEILING = 15000
LAT_SLACK = 0.1
DIRS = ("zh2en", "en2zh")


def load_dir(name: str, held: bool) -> dict:
    """one direction's dev5 proxy outputs -> {Q, O, L[, H]}; raises OSError / KeyError / ValueError if incomplete."""
    s = json.load(open(DEC / name / "score_text.json"))
    lat = json.load(open(TE / name / f"textlat_{name}.json"))
    out = {"Q": float(s["tcpBLEU_all"]), "O": float(s["tcpBLEU_ov"]), "one": s.get("tcpBLEU_1spk"),
           "L": float(lat["unit"]["text_laal_online"]["all"]["mean"])}
    if held:
        f = TE / name / f"heldout_{name}.json"
        out["H"] = float(json.load(open(f))["ce_label"]) if f.exists() else None
    return out


def load_ref(name: str) -> float:
    return float(json.load(open(TE / name / f"textlat_{name}.json"))["unit"]["text_laal_online"]["all"]["mean"])


def decide(points: list[dict], ref: dict, stage: str) -> dict:
    """points: [{step, status: ok|failed|pending, zh2en: {Q,O,L[,H]} | None, en2zh: ...}] (any order); ref: {d: L_ref}.
    -> the decision dict (see the module doc)."""
    pts = sorted(points, key=lambda p: p["step"])
    for p in pts:
        if p["status"] == "ok" and not (p.get("zh2en") and p.get("en2zh")):
            p["status"] = "failed"
        if p["status"] == "ok":
            p["feasible"] = all(p[d]["L"] <= ref[d] + LAT_SLACK + 1e-12 for d in DIRS)
            p["J"] = (p["zh2en"]["Q"] + p["en2zh"]["Q"]) / 2
            p["O_mean"] = (p["zh2en"]["O"] + p["en2zh"]["O"]) / 2
            p["dL_mean"] = sum(p[d]["L"] - ref[d] for d in DIRS) / 2
            hs = [p[d].get("H") for d in DIRS]
            p["H_mean"] = sum(hs) / 2 if stage == "s2" and all(h is not None for h in hs) else None
    prefix = []
    for i, p in enumerate(pts):
        if p["step"] != INTERVAL * (i + 1) or p["status"] == "pending":
            break
        prefix.append(p)
    best, last_imp, action, stop_at = -math.inf, 0, "continue", None
    for p in prefix:
        p["improved"] = bool(p["status"] == "ok" and p["feasible"] and p["J"] > best + EPS)
        if p["improved"]:
            best, last_imp = p["J"], p["step"]
        if p["step"] >= S_MIN and p["step"] - last_imp >= PATIENCE * INTERVAL:
            action, stop_at = "stop", p["step"]
            break
    end = prefix[-1]["step"] if prefix else 0
    if action == "continue" and end >= CEILING:
        action = "ceiling"
    res = {"stage": stage, "action": action, "prefix_end": end, "decision_step": stop_at,
           "best_J": None if best == -math.inf else best, "last_improvement": last_imp, "ref_L": ref,
           "rule": f"eps {EPS}, s_min {S_MIN}, patience {PATIENCE} x {INTERVAL}, L <= L_ref + {LAT_SLACK} both dirs, "
                   f"ceiling {CEILING}", "points": pts}
    if action == "stop":
        cands = [p for p in pts if p["status"] == "ok" and p["feasible"]]
        if not cands:
            res["action"] = "no_feasible"
            return res

        def key(p):
            h = p["H_mean"] if stage == "s2" else 0.0
            return (round(p["J"], 2), -(h if h is not None else math.inf), round(p["O_mean"], 2),
                    -round(p["dL_mean"], 3), p["step"])

        ch = max(cands, key=key)
        res.update(chosen=ch["step"], chosen_names={d: ch["names"][d] for d in DIRS},
                   candidates=[p["step"] for p in cands])
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["s1", "s2"], required=True)
    ap.add_argument("--ref-zh2en", required=True)
    ap.add_argument("--ref-en2zh", required=True)
    ap.add_argument("--cand", nargs=4, action="append", default=[], metavar=("STEP", "ZH2EN", "EN2ZH", "STATUS"))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.out and Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    try:
        ref = {"zh2en": load_ref(a.ref_zh2en), "en2zh": load_ref(a.ref_en2zh)}
    except (OSError, KeyError, ValueError) as e:
        print(json.dumps({"stage": a.stage, "action": "wait_ref", "error": repr(e)}))
        return
    pts = []
    for step, zn, en, st in a.cand:
        if st not in ("ok", "failed", "pending"):
            raise SystemExit(f"status ok|failed|pending, not {st}")
        p = {"step": int(step), "status": st, "names": {"zh2en": zn, "en2zh": en}, "zh2en": None, "en2zh": None}
        if st == "ok":
            try:
                for d, n in (("zh2en", zn), ("en2zh", en)):
                    p[d] = load_dir(n, a.stage == "s2")
            except (OSError, KeyError, ValueError) as e:
                p["error"] = repr(e)
        pts.append(p)
    res = decide(pts, ref, a.stage)
    if a.out and res["action"] in ("stop", "no_feasible", "ceiling"):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "x", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
