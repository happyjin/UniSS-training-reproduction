"""The registered decisions that wait for T4 / T5's dev5 quick checks (CPU):
  J1 base   reports/mist_w3po_v1/thinker_joint_crf_prereg_v1.zh-CN.md §3
  D1 base   reports/mist_w3po_v1/thinker_dagger_d1_prereg_v1.zh-CN.md §2
  schedule  thinker_dagger_d1_prereg_v1 §3 (does D1 run; D1 or J1 first)
Inputs per candidate t in {t4, t5}: the training log (CRF_RUN_EXIT rc) and step_03000/DONE, the quick check's go.json
and stall_diag.txt (line 1 = the model, line 2 = A' on the same windows), its dev5 decode (for the speech proxy).
Writes (noclobber) logs/mist_w3po_v1/j1/base_decision.json, logs/mist_w3po_v1/d1/base_decision.json,
logs/mist_w3po_v1/d1/schedule.json. --wait polls every 60 s until both candidates are resolved (go.json written, or the
training ended without a clean step_03000).

    python -m experiments.mist_w3po_v1.crf.j1_decide [--wait] [--dry]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

C = "eval_outputs/mist_w3po_v1/crf"
RUNS = {"t4": "t4_dart_t0_w10_20261111T173134Z", "t5": "t5_evid_p4_20261111T154210Z"}
BASES = {"t4": {"sched": f"{C}/sched_piece_t0_g1000_v1", "dart": f"{C}/t4_dart_w10_v1", "lw_mode": "t2"},
         "t5": {"sched": f"{C}/sched_t5_pieces4_w10_v1", "dart": f"{C}/t5_dart_pieces4_w10_v1", "lw_mode": "t5"}}
WIN5 = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl"
A5 = "eval_outputs/mist_w3po_v1/decode/plcrf_dev5_aprule"
OUT_J1 = Path("logs/mist_w3po_v1/j1/base_decision.json")
OUT_D1 = Path("logs/mist_w3po_v1/d1/base_decision.json")
OUT_SCH = Path("logs/mist_w3po_v1/d1/schedule.json")
FINAL_SHARE_A = 0.497      # A' dev5 final-read word share (T4 / T5 GO line 1)
A6S_LINE = 0.28            # rule C: share of class A among >= 6 s windows
S3_FACTOR = 2.0            # B-class silent reads >= 2 x A''s
SEG_WORDS_MIN = 4.0        # "fragmented": talker segment mean words < 4.0
TIE_PROXY_S = 0.05
TIE_BLEU_J1 = 0.1
TIE_BLEU_D1 = 0.1


# ------------------------------------------------------------------------------------------------ pure rules
def parse_stall(text: str) -> tuple[dict, dict]:
    """t1_stall_diag.py output: line 1 = the model, line 2 = A' (each '<label> {json}')."""
    rows = [json.loads(l[l.index("{"):]) for l in text.splitlines() if "{" in l]
    return rows[0], rows[1]


def a6s_share(d: dict) -> float:
    b = d.get("by_dur", {}).get(">=6s", {})
    n = sum(b.values())
    return b.get("A", 0) / n if n else 0.0


def stall_flags(go: dict, model: dict, aprime: dict) -> dict:
    s1 = go["final_share"] > FINAL_SHARE_A
    s2 = a6s_share(model) >= A6S_LINE
    bm, ba = model.get("B_mean_silent_reads"), aprime.get("B_mean_silent_reads")
    s3 = bm is not None and ba is not None and bm >= S3_FACTOR * ba
    return {"S1": bool(s1), "S2": bool(s2), "S3": bool(s3), "stall": bool(s1 or s2 or s3),
            "final_share": go["final_share"], "ge6s_A_share": round(a6s_share(model), 4),
            "B_silent_model": bm, "B_silent_aprime": ba}


def frag_or_quality(go: dict) -> dict:
    seg = go.get("talker_blocks", {}).get("segment", {}).get("mean")
    f = (not go["go_text"]) or (seg is not None and seg < SEG_WORDS_MIN)
    return {"frag_or_quality": bool(f), "go_text": bool(go["go_text"]), "segment_words_mean": seg}


def j1_base(info: dict) -> dict:
    """info[t] = {"ok": trained cleanly, "GO": bool, "text_all": float, "proxy": float | None}; §3 of J1's prereg."""
    ok = {t: v for t, v in info.items() if v["ok"]}
    go = [t for t, v in ok.items() if v["GO"]]
    if len(go) == 1:
        return {"base": go[0], "rule": "only one GO"}
    if len(go) == 2:
        a, b = go
        pa, pb = ok[a]["proxy"], ok[b]["proxy"]
        if abs(pa - pb) >= TIE_PROXY_S:
            return {"base": a if pa < pb else b, "rule": "both GO: lower dev5 speech proxy"}
        return {"base": a if ok[a]["text_all"] >= ok[b]["text_all"] else b,
                "rule": f"both GO, proxy within {TIE_PROXY_S} s: higher dev5 text all"}
    if not ok:
        return {"base": None, "rule": "neither trained cleanly"}
    if len(ok) == 1:
        return {"base": next(iter(ok)), "rule": "neither GO; the only clean training"}
    d = ok["t4"]["text_all"] - ok["t5"]["text_all"]
    if abs(d) < TIE_BLEU_J1:
        return {"base": "t5", "rule": f"neither GO; text all within {TIE_BLEU_J1}: T5"}
    return {"base": "t4" if d > 0 else "t5", "rule": "neither GO: higher dev5 text all"}


def d1_base(info: dict) -> dict:
    """D1 prereg §2: higher dev5 text all among the clean trainings; within 0.1 -> lower dev5 text LAAL."""
    ok = {t: v for t, v in info.items() if v["ok"]}
    if not ok:
        return {"base": None, "rule": "neither trained cleanly"}
    if len(ok) == 1:
        return {"base": next(iter(ok)), "rule": "the only clean training"}
    d = ok["t4"]["text_all"] - ok["t5"]["text_all"]
    if abs(d) < TIE_BLEU_D1:
        return {"base": "t4" if ok["t4"]["text_laal"] <= ok["t5"]["text_laal"] else "t5",
                "rule": f"text all within {TIE_BLEU_D1}: lower dev5 text LAAL"}
    return {"base": "t4" if d > 0 else "t5", "rule": "higher dev5 text all"}


def schedule(info: dict) -> dict:
    """D1 prereg §3: D1 runs iff any clean candidate stalls or none is GO; D1 first iff any stalls."""
    ok = {t: v for t, v in info.items() if v["ok"]}
    stall = any(v["stall"]["stall"] for v in ok.values())
    fq = any(v["frag"]["frag_or_quality"] for v in ok.values())
    any_go = any(v["GO"] for v in ok.values())
    run_d1 = stall or not any_go
    return {"any_stall": stall, "any_frag_or_quality": fq, "any_GO": any_go, "run_D1": run_d1, "run_J1": True,
            "first": "D1" if stall else "J1"}


# ------------------------------------------------------------------------------------------------ inputs
def trained(t: str) -> str:
    """'ok', 'failed' or 'running'."""
    log = Path(f"logs/mist_w3po_v1/{RUNS[t]}.log")
    txt = log.read_text(errors="replace") if log.exists() else ""
    m = re.findall(r"CRF_RUN_EXIT rc=(\d+)", txt)
    if not m:
        return "running"
    done = Path(f"{C}/train/{RUNS[t]}/ckpt/step_03000/DONE").exists()
    return "ok" if m[-1] == "0" and done else "failed"


def quick_dir(t: str) -> Path:
    return Path(f"{C}/t_eval/dev5_{t}_s3000")


def resolved(t: str) -> bool:
    s = trained(t)
    return s == "failed" or (s == "ok" and (quick_dir(t) / "go.json").exists() and (quick_dir(t) / "stall_diag.txt").exists())


def gather(proxy_root: Path, need_proxy: bool) -> dict:
    from experiments.mist_w3po_v1.crf import joint_speech_proxy as JP

    info = {}
    for t in ("t4", "t5"):
        if trained(t) != "ok" or not (quick_dir(t) / "go.json").exists():
            info[t] = {"ok": False, "GO": False, "trained": trained(t)}
            continue
        go = json.loads((quick_dir(t) / "go.json").read_text())
        m, a = parse_stall((quick_dir(t) / "stall_diag.txt").read_text())
        v = {"ok": True, "trained": "ok", "GO": bool(go["GO"]), "text_all": go["text_all"], "text_ov": go["text_ov"],
             "text_laal": go["text_laal"], "final_share": go["final_share"], "stall": stall_flags(go, m, a),
             "frag": frag_or_quality(go), "proxy": None}
        if need_proxy:
            p = proxy_root / f"proxy_dev5_{t}_rl.json"
            v["proxy"] = JP.run_proxy(f"eval_outputs/mist_w3po_v1/decode/dev5_{t}_s3000", WIN5, True, str(p))["speech_laal"]
        info[t] = v
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the current state, write nothing")
    a = ap.parse_args()
    if a.dry:
        print(json.dumps({t: {"trained": trained(t), "resolved": resolved(t)} for t in RUNS}))
        return
    for p in (OUT_J1, OUT_D1, OUT_SCH):
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
    while not all(resolved(t) for t in RUNS):
        if not a.wait:
            raise SystemExit(f"not resolved yet: {[t for t in RUNS if not resolved(t)]}")
        time.sleep(60)
    root = Path("logs/mist_w3po_v1/j1/proxy")
    root.mkdir(parents=True, exist_ok=True)
    info = gather(root, need_proxy=True)
    from experiments.mist_w3po_v1.crf import joint_speech_proxy as JP

    ap5 = root / "proxy_dev5_aprime.json"   # A' dev5, its own segments as blocks (J1 / D1 GO line 4's reference)
    aprime_proxy = (json.loads(ap5.read_text()) if ap5.exists() else JP.run_proxy(A5, WIN5, False, str(ap5)))["speech_laal"]
    jb, db, sc = j1_base(info), d1_base(info), schedule(info)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    common = {"utc": now, "candidates": info, "aprime_dev5_proxy": aprime_proxy}
    for p, d in ((OUT_J1, {**jb, **({"data": BASES[jb["base"]]} if jb["base"] else {})}),
                 (OUT_D1, {**db, **({"data": BASES[db["base"]], "run": RUNS[db["base"]]} if db["base"] else {})}),
                 (OUT_SCH, sc)):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x", encoding="utf-8") as fh:
            json.dump({**d, **common}, fh, indent=1)
    print(json.dumps({"J1": jb, "D1": db, "schedule": sc}))


if __name__ == "__main__":
    main()
