"""J1-v2's CPU precheck PC2-PC4 (reports/mist_w3po_v1/thinker_joint_crf_v2_prereg_v1.zh-CN.md §3; thresholds there and in
TH, fixed in the pre-registration 625a1926 before any number). PC1 = crf/test_j1v2.py + the regression tests, whose logs
are checked here (--pc1-logs: every log must end with an ALL ... PASSED line).

  PC2  the talker model (T3's json, unchanged) on the talker J1-v2 is evaluated with (read-level blocks + PL-EM):
       A'rl + PL-EM and VE1rl + PL-EM, seed 1; every block's duration replaced by the model's (j1_precheck.pc2's
       construction): |model - actual| per-stream speech LAAL <= 0.30 s and unit Pearson r >= 0.8, both runs
  PC3  VE1's data (crf/sched_t5_merge3_w10_v1, lattice v2p), every window of the 3000-step plan: coarsened-grid share
       <= 2 %; J1-v2's per-step CPU time (window_parts_shift + budget + lam* + posterior of every window of the step,
       one process, neutral scores psi = 0) over the first 200 steps: p95 <= 3.0 s
  PC4  first 200 steps, neutral scores: J1-v2's posterior (speech l, talker factor, the shift budget's lam*) vs the base
       (the same chain with the playback terms zeroed = VE1's v2p posterior, its own lam* = 0): (i) E[speech l] / word
       J1-v2 < base; (ii) E[blocks] J1-v2 < base; (iii) E[text Lat] / word J1-v2 <= base + 0.25 s
  PC4 descriptive (not a line): a text-timeline posterior whose lam_text gives the same E[speech l] as J1-v2 (per step,
       bisection): its E[blocks] and E[text Lat] next to J1-v2's

    python -m experiments.mist_w3po_v1.crf.j1v2_precheck --out eval_outputs/mist_w3po_v1/crf/j1v2_precheck_v1.json \
        --pc1-logs <test logs> [--procs 64] [--steps4 200]
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import j1_precheck as JPC
from experiments.mist_w3po_v1.crf import joint_speech_crf as JS
from experiments.mist_w3po_v1.crf import joint_speech_proxy as JP

TH = {"pc2_laal": 0.30, "pc2_r": 0.8, "pc3_coarse": 0.02, "pc3_step_p95_s": 3.0, "pc4_text_slack": 0.25}
C = "eval_outputs/mist_w3po_v1/crf"
SCHED, MODE = f"{C}/sched_t5_merge3_w10_v1", "v2p"
DEC, SP = "eval_outputs/mist_w3po_v1/decode", "eval_outputs/mist_w3po_v1/speech"
PC2_RUNS_V2 = {"Aprime_rl_plem": (f"{DEC}/dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem",
                                  f"{SP}/dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem/talker"),
               "VE1_rl_plem": (f"{DEC}/dev_ve1_s3000__rlplem", f"{SP}/dev_ve1_s3000__rlplem/talker")}
_G: dict = {}


def matched_text(zp: list[dict], pf: list[dict], recs: list[dict], pls: list[dict], target: float) -> dict:
    """lam_text >= 0 on the zero-playback chains with sum_w E_{q_text}[speech l] = target (doubling from 1 up to 1e3,
    then 50 bisections); its E[blocks] and E[text Lat]. unmatched when even lam_text = 1e3 stays above the target."""
    def ev(lam):
        sp = bl = tx = 0.0
        for z, p, rec, pl in zip(zp, pf, recs, pls):
            fb = JS.forward_backward(z["aug"], lam)
            s_, b_ = JPC._expect(z["aug"], fb, p["aug"], rec, pl)
            sp, bl, tx = sp + s_, bl + b_, tx + fb["Eq_l"]
        return sp, bl, tx
    s0 = ev(0.0)
    if s0[0] <= target:
        return {"lam_text": 0.0, "speech": s0[0], "blocks": s0[1], "text": s0[2], "matched": True}
    lo, hi = 0.0, 1.0
    while ev(hi)[0] > target:
        lo, hi = hi, 2.0 * hi
        if hi > 1e3:
            s = ev(1e3)
            return {"lam_text": 1e3, "speech": s[0], "blocks": s[1], "text": s[2], "matched": False}
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ev(mid)[0] > target:
            lo = mid
        else:
            hi = mid
    s = ev(hi)
    return {"lam_text": hi, "speech": s[0], "blocks": s[1], "text": s[2], "matched": True}


def _step_job(s: int) -> dict:
    ids, tm, delta = _G["plan"][s], _G["tm"], _G["delta"]
    recs = [_G["recs"][i] for i in ids]
    pls = [JPC.neutral_plan(rec, MODE) for rec in recs]
    t0 = time.perf_counter()    # J1-v2's per-step CPU work
    full = [JS.window_parts_shift(JPC.W_(rec), pl, {}, tm, JS.TAU) for rec, pl in zip(recs, pls)]
    tgt, bi = JS.budget_shift(full, delta)
    lam_j, info_j = JS.solve_lambda_to(full, tgt)
    for rec, pl, p in zip(recs, pls, full):
        JS.posterior_from_parts(JPC.W_(rec), pl, p, JS.TAU, lam_j)
    t_step = time.perf_counter() - t0
    zp = [JS.window_parts(JPC.W_(rec), pl, {}, tm, JS.TAU, zero_play=True) for rec, pl in zip(recs, pls)]
    lam_b, info_b = JS.solve_lambda(zp)
    acc = {"words": 0.0, "j_sp": 0.0, "j_tx": 0.0, "j_bl": 0.0, "b_sp": 0.0, "b_tx": 0.0, "b_bl": 0.0}
    for rec, pl, pf, pz in zip(recs, pls, full, zp):
        fj = JS.forward_backward(pf["aug"], lam_j)
        fb = JS.forward_backward(pz["aug"], lam_b)
        acc["j_sp"] += fj["Eq_l"]
        acc["b_tx"] += fb["Eq_l"]
        tx, bl = JPC._expect(pf["aug"], fj, pz["aug"], rec, pl)
        acc["j_tx"] += tx
        acc["j_bl"] += bl
        sp, bl2 = JPC._expect(pz["aug"], fb, pf["aug"], rec, pl)
        acc["b_sp"] += sp
        acc["b_bl"] += bl2
        acc["words"] += pf["sum_words"]
    mt = matched_text(zp, full, recs, pls, acc["j_sp"])
    return {"step": s, "t_step": t_step, "lam_j": lam_j, "lam_b": lam_b, "bind_j": bool(info_j["binding"]),
            "cap_j": bool(info_j["capped"]), "bind_b": bool(info_b["binding"]), "floor": bi["floor_active"],
            "T": tgt, "E0": bi["E0"], "l_e": bi["l_e"], "m_sp": mt["speech"], "m_bl": mt["blocks"], "m_tx": mt["text"],
            "m_lam": mt["lam_text"], "m_ok": mt["matched"], **acc}


def pc34(procs: int, steps4: int, tm_d: dict, delta: float) -> dict:
    recs, plan = JPC.load_plan(SCHED)
    used = sorted({i for s in plan for i in s})
    t0 = time.time()
    JPC._init(recs, plan, MODE, tm_d)          # j1_precheck's workers (sizes) read JPC._G
    _G.update(recs=recs, plan=plan, tm=JS.TalkerModel(tm_d), delta=delta)
    with get_context("fork").Pool(procs) as pool:
        sz = pool.map(JPC._size_job, used, chunksize=16)
        stp = pool.map(_step_job, list(range(steps4)), chunksize=1)
    co = sum(r["coarsened"] for r in sz)
    keys = ("words", "j_sp", "j_tx", "j_bl", "b_sp", "b_tx", "b_bl", "m_sp", "m_bl", "m_tx", "T", "E0", "l_e")
    tot = {k: sum(r[k] for r in stp) for k in keys}
    W = tot["words"]
    pc4 = {"steps": steps4, "words": W, "delta": delta,
           "J1v2": {"speech_l_per_word": tot["j_sp"] / W, "text_lat_per_word": tot["j_tx"] / W, "blocks": tot["j_bl"]},
           "base": {"speech_l_per_word": tot["b_sp"] / W, "text_lat_per_word": tot["b_tx"] / W, "blocks": tot["b_bl"]},
           "budget": {"T_per_word": tot["T"] / W, "E0_per_word": tot["E0"] / W, "label_path_per_word": tot["l_e"] / W,
                      "floor_active_steps": sum(r["floor"] for r in stp)},
           "lam_J1v2": JPC.st([r["lam_j"] for r in stp]), "lam_base": JPC.st([r["lam_b"] for r in stp]),
           "binding_J1v2": sum(r["bind_j"] for r in stp), "capped_J1v2": sum(r["cap_j"] for r in stp),
           "binding_base": sum(r["bind_b"] for r in stp),
           "descriptive_matched_text_timeline": {
               "speech_l_per_word": tot["m_sp"] / W, "text_lat_per_word": tot["m_tx"] / W, "blocks": tot["m_bl"],
               "lam_text": JPC.st([r["m_lam"] for r in stp]), "unmatched_steps": sum(not r["m_ok"] for r in stp)}}
    pc4["i_speech"] = bool(pc4["J1v2"]["speech_l_per_word"] < pc4["base"]["speech_l_per_word"])
    pc4["ii_blocks"] = bool(pc4["J1v2"]["blocks"] < pc4["base"]["blocks"])
    pc4["iii_text"] = bool(pc4["J1v2"]["text_lat_per_word"] <= pc4["base"]["text_lat_per_word"] + TH["pc4_text_slack"])
    pc4["pass"] = pc4["i_speech"] and pc4["ii_blocks"] and pc4["iii_text"]
    step_t = [r["t_step"] for r in stp]
    pc3 = {"windows": len(sz), "coarsened": co, "coarsened_share": round(co / len(sz), 4),
           "delta_hist": {str(d): sum(1 for r in sz if r["delta"] == d) for d in sorted({r["delta"] for r in sz})},
           "aug_states_max_per_window": JPC.st([r["states_max"] for r in sz]), "aug_edges": JPC.st([r["edges"] for r in sz]),
           "t_parts_per_window_s": JPC.st([r["t_parts"] for r in sz]), "step_cpu_s_first_steps": JPC.st(step_t)}
    pc3["pass"] = bool(pc3["coarsened_share"] <= TH["pc3_coarse"] and pc3["step_cpu_s_first_steps"]["p95"]
                       <= TH["pc3_step_p95_s"])
    return {"sched": SCHED, "mode": MODE, "steps_plan": len(plan), "PC3": pc3, "PC4": pc4,
            "cpu_s": round(time.time() - t0, 1)}


def pc1(logs: list[str]) -> dict:
    out = {}
    for f in logs:
        lines = [l.strip() for l in open(f, encoding="utf-8", errors="replace") if l.strip()]
        out[f] = bool(lines) and any(l.startswith("ALL") and "PASSED" in l for l in lines[-3:])
    return {"logs": out, "pass": bool(out) and all(out.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pc1-logs", nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--steps4", type=int, default=200)
    ap.add_argument("--delta", type=float, default=JS.JS_DELTA)
    ap.add_argument("--talker-model", default=JP.TALKER_MODEL)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    if a.delta != JS.JS_DELTA:
        raise SystemExit("delta is registered as 0.134")
    scratch = Path(str(out) + ".work")
    scratch.mkdir(parents=True)
    tm_d = json.loads(Path(a.talker_model).read_text())
    res = {"thresholds": TH, "talker_model": a.talker_model, "prereg": "thinker_joint_crf_v2_prereg_v1 (625a1926)"}
    res["PC1"] = pc1(a.pc1_logs)
    print(json.dumps({"PC1": res["PC1"]}), flush=True)
    JPC.PC2_RUNS = PC2_RUNS_V2
    res["PC2"] = JPC.pc2(scratch, JS.TalkerModel(tm_d))
    print(json.dumps({"PC2": res["PC2"]}), flush=True)
    res["v2p"] = pc34(a.procs, a.steps4, tm_d, a.delta)
    print(json.dumps({"v2p": res["v2p"]}), flush=True)
    res["pass"] = bool(res["PC1"]["pass"] and res["PC2"]["pass"] and res["v2p"]["PC3"]["pass"]
                       and res["v2p"]["PC4"]["pass"])
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"J1V2_PRECHECK pass={res['pass']}", flush=True)


if __name__ == "__main__":
    main()
