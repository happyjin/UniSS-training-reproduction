"""J1's CPU precheck PC2-PC4 (reports/mist_w3po_v1/thinker_joint_crf_prereg_v1.zh-CN.md §5; thresholds there and in
TH below, fixed in the pre-registration d942b122 before any number). PC1 = the unit tests (crf/test_j1.py and the
regression runs of test_joint_speech_crf / test_train_crf_sft / test_merge_read_segments).

  PC2  the talker model (T3's json, unchanged) on the EVAL talker c1+b1+s2: A' (seed 1) and T2 (seed 1, fragmented);
       every block's dur_ms replaced by the model's expected duration at the model's own backlog (grid queue),
       speech_latency_lit unchanged: |model - actual| per-stream speech LAAL (all) <= 0.30 s and unit Pearson r >= 0.8,
       both runs
  PC3  both candidate J1 data sets (T4: sched_piece_t0_g1000_v1, lattice t2; T5: sched_t5_pieces4_w10_v1, lattice t5),
       every window of the 3000-step plan: share of windows whose augmented chain needed a coarser grid <= 2 %; the
       joint chain's CPU time per step (window_parts + lam solve + posterior of every window of the step, one process,
       neutral Thinker scores psi = 0) over the first 200 steps: p95 <= 3.0 s
  PC4  first 200 steps, neutral scores: J1's posterior (speech-timeline l, talker factor, its own lam*: T1 sub-lattice
       target) vs the base's (the same chain with the playback terms zeroed = the T4 / T5 lattice posterior on the text
       timeline, its own lam*): (i) E[speech l] / word J1 < base; (ii) E[blocks] J1 < base; (iii) E[text Lat] / word
       J1 <= base + 0.25 s. GO on the data set the base decision picks; both reported.

    python -m experiments.mist_w3po_v1.crf.j1_precheck --out <new json> [--procs 64] [--steps4 200]
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import joint_speech_checks as JC
from experiments.mist_w3po_v1.crf import joint_speech_crf as JS
from experiments.mist_w3po_v1.crf import joint_speech_proxy as JP

TH = {"pc2_laal": 0.30, "pc2_r": 0.8, "pc3_coarse": 0.02, "pc3_step_p95_s": 3.0, "pc4_text_slack": 0.25}
C = "eval_outputs/mist_w3po_v1/crf"
SCHEDS = {"t4": (f"{C}/sched_piece_t0_g1000_v1", "t2"), "t5": (f"{C}/sched_t5_pieces4_w10_v1", "t5")}
WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
SP = "eval_outputs/mist_w3po_v1/speech"
PC2_RUNS = {"Aprime": ("eval_outputs/mist_w3po_v1/decode/dev_crf_v2p_s3000_finalpost_s8_c2g06",
                       f"{SP}/dev_crf_v2p_s3000_finalpost_s8_c2g06/talker"),
            "T2": ("eval_outputs/mist_w3po_v1/decode/dev_t2_latent_t0_s3000_w05",
                   f"{SP}/dev_t2_latent_t0_s3000_w05/talker")}
_G: dict = {}


class W_:
    def __init__(self, rec, idx=0):
        self.idx, self.rec = idx, rec


def load_plan(sched: str, steps: int = 3000) -> tuple[list[dict], list[list[int]]]:
    """the trainer's windows (train_crf_sft.load_windows order) and its plan_steps(seed 20261109, 64 rows)."""
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    recs = []
    for f in sorted(Path(sched).glob("shard_*.jsonl")):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    recs = [r for r in recs if r["n_tok"] > 0]
    recs.sort(key=lambda r: r["id"])
    return recs, T.plan_steps([r["n_read"] for r in recs], 20261109, steps, 64)


def neutral_plan(rec: dict, mode: str) -> dict:
    """the lattice plan with every request unscored: LW.full_res gives psi = 0 everywhere."""
    from experiments.mist_w3po_v1.crf import latent_write_crf as LW

    pl = LW.plan_window(rec, LW.lattice(rec, mode), skip_forced=False, mode=mode)
    for q in pl["reqs"]:
        q["scored"] = False
    return pl


# ------------------------------------------------------------------------------------------------ PC2
def pc2(scratch: Path, tm: JS.TalkerModel) -> dict:
    wins = {w["id"]: w for w in map(json.loads, open(WIN, encoding="utf-8"))}
    out = {}
    for name, (run, talker) in PC2_RUNS.items():
        d = scratch / f"pc2_{name}_model_talker"
        JP.write_rows(JP.model_durations_on_talker(talker, wins, tm), d)
        runs = JC._laal_runs(run, [(talker, str(scratch / f"pc2_{name}_actual.json")),
                                   (str(d), str(scratch / f"pc2_{name}_model.json"))], WIN)
        x, y = np.asarray(runs[0]["units"]), np.asarray(runs[1]["units"])
        assert len(x) == len(y)
        r = float(np.corrcoef(x, y)[0, 1])
        diff = runs[1]["mean"] - runs[0]["mean"]
        out[name] = {"actual_speech_laal": runs[0]["mean"], "model_speech_laal": runs[1]["mean"], "diff_s": round(diff, 3),
                     "units": len(x), "pearson_r": round(r, 4),
                     "pass": bool(abs(diff) <= TH["pc2_laal"] and r >= TH["pc2_r"])}
    out["pass"] = all(v["pass"] for v in out.values())
    return out


# ------------------------------------------------------------------------------------------------ PC3 sizes
def _size_job(i: int) -> dict:
    rec, mode, tm = _G["recs"][i], _G["mode"], _G["tm"]
    pl = neutral_plan(rec, mode)
    t0 = time.perf_counter()
    parts = JS.window_parts(W_(rec), pl, {}, tm, JS.TAU)
    a = parts["aug"]
    return {"i": i, "states_max": int(max(a["states"])), "coarsened": bool(a["coarsened"]), "delta": a["delta"],
            "edges": a["edges"], "t_parts": time.perf_counter() - t0}


# ------------------------------------------------------------------------------------------------ PC4 + step CPU
def _expect(aug: dict, fb: dict, l_of: dict, rec: dict, pl: dict) -> tuple[float, float]:
    """(E_q[l] with l from another chain of the same structure, E_q[blocks]) under the posterior fb of aug."""
    el, eb = 0.0, 0.0
    for r, (rd, ro) in enumerate(zip(aug["reads"], l_of["reads"])):
        assert len(rd["l"]) == len(ro["l"]) and np.array_equal(rd["te"], ro["te"]) and np.array_equal(rd["src"], ro["src"])
        el += float((fb["q"][r] * ro["l"]).sum())
        qe = np.bincount(rd["te"], weights=fb["q"][r], minlength=rd["nprev"] * rd["ncur"]).reshape(rd["nprev"],
                                                                                                  rd["ncur"])
        eb += float((qe * JS.edge_blocks(rec, pl, r, rd["nprev"], rd["ncur"])).sum())
    return el, eb


def _step_job(s: int) -> dict:
    ids, mode, tm = _G["plan"][s], _G["mode"], _G["tm"]
    recs = [_G["recs"][i] for i in ids]
    pls = [neutral_plan(rec, mode) for rec in recs]
    t0 = time.perf_counter()     # J1's per-step CPU work: parts + lam* + posterior of every window
    full = [JS.window_parts(W_(rec), pl, {}, tm, JS.TAU) for rec, pl in zip(recs, pls)]
    lam_j, info_j = JS.solve_lambda(full)
    for rec, pl, p in zip(recs, pls, full):
        JS.posterior_from_parts(W_(rec), pl, p, JS.TAU, lam_j)
    t_step = time.perf_counter() - t0
    zp = [JS.window_parts(W_(rec), pl, {}, tm, JS.TAU, zero_play=True) for rec, pl in zip(recs, pls)]
    lam_b, info_b = JS.solve_lambda(zp)
    acc = {"words": 0.0, "j_sp": 0.0, "j_tx": 0.0, "j_bl": 0.0, "b_sp": 0.0, "b_tx": 0.0, "b_bl": 0.0}
    for rec, pl, pf, pz in zip(recs, pls, full, zp):
        fj = JS.forward_backward(pf["aug"], lam_j)
        fb = JS.forward_backward(pz["aug"], lam_b)
        acc["j_sp"] += fj["Eq_l"]
        acc["b_tx"] += fb["Eq_l"]
        tx, bl = _expect(pf["aug"], fj, pz["aug"], rec, pl)
        acc["j_tx"] += tx
        acc["j_bl"] += bl
        sp, bl2 = _expect(pz["aug"], fb, pf["aug"], rec, pl)
        acc["b_sp"] += sp
        acc["b_bl"] += bl2
        acc["words"] += float(sum(len(c["text"].split()) if "n_words" not in c else c["n_words"] for c in rec["chunks"]))
    return {"step": s, "t_step": t_step, "lam_j": lam_j, "lam_b": lam_b, "bind_j": bool(info_j["binding"]),
            "bind_b": bool(info_b["binding"]), **acc}


def _init(recs, plan, mode, tm_d):
    _G.update(recs=recs, plan=plan, mode=mode, tm=JS.TalkerModel(tm_d))


def st(v) -> dict:
    v = np.asarray(v, np.float64)
    return {"mean": round(float(v.mean()), 4), "p50": round(float(np.median(v)), 4),
            "p95": round(float(np.percentile(v, 95)), 4), "max": round(float(v.max()), 4), "n": int(len(v))}


def pc34(name: str, procs: int, steps4: int, tm_d: dict) -> dict:
    sched, mode = SCHEDS[name]
    recs, plan = load_plan(sched)
    used = sorted({i for s in plan for i in s})
    t0 = time.time()
    _init(recs, plan, mode, tm_d)            # fork: the workers inherit the windows (no pickling)
    with get_context("fork").Pool(procs) as pool:
        sz = pool.map(_size_job, used, chunksize=16)
        stp = pool.map(_step_job, list(range(steps4)), chunksize=1)
    co = sum(r["coarsened"] for r in sz)
    tot = {k: sum(r[k] for r in stp) for k in ("words", "j_sp", "j_tx", "j_bl", "b_sp", "b_tx", "b_bl")}
    W = tot["words"]
    pc4 = {"steps": steps4, "words": W,
           "J1": {"speech_l_per_word": tot["j_sp"] / W, "text_lat_per_word": tot["j_tx"] / W, "blocks": tot["j_bl"]},
           "base": {"speech_l_per_word": tot["b_sp"] / W, "text_lat_per_word": tot["b_tx"] / W, "blocks": tot["b_bl"]},
           "lam_J1": st([r["lam_j"] for r in stp]), "lam_base": st([r["lam_b"] for r in stp]),
           "binding_J1": sum(r["bind_j"] for r in stp), "binding_base": sum(r["bind_b"] for r in stp)}
    pc4["i_speech"] = bool(pc4["J1"]["speech_l_per_word"] < pc4["base"]["speech_l_per_word"])
    pc4["ii_blocks"] = bool(pc4["J1"]["blocks"] < pc4["base"]["blocks"])
    pc4["iii_text"] = bool(pc4["J1"]["text_lat_per_word"] <= pc4["base"]["text_lat_per_word"] + TH["pc4_text_slack"])
    pc4["pass"] = pc4["i_speech"] and pc4["ii_blocks"] and pc4["iii_text"]
    step_t = [r["t_step"] for r in stp]
    pc3 = {"windows": len(sz), "coarsened": co, "coarsened_share": round(co / len(sz), 4),
           "delta_hist": {str(d): sum(1 for r in sz if r["delta"] == d) for d in sorted({r["delta"] for r in sz})},
           "aug_states_max_per_window": st([r["states_max"] for r in sz]), "aug_edges": st([r["edges"] for r in sz]),
           "t_parts_per_window_s": st([r["t_parts"] for r in sz]), "step_cpu_s_first_steps": st(step_t)}
    pc3["pass"] = bool(pc3["coarsened_share"] <= TH["pc3_coarse"] and pc3["step_cpu_s_first_steps"]["p95"]
                       <= TH["pc3_step_p95_s"])
    return {"sched": sched, "mode": mode, "steps_plan": len(plan), "PC3": pc3, "PC4": pc4,
            "cpu_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--steps4", type=int, default=200)
    ap.add_argument("--talker-model", default=JP.TALKER_MODEL)
    ap.add_argument("--only", default="pc2,t4,t5")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    scratch = Path(str(out) + ".work")
    scratch.mkdir(parents=True)
    tm_d = json.loads(Path(a.talker_model).read_text())
    res = {"thresholds": TH, "talker_model": a.talker_model}
    todo = a.only.split(",")
    if "pc2" in todo:
        res["PC2"] = pc2(scratch, JS.TalkerModel(tm_d))
        print(json.dumps({"PC2": res["PC2"]}), flush=True)
    for name in ("t4", "t5"):
        if name in todo:
            res[name] = pc34(name, a.procs, a.steps4, tm_d)
            print(json.dumps({name: {k: v for k, v in res[name].items()}}), flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
