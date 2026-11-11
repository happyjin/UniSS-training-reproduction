"""post-hoc diagnostic (NOT a registered GO line): PC4 with J1's lam target = the base posterior's own E[speech l]
(per window), i.e. J1 at the base's expected speech latency; same steps / neutral scores as j1_precheck PC4."""
import json, sys
from multiprocessing import get_context
import numpy as np
from experiments.mist_w3po_v1.crf import j1_precheck as P
from experiments.mist_w3po_v1.crf import joint_speech_crf as JS

def job(s):
    ids, mode, tm = P._G["plan"][s], P._G["mode"], P._G["tm"]
    recs = [P._G["recs"][i] for i in ids]
    pls = [P.neutral_plan(r, mode) for r in recs]
    full = [JS.window_parts(P.W_(r), pl, {}, tm, JS.TAU) for r, pl in zip(recs, pls)]
    zp = [JS.window_parts(P.W_(r), pl, {}, tm, JS.TAU, zero_play=True) for r, pl in zip(recs, pls)]
    lam_b, _ = JS.solve_lambda(zp)
    acc = dict(words=0.0, b_sp=0.0, b_tx=0.0, b_bl=0.0, m_sp=0.0, m_tx=0.0, m_bl=0.0, t1_sp=0.0)
    fbs = []
    for r, pl, pf, pz in zip(recs, pls, full, zp):
        fb = JS.forward_backward(pz["aug"], lam_b)
        sp, bl = P._expect(pz["aug"], fb, pf["aug"], r, pl)
        acc["b_sp"] += sp; acc["b_tx"] += fb["Eq_l"]; acc["b_bl"] += bl
        acc["t1_sp"] += pf["target"]
        pf["target"] = sp            # matched: J1 held at the base posterior's E[speech l]
        acc["words"] += float(sum(c.get("n_words", len(c["text"].split())) for c in r["chunks"]))
    lam_m, info = JS.solve_lambda(full)
    for r, pl, pf, pz in zip(recs, pls, full, zp):
        fj = JS.forward_backward(pf["aug"], lam_m)
        tx, bl = P._expect(pf["aug"], fj, pz["aug"], r, pl)
        acc["m_sp"] += fj["Eq_l"]; acc["m_tx"] += tx; acc["m_bl"] += bl
    return {**acc, "lam_m": lam_m, "bind": bool(info["binding"])}

if __name__ == "__main__":
    out = {}
    tm_d = json.load(open(P.JP.TALKER_MODEL))
    for name in ("t4", "t5"):
        sched, mode = P.SCHEDS[name]
        recs, plan = P.load_plan(sched)
        P._init(recs, plan, mode, tm_d)
        with get_context("fork").Pool(64) as pool:
            rs = pool.map(job, list(range(200)), chunksize=1)
        t = {k: sum(r[k] for r in rs) for k in ("words", "b_sp", "b_tx", "b_bl", "m_sp", "m_tx", "m_bl", "t1_sp")}
        W = t["words"]
        out[name] = {"base": {"speech_l_per_word": t["b_sp"] / W, "text_lat_per_word": t["b_tx"] / W, "blocks": t["b_bl"]},
                     "J1_matched": {"speech_l_per_word": t["m_sp"] / W, "text_lat_per_word": t["m_tx"] / W, "blocks": t["m_bl"]},
                     "T1_speech_target_per_word": t["t1_sp"] / W, "binding_steps": sum(r["bind"] for r in rs),
                     "lam_mean": float(np.mean([r["lam_m"] for r in rs]))}
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(sys.argv[1], "x"), indent=1)
