"""Robust wave 17 mechanism checks (reports/mist_w3po_v1/robust_wave17_enclora_prereg.zh-CN.md, implementation note 6;
predictions written in robust_mainline_plan_20261120 4.3). Text BLEU (wave2_s1_score.Bleu: normalize_for_bleu, sacrebleu
zh / 13a), paired bootstrap 2000, seed 0, units = sentences; the same resamples for every system of a cell (wave13_score.py).
  Prediction 2 (no memorisation in the encoder): on wave 17's 600 zh2en UniST training clips (wave17_items.py),
      enc - base (la2 w1; wave17/la2_{base,enc}_train600); "memorisation" = point >= 3.0 and CI lower > 0 (wave 13 H4
      threshold); the prediction holds unless that is met. Shown next to wave 13's erm - base on its CoVoST train600
      (wave13/summary.json). Control (auxiliary): unist - base on the same clips (wave17/la2_unist_train600, if run).
      Auxiliary split: clips the sampler drew at least once (n_sampled > 0) / never.
  Prediction 3 (en2zh noisy gain from offline translation, not from committing): set a of both directions x clean/noisy
      offline = oracle-commit output (base: wave3_combos/D/oracle_<dir>; enc: wave17/oracle_enc_a), checked == final
      draft row by row; la2 = base_mean of 3 base runs / wave17/eval_enc; CL = offline - la2, dCL = CL(enc) - CL(base),
      la2 diff = offline diff - dCL. en2zh/noisy holds if offline diff CI lower > 0 and offline diff >= -dCL; "not
      applicable" if the en2zh/noisy la2 diff CI lower <= 0 (no gain to decompose).
Self-check: base offline equals wave3 D/oracle.json to 2 decimals; base la2 values equal wave13/summary.json.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave17_mech_score.py --out eval_outputs/mist_w3po_v1/wave17/mech.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave2_s1_score as S  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave13_score import boot, ci, load  # noqa: E402

E = REPO / "eval_outputs/mist_w3po_v1"
W17 = E / "wave17"
BASES = ("base", "base_r1", "base_r2")


def la2_path(sysn, d):
    root = {"base": E / "wave2_S1", "base_r1": E / "wave5/base_rerun1", "base_r2": E / "wave5/base_rerun2",
            "enc": W17 / "eval_enc"}[sysn]
    return root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"


def or_path(sysn, d):
    if sysn == "base":
        return E / f"wave3_combos/D/oracle_{d}/blocks_{d}_w1_oraclecommit.jsonl"
    return W17 / f"oracle_{sysn}_a/thinker_{d}/blocks_{d}_w1_oraclecommit.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    ref3 = json.load(open(E / "wave3_combos/D/oracle.json"))
    ref13 = json.load(open(E / "wave13/summary.json"))
    res = {"metric": "text BLEU", "bootstrap": "paired, 2000, seed 0, units = sentences"}
    # ---- prediction 3: oracle decomposition on set a
    dec = {}
    for d in ("zh2en", "en2zh"):
        data = E / "wave2_S1/data"
        meta = {m["id"]: m for m in S.jl(data / f"meta_{d}.jsonl") if m["set"] == "a"}
        items = {r["id"]: r for r in S.jl(data / f"items_{d}.jsonl")}
        refs = {i: items[i]["translation"] for i in meta}
        pers = {f"la2_{s}": load(la2_path(s, d)) for s in BASES + ("enc",)}
        pers.update({f"off_{s}": load(or_path(s, d), offline=True) for s in ("base", "enc")})
        for k, v in pers.items():
            assert set(v) >= set(meta), f"{k} {d}: missing items"
        dec[d] = {}
        for c in ("clean", "noisy"):
            ids = sorted(i for i, m in meta.items() if m["cond"] == c)
            vals, nu = boot(S.Bleu(S.LANG[d]), ids, {i: meta[i]["stream"] for i in ids}, refs, pers)
            vals["la2_base_mean"] = np.mean([vals[f"la2_{b}"] for b in BASES], axis=0)
            cl_b, cl_e = vals["off_base"] - vals["la2_base_mean"], vals["off_enc"] - vals["la2_enc"]
            r = {"n_sentences": len(ids), "n_units": nu, "bleu": {k: round(float(v[0]), 3) for k, v in vals.items()},
                 "offline_diff_enc-base": ci(vals["off_enc"] - vals["off_base"]),
                 "la2_diff_enc-base_mean": ci(vals["la2_enc"] - vals["la2_base_mean"]),
                 "commit_loss": {"base": ci(cl_b), "enc": ci(cl_e)},
                 "dCL_enc-base": ci(cl_e - cl_b)}
            assert round(r["bleu"]["off_base"], 2) == round(ref3[d]["a"]["oracle"][c], 2), (d, c, r["bleu"]["off_base"])
            assert r["bleu"]["la2_base_mean"] == ref13[d][c]["bleu"]["la2_base_mean"], (d, c)
            dec[d][c] = r
    res["oracle_decomposition_set_a"] = dec
    z = dec["en2zh"]["noisy"]
    off, dcl, la2 = z["offline_diff_enc-base"], z["dCL_enc-base"], z["la2_diff_enc-base_mean"]
    if la2[1] <= 0:
        p3 = "不适用（英→中带噪的 LA2 增益不显著，没有可分解的增益）"
    else:
        p3 = "成立" if (off[1] > 0 and off[0] >= -dcl[0]) else "不成立"
    res["prediction3_en2zh_noisy"] = {"offline_diff": off, "minus_dCL (commit contribution)": round(-dcl[0], 3),
                                      "la2_diff": la2, "VERDICT": p3}
    # ---- prediction 2: training subset
    ti = W17 / "data/items_zh2en_train600.jsonl"
    titems = {r["id"]: r for r in S.jl(ti)}
    ids = sorted(titems)
    sysl = ["base", "enc"] + (["unist"] if (W17 / "la2_unist_train600/thinker_zh2en/blocks_zh2en_w1.jsonl").exists() else [])
    tp = {s: load(W17 / f"la2_{s}_train600/thinker_zh2en/blocks_zh2en_w1.jsonl") for s in sysl}
    for s, v in tp.items():
        assert set(v) == set(ids), f"train600 {s}: rows differ"
    refs = {i: titems[i]["translation"] for i in ids}
    t = {}
    for name, sub in (("all", ids), ("sampled", [i for i in ids if titems[i]["n_sampled"] > 0]),
                      ("never_sampled", [i for i in ids if titems[i]["n_sampled"] == 0])):
        tv, nu = boot(S.Bleu("eng"), sub, {i: i for i in sub}, refs, tp)
        t[name] = {"n_sentences": len(sub), "bleu": {k: round(float(v[0]), 3) for k, v in tv.items()},
                   "enc-base": ci(tv["enc"] - tv["base"])}
        if "unist" in tv:
            t[name]["unist-base"] = ci(tv["unist"] - tv["base"])
            t[name]["enc-unist"] = ci(tv["enc"] - tv["unist"])
    eb = t["all"]["enc-base"]
    mem = bool(eb[0] >= 3.0 and eb[1] > 0)
    res["train600_zh2en_clean_unist"] = t
    res["prediction2"] = {"enc-base": eb, "wave13_erm-base_covost_train600": ref13["train600_zh2en_clean"]["erm-base"],
                          "control_unist-base": t["all"].get("unist-base"), "memorisation_threshold": "point >= 3.0 and CI lower > 0",
                          "VERDICT": "不成立（编码器有记忆）" if mem else "成立"}
    res["self_check"] = "base offline equals wave3 D/oracle.json; base la2 mean equals wave13/summary.json; offline == final draft"
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("prediction2", "prediction3_en2zh_noisy")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
