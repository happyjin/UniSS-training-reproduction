"""Robust wave 13 scoring (reports/mist_w3po_v1/robust_wave13_cleandrop_diag_prereg.zh-CN.md, implementation notes):
where ERM's clean zh2en drop comes from. Text BLEU in the wave-5 cell (wave2_s1_score.Bleu: normalize_for_bleu,
sacrebleu zh / 13a; units = sentences (stream of set a); paired bootstrap 2000, seed 0; the same resamples for every
system of a cell).
  offline  = the oracle-commit run's output (== its final draft after the whole sentence, checked row by row):
             base = wave3_combos/D/oracle_<dir> (single run), erm / erm_wise05 = wave13/oracle_<sys>_a
  la2      = base_mean of 3 base runs (S1, base_rerun1, base_rerun2; S1 alone as sensitivity), erm / erm_wise05 from
             wave5/M/eval_*, dzh = wave13/la2_dzh_a (zh2en only)
  commit loss CL = offline - la2 of the same system; dCL(sys) = CL(sys) - CL(base)
  train600 = wave13/la2_{base,erm}_train600 (600 clean zh2en training clips; no base noise floor)
Self-checks (else no output): erm / erm_wise05 la2 and their - base_mean equal wave5/M/summary.json; base offline
zh2en/a/clean equals wave3 D/oracle.json to 2 decimals; offline == last draft for every scored row.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave13_score.py --out eval_outputs/mist_w3po_v1/wave13/summary.json
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

E = REPO / "eval_outputs/mist_w3po_v1"
W13 = E / "wave13"
NB = 2000
BASES = ("base", "base_r1", "base_r2")


def la2_path(sysn, d):
    root = {"base": E / "wave2_S1", "base_r1": E / "wave5/base_rerun1", "base_r2": E / "wave5/base_rerun2",
            "erm": E / "wave5/M/eval_erm", "erm_wise05": E / "wave5/M/eval_erm_wise05", "dzh": W13 / "la2_dzh_a"}[sysn]
    return root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"


def or_path(sysn, d):
    if sysn == "base":
        return E / f"wave3_combos/D/oracle_{d}/blocks_{d}_w1_oraclecommit.jsonl"
    return W13 / f"oracle_{sysn}_a/thinker_{d}/blocks_{d}_w1_oraclecommit.jsonl"


def load(p, offline=False):
    out = {}
    for r in S.jl(p):
        t = "".join(b["text"] for b in r["blocks"])
        if offline:
            assert t.strip() == r["drafts"][-1]["draft"].strip(), f"{p} {r['id']}: output != final draft"
        out[r["id"]] = t
    return out


def boot(bl, ids, units_of, refs, pers):
    units = sorted(set(units_of[i] for i in ids))
    up = {u: k for k, u in enumerate(units)}
    pu = np.array([up[units_of[i]] for i in ids])
    rng = np.random.default_rng(0)
    W = [np.ones(len(units))] + [np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)).astype(float)
                                 for _ in range(NB)]
    vals = {}
    for s, per in pers.items():
        x = bl.stats([per[i] for i in ids], [refs[i] for i in ids])
        U = np.zeros((len(units), x.shape[1]), dtype=np.int64)
        np.add.at(U, pu, x)
        vals[s] = np.array([bl.score((U * w[:, None]).sum(0)) for w in W])
    return vals, len(units)


def ci(v):
    return [round(float(v[0]), 3), round(float(np.percentile(v[1:], 2.5)), 3), round(float(np.percentile(v[1:], 97.5)), 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    ref5 = json.load(open(E / "wave5/M/summary.json"))
    ref3 = json.load(open(E / "wave3_combos/D/oracle.json"))
    res = {"metric": "text BLEU", "unit": "a", "bootstrap": "paired, 2000, seed 0, units = sentences"}
    for d in ("zh2en", "en2zh"):
        data = E / "wave2_S1/data"
        meta = {m["id"]: m for m in S.jl(data / f"meta_{d}.jsonl") if m["set"] == "a"}
        items = {r["id"]: r for r in S.jl(data / f"items_{d}.jsonl")}
        refs = {i: items[i]["translation"] for i in meta}
        la2s = BASES + ("erm", "erm_wise05") + (("dzh",) if d == "zh2en" else ())
        pers = {f"la2_{s}": load(la2_path(s, d)) for s in la2s}
        pers.update({f"off_{s}": load(or_path(s, d), offline=True) for s in ("base", "erm", "erm_wise05")})
        for k, v in pers.items():
            assert set(v) >= set(meta), f"{k} {d}: missing items"
        res[d] = {}
        for c in ("clean", "noisy"):
            ids = sorted(i for i, m in meta.items() if m["cond"] == c)
            vals, nu = boot(S.Bleu(S.LANG[d]), ids, {i: meta[i]["stream"] for i in ids}, refs, pers)
            vals["la2_base_mean"] = np.mean([vals[f"la2_{b}"] for b in BASES], axis=0)
            r = {"n_sentences": len(ids), "n_units": nu, "bleu": {k: round(float(v[0]), 3) for k, v in vals.items()},
                 "la2_base_floor": round(max(abs(vals[f"la2_{x}"][0] - vals[f"la2_{y}"][0])
                                             for i, x in enumerate(BASES) for y in BASES[i + 1:]), 3)}
            cl = {"base": vals["off_base"] - vals["la2_base_mean"], "base_S1": vals["off_base"] - vals["la2_base"],
                  "erm": vals["off_erm"] - vals["la2_erm"], "erm_wise05": vals["off_erm_wise05"] - vals["la2_erm_wise05"]}
            r["offline_diff"] = {"erm-base": ci(vals["off_erm"] - vals["off_base"]),
                                 "erm_wise05-base": ci(vals["off_erm_wise05"] - vals["off_base"]),
                                 "erm_wise05-erm": ci(vals["off_erm_wise05"] - vals["off_erm"])}
            r["la2_diff"] = {f"{s}-base_mean": ci(vals[f"la2_{s}"] - vals["la2_base_mean"]) for s in la2s[3:]}
            r["la2_diff"].update({f"{s}-base_S1": ci(vals[f"la2_{s}"] - vals["la2_base"]) for s in la2s[3:]})
            r["commit_loss"] = {k: ci(v) for k, v in cl.items()}
            r["commit_loss_diff"] = {"erm-base": ci(cl["erm"] - cl["base"]), "erm_wise05-base": ci(cl["erm_wise05"] - cl["base"]),
                                     "erm_wise05-erm": ci(cl["erm_wise05"] - cl["erm"]),
                                     "erm-base_S1": ci(cl["erm"] - cl["base_S1"]),
                                     "erm_wise05-base_S1": ci(cl["erm_wise05"] - cl["base_S1"])}
            res[d][c] = r
            t5 = ref5[d]["a"][c]["text"]   # self-checks against wave 5 / wave 3
            for s in ("erm", "erm_wise05"):
                assert r["bleu"][f"la2_{s}"] == t5[s], (d, c, s, r["bleu"][f"la2_{s}"], t5[s])
                assert r["la2_diff"][f"{s}-base_mean"] == t5[f"{s}-base_mean"], (d, c, s, r["la2_diff"][f"{s}-base_mean"], t5[f"{s}-base_mean"])
            assert round(r["bleu"]["off_base"], 2) == round(ref3[d]["a"]["oracle"][c], 2), (d, c, r["bleu"]["off_base"])
    res["self_check"] = ("erm / erm_wise05 LA2 and - base_mean identical to wave5/M/summary.json (2 dirs x 2 conds); base "
                         "offline equals wave3 D/oracle.json; offline == final draft on every row")
    # H4: training subset
    ti = W13 / "data/items_zh2en_train600.jsonl"
    titems = {r["id"]: r for r in S.jl(ti)}
    tp = {s: load(W13 / f"la2_{s}_train600/thinker_zh2en/blocks_zh2en_w1.jsonl") for s in ("base", "erm")}
    ids = sorted(titems)
    for s, v in tp.items():
        assert set(v) == set(ids), f"train600 {s}: rows differ"
    tv, nu = boot(S.Bleu("eng"), ids, {i: i for i in ids}, {i: titems[i]["translation"] for i in ids}, tp)
    res["train600_zh2en_clean"] = {"n_sentences": len(ids), "n_units": nu, "n_speakers_note": "22 (all wave-5 zh2en speakers)",
                                   "bleu": {k: round(float(v[0]), 3) for k, v in tv.items()},
                                   "erm-base": ci(tv["erm"] - tv["base"]), "noise_floor": "not available (single base run)"}
    # verdicts (prereg rules; implementation notes)
    z = res["zh2en"]["clean"]
    off, dcl = z["offline_diff"]["erm-base"], z["commit_loss_diff"]["erm-base"]
    dzh, erm = z["la2_diff"]["dzh-base_mean"], z["la2_diff"]["erm-base_mean"]
    t4 = res["train600_zh2en_clean"]["erm-base"]
    h3 = ("成立" if dzh[0] >= -0.3 and erm[0] <= -2.0 else "不成立" if dzh[0] <= -2.0 else "不确定")
    res["verdict"] = {
        "H1": {"offline_erm-base": off, "dCL_erm-base": dcl,
               "holds": bool(off[0] >= -1.0 and off[2] >= 0 and dcl[0] >= 2.0)},
        "H2": {"offline_erm-base": off, "holds": bool(off[0] <= -2.0 and off[2] < 0)},
        "H3": {"dzh-base_mean": dzh, "erm-base_mean": erm, "VERDICT": h3},
        "H4": {"train600_erm-base": t4, "holds": bool(t4[0] >= 3.0 and t4[1] > 0)},
    }
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
