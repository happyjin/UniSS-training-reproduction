"""Robust wave 12 scoring (reports/mist_w3po_v1/robust_wave12_overfit_diag_prereg.zh-CN.md): text BLEU of ERM's
checkpoints 500 / 1000 / 1500 / 2000 / final (all alpha = 1, Thinker only, (a) unit) against the mean of the 3 base runs
(wave2_S1, base_rerun1, base_rerun2; wave 5 change 3), plus final - ckpt_00500. Same cell as wave11_score.py /
wave5_m_score.py: normalize_for_bleu, sacrebleu zh / 13a (wave2_s1_score.Bleu), units = sentences (stream of set a),
paired bootstrap 2000, seed 0. final = wave5/M/eval_erm (reused). Self-check: final's erm - base_mean in the 4 (a) cells
must equal wave5/M/summary.json, else no output.
Verdict (implementation notes of the prereg): curve = zh2en/a/clean text diff vs base_mean over 500..final;
monotone = at most one of the 4 steps rises and that rise <= 0.3; early = ckpt_00500 - base_mean CI upper < 0;
early -> "early shift, not pure overfitting"; else monotone and (final - ckpt_00500) CI upper < 0 -> "supports
overfitting"; else "does not support overfitting".
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave12_score.py --out eval_outputs/mist_w3po_v1/wave12/summary.json
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
CKPTS = ("ckpt_00500", "ckpt_01000", "ckpt_01500", "ckpt_02000", "erm")
ROOTS = {"base": E / "wave2_S1", "base_r1": E / "wave5/base_rerun1", "base_r2": E / "wave5/base_rerun2",
         **{c: E / f"wave12/eval_{c}" for c in CKPTS[:-1]}, "erm": E / "wave5/M/eval_erm"}
BASES = ("base", "base_r1", "base_r2")
NB = 2000


def cell(meta, pers, lang, cond):
    bl = S.Bleu(lang)
    ids = sorted(i for i, m in meta.items() if m["set"] == "a" and m["cond"] == cond)
    units = sorted({meta[i]["stream"] for i in ids})
    up = {u: k for k, u in enumerate(units)}
    pu = np.array([up[meta[i]["stream"]] for i in ids])
    st = {}
    for sysn, per in pers.items():
        x = bl.stats([per[i] for i in ids], [meta[i]["ref"] for i in ids])
        U = np.zeros((len(units), x.shape[1]), dtype=np.int64)
        np.add.at(U, pu, x)
        st[sysn] = U
    rng = np.random.default_rng(0)
    W = [np.ones(len(units))] + [np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)).astype(float)
                                 for _ in range(NB)]
    vals = {s: np.array([bl.score((st[s] * w[:, None]).sum(0)) for w in W]) for s in pers}
    vals["base_mean"] = np.mean([vals[b] for b in BASES], axis=0)
    r = {"n_sentences": len(ids), "n_units": len(units), "bleu": {s: round(float(v[0]), 3) for s, v in vals.items()},
         "base_floor": round(max(abs(vals[x][0] - vals[y][0]) for i, x in enumerate(BASES) for y in BASES[i + 1:]), 3)}

    def ci(d):
        return [round(float(d[0]), 3), round(float(np.percentile(d[1:], 2.5)), 3), round(float(np.percentile(d[1:], 97.5)), 3)]
    r["vs_base_mean"] = {c: ci(vals[c] - vals["base_mean"]) for c in CKPTS}
    r["final-ckpt_00500"] = ci(vals["erm"] - vals["ckpt_00500"])
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    ref5 = json.load(open(E / "wave5/M/summary.json"))
    res = {"systems": {k: str(v.relative_to(REPO)) for k, v in ROOTS.items()}, "unit": "a", "metric": "text BLEU"}
    for d in ("en2zh", "zh2en"):
        data = E / "wave2_S1/data"
        meta = {m["id"]: m for m in S.jl(data / f"meta_{d}.jsonl") if m["set"] == "a"}
        items = {r["id"]: r for r in S.jl(data / f"items_{d}.jsonl")}
        for i in meta:
            meta[i]["ref"] = items[i]["translation"]
        pers = {}
        for sysn, root in ROOTS.items():
            blk = {r["id"]: "".join(b["text"] for b in r["blocks"]) for r in S.jl(root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
            assert set(blk) >= set(meta), f"{sysn} {d}: missing items"
            pers[sysn] = {i: blk[i] for i in meta}
        res[d] = {c: cell(meta, pers, S.LANG[d], c) for c in ("clean", "noisy")}
        for c in ("clean", "noisy"):   # self-check against wave 5
            mine, theirs = res[d][c]["vs_base_mean"]["erm"], ref5[d]["a"][c]["text"]["erm-base_mean"]
            assert mine == theirs, f"self-check {d}/{c}: {mine} != wave5 {theirs}"
            assert res[d][c]["bleu"]["erm"] == ref5[d]["a"][c]["text"]["erm"]
    res["self_check_vs_wave5_summary"] = "erm - base_mean (a) x 2 dirs x clean/noisy identical to wave5/M/summary.json"
    z = res["zh2en"]["clean"]
    curve = [z["vs_base_mean"][c][0] for c in CKPTS]
    rises = [round(curve[k + 1] - curve[k], 3) for k in range(4) if curve[k + 1] > curve[k]]
    monotone = len(rises) == 0 or (len(rises) == 1 and rises[0] <= 0.3)
    fin_hi = z["final-ckpt_00500"][2]
    early = z["vs_base_mean"]["ckpt_00500"][2] < 0
    v = ("早期就有的偏移（风格或领域），不支持纯过拟合" if early else
         "支持过拟合" if monotone and fin_hi < 0 else "不支持过拟合")
    res["verdict"] = {"zh2en_a_clean_curve_500..final": curve, "rises": rises, "monotone_allow_one_rise_le_0.3": monotone,
                      "final-ckpt_00500_ci_upper_lt_0": fin_hi < 0, "early_sig_drop_at_500": early, "VERDICT": v}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
