"""Robust wave 10 mechanism measurements (CPU, auxiliary; reports/mist_w3po_v1/robust_wave10_ties_prereg.zh-CN.md).

tau_d = (alpha / rank) b_d a_d of the two single-direction LoRAs; Delta_erm = the joint wave-5 ERM LoRA delta (its own
alpha / rank). Per text layer (the 4 adapted matrices q/k/v/o concatenated) and over all layers:
  cos(tau_zh2en, tau_en2zh);
  after the TIES trim (top 20 % |.| per matrix, as wave10_ties_merge.py): both_kept = share of entries kept in both,
  conflict = share of the both-kept entries whose signs differ (and the same over the union of kept entries);
  cos(Delta_erm, tau_zh2en + tau_en2zh), and (extra) cos(Delta_erm, tau*) with tau* the TIES-merged vector (lambda 1);
  norms |tau_zh2en|, |tau_en2zh|, |Delta_erm|.
Per projection type (q/k/v/o) the same over all layers. Refuses to overwrite.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave10_mech.py --zh2en .. --en2zh .. --erm .. --out <json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.wave10_ties_merge import taus, ties, trim  # noqa: E402


class Acc:
    def __init__(self):
        self.d = defaultdict(float)

    def add(self, z, e, erm, tstar, k):
        s = z + e
        tz, te = trim(z, k), trim(e, k)
        both = (tz != 0) & (te != 0)
        union = (tz != 0) | (te != 0)
        conf = both & (torch.sign(tz) != torch.sign(te))
        d = self.d
        d["zz"] += float((z * z).sum()); d["ee"] += float((e * e).sum()); d["ze"] += float((z * e).sum())
        d["rr"] += float((erm * erm).sum()); d["ss"] += float((s * s).sum()); d["rs"] += float((erm * s).sum())
        d["tt"] += float((tstar * tstar).sum()); d["rt"] += float((erm * tstar).sum())
        d["n"] += z.numel(); d["both"] += int(both.sum()); d["union"] += int(union.sum()); d["conf"] += int(conf.sum())

    def out(self):
        d = self.d
        c = lambda xy, xx, yy: round(d[xy] / max((d[xx] * d[yy]) ** 0.5, 1e-30), 4)
        return {"cos_zh2en_en2zh": c("ze", "zz", "ee"), "cos_erm_sum": c("rs", "rr", "ss"), "cos_erm_ties": c("rt", "rr", "tt"),
                "both_kept_frac": round(d["both"] / d["n"], 5), "conflict_of_both": round(d["conf"] / max(d["both"], 1), 4),
                "conflict_of_union": round(d["conf"] / max(d["union"], 1), 5),
                "norm_zh2en": round(d["zz"] ** 0.5, 4), "norm_en2zh": round(d["ee"] ** 0.5, 4), "norm_erm": round(d["rr"] ** 0.5, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh2en", required=True)
    ap.add_argument("--en2zh", required=True)
    ap.add_argument("--erm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=0.2)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    Z, E, R = taus(a.zh2en), taus(a.en2zh), taus(a.erm)
    assert set(Z) == set(E) == set(R)
    layer, proj, tot = defaultdict(Acc), defaultdict(Acc), Acc()
    for key in sorted(Z, key=lambda x: (int(re.search(r"layers\.(\d+)\.", x).group(1)), x)):
        li = int(re.search(r"layers\.(\d+)\.", key).group(1))
        pj = re.search(r"\.(\w)_proj\.", key).group(1)
        tstar, _ = ties([Z[key], E[key]], a.k, 1.0)
        for acc in (layer[li], proj[pj], tot):
            acc.add(Z[key], E[key], R[key], tstar, a.k)
    res = {"k": a.k, "inputs": {"zh2en": a.zh2en, "en2zh": a.en2zh, "erm": a.erm}, "all": tot.out(),
           "per_proj": {p: v.out() for p, v in sorted(proj.items())},
           "per_layer": {str(li): layer[li].out() for li in sorted(layer)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    pl = res["per_layer"]
    q = lambda f: [pl[str(i)][f] for i in sorted(layer)]
    print(json.dumps({"all": res["all"], "per_proj": res["per_proj"],
                      "per_layer_cos_zh2en_en2zh_min_med_max": [min(q("cos_zh2en_en2zh")), sorted(q("cos_zh2en_en2zh"))[len(pl) // 2], max(q("cos_zh2en_en2zh"))],
                      "per_layer_conflict_of_both_min_med_max": [min(q("conflict_of_both")), sorted(q("conflict_of_both"))[len(pl) // 2], max(q("conflict_of_both"))],
                      "per_layer_cos_erm_sum_min_med_max": [min(q("cos_erm_sum")), sorted(q("cos_erm_sum"))[len(pl) // 2], max(q("cos_erm_sum"))]},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
