"""Audit A8 of plan v9 §11.2: does the diarizer track speakers on the Chinese AMI meetings as well as on the English ones?

Per meeting, Nemotron-3's offline-preset posteriors (> 0.5, 80 ms frames) against the reference speaker activity of the
same meeting, slots mapped to speakers by maximal overlap over the whole meeting (Hungarian), collar 0, overlap scored:
  zh  the Chinese mix (the kept clips on their Chinese times) with its own posteriors
  en  the English IHM-mix (the same utterances on their English times) with the original posteriors
plus the number of slots that are active for at least 1% of the meeting's speech against the number of speakers.
Meetings whose zh DER exceeds en DER by more than --max-gap points are listed as failing.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.audit_zh_diar --mix zh_mix_v5 --tag zh5 --out <json>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_zh_mix import removed

FRAME = 0.08


def der(ref: dict, post: np.ndarray) -> tuple[float, int]:
    on = post > 0.5
    n = len(on)
    spk = sorted(ref)
    R = np.stack([ref[s][:n] if len(ref[s]) >= n else np.pad(ref[s], (0, n - len(ref[s]))) for s in spk], 1)
    M = np.array([[float((R[:, i] & on[:, j]).sum()) for j in range(on.shape[1])] for i in range(len(spk))])
    ri, ci = linear_sum_assignment(-M)
    H = np.zeros_like(R)
    extra = np.zeros(n, np.int32)
    used = set()
    for i, j in zip(ri, ci):
        H[:, i] = on[:, j]
        used.add(j)
    for j in range(on.shape[1]):
        if j not in used:
            extra += on[:, j]
    nr, nh = R.sum(1), H.sum(1) + extra
    corr = (R & H).sum(1)
    err = np.maximum(0, nr - nh).sum() + np.maximum(0, nh - nr).sum() + (np.minimum(nr, nh) - corr).sum()
    speech = max(1, int(nr.sum()))
    active = int(sum(on[:, j].sum() >= 0.01 * (R.any(1).sum()) for j in range(on.shape[1])))
    return 100.0 * err / speech, active


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--mix", default="zh_mix_v5")
    ap.add_argument("--tag", default="zh5")
    ap.add_argument("--max-gap", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    split_of = json.load(open(data / "meetings.json"))
    res, agg = {}, defaultdict(lambda: defaultdict(list))
    for f in sorted((data / args.mix / "utts").glob("*.jsonl")):
        m = f.stem
        us = [json.loads(l) for l in open(f, encoding="utf-8")]
        kept = [u for u in us if not u["hes"] and not removed(u) and u["placed"]]
        if not kept:
            continue
        out = {}
        for side, post_dir, b, e in (("zh", f"nemotron_offline_{args.tag}", "zh_begin", "zh_end"),
                                     ("en", "nemotron_offline", "en_begin", "en_end")):
            post = torch.load(data / "diar" / post_dir / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
            ref = defaultdict(lambda: np.zeros(len(post), bool))
            # zh: the clips that are in the Chinese mix; en: every non-hesitation utterance, all of them are in the
            # English recording (scoring only the kept ones would count the rest as false alarms)
            for u in (kept if side == "zh" else [u for u in us if not u["hes"]]):
                ref[u["speaker_id"]][int(u[b] / FRAME): int(np.ceil(u[e] / FRAME))] = True
            d, act = der(dict(ref), post)
            out[f"der_{side}"], out[f"active_slots_{side}"] = round(d, 2), act
        out["speakers"] = len({u["speaker_id"] for u in kept})
        out["split"] = split_of[m]
        out["fails"] = bool(out["der_zh"] - out["der_en"] > args.max_gap)
        res[m] = out
        for k in ("der_zh", "der_en"):
            agg[out["split"]][k].append(out[k])
        agg[out["split"]]["slot_err_zh"].append(abs(out["active_slots_zh"] - out["speakers"]))
        agg[out["split"]]["slot_err_en"].append(abs(out["active_slots_en"] - out["speakers"]))
    summ = {sp: {k: round(float(np.mean(v)), 2) for k, v in a.items()} for sp, a in agg.items()}
    summ["failing_meetings"] = sorted(m for m, v in res.items() if v["fails"])
    p = Path(args.out)
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    p.write_text(json.dumps({"summary": summ, "meetings": res}, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
