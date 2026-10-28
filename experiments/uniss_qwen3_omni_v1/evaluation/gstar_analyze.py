"""Where G-STAR-SI output goes wrong (docs/multi_speaker_si_plan_v8_streaming.zh-CN.md §1, the offline gate).

Per window, against the AMI reference and the slot map the targets were built with:
  format      raw output present but no segment parses / empty output
  count       more or fewer speakers than the reference
  slot        segments whose slot differs from the slot of the reference utterance they match (time IoU plus
              text similarity, so overlapped speakers are told apart)
  time        |start| and |end| error of segments matched to a reference utterance, seconds
  omission    reference utterances no output segment overlaps; insertion: segments overlapping no reference
Grouped by single-speaker / multi-speaker / overlapped windows; a few worst examples per class are listed.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_analyze --windows ... --hyp ... [--preset offline]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--slot-map", default="data/ami_multispk_si_v1/e2e_gstar_v1/slot_map.json")
    ap.add_argument("--preset", default="offline")
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    smap = json.load(open(args.slot_map))
    smap = smap.get(args.preset, smap)
    H = {h["id"]: h for h in map(json.loads, open(args.hyp, encoding="utf-8"))}
    stats = defaultdict(lambda: defaultdict(list))
    examples = defaultdict(list)
    for w in map(json.loads, open(args.windows, encoding="utf-8")):
        if w["id"] not in H:
            continue
        h = H[w["id"]]
        ref = [dict(u, slot=smap[w["meeting"]][u["speaker_id"]]) for u in w["utts"] if not u["hesitation"]]
        segs = h["segments"]
        sp = [(u["begin_s"], u["end_s"]) for u in ref]
        overlap = any(min(a[1], b[1]) > max(a[0], b[0]) for i, a in enumerate(sp) for b in sp[i + 1:])
        n_ref = len({u["slot"] for u in ref})
        groups = ["all", "single_spk" if n_ref == 1 else "multi_spk"] + (["overlap"] if overlap else [])
        n_hyp = len({s["spk"] for s in segs})
        rec = {"format_fail": bool(h.get("raw", "").replace("<|im_end|>", "").strip()) and not segs,
               "empty": not segs, "count_over": n_hyp > n_ref, "count_under": n_hyp < n_ref}
        wrong, terr_s, terr_e, ins = 0, [], [], 0

        def fit(u, s):   # overlapped utterances are told apart by what they say as well as by when
            ov = min(u["end_s"], s["end_s"]) - max(u["begin_s"], s["start_s"])
            if ov <= 0:
                return -1.0
            iou = ov / (max(u["end_s"], s["end_s"]) - min(u["begin_s"], s["start_s"]))
            return iou + SequenceMatcher(None, u.get("ref", u["zh"]), s["text"]).ratio()

        for s in segs:
            best = max(ref, key=lambda u: fit(u, s), default=None)
            if best is None or fit(best, s) < 0:
                ins += 1
                continue
            wrong += s["spk"] != best["slot"]
            terr_s.append(abs(s["start_s"] - best["begin_s"]))
            terr_e.append(abs(s["end_s"] - best["end_s"]))
        omit = sum(1 for u in ref if not any(min(u["end_s"], s["end_s"]) > max(u["begin_s"], s["start_s"]) for s in segs))
        for g in groups:
            st = stats[g]
            for k, v in rec.items():
                st[k].append(v)
            st["seg_slot_wrong"] += [1] * wrong + [0] * (len(segs) - wrong - ins)
            st["start_err_s"] += terr_s
            st["end_err_s"] += terr_e
            st["ref_omitted"] += [1] * omit + [0] * (len(ref) - omit)
            st["seg_inserted"] += [1] * ins + [0] * (len(segs) - ins)
        if wrong and len(examples["slot"]) < args.examples:
            examples["slot"].append({"id": w["id"], "ref": [(u["slot"], u["begin_s"], u["en"][:60]) for u in ref],
                                     "hyp": [(s["spk"], s["start_s"], s["text"][:30]) for s in segs]})
        if omit and len(examples["omission"]) < args.examples:
            examples["omission"].append({"id": w["id"], "ref": [(u["slot"], u["begin_s"], u.get("ref", u["zh"])[:30]) for u in ref],
                                         "hyp": [(s["spk"], s["start_s"], s["text"][:30]) for s in segs]})
        if rec["format_fail"] and len(examples["format"]) < args.examples:
            examples["format"].append({"id": w["id"], "raw": h.get("raw", "")[:200]})
    res = {g: {k: (round(float(np.mean(v)), 4) if v else None) for k, v in st.items()} | {"windows": len(st["empty"])}
           for g, st in stats.items()}
    res["examples"] = examples
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "x", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
