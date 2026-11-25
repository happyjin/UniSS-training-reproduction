import json, sys
import numpy as np
M = ["asr", "text_bleu", "d2", "d1", "EndOffset", "NP_in", "PT_in", "NC", "sim", "pesq"]
def val(m, x):
    if m in ("sim", "pesq"): return x["paired_median_diff"]
    return x["diff"]
print("root  dir  " + "  ".join(M) + "  | N_vs_B_b asr d2")
for root in sys.argv[1:]:
    s = json.load(open(f"eval_outputs/sst_hsmm_v1/{root}/summary.json"))["cells"]
    for d in ("en2zh", "zh2en"):
        cs = [v for k, v in s.items() if k.startswith(d)]
        row = [np.mean([val(m, c["NEW_vs_B_b"][m]) for c in cs]) for m in M]
        # N vs B_b = NEW_vs_B_b - NEW_vs_N
        nb = [np.mean([c["NEW_vs_B_b"][m]["diff"] - c["NEW_vs_N"][m]["diff"] for c in cs]) for m in ("asr", "d2")]
        print(f"{root:6s} {d} " + " ".join(f"{v:+.3f}" for v in row) + f" | {nb[0]:+.2f} {nb[1]:+.3f}  ({len(cs)} cells)")
