"""gate_2seed: (1) one draw = gate.py exactly; (2) the same draw twice = gate.py exactly (averaging is a no-op);
(3) pooled() on hand-made stats is the element-wise mean over the meetings common to all draws.

    python -m experiments.mist_w3po_v1.evaluation.test_gate_2seed <cand dir> <v8 ctx0 dir> <v8 ctx2 dir>
"""
import sys
from pathlib import Path

from experiments.mist_w3po_v1.evaluation.gate import gate
from experiments.mist_w3po_v1.evaluation.gate_2seed import gate2, pooled


def main():
    c, r0, r2 = sys.argv[1:4]
    g = gate([Path(r0), Path(r2)], Path(c), n=2000)
    for draws in (1, 2):
        g2 = gate2([[r0] * draws, [r2] * draws], [c] * draws, n=2000)
        for k, v in g["items"].items():
            for f in ("A", "B", "diff", "ci95", "p_le0", "pass"):
                assert g2["items"][k][f] == v[f], (draws, k, f, g2["items"][k][f], v[f])
        assert g2["pass"] == g["pass"]
    a = {"per_meeting": {"m1": {"all_stats": [2.0] * 10}, "m2": {"all_stats": [1.0] * 10}}}
    b = {"per_meeting": {"m1": {"all_stats": [4.0] * 10}, "m3": {"all_stats": [9.0] * 10}}}
    p = pooled([a, b])
    assert list(p["per_meeting"]) == ["m1"] and p["per_meeting"]["m1"]["all_stats"] == [3.0] * 10, p
    print("test_gate_2seed OK")


if __name__ == "__main__":
    main()
