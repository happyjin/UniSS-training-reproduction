"""D1 RealSI test (prereg_dirsel.zh-CN.md), text side on LA2's 15 windows: the dev choice per direction against LA2,
E1/E2 with the paired bootstrap (B = 1000, one-sided), Holm over the two directions. En->Zh kept the default (A0), Zh->En
uses bma3 at b0 = 2 with SDR. Also the default A0 in Zh->En on the same draws, for comparison. Writes the json given.

    PYTHONPATH=. python3 reports/uniss_si_v4/test_d1.py reports/uniss_si_v4/diag/d1_test.json
"""
import json
import sys

sys.path.insert(0, "paper/bayes_simul_s2st_v4/scripts")
from endpoints import endpoints  # noqa: E402
from realsi_text import POLICIES, V4, points  # noqa: E402

CHOSEN = {"en2zh": ("A0", POLICIES["A0"]), "zh2en": ("bma3_b2", [(V4 / "realsi_grid_bma3", "_swbma3b2q{r}")])}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    out = {}
    for d, (name, spec) in CHOSEN.items():
        la2, ours = points(POLICIES["LA2"], d), points(spec, d)
        assert len(ours) == 15, (d, sorted(ours))
        r = endpoints({"LA2": list(la2.values()), name: list(ours.values())}, name, "LA2", B=1000)
        out[d] = {"system": name, **{k: r[k] for k in ("E1", "E1_ci", "E1_p", "E2", "E2_ci", "E2_p")}}
        print(d, name, {k: out[d][k] for k in ("E1", "E1_ci", "E1_p", "E2", "E2_p")}, flush=True)
    for e in ("E1", "E2"):
        for d, p in zip(out, holm([out[d][f"{e}_p"] for d in out])):
            out[d][f"{e}_p_holm"] = p
    with open(sys.argv[1], "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
