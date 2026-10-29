"""P2 RealSI text test (make_p2_jobs.py): A2 (+SDR and without, b0 = 1) on LA2's 15 windows against LA2, E1/E2 with the
paired bootstrap (B = 1000, one-sided), and A2's whole-sentence BLEU beside s0's. Writes the json given.

    PYTHONPATH=. python3 reports/uniss_si_v4/test_p2.py reports/uniss_si_v4/diag/p2_test.json
"""
import json
import sys

sys.path.insert(0, "paper/bayes_simul_s2st_v4/scripts")
from endpoints import endpoints, point_score  # noqa: E402
from realsi_text import POLICIES, V4, points  # noqa: E402

SYS = {"A2": [(V4 / "realsi_grid_a2", "_swa2b1q{r}")], "A2-noSDR": [(V4 / "realsi_grid_a2", "_swa2b1")]}
OFF = {"A2": (V4 / "realsi_off_a2", "_swa2b0"), "s0": (V4 / "realsi_off_s0", "_sws0b0")}


def main():
    out = {}
    for d in ("en2zh", "zh2en"):
        la2 = points(POLICIES["LA2"], d)
        for name, spec in SYS.items():
            ours = points(spec, d)
            assert len(ours) == 15, (name, d, sorted(ours))
            r = endpoints({"LA2": list(la2.values()), name: list(ours.values())}, name, "LA2", B=1000)
            out.setdefault(name, {})[d] = {k: r[k] for k in ("E1", "E1_ci", "E1_p", "E2", "E2_ci", "E2_p")}
            print(d, name, {k: round(r[k], 3) for k in ("E1", "E1_p", "E2", "E2_p")}, flush=True)
        for name, (folder, suf) in OFF.items():
            p = points([(folder, suf)], d)
            out.setdefault(f"offline_{name}", {})[d] = [round(x, 2) for x in point_score(p[60.0], lambda r: slice(None))]
            print(d, "offline", name, out[f"offline_{name}"][d], flush=True)
    with open(sys.argv[1], "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
