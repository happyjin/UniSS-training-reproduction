"""Robust wave 16 NCWI, group ncwi_erm (change B in reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md): wave16_fit.py
unchanged except the endpoints, alpha 0.5 = wave5/M/eval_erm_wise05 and alpha 1 = wave5/M/eval_erm (alpha 0 = the same 3
base runs). b-hat (wave16/noise) and the halves are the same, so the bin edges are those of the UniST fit; this is asserted
against wave16/fit/fit.json when it exists.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave16e_fit.py --out eval_outputs/mist_w3po_v1/wave16/erm/fit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave16_fit as F  # noqa: E402

W5M = REPO / "eval_outputs/mist_w3po_v1/wave5/M"


def main() -> None:
    F.ENDS = {0.5: W5M / "eval_erm_wise05", 1.0: W5M / "eval_erm"}
    F.main()
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    ref = REPO / "eval_outputs/mist_w3po_v1/wave16/fit/fit.json"
    if out.exists() and ref.exists():
        a, b = json.load(open(out / "fit.json")), json.load(open(ref))
        for d in ("en2zh", "zh2en"):
            for h in (0, 1):
                assert a[d][f"fit_half{h}"]["edges_bhat"] == b[d][f"fit_half{h}"]["edges_bhat"], (d, h)
                assert [x["n_fit_a"] for x in a[d][f"fit_half{h}"]["bins"]] == [x["n_fit_a"] for x in b[d][f"fit_half{h}"]["bins"]]
        print("bin edges and bin sizes equal those of wave16/fit (UniST endpoints)", flush=True)


if __name__ == "__main__":
    main()
