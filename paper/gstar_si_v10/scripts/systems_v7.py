"""Systems of paper v7 (prereg_v7), on top of systems_v6.py / systems.py (unchanged).

  C-turn / OC / CC / SC   the cascades decoded in v7 (turn-trained translator; reference turns; 2 s context; streaming)
  S2dl_LA2_full           LA2 on the prefix-trained En->Zh model, all 16 meetings (v6's 8 + v7's 8)
  S2dl_seed2_full         En->Zh MIST-Simul seed 20261101, all 16 meetings (v6's 8 + v7's 8)
  S2dl_seed3_d1           En->Zh MIST-Simul seed 20261102
  S2dlZ_seed2_d1          Zh->En MIST-Simul (3k recipe) seed 20261101 (zh2en and ali)
  S2dlZ_LA2_d1            LA2 on the prefix-trained Zh->En model (zh2en and ali)
  S2dl_lag<L>_sub         fixed lag L s, En->Zh, the 8-meeting subset (prereg_v7 amendment 2)
Whole-meeting online runs: eval_outputs/.../gstar_v7_20261101/on_<set>_<sys>_full/ and scores/on_<set>_<sys>_full.json.
"""
from __future__ import annotations

from pathlib import Path

from systems_v6 import NAMES_V6, ON, ROOT, SP6, V6, hyp_dir as hyp_dir_v6, windows as windows_v6  # noqa: F401

EV = ROOT / "eval_outputs/uniss_qwen3_omni_v1"
V7 = EV / "gstar_v7_20261101"
SP7 = EV / "gstar_v7_speech_20261101"
SC7 = V7 / "scores"
D = ROOT / "data/ami_multispk_si_v1"

NAMES_V7 = {
    "C-turn": r"Cascade, turn-trained translator",
    "OC": r"Cascade, reference turns (oracle)",
    "CC": r"Cascade, turns + 2\,s context",
    "SC": r"Streaming cascade (turns + LA2)",
    "S2dl_LA2_full": r"\sys{}-Simul model + LA2",
    "S2dlZ_LA2_d1": r"\sys{}-Simul model + LA2",
    "S2dl_seed2_full": r"\sys{}-Simul, seed 2",
    "S2dl_seed3_d1": r"\sys{}-Simul, seed 3",
    "S2dlZ_seed2_d1": r"\sys{}-Simul, seed 2",
    "S2dl_lag1_sub": r"\sys{}-Simul model, fixed lag 1\,s",
    "S2dl_lag2_sub": r"\sys{}-Simul model, fixed lag 2\,s",
}
STREAM_V7 = {k for k in NAMES_V7 if k not in ("C-turn", "OC", "CC")}


def hyp_dir(dr: str, name: str) -> Path | None:
    if name in NAMES_V7:
        p = V7 / f"{dr}_{name}"
        return p if (p / "hyp.jsonl").exists() else None
    return hyp_dir_v6(dr, name)


def windows(dr: str, name: str) -> Path:
    if name in NAMES_V7:
        if name.endswith("_sub"):
            return ON / "windows_test_ref_en2zh_stream_sub.jsonl"
        if dr == "ali":
            return ROOT / "data/alimeeting_si_v1/windows_test_ref.jsonl"
        if dr == "zh2en":
            return D / "e2e_gstar_zh2en_v1/windows_test_ref.jsonl"
        return D / ("e2e_gstar_stream_v1" if name in STREAM_V7 else "e2e_gstar_v1") / "windows_test_ref.jsonl"
    return windows_v6(dr, name)
