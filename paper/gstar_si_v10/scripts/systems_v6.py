"""Systems added in paper v6 (review of v5), on top of systems.py (which keeps v5's systems unchanged).

  O2-nosd-relabel   the no-speaker-stream model relabelled after the fact by the diarizer (scripts/v6_analyses.py)
  S2dl_LA2_d1       LocalAgreement on the prefix-trained MIST-Simul (M04 1); S2dlm_LA2_d1 the same on MIST-Simul-M
  S2dl_lag<L>       fixed lag on the model's own timestamps, L s (M04 2)
  SC                streaming cascade: low-latency diarizer turns, LocalAgreement on each turn (M04 4)
  OC / CC           cascade on reference turns and speakers / on diarizer turns with 2 s context (M05 2-3)
  C-turn            cascade whose translator is trained on its own turns (M05 1)
  S2dlabc_d1        deadline labels on the stable-prefix model's windows and reads (M04 3)
  S2dlchar_d1       Zh->En deadlines from forced-aligned character times (M08)
  on:<name>         whole-meeting online runs (M01), scored by reports/gstar_si_v6/meeting_score.py
"""
from __future__ import annotations

from pathlib import Path

from systems import EV, NAMES, ROOT, V3, hyp_dir as hyp_dir_v5, windows as windows_v5  # noqa: F401

V6 = EV / "gstar_v6_20261031"
SP6 = EV / "gstar_v6_speech_20261031"
ON = ROOT / "data/online_v6_20261031"

NAMES_V6 = {
    "O2-nosd-relabel": r"\sys{} w/o stream + diarizer relabelling",
    "S2dl_LA2_d1": r"\sys{}-Simul model + LA2",
    "S2dlm_LA2_d1": r"\sys{}-Simul-M model + LA2",
    "S2dl_lag1": r"\sys{}-Simul model, fixed lag 1\,s",
    "S2dl_lag2": r"\sys{}-Simul model, fixed lag 2\,s",
    "S2dl_lag3": r"\sys{}-Simul model, fixed lag 3\,s",
    "SC": r"Streaming cascade (turns + LA2)",
    "OC": r"Cascade, reference turns",
    "CC": r"Cascade, turns + 2\,s context",
    "C-turn": r"Cascade, turn-trained translator",
    "S2dlabc_d1": r"Deadline labels, ABC windows/reads",
    "S2dlchar_d1": r"\sys{}-Simul, character-timed deadlines",
    "S2dl_seed1_d1": r"\sys{}-Simul, seed 20261101",
}
STREAM_V6 = {"S2dl_seed1_d1", "S2dl_LA2_d1", "S2dlm_LA2_d1", "S2dl_lag1", "S2dl_lag2", "S2dl_lag3", "SC", "S2dlabc_d1", "S2dlchar_d1"}
D = ROOT / "data/ami_multispk_si_v1"


def hyp_dir(dr: str, name: str) -> Path | None:
    if name in NAMES_V6:
        p = V6 / f"{dr}_{name}"
        return p if (p / "hyp.jsonl").exists() else None
    return hyp_dir_v5(dr, name)


def windows(dr: str, name: str) -> Path:
    if name in NAMES_V6:
        if dr == "ali":
            return ROOT / "data/alimeeting_si_v1/windows_test_ref.jsonl"
        if dr == "zh2en":
            return D / "e2e_gstar_zh2en_v1/windows_test_ref.jsonl"
        return D / ("e2e_gstar_stream_v1" if name in STREAM_V6 else "e2e_gstar_v1") / "windows_test_ref.jsonl"
    return windows_v5(dr, name)
