"""Where every system of paper v5 lives, for both directions (shared by make_tables.py and bootstrap.py).

  v3 runs      eval_outputs/.../gstar_v3_20261029/<dir>_<name>/            (reports/gstar_si_paper_v1/make_v3_jobs.py)
  v4 runs      eval_outputs/.../gstar_v4_20261030/<dir>_<name>/            (make_v4_runs.py / make_v4b_runs.py):
               the deadline-label simultaneous model at other read intervals and its Zh->En and AliMeeting runs (the
               mixed-interval model of make_v4b_runs.py was cancelled at the compute deadline)
  en->zh v1    prompted Thinker, its cascade and its LA2 run were not rerun (same test windows, same scorer)
  speech       eval_outputs/.../gstar_v4_speech_20261030/<dir>_<name>/      (make_v4_speech_jobs.py); the prompted
               En->Zh offline speech of v2 is reused (same voices, byte-identical prompts)
  v5 runs      eval_outputs/.../gstar_v5_20261030/<dir>_<name>/            (reports/gstar_si_paper_v1/make_v5_runs.py, new
               server): the Zh->En deadline model retrained for 3,000 steps (as En->Zh) at 0.5 / 1 / 2 s and on
               AliMeeting, the mixed-interval En->Zh model S2-dl-m, and the diarizer-defined windows for the v4 models;
               speech of the 3k Zh->En model in gstar_v5_speech_20261030. The 1,000-step Zh->En model of v4 is kept as
               S2dl1k_d1 (appendix).
v4 drops the commit-time controls of v3 (wait bias, queue-dependent release, fluent realisation): the simultaneous
system is the deadline-label model S2dl; the stable-prefix model S2 (no bias) appears only as a label ablation.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EV = ROOT / "eval_outputs/uniss_qwen3_omni_v1"
V3 = EV / "gstar_v3_20261029"
V4 = EV / "gstar_v4_20261030"
SP4 = EV / "gstar_v4_speech_20261030"
V5 = EV / "gstar_v5_20261030"
SP5 = EV / "gstar_v5_speech_20261030"
OFF1, STR1, CAS1 = (EV / "gstar_offline_eval_o1_20261026T195403Z", EV / "gstar_stream_eval_o1_20261026T195403Z",
                    EV / "gstar_cascade_20261028")
SP2 = EV / "gstar_speech_full_20261028"
D = ROOT / "data/ami_multispk_si_v1"
DIRS = ("en2zh", "zh2en")
ARROW = {"en2zh": r"En$\rightarrow$Zh", "zh2en": r"Zh$\rightarrow$En", "ali": r"Zh$\rightarrow$En, AliMeeting"}
# real Mandarin meetings (AliMeeting test, Zh->En); scored like zh2en
REAL = ["B-off", "C-ft2", "O2-nosd", "O2", "O2_LA2_d1", "S2dl_d1"]

HEAD = "S2dl_d1"   # the simultaneous system: deadline-label reads at the 1 s interval it was trained on
LABELS = "S2_b0_d1"   # label ablation: stable-prefix reads (the model's own drafts), plain wait decoding

NAMES = {
    "B-off": r"Qwen3-Omni, prompted",
    "C-van": r"Cascade: diarizer $\rightarrow$ Qwen3-Omni",
    "C-ft2": r"Cascade: diarizer $\rightarrow$ same-data model",
    "O2-nosd": r"\sys{} w/o speaker stream",
    "O2": r"\sys{} (offline)",
    "O2-oracle": r"\sys{} (offline), oracle stream",
    "B-LA2": r"Qwen3-Omni + LA2",
    "O2_LA2_d1": r"\sys{} (offline) + LA2",
    "O2_LA2_d2": r"\sys{} (offline) + LA2, $\Delta{=}2$\,s",
    "S2dl_d1": r"\sys{}-Simul",
    "S2dl_d0.5": r"\sys{}-Simul, $\Delta{=}0.5$\,s",
    "S2dl_d2": r"\sys{}-Simul, $\Delta{=}2$\,s",
    "S2dl1k_d1": r"\sys{}-Simul, 1{,}000 steps",
    "S2dlm_d0.5": r"\sys{}-Simul-M, $\Delta{=}0.5$\,s",
    "S2dlm_d1": r"\sys{}-Simul-M, $\Delta{=}1$\,s",
    "S2dlm_d2": r"\sys{}-Simul-M, $\Delta{=}2$\,s",
    LABELS: r"Stable-prefix labels",
}

STREAMING = {"B-LA2", "O2_LA2_d1", "O2_LA2_d2", LABELS} | {k for k in NAMES if k.startswith("S2dl")}
OFFLINE = [k for k in NAMES if k not in STREAMING]
STREAM = sorted(STREAMING)


def hyp_dir(dr: str, name: str) -> Path | None:
    if dr == "en2zh":
        old = {"B-off": OFF1 / "B-off", "C-van": CAS1 / "C-van", "B-LA2": STR1 / "B-LA2_d1"}
        if name in old:
            return old[name]
        if name == "S2dl_d1":
            return V3 / "en2zh_S2dl_d1"
    if name.startswith("S2dlm"):
        return V5 / f"{dr}_{name}" if dr == "en2zh" else None
    if name == "S2dl1k_d1":
        return V4 / "zh2en_S2dl_d1" if dr == "zh2en" else None
    if dr == "ali" and name not in REAL:
        return None
    if name.startswith("S2dl") and dr in ("zh2en", "ali"):
        return V5 / f"{dr}_S2dl3k_{name.split('_', 1)[1]}"   # the 3,000-step Zh->En model
    if name.startswith("S2dl"):
        return V4 / f"{dr}_{name}"
    return V3 / f"{dr}_{name}" if name != "B-LA2" else V3 / f"{dr}_B-LA2_d1"


def windows(dr: str, name: str) -> Path:
    if dr == "ali":
        return ROOT / "data/alimeeting_si_v1/windows_test_ref.jsonl"
    if dr == "zh2en":
        return D / "e2e_gstar_zh2en_v1/windows_test_ref.jsonl"
    return D / ("e2e_gstar_stream_v1" if name in STREAMING else "e2e_gstar_v1") / "windows_test_ref.jsonl"


def SPEECH_DIR(dr: str, name: str) -> Path | None:  # noqa: N802
    if dr == "zh2en" and name == "S2dl_d1":
        p = SP5 / "zh2en_S2dl3k_d1"
        return p if (p / "score.json").exists() else None
    p = SP4 / f"{dr}_{name}"
    if p.exists():
        return p
    if dr == "en2zh" and name == "B-off":
        return SP2 / "B-off"
    return None


def ASR_FILE(sp: Path) -> Path:  # noqa: N802
    """The speech transcripts: v4's Zh->En Whisper step stopped at ~40%; v5 completed it in asr_speech.v5full.jsonl."""
    full = sp / "talker/asr_speech.v5full.jsonl"
    return full if full.exists() else sp / "talker/asr_speech.jsonl"


# paired bootstrap comparisons (a better than b?); "sp:" compares the speech
PAIRS = []
for dr in DIRS:
    t = lambda k: f"{dr}:{k}"   # noqa: E731
    PAIRS += [(t("O2"), t("C-ft2")), (t("O2"), t("C-van")), (t("O2"), t("B-off")), (t("O2"), t("O2-nosd")),
              (t("O2-oracle"), t("O2")), (t(HEAD), t("O2_LA2_d1")), (t(HEAD), t("B-LA2")),
              (t("S2dl_d2"), t("O2_LA2_d2")), (t("O2"), t(HEAD)), (t(HEAD), t(LABELS)),
              ("sp:" + t("O2"), "sp:" + t("C-ft2")), ("sp:" + t(HEAD), "sp:" + t("O2_LA2_d1"))]
PAIRS += [("en2zh:S2dlm_d1", "en2zh:S2dl_d1"), ("en2zh:S2dlm_d0.5", "en2zh:S2dl_d0.5"),
          ("en2zh:S2dlm_d2", "en2zh:S2dl_d2"), ("en2zh:S2dlm_d1", "en2zh:O2_LA2_d1"),
          ("zh2en:S2dl_d1", "zh2en:S2dl1k_d1"), ("zh2en:O2_LA2_d1", "zh2en:S2dl_d1"),
          ("zh2en:S2dl_d2", "zh2en:O2_LA2_d2")]
PAIRS += [("ali:O2", "ali:C-ft2"), ("ali:C-ft2", "ali:O2"), ("ali:O2", "ali:O2-nosd"), ("ali:O2", "ali:B-off"),
          ("ali:S2dl_d1", "ali:O2_LA2_d1")]
