"""Shared constants of the QABC + Qwen3-Omni native Talker run on RealSI (registration:
paper/baselines_run_plan_qabc_omnitalker_v1.md, tracked copy reports/s2st_baselines_v1/).

The text side is fixed: the committed blocks (text, write_ms) that QABC wrote at the paper's RealSI tier points
(tab_realsi_main, m1-m4, both directions), read from the paper's stored runs. Only the speech generator changes: every
block is spoken by Qwen3-Omni-30B-A3B-Instruct's own Thinker -> Talker -> code2wav (vLLM-Omni 0.30.0), prompted to read
the block verbatim, preset voice Ethan. Everything after synthesis (queue model, speech-only ASR, BLEU, D1 / D0w) is
the paper's.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PAPER_E = REPO / "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500"
OUT = REPO / "eval_outputs/s2st_baselines_v1/qabc_omnitalker_realsi"
RESULTS = REPO / "paper/baselines_results/qabc_omnitalker"
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")

# the paper's tier table (tab_realsi_main.tex, QABC column; laal_definitions_compare_v1 §4): direction, tier, budget (s),
# run family (collect_results.FAMILY key), read window m (s), b0, the paper's printed ASR-BLEU and D1 (3-seed means)
POINTS = [
    ("en2zh", "m1", 1.18, "swb-q-bm1", 0.5, -1, 25.33, 1.13),
    ("en2zh", "m2", 1.98, "swb-q-b1", 0.5, 1, 28.26, 1.77),
    ("en2zh", "m3", 2.46, "swb-q-b1", 1.0, 1, 30.33, 2.25),
    ("en2zh", "m4", 2.98, "swb-q-b1", 2.0, 1, 30.51, 2.75),
    ("zh2en", "m1", 1.18, "swb-q-bm1-zh", 0.5, -1, 17.83, 1.14),
    ("zh2en", "m2", 1.92, "swb-q-b0-zh", 1.0, 0, 21.42, 1.70),
    ("zh2en", "m3", 2.40, "swb-q-b1-zh", 1.0, 1, 22.46, 2.18),
    ("zh2en", "m4", 2.98, "swb-q-b1-zh", 2.0, 1, 22.65, 2.67),
]
PAPER_SEEDS = ("", "-seed1", "-seed2")   # the paper talker's three seeds (same committed blocks in all three)
SEEDS = (0, 1, 2)                         # Omni talker seeds; vLLM per-request seed = SEED_BASE + seed (all stages)
SEED_BASE = 20261020                      # seed 0 = the native CVSS-T run's seed (qwen3omni_native_cvss.py)
LANGS = {"en2zh": ("eng", "cmn"), "zh2en": ("cmn", "eng")}
SPEAKER = "Ethan"                         # Qwen3-Omni's default preset voice (transformers generate(speaker="Ethan"))

# fixed prompt: the instruction is the SYSTEM message (vLLM-Omni's talker skips system turns), the user turn is the
# block text alone, so the talker is conditioned on the block text only
SYSTEM_PROMPT = ("You are a text-to-speech reader. Say the user's message aloud exactly as it is written, word for word, "
                 "in its own language. Do not answer it, translate it, complete it, correct it, or add anything. "
                 "Reply with the user's text only.")
THINKER_EOS = 151645
CODEC_EOS = 2150
SPEAKABLE = re.compile(r"[0-9A-Za-z㐀-䶿一-鿿]")


def point_name(d: str, tier: str) -> str:
    return f"{d}_{tier}"


def paper_run(p: tuple, seed_suffix: str = "") -> Path:
    d, _, _, fam, w, *_ = p
    return PAPER_E / f"system_{fam}{seed_suffix}_m{w:g}" / f"realsi_{d}"


def omni_run(p: tuple, seed: int) -> Path:
    return OUT / "runs" / f"{point_name(p[0], p[1])}_s{seed}" / f"realsi_{p[0]}"


def tts_text(text: str) -> str | None:
    """the text sent for one block: stripped of surrounding whitespace; None (no request, no audio) when nothing
    speakable is left (punctuation-only blocks), as SimulInfer's tts.Speech.run and R3."""
    t = text.strip()
    return t if SPEAKABLE.search(t) else None
