"""E10 shared constants and helpers (no swift / peft imports, so data tools can use it).

E10 = the LiveInterpret S5 Thinker-Talker trained in baseline B's form on B's AMI data
(reports/mist_w3po_v1/e10_liveinterpret_multispk_plan.zh-CN.md).

* Speaker tags. The S5 tokenizer already holds 38 stream tokens of which S5 uses 4; the first eight
  `<|dur:N|>` rows (ids 248080-248087 in the 4B variant) are never used by S5 training or inference, so they are
  re-purposed as `<|spk1|>`..`<|spk8|>`. No tokenizer or embedding resize; the four S5 ids stay unchanged.
* Speaker posteriors. B's loader gives a read at time t the Nemotron lowlat posteriors of frames
  f < (t - 1040) // 80 (`post_until_ms`). The S5 Thinker is fed incrementally with a KV cache, so a frame's
  posterior cannot be added later; instead audio frame j (80 ms) gets the posterior of frame j - 13, which under
  B's rule is always available when frame j is fed (640 ms chunks end on 80 ms multiples). Added through a
  zero-initialised Linear(8 -> hidden).
"""
from __future__ import annotations

import re

import numpy as np

N_SLOT = 8
SD_DELAY_FRAMES = 13                 # 1040 ms lookahead of B's lowlat preset / 80 ms frames
FRAME_MS = 80
LOOKAHEAD_MS = 1040
BASE_CHUNK_MS = 640

# <|dur:N|> buckets of streaming_si.model.base_model.STREAM_DUR_BUCKETS, first eight
_DUR_FOR_SLOT = (1, 2, 3, 5, 8, 12, 18, 25)
SPK_TOKENS = tuple(f"<|dur:{n}|>" for n in _DUR_FOR_SLOT)      # SPK_TOKENS[k-1] is <|spk k|>

INSTR_PLAIN = "\nTranslate the streaming speech into {tgt} simultaneously:\n"   # = S5 default (sd rows)
INSTR_GSTAR = ("\nSeveral people are speaking. Translate the streaming speech into {tgt} simultaneously, "
               "starting every utterance with its speaker tag:\n")

PIECE_RE = re.compile(r"<\|spk(\d+)\|>(?:<\|t\d+\|>)?(.*?)(?:<\|t\d+\|>)?(?=<\|spk\d+\|>|$)", re.S)


def spk_token(k: int) -> str:
    if not 1 <= int(k) <= N_SLOT:
        raise ValueError(f"speaker slot {k} outside 1..{N_SLOT}")
    return SPK_TOKENS[int(k) - 1]


def spk_token_ids(tokenizer) -> list[int]:
    """Single-token ids of the eight tags; raises if any tag is not one token."""
    ids = []
    for t in SPK_TOKENS:
        got = tokenizer.encode(t, add_special_tokens=False)
        if len(got) != 1:
            raise RuntimeError(f"{t} is not a single token: {got} (add_s4_stream_tokens first)")
        ids.append(int(got[0]))
    return ids


def parse_pieces(target: str) -> list[tuple[int, str]]:
    """B's target string `<|spk k|><|t a|>text<|t b|>...` -> [(k, text)] in order."""
    out = []
    for m in PIECE_RE.finditer(target or ""):
        txt = m.group(2).strip()
        if txt:
            out.append((int(m.group(1)), txt))
    return out


def perturb(post: np.ndarray, rng: np.random.Generator, p: float = 0.2) -> np.ndarray:
    """B's perturbation (pretrain_thinker_gstar.perturb): with prob. p, N(0, 0.1) noise clipped to [0, 1]
    and a roll of -1/0/+1 frames."""
    if rng.random() >= p:
        return post
    out = np.clip(post + rng.normal(0, 0.1, post.shape), 0, 1)
    s = int(rng.integers(-1, 2))
    return np.roll(out, s, axis=0) if s else out


def delayed_post(post: np.ndarray | None, n_frames: int, delay: int = SD_DELAY_FRAMES) -> np.ndarray:
    """(n_frames, N_SLOT) float32: row j is post[j - delay] (zeros where unavailable or missing)."""
    out = np.zeros((max(0, int(n_frames)), N_SLOT), dtype=np.float32)
    if post is None or n_frames <= delay:
        return out
    src = np.asarray(post, dtype=np.float32)[: n_frames - delay, :N_SLOT]
    out[delay: delay + len(src), : src.shape[1]] = src
    return out


def instruction(tgt_lang: str, gstar: bool) -> str:
    name = {"zh": "Chinese", "cmn": "Chinese", "en": "English", "eng": "English"}.get((tgt_lang or "").lower(),
                                                                                       "English")
    return (INSTR_GSTAR if gstar else INSTR_PLAIN).format(tgt=name)
