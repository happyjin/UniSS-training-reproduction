"""T1's data: streaming trajectories from the read/write data, with offline replay.

Two pools, one per step for every rank (as Stage2Dataset: ranks draw the same
cost of work, so none waits at the all-reduce):

  si       a row of data/unist_simul_train_v1 (build_si_train_index.py), cut
           into blocks. Each draw picks a timing rule (hybrid 0.5, omni 0.3,
           argmax 0.1, suff 0.1) and a read window of m seconds (1-6, as
           SimulS2ST-Omni's latency multipliers). A segment is written at the
           end of the first window that reaches its tau = max over its words
           of support + delta, never after the source ends; segments written
           at the same moment are one block. A block carries the source GLM
           tokens heard by then less a holdback (the GLM-4 encoder attends
           both ways: a prefix's last second of tokens disagrees with the
           full tokenisation, so inference uses only tokens a second behind
           the audio), the block's slice of the translation, and its codes.
  offline  a row of the clean index exactly as Plan 1 trained on it.

With word_share > 0 (T1c), a share of si draws is cut the way inference
releases text word by word instead: at the end of each read window (1-3 s)
the words whose tau has passed become a block once there are k of them
(1-3), and the block's codes end between its last word and the next, at the
middle of the gap in the target speech. The codes are one continuous stream
either way, so nothing is cut that is heard; what the talker learns is to
say exactly a fragment and stop -- T1b, which had only seen blocks ending at
real pauses, kept talking after a fragment ("manipulating their" over and
over) and lost 9-13 ASR-BLEU when RealSI was handed over word by word.

Rows are a real permutation per pass. Within the si pool each pass keeps
every A2 row and a fresh random share of A1 and B rows, so the pool is
45:15:10 by tier, and read-speech sources into Chinese are thinned to a
quarter of that direction.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

RULES = ("omni", "argmax", "suff", "hybrid")
RULE_P = {"hybrid": 0.5, "omni": 0.3, "argmax": 0.1, "suff": 0.1}
TIER_SHARE = (45.0, 15.0, 10.0)   # A2, A1, B
GLM_MS = 80                       # GLM-4 tokens: 12.5 a second


def _flat(root: Path, name: str):
    return np.load(root / f"{name}_values.npy", mmap_mode="r"), np.load(root / f"{name}_offsets.npy")


class SIIndex:
    def __init__(self, root: str | Path):
        root = Path(root)
        self.meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.clean_idx = np.load(root / "clean_idx.npy")
        self.tier = np.load(root / "tier.npy")
        self.tgt_eng = np.load(root / "tgt_eng.npy")
        self.read = np.load(root / "read.npy")
        self.src_ms = np.load(root / "src_ms.npy")
        self.word_start = _flat(root, "word_start")
        self.support = {r: _flat(root, f"support_{r}") for r in RULES}
        self.seg_w1 = _flat(root, "seg_w1")
        self.seg_code1 = _flat(root, "seg_code1")
        timed = (root / "word_start_ms_values.npy").exists()   # v2 on: when each target word is said
        self.word_start_ms = _flat(root, "word_start_ms") if timed else None
        self.word_end_ms = _flat(root, "word_end_ms") if timed else None

    def __len__(self) -> int:
        return int(self.clean_idx.size)

    @staticmethod
    def _get(flat, i):
        values, offsets = flat
        return np.asarray(values[offsets[i]: offsets[i + 1]]).tolist()

    def keep_probs(self, read_cap: float = 0.25) -> np.ndarray:
        """Per row, the chance a pass keeps it: tiers to TIER_SHARE, read speech into Chinese to read_cap."""
        counts = np.bincount(self.tier, minlength=3).astype(np.float64)
        total = counts[0] / (TIER_SHARE[0] / sum(TIER_SHARE))
        tier_p = np.minimum(1.0, np.asarray(TIER_SHARE) / sum(TIER_SHARE) * total / np.maximum(counts, 1))
        p = tier_p[self.tier]
        into_cmn = ~self.tgt_eng
        read, plain = (into_cmn & self.read), (into_cmn & ~self.read)
        read_mass, plain_mass = p[read].sum(), p[plain].sum()
        if read_mass > 0:
            p = np.where(read, p * min(1.0, read_cap / (1 - read_cap) * plain_mass / read_mass), p)
        return p


def blocks_for(index: SIIndex, i: int, rule: str, m: int, n_codes: int, n_glm: int, translation: str,
               delta_ms: int, holdback: int) -> list[dict]:
    """The row's segments grouped by write moment: [{write_ms, glm_end, text, code_end, final}]."""
    src = int(index.src_ms[i])
    sup = index._get(index.support[rule], i)
    nxt = src
    for j in range(len(sup) - 1, -1, -1):       # unsupported words wait for the next supported one
        if sup[j] < 0:
            sup[j] = nxt
        nxt = sup[j]
    tau, run = [], 0
    for s in sup:
        run = max(run, s + delta_ms)
        tau.append(min(run, src + delta_ms))
    starts = index._get(index.word_start, i)
    n_words = len(starts)

    def boundary(w: int) -> int:                  # a segment's text ends where the next word's whitespace begins
        if w >= n_words:
            return len(translation)
        b = starts[w]
        while b > 0 and translation[b - 1].isspace():
            b -= 1
        return b

    window = m * 1000
    blocks = []
    for w1, c1 in zip(index._get(index.seg_w1, i), index._get(index.seg_code1, i)):
        write = min(math.ceil(tau[w1 - 1] / window) * window, src)
        if blocks and blocks[-1]["write_ms"] == write:
            blocks[-1].update(w1=w1, code_end=c1)
        else:
            blocks.append({"write_ms": write, "w1": w1, "code_end": c1})
    blocks[-1]["code_end"] = n_codes
    out, w0, c_prev, g_prev = [], 0, 0, 0
    for k, b in enumerate(blocks):
        final = k == len(blocks) - 1
        g = n_glm if b["write_ms"] >= src else max(0, min(n_glm, round(b["write_ms"] / GLM_MS)) - holdback)
        g = max(g, g_prev)
        code_end = max(min(b["code_end"], n_codes), c_prev)
        out.append({"write_ms": b["write_ms"], "glm_end": g, "code_end": n_codes if final else code_end,
                    "text": translation[boundary(w0) if w0 else 0: boundary(b["w1"])], "final": final})
        w0, c_prev, g_prev = b["w1"], code_end, g
    return out


def word_blocks_for(index: SIIndex, i: int, rule: str, window_ms: int, min_words: int, n_codes: int, n_glm: int,
                    translation: str, delta_ms: int, holdback: int) -> list[dict]:
    """Blocks as word-by-word release makes them (evaluation/si_infer.py plan_word_blocks), codes cut between words."""
    src = int(index.src_ms[i])
    sup = index._get(index.support[rule], i)
    nxt = src
    for j in range(len(sup) - 1, -1, -1):
        if sup[j] < 0:
            sup[j] = nxt
        nxt = sup[j]
    tau, run = [], 0
    for s in sup:
        run = max(run, s + delta_ms)
        tau.append(min(run, src + delta_ms))
    starts = index._get(index.word_start, i)
    w_on, w_off = index._get(index.word_start_ms, i), index._get(index.word_end_ms, i)
    n = len(starts)

    def boundary(w: int) -> int:
        if w >= n:
            return len(translation)
        b = starts[w]
        while b > 0 and translation[b - 1].isspace():
            b -= 1
        return b

    out, done, c_prev, g_prev, t = [], 0, 0, 0, window_ms
    while done < n:
        t = min(t, src)
        ready = sum(1 for x in tau if x <= t) if t < src else n
        final = ready >= n
        code_end = n_codes if final else min(n_codes, round((w_off[ready - 1] + w_on[ready]) / 2 / 20)) if ready else 0
        if (ready - done >= min_words or final) and code_end > c_prev:
            g = n_glm if t >= src else max(0, min(n_glm, round(t / GLM_MS)) - holdback)
            g_prev = max(g, g_prev)
            out.append({"write_ms": t, "glm_end": g_prev, "code_end": code_end,
                        "text": translation[boundary(done) if done else 0: boundary(ready)], "final": final})
            done, c_prev = ready, code_end
        elif final:   # nothing left to say after the last cut: the words join the block before
            out[-1].update(code_end=n_codes, final=True, text=out[-1]["text"] + translation[boundary(done): boundary(ready)])
            done = ready
        t += window_ms
    return out


class SIDataset(Dataset):
    """Indexable over (step, position); the pool comes from a seeded per-step schedule."""

    def __init__(self, corpus, si_root: str | Path, steps: int, *, global_batch: int, seed: int,
                 si_share: float = 0.7, delta_ms: int = 200, holdback: int = 12, m_max: int = 6,
                 read_cap: float = 0.25, word_share: float = 0.0, word_windows=(1, 2, 3), word_min=(1, 2, 3)):
        self.corpus, self.global_batch, self.seed = corpus, int(global_batch), int(seed)
        self.si = SIIndex(si_root)
        self.word_share, self.word_windows, self.word_min = float(word_share), tuple(word_windows), tuple(word_min)
        if self.word_share > 0 and self.si.word_start_ms is None:
            raise SystemExit(f"{si_root} has no word times; word blocks need an index from build_si_train_index.py v2")
        self.delta_ms, self.holdback, self.m_max = int(delta_ms), int(holdback), int(m_max)
        rng = np.random.default_rng(seed)
        self.schedule = np.where(rng.random(steps) < si_share, "si", "offline").tolist()
        self.keep_p = self.si.keep_probs(read_cap)
        self.n_offline = len(corpus.audio_rows)
        self.offsets, consumed = [], {"si": 0, "offline": 0}
        for kind in self.schedule:
            self.offsets.append(consumed[kind])
            consumed[kind] += self.global_batch
        self._orders: dict[tuple[str, int], np.ndarray] = {}
        self._si_sizes: list[int] = []
        expected = float(self.keep_p.sum())
        self.epochs = {"si": consumed["si"] / max(expected, 1), "offline": consumed["offline"] / max(self.n_offline, 1)}
        self.collate_fn = self.collate

    def __len__(self) -> int:
        return len(self.schedule) * self.global_batch

    def _order(self, pool: str, epoch: int) -> np.ndarray:
        key = (pool, epoch)
        if key not in self._orders:
            rng = np.random.default_rng([self.seed, epoch, 0 if pool == "si" else 1])
            if pool == "si":
                kept = np.flatnonzero(rng.random(len(self.si)) < self.keep_p)
                order = kept[rng.permutation(kept.size)]
            else:
                order = rng.permutation(self.n_offline)
            if len(self._orders) >= 4:
                self._orders.pop(next(iter(self._orders)))
            self._orders[key] = order
        return self._orders[key]

    def _row(self, pool: str, cursor: int) -> int:
        epoch = 0
        while True:
            order = self._order(pool, epoch)
            if cursor < order.size:
                return int(order[cursor])
            cursor -= order.size
            epoch += 1

    def __getitem__(self, index: int) -> dict:
        step, position = divmod(index, self.global_batch)
        step = min(step, len(self.schedule) - 1)
        kind = self.schedule[step]
        row_rng = random.Random(self.seed * 1_000_003 + index)
        rows = self.corpus.audio_rows
        if kind == "offline":
            ci = self._row("offline", self.offsets[step] + position) % self.n_offline
        else:
            i = self._row("si", self.offsets[step] + position)
            ci = int(self.si.clean_idx[i])
        sample = self.corpus.audio_sample(ci)
        ci = ci % len(rows)
        glm = rows.teacher_glm(ci)
        out = {"kind": kind, "codes": sample["codes"], "globals": sample["globals"], "source_glm": glm,
               "translation": sample["target_text"], "target_lang": sample["target_lang"],
               "source_seconds": float(rows.source_ms[ci]) / 1000}
        if kind == "si":
            rule = row_rng.choices(list(RULE_P), weights=list(RULE_P.values()))[0]
            if self.word_share > 0 and row_rng.random() < self.word_share:
                m, k = row_rng.choice(self.word_windows), row_rng.choice(self.word_min)
                out["blocks"] = word_blocks_for(self.si, i, rule, int(round(m * 1000)), k, len(sample["codes"]), len(glm),
                                                sample["target_text"], self.delta_ms, self.holdback)
                out["rule"], out["m"] = f"{rule}/word{k}", m
            else:
                m = row_rng.randint(1, self.m_max)
                out["blocks"] = blocks_for(self.si, i, rule, m, len(sample["codes"]), len(glm), sample["target_text"],
                                           self.delta_ms, self.holdback)
                out["rule"], out["m"] = rule, m
        return out

    def collate(self, samples: list[dict]) -> dict:
        kind = samples[0]["kind"]
        if any(s["kind"] != kind for s in samples):
            raise ValueError("a batch must hold one pool; the schedule drifted")
        return {
            "kind": kind,
            "translation": [s["translation"] for s in samples],
            "target_lang": [s["target_lang"] for s in samples],
            "codes": [torch.as_tensor(np.asarray(s["codes"], dtype=np.int64)) for s in samples],
            "globals": torch.stack([torch.as_tensor(np.asarray(s["globals"], dtype=np.int64)) for s in samples]),
            "source_glm": [torch.as_tensor(np.asarray(s["source_glm"], dtype=np.int64)) for s in samples],
            "source_seconds": torch.tensor([s["source_seconds"] for s in samples], dtype=torch.float32),
            "blocks": [s.get("blocks") for s in samples],
            "rule": [s.get("rule") for s in samples], "m": [s.get("m") for s in samples],
        }
