"""TurnCRF end-to-end (scheme 2): reference frame labels y for the CRF auxiliary loss of a v8-format training row.

Slot conventions (checked, reports/mist_w3po_v1/turn_crf_e2e_code.zh-CN.md §2):
  * v8's rows (data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1, build_ami_gstar_stream_rows_zh2en.py) write
    <|spk k|> with k = slot_map.json[preset][meeting][speaker], a SESSION-level (meeting-level) Hungarian map
    (build_ami_gstar_si_rows.slot_map over the whole session, 1-based: npy column k - 1). The projector reads the
    window's posterior columns as they are (no remapping), so the LLM learns "column k-1 active -> <|spk k|>" only up
    to the diarizer's within-session slot drift.
  * The CPU stage (turn_data.py) labels frames with the WINDOW-level map (S0-5, build_windows_zh2en.window_slot_map:
    Hungarian on frame overlap with P > 0.5 inside the window; a speaker without overlap keeps its session slot unless
    taken, then its best free slot).
  * Here (one pre-registered setting): the window's speakers are the row-format segments <|spk k|><|t a|>..<|t b|>
    of the window's COMPLETE target (the window's last-read row, end == "eos": prefix + target; an offline row: its
    own target), identified by their session slot k, and re-mapped to diarizer columns with the window-level map on
    the window's full posteriors (the CPU stage's convention, so the CRF is asked to clean P on the diarizer's own
    columns, never to move mass between columns, which its slot-shared potentials cannot do). The complete target is
    used because a read row's own prefix + target lacks the chunks committed after the read, so its frames would be
    labelled empty while someone speaks. Labels are supervision only; the model input is unchanged.
  * Frames: a segment covers frames [a, b) of its time tokens (80 ms, window-relative). More than 2 active slots: the
    2 with the longest coverage in the window (ties -> lower slot), turn_data.joint_labels. y = joint label in C
    (turn_crf.LabelSpace, |C| = 37). Hesitations are not in v8's targets, so they are "empty" (as the CPU stage).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.mist_w3po_v1.crf.turn_crf import LabelSpace

K = 8
SEG_RE = re.compile(r"<\|spk(\d+)\|><\|t(\d+)\|>(.*?)<\|t(\d+)\|>", re.S)


def parse_segments(s: str) -> list[tuple[int, int, int]]:
    """Row-format text -> [(slot 1-based, a, b)] (frames [a, b))."""
    return [(int(m.group(1)), int(m.group(2)), int(m.group(4))) for m in SEG_RE.finditer(s)]


def window_key(row: dict) -> tuple[str, str]:
    parts = row["id"].split("|")
    return parts[0], row.get("preset") or (parts[1] if len(parts) > 1 else "")


def session_matrix(segs, T: int) -> np.ndarray:
    R = np.zeros((T, K), dtype=bool)
    for k, a, b in segs:
        R[max(0, a):min(T, b), k - 1] = True
    return R


def window_map(segs, post: np.ndarray) -> dict[int, int]:
    """Session slot -> window-level slot (both 1-based); build_windows_zh2en.window_slot_map on integer frames."""
    T = len(post)
    spks = sorted({k for k, _, _ in segs})
    if not spks:
        return {}
    on = post > 0.5
    act = {s: np.zeros(T, dtype=bool) for s in spks}
    for k, a, b in segs:
        act[k][max(0, a):min(T, b)] = True
    M = np.array([[float((act[s] & on[:, j]).sum()) for j in range(post.shape[1])] for s in spks])
    r, c = linear_sum_assignment(-M)
    out = {spks[i]: int(j) + 1 for i, j in zip(r, c) if M[i, j] > 0}
    for s in spks:
        if s not in out:
            taken = set(out.values())
            k = s
            if k in taken:
                free = [j + 1 for j in np.argsort(-M[spks.index(s)], kind="stable") if j + 1 not in taken]
                k = free[0] if free else k
            out[s] = int(k)
    return out


def joint_labels(R: np.ndarray, space: LabelSpace) -> tuple[np.ndarray, int]:
    """turn_data.joint_labels (copied to keep this module import-light): top-2 slots by window coverage."""
    cov = R.sum(0)
    y = np.zeros(len(R), dtype=np.int64)
    n_trunc = 0
    for t in range(len(R)):
        ks = np.flatnonzero(R[t])
        if len(ks) > 2:
            n_trunc += 1
            ks = sorted(sorted(ks, key=lambda k: (-cov[k], k))[:2])
        y[t] = space.label_of(ks)
    return y, n_trunc


class RefIndex:
    """(window id, preset) -> the window's complete row-format segments; labels(row) -> y over the window's frames."""

    def __init__(self, rows, mode: str = "window"):
        assert mode in ("window", "session")
        self.mode = mode
        self.segs: dict[tuple[str, str], list] = {}
        self.space = LabelSpace(K)
        self.cache: dict[tuple[str, str], np.ndarray] = {}
        self.stats = {"windows": 0, "remapped_speakers": 0, "speakers": 0, "frames_truncated": 0, "missing": 0}
        for r in rows:
            if r.get("inst") == "gstar" and r.get("end") == "eos":
                self.segs[window_key(r)] = parse_segments(r.get("prefix", "") + r.get("target", ""))

    @classmethod
    def from_jsonl(cls, path: str, mode: str = "window") -> "RefIndex":
        return cls(map(json.loads, open(path, encoding="utf-8")), mode)

    def window_labels(self, key, post: np.ndarray) -> np.ndarray | None:
        if key in self.cache:
            return self.cache[key]
        segs = self.segs.get(key)
        if segs is None:
            self.stats["missing"] += 1
            return None
        T = len(post)
        if self.mode == "window":
            wm = window_map(segs, post)
            self.stats["speakers"] += len(wm)
            self.stats["remapped_speakers"] += sum(int(a != b) for a, b in wm.items())
            segs = [(wm[k], a, b) for k, a, b in segs]
        y, n_tr = joint_labels(session_matrix(segs, T), self.space)
        self.stats["windows"] += 1
        self.stats["frames_truncated"] += n_tr
        self.cache[key] = y
        return y

    def labels(self, row: dict, tau: int, post_full: np.ndarray | None = None) -> np.ndarray | None:
        """y[:tau] for a row (None when its window has no complete target in the index)."""
        if post_full is None:
            post_full = np.load(row["post"]).astype(np.float32)
        y = self.window_labels(window_key(row), post_full)
        if y is None:
            return None
        out = np.zeros(tau, dtype=np.int64)
        n = min(tau, len(y))
        out[:n] = y[:n]
        return out


class RecRefs:
    """The same labels for the shared trainer's rows (train_crf_sft.py): meta["rec"] is the window's sched_v1 record,
    whose "S" is the window's complete v8 target in row format (all chunks, commit order) and "post" its lowlat
    posteriors (unperturbed, whole window). Cached per window id."""

    def __init__(self, mode: str = "window"):
        assert mode in ("window", "session")
        self.mode = mode
        self.space = LabelSpace(K)
        self.cache: dict[str, np.ndarray] = {}
        self.stats = {"windows": 0, "remapped_speakers": 0, "speakers": 0, "frames_truncated": 0}

    def window_labels(self, rec: dict) -> np.ndarray:
        y = self.cache.get(rec["id"])
        if y is None:
            post = np.load(rec["post"]).astype(np.float32)
            segs = parse_segments(rec["S"])
            if self.mode == "window":
                wm = window_map(segs, post)
                self.stats["speakers"] += len(wm)
                self.stats["remapped_speakers"] += sum(int(a != b) for a, b in wm.items())
                segs = [(wm[k], a, b) for k, a, b in segs]
            y, n_tr = joint_labels(session_matrix(segs, len(post)), self.space)
            self.stats["windows"] += 1
            self.stats["frames_truncated"] += n_tr
            self.cache[rec["id"]] = y
        return y

    def labels(self, rec: dict, tau: int) -> np.ndarray:
        y = self.window_labels(rec)
        out = np.zeros(tau, dtype=np.int64)
        n = min(tau, len(y))
        out[:n] = y[:n]
        return out


def main() -> None:
    """Slot-convention statistics on v8's train rows (CPU): how often the window-level map differs from the session
    slot, per speaker-window and per labelled speech frame."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--rows", default="data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1/rows_train.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preset", default="lowlat")
    args = ap.parse_args()
    rows = [r for r in map(json.loads, open(args.rows, encoding="utf-8"))
            if r.get("inst") == "gstar" and r.get("end") == "eos" and r.get("preset") == args.preset]
    if args.limit:
        rows = rows[: args.limit]
    idx = RefIndex(rows, "window")
    st = {"windows": 0, "speaker_windows": 0, "speaker_windows_remapped": 0, "windows_with_remap": 0,
          "speech_frames": 0, "speech_frames_label_differs": 0, "seg_frames_outside_post": 0}
    for r in rows:
        key = window_key(r)
        post = np.load(r["post"]).astype(np.float32)
        segs = idx.segs[key]
        wm = window_map(segs, post)
        ys = joint_labels(session_matrix(segs, len(post)), idx.space)[0]
        yw = idx.window_labels(key, post)
        st["windows"] += 1
        st["speaker_windows"] += len(wm)
        nre = sum(int(a != b) for a, b in wm.items())
        st["speaker_windows_remapped"] += nre
        st["windows_with_remap"] += int(nre > 0)
        sp = ys != 0
        st["speech_frames"] += int(sp.sum())
        st["speech_frames_label_differs"] += int((ys != yw).sum())
        st["seg_frames_outside_post"] += sum(max(0, b - len(post)) for _, _, b in segs)
    st["share_speaker_windows_remapped"] = st["speaker_windows_remapped"] / max(1, st["speaker_windows"])
    st["share_windows_with_remap"] = st["windows_with_remap"] / max(1, st["windows"])
    st["share_frames_label_differs"] = st["speech_frames_label_differs"] / max(1, st["speech_frames"])
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
