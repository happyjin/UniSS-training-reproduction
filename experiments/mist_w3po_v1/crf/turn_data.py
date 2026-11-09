"""TurnCRF CPU stage (crf_novel_ideas.zh-CN.md §3.4 / §11): reference frame labels and packed diarizer posteriors.

Windows: AMI train = windows_train_online.jsonl (all 28106 online windows; windows_train_diar.jsonl is the subset of
23742 with "utts"), dev = windows_dev_online.jsonl (validation meetings). Only these two splits are read.

Reference per window (80 ms frames, frame t = [0.08 t, 0.08 (t+1)) s of the window):
  * utterances: windows_train_diar.jsonl "utts" when the window has them; otherwise rebuilt exactly as
    experiments/mist_w3po_v1/data/build_windows_zh2en.py builds them (non-hesitation, not removed, placed utterances of
    zh_mix_v8 overlapping the window, clipped to it; slot by its window_slot_map = window-level Hungarian matching,
    1-based slot_win -> npy column slot_win - 1). The rebuild is checked against every train_diar window.
  * R[t, k] = 1 when an utterance with slot k covers frame t (frames int(begin / 0.08) .. ceil(end / 0.08) - 1, the
    builder's convention); y[t] = joint label in C. More than 2 active slots: keep the 2 with the longest coverage in
    the window (ties -> lower slot); counted. No utterance -> empty.

Writes --out/{train,dev}.npz and stats.json (refuses an existing --out).

    PYTHONPATH=. python -m experiments.mist_w3po_v1.crf.turn_data --out eval_outputs/mist_w3po_v1/crf/turn_refs_v1
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf.turn_crf import LabelSpace
from experiments.mist_w3po_v1.data.build_windows_zh2en import FRAME, window_slot_map
from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_zh_mix import removed

K = 8
WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1"


def meeting_talk(data: Path, mix: str, m: str) -> list[dict]:
    talk = []
    for u in map(json.loads, open(data / mix / "utts" / f"{m}.jsonl", encoding="utf-8")):
        if u["hes"] or removed(u) or not u["placed"]:
            continue
        talk.append(u)
    return sorted(talk, key=lambda u: (u["zh_begin"], u["zh_end"]))


def rebuild_members(rec: dict, talk: list[dict], post: np.ndarray, sess: dict) -> list[dict]:
    ws = rec["start_s"]
    we = ws + rec["dur_ms"] / 1000.0
    members = []
    for u in talk:
        if min(we, u["zh_end"]) - max(ws, u["zh_begin"]) <= 0:
            continue
        members.append({"speaker_id": u["speaker_id"], "begin_s": round(max(0.0, u["zh_begin"] - ws), 3),
                        "end_s": round(min(we, u["zh_end"]) - ws, 3)})
    if members:
        wmap = window_slot_map(members, post.astype(np.float32), sess or {})
        for u in members:
            u["slot_win"] = int(wmap[u["speaker_id"]])
    return members


def slot_matrix(members: list[dict], T: int) -> np.ndarray:
    R = np.zeros((T, K), dtype=bool)
    for u in members:
        a, b = max(0, int(u["begin_s"] / FRAME)), min(T, int(np.ceil(u["end_s"] / FRAME)))
        R[a:b, u["slot_win"] - 1] = True
    return R


def joint_labels(R: np.ndarray, space: LabelSpace) -> tuple[np.ndarray, int]:
    cov = R.sum(0)
    y = np.zeros(len(R), dtype=np.int16)
    n_trunc = 0
    for t in range(len(R)):
        ks = np.flatnonzero(R[t])
        if len(ks) > 2:
            n_trunc += 1
            ks = sorted(sorted(ks, key=lambda k: (-cov[k], k))[:2])
        y[t] = space.label_of(ks)
    return y, n_trunc


def runs(x: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], x.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default=WIN)
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--mix", default="zh_mix_v8")
    ap.add_argument("--ref", default="data/ami_multispk_si_v1/e2e_gstar_zh2en_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    wd, data = Path(args.windows), Path(args.data)
    sess_map = json.loads((Path(args.ref) / "slot_map.json").read_text())["lowlat"]
    space = LabelSpace(K)
    diar = {r["id"]: r for r in map(json.loads, open(wd / "windows_train_diar.jsonl", encoding="utf-8"))}
    stats: dict = {}
    packs = {}
    for split, fname in (("train", "windows_train_online.jsonl"), ("dev", "windows_dev_online.jsonl")):
        recs = [json.loads(l) for l in open(wd / fname, encoding="utf-8")]
        assert all(r["split"] == ("train" if split == "train" else "validation") for r in recs)
        talk_cache: dict = {}
        ids, meet, Ts, Ps, Rs, ys, src = [], [], [], [], [], [], []
        st = defaultdict(int)
        chk = Counter()
        for r in recs:
            post = np.load(r["post"]["lowlat"])
            T = post.shape[0]
            m = r["meeting"]
            if m not in talk_cache:
                talk_cache[m] = meeting_talk(data, args.mix, m)
            rebuilt = rebuild_members(r, talk_cache[m], post, sess_map.get(m))
            if r["id"] in diar:
                members = diar[r["id"]]["utts"]
                Rr = slot_matrix(rebuilt, T)
                Rp = slot_matrix(members, T)
                chk["windows"] += 1
                chk["frames_equal_windows"] += bool((Rr == Rp).all())
                chk["n_utts_equal"] += len(rebuilt) == len(members)
                chk["slot_equal_utts"] += sum(a["slot_win"] == b["slot_win"] for a, b in zip(
                    sorted(rebuilt, key=lambda u: (u["begin_s"], u["end_s"], u["speaker_id"])),
                    sorted(members, key=lambda u: (u["begin_s"], u["end_s"], u["speaker_id"]))))
                chk["utts"] += len(members)
                src.append(0)
            else:
                members = rebuilt
                src.append(1)
            R = slot_matrix(members, T)
            y, n_tr = joint_labels(R, space)
            nact = R.sum(1)
            st["windows"] += 1
            st["frames"] += T
            st["frames_empty"] += int((nact == 0).sum())
            st["frames_1"] += int((nact == 1).sum())
            st["frames_2"] += int((nact == 2).sum())
            st["frames_gt2"] += int((nact > 2).sum())
            st["frames_truncated"] += n_tr
            st["windows_with_gt2"] += int(n_tr > 0)
            st["windows_no_ref_speech"] += int(not R.any())
            for k in range(K):
                for a, b in runs(R[:, k]):
                    st["ref_turns"] += 1
                    st["ref_turns_le_1s"] += int((b - a) * FRAME <= 1.0 + 1e-9)
            st["max_slot_win"] = max(st["max_slot_win"], max((u["slot_win"] for u in members), default=0))
            ids.append(r["id"])
            meet.append(m)
            Ts.append(T)
            Ps.append(post.astype(np.float16))
            Rs.append(R)
            ys.append(y)
        Ts = np.array(Ts, dtype=np.int64)
        offs = np.concatenate([[0], np.cumsum(Ts)])
        durs = np.array([r["dur_ms"] for r in recs])
        st = dict(st)
        st["share_frames_gt2"] = st["frames_gt2"] / st["frames"]
        st["share_speech_frames_gt2"] = st["frames_gt2"] / max(1, st["frames"] - st["frames_empty"])
        st["meetings"] = len(set(meet))
        st["dur_ms_percentiles"] = {str(p): float(np.percentile(durs, p)) for p in (0, 5, 25, 50, 75, 95, 99, 100)}
        st["dur_ms_mean"] = float(durs.mean())
        st["T_frames_max"] = int(Ts.max())
        st["windows_at_20s_cap"] = int((Ts >= 250).sum())
        st["rebuilt_windows"] = int(sum(src))
        if split == "train":
            st["rebuild_check_vs_train_diar"] = dict(chk)
        stats[split] = st
        packs[split] = dict(ids=np.array(ids), meeting=np.array(meet), T=Ts, offs=offs,
                            P=np.concatenate(Ps), R=np.concatenate(Rs), y=np.concatenate(ys).astype(np.int16),
                            src=np.array(src, dtype=np.int8))
        print(split, json.dumps(st), flush=True)
    out.mkdir(parents=True)
    for split, pk in packs.items():
        np.savez(out / f"{split}.npz", **pk)
    stats["args"] = vars(args)
    stats["slot_win_indexing"] = "1-based: npy column = slot_win - 1 (build_windows_zh2en.window_slot_map: int(j) + 1)"
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
