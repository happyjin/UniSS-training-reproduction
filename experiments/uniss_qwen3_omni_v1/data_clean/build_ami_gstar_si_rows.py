"""G-STAR-SI data (docs/multi_speaker_si_plan_v7.zh-CN.md, steps D3-D7): the multi-speaker translation targets in the
gstar-nemotron branch's format, with the Nemotron-3 slots as speaker labels, and the rows the Thinker trains on.

  D3  reference speaker activity per session on the 80 ms grid (every AMI utterance, hesitations included)
  D4  reference speaker -> Nemotron slot, per session and preset, by Hungarian matching of frame overlap (the
      branch's map_speakers_to_slots on the whole session, so a label means the same speaker all meeting long);
      the oracle posteriors put each speaker's activity on its slot's channel
  D5  windows: "ref" = the reviewed utterance groups of e2e_v0; "diar_<preset>" = causal windows on Nemotron
      activity (a window opens when any slot is active and closes after 1 s without activity or at 20 s). A window
      is kept only when every utterance overlapping it lies (>= --inside) within it and every non-hesitation
      utterance passed the review (build_ami_ms_rows.review, --review text)
  D6  target: every non-hesitation utterance of the window, by start, as <|spk k|><|t a|>translation<|t b|> --
      k its speaker's slot, a / b its start / end on the 80 ms grid from the window start (t0..t250, <= 20 s)
  D7  rows (offline: the whole window at once): ref windows with the offline or the streaming preset's posteriors
      (half each), diar windows with their own preset's; plus a share of single-speaker whole-sentence rows
      (Plan 4A's offline rows, "inst": "sd", no posteriors) so plain translation is kept

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows --out data/ami_multispk_si_v1/e2e_gstar_v1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment

from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import is_hesitation, review

FRAME = 0.08
N_TIME = 251
N_SLOT = 8


def q(t: float) -> int:
    return int(min(N_TIME - 1, max(0, round(t / FRAME))))


def seg(slot: int, a: float, b: float, zh: str) -> str:
    return f"<|spk{slot}|><|t{q(a)}|>{zh}<|t{q(b)}|>"


def slot_map(act: dict, post: np.ndarray, thr: float = 0.5) -> dict:
    """reference speaker -> 1-based slot, Hungarian on frame overlap; speakers beyond the slots take their best."""
    spks = sorted(act)
    on = post > thr
    n = min(len(on), min(len(a) for a in act.values()))
    M = np.array([[float((act[s][:n] & on[:n, j]).sum()) for j in range(post.shape[1])] for s in spks])
    r, c = linear_sum_assignment(-M)
    out = {spks[i]: int(j) + 1 for i, j in zip(r, c)}
    for i, s in enumerate(spks):
        out.setdefault(s, int(M[i].argmax()) + 1)
    return out


def diar_windows(post: np.ndarray, max_s: float, gap_s: float) -> list[tuple[float, float]]:
    """Causal windows on activity: open at the first active frame, close after gap_s without activity or at max_s."""
    act = (post > 0.5).any(1)
    gap, cap = round(gap_s / FRAME), round(max_s / FRAME)
    out, i, n = [], 0, len(act)
    while i < n:
        if not act[i]:
            i += 1
            continue
        st, last, j = i, i, i
        while j < n and j - st < cap and j - last <= gap:
            if act[j]:
                last = j
            j += 1
        out.append((st * FRAME, (last + 1) * FRAME))
        i = last + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--v0", default="data/ami_multispk_si_v1/e2e_v0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--presets", default="offline,lowlat")
    ap.add_argument("--max-win", type=float, default=20.0)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--inside", type=float, default=0.9)
    ap.add_argument("--max-cer", type=float, default=0.3)
    ap.add_argument("--sd-rows", default="data/si_thinker_selfdistill_v2/rows_train.jsonl")
    ap.add_argument("--sd-share", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    presets = args.presets.split(",")
    split_of = json.load(open(data / "meetings.json"))
    rev = review(data, args.max_cer)
    src = pads.dataset(data / "index/source_ihm", format="parquet", partitioning="hive").to_table(
        columns=["record_id", "meeting_id", "speaker_id", "begin_time", "end_time", "transcript"]).to_pylist()
    utts = defaultdict(list)
    for u in src:
        ok, zh, why = rev.get(u["record_id"], (False, "", "no_translation"))
        u.update(hes=is_hesitation(u["transcript"]), zh=zh, passed=why in ("ok", "cer", "over_long", "no_cer"))
        utts[u["meeting_id"]].append(u)

    stats = defaultdict(lambda: defaultdict(int))
    slot_maps = {p: {} for p in presets}
    windows = defaultdict(list)   # (split, wtype) -> windows
    for m in sorted(utts):
        split = split_of[m]
        us = sorted(utts[m], key=lambda u: u["begin_time"])
        posts = {p: torch.load(data / f"diar/nemotron_{p}/{m}.post.pt", weights_only=False)["post80"].float().numpy()
                 for p in presets}
        T = max(len(v) for v in posts.values())
        act = defaultdict(lambda: np.zeros(T, dtype=bool))
        for u in us:   # D3
            act[u["speaker_id"]][int(u["begin_time"] / FRAME): int(np.ceil(u["end_time"] / FRAME))] = True
        act = dict(act)
        (out / "oracle").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "oracle" / f"{m}.npz", speakers=np.array(sorted(act)),
                            act=np.stack([act[s] for s in sorted(act)], 1))
        oracle = {}
        for p in presets:   # D4
            mp = slot_map(act, posts[p])
            slot_maps[p][m] = mp
            o = np.zeros((len(posts[p]), N_SLOT), dtype=np.float16)
            for s, k in mp.items():
                n = min(len(o), len(act[s]))
                o[:n, k - 1] = np.maximum(o[:n, k - 1], act[s][:n])
            oracle[p] = o
        sess = data / "session_audio" / "ihm_mix" / f"{m}.Mix-Headset.wav"
        x = None

        def take(ws: float, we: float, wtype: str, wid: str, audio: str | None):
            """Utterances of [ws, we]; None if one straddles the window or one fails the review."""
            inside = []
            for u in us:
                ov = min(we, u["end_time"]) - max(ws, u["begin_time"])
                if ov <= 0:
                    continue
                if ov < args.inside * (u["end_time"] - u["begin_time"]):
                    stats[(split, wtype)]["drop_straddle"] += 1
                    return None
                inside.append(u)
            talk = [u for u in inside if not u["hes"]]
            if not talk:
                stats[(split, wtype)]["drop_no_speech"] += 1
                return None
            if any(not u["passed"] for u in talk):
                stats[(split, wtype)]["drop_review"] += 1
                return None
            return {"id": wid, "meeting": m, "split": split, "wtype": wtype, "start_s": round(ws, 3),
                    "dur_ms": round((we - ws) * 1000), "audio": audio,
                    "utts": [{"record_id": u["record_id"], "speaker_id": u["speaker_id"],
                              "begin_s": round(max(0.0, u["begin_time"] - ws), 3),
                              "end_s": round(min(we, u["end_time"]) - ws, 3), "en": u["transcript"],
                              "zh": "" if u["hes"] else u["zh"], "hesitation": u["hes"]} for u in inside]}

        # D5: the causal Nemotron windows here, the reviewed reference groups (e2e_v0 pieces) in stash_ref
        for p in presets:
            for k, (ws, we) in enumerate(diar_windows(posts[p], args.max_win, args.gap)):
                ws, we = max(0.0, ws - 0.2), we + 0.3
                w = take(ws, we, f"diar_{p}", f"{m}_{p}_{k:04d}", None)
                if w is None:
                    continue
                if x is None:
                    x, sr = sf.read(str(sess), dtype="float32", always_2d=True)
                    x = x.mean(1)
                wav = out / "audio" / split / f"{w['id']}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(wav), x[round(ws * 16000): round(we * 16000)], 16000)
                w["audio"] = str(wav.resolve())
                windows[(split, f"diar_{p}")].append(w)
                stats[(split, f"diar_{p}")]["kept"] += 1
        meeting_post = {"post": posts, "oracle": oracle}
        # posterior slices for every window of this meeting (both presets for ref windows, own preset for diar ones)
        for key in [(split, f"diar_{p}") for p in presets]:
            for w in windows[key]:
                if w["meeting"] != m or "post" in w:
                    continue
                p = w["wtype"].split("_", 1)[1]
                w["post"] = {p: save_slice(out, p, w, meeting_post["post"][p]),
                             f"oracle_{p}": save_slice(out, f"oracle_{p}", w, meeting_post["oracle"][p])}
        stash_ref(args, out, m, split, us, take, windows, stats, meeting_post, presets)
        print(f"{m} {split}: " + " ".join(f"{k[1]}={stats[k]['kept']}" for k in stats if k[0] == split), flush=True)

    (out / "slot_map.json").write_text(json.dumps(slot_maps, indent=0), encoding="utf-8")
    write_outputs(args, out, rng, presets, windows, slot_maps, stats)


def save_slice(out: Path, name: str, w: dict, post: np.ndarray) -> str:
    f0 = round(w["start_s"] / FRAME)
    n = int(np.ceil(w["dur_ms"] / 1000 / FRAME))
    s = post[f0: f0 + n]
    if len(s) < n:
        s = np.concatenate([s, np.zeros((n - len(s), post.shape[1]), dtype=s.dtype)])
    p = out / "post" / name / w["split"] / f"{w['id']}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, s.astype(np.float16))
    return str(p.resolve())


def stash_ref(args, out, m, split, us, take, windows, stats, meeting_post, presets):
    """The reviewed reference groups of e2e_v0 for this meeting, as ref windows (their audio already cut)."""
    if not hasattr(stash_ref, "pieces"):
        stash_ref.pieces = defaultdict(list)
        for sp in ("train", "validation", "test"):
            for line in open(Path(args.v0) / f"pieces_{sp}.jsonl", encoding="utf-8"):
                p = json.loads(line)
                stash_ref.pieces[p["meeting"]].append(p)
    for p in stash_ref.pieces.get(m, []):
        ws, we = p["offset_ms"] / 1000, (p["offset_ms"] + p["dur_ms"]) / 1000
        if we - ws > args.max_win + 0.5:
            stats[(split, "ref")]["drop_too_long"] += 1
            continue
        w = take(ws, we, "ref", f"{p['id']}", p["audio"])
        if w is None:
            continue
        w["strict_review"] = p.get("strict_review")
        w["post"] = {}
        for pr in presets:
            w["post"][pr] = save_slice(out, pr, w, meeting_post["post"][pr])
            w["post"][f"oracle_{pr}"] = save_slice(out, f"oracle_{pr}", w, meeting_post["oracle"][pr])
        windows[(split, "ref")].append(w)
        stats[(split, "ref")]["kept"] += 1


def target_of(w: dict, smap: dict) -> str:
    talk = sorted((u for u in w["utts"] if not u["hesitation"]), key=lambda u: (u["begin_s"], u["end_s"]))
    return "".join(seg(smap[u["speaker_id"]], u["begin_s"], u["end_s"], u["zh"].strip()) for u in talk)


def write_outputs(args, out, rng, presets, windows, slot_maps, stats):
    rows = {"train": [], "validation": []}
    for (split, wtype), ws in sorted(windows.items()):
        with open(out / f"windows_{split}_{wtype}.jsonl", "x", encoding="utf-8") as fh:
            for w in ws:
                w["target"] = {p: target_of(w, slot_maps[p][w["meeting"]]) for p in presets}
                fh.write(json.dumps(w, ensure_ascii=False) + "\n")
        if split not in rows:
            continue
        for w in ws:
            p = rng.choice(presets) if wtype == "ref" else wtype.split("_", 1)[1]
            rows[split].append({"id": f"{w['id']}|{p}", "audio": w["audio"], "dir": "en2zh", "t_ms": w["dur_ms"],
                                "dur_ms": w["dur_ms"], "post": w["post"][p], "prefix": "", "target": w["target"][p],
                                "end": "eos", "kind": "offline", "inst": "gstar", "wtype": wtype, "preset": p})
    n_ms = len(rows["train"])
    if args.sd_share > 0 and Path(args.sd_rows).exists():
        sd = [r for r in map(json.loads, open(args.sd_rows, encoding="utf-8")) if r.get("kind") == "offline"]
        pick = rng.sample(sd, min(len(sd), round(args.sd_share * n_ms)))
        rows["train"] += [{**r, "inst": "sd"} for r in pick]
    for split, rs in rows.items():
        rng.shuffle(rs)
        with open(out / f"rows_{split}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    summary = {f"{k[0]}/{k[1]}": dict(v) for k, v in sorted(stats.items())}
    summary["rows"] = {"train": len(rows["train"]), "train_gstar": n_ms, "train_sd": len(rows["train"]) - n_ms,
                       "validation": len(rows["validation"])}
    summary["args"] = vars(args)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
