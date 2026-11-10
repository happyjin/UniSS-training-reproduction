"""W3PO (docs/mist_grpo_plan.zh-CN.md §4 S0-3, §5): online diarizer windows on the Chinese AMI meetings (zh_mix_v8,
low-latency Nemotron posteriors nemotron_lowlat_zh8) for the dev (validation) and train splits, with the test rule of
reports/gstar_si_v6/build_online_windows.py: a window opens at the first frame where any slot is > 0.5 and closes
after 1.0 s without activity or at 20 s; audio [open - 0.2 s, last active frame + 0.3 s]; open/close detection times.

  dev    every window of every validation meeting that has scored references (meetings in
         e2e_gstar_zh2en_v1/windows_validation_ref.jsonl; the A8-failing ones have none): the dev protocol of the
         whole-meeting online evaluation, scored with reports/gstar_si_v8/overlap_score.py against the validation
         reference windows. windows_dev_online.jsonl
  train  every window of every train meeting that passes A8 (as build_ami_gstar_zh2en_rows.py), and, for training
         (SFT rows S0-3, RL rollouts), the subset whose overlapping placed non-hesitation utterances all lie >= 0.9
         inside the window, none failing the §11.5 gate, at least one present: windows_train_online.jsonl (all) and
         windows_train_diar.jsonl (with "utts" in the reference-window format, the session slot of each speaker
         "slot_sess" and the window-level Hungarian slot "slot_win" (S0-5)).

Writes under --out (refuses an existing --out).

    python -m experiments.mist_w3po_v1.data.build_windows_zh2en --out eval_outputs/mist_w3po_v1/data/windows_zh2en_v1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment

from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import slot_map
from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_zh_mix import bad_reason, cer_of, removed

FRAME, LOOKAHEAD, GAP, CAP = 0.08, 1.04, 1.0, 20.0


def online_windows(post: np.ndarray, gap_s: float = GAP) -> list[tuple[int, int, int]]:
    """reports/gstar_si_v6/build_online_windows.py online_windows, unchanged at the default gap_s = GAP (1.0 s);
    speech-gap plan C2 (--gap): a window closes after gap_s without activity."""
    act = (post > 0.5).any(1)
    gap, cap = round(gap_s / FRAME), round(CAP / FRAME)
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
        out.append((st, last + 1, j))
        i = last + 1
    return out


def window_slot_map(members: list[dict], post_w: np.ndarray, sess: dict) -> dict:
    """S0-5: reference speaker -> slot by Hungarian matching of frame overlap inside the window only; a speaker the
    window's posteriors do not cover keeps its session slot unless that slot is taken, then its best free slot."""
    spks = sorted({u["speaker_id"] for u in members})
    n = len(post_w)
    on = post_w > 0.5
    act = {s: np.zeros(n, dtype=bool) for s in spks}
    for u in members:
        act[u["speaker_id"]][max(0, int(u["begin_s"] / FRAME)): int(np.ceil(u["end_s"] / FRAME))] = True
    M = np.array([[float((act[s] & on[:, j]).sum()) for j in range(post_w.shape[1])] for s in spks])
    r, c = linear_sum_assignment(-M)
    out = {spks[i]: int(j) + 1 for i, j in zip(r, c) if M[i, j] > 0}
    for s in spks:
        if s not in out:
            taken = set(out.values())
            k = sess.get(s)
            if k is None or k in taken:
                free = [j + 1 for j in np.argsort(-M[spks.index(s)]) if j + 1 not in taken]
                k = free[0] if free else (k or 1)
            out[s] = int(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--mix", default="zh_mix_v8")
    ap.add_argument("--recheck", default="data/ami_multispk_si_v1/zh_revoice_v1/talker")
    ap.add_argument("--en-norm", default="data/ami_multispk_si_v1/en_norm_v1.jsonl")
    ap.add_argument("--diar", default="nemotron_lowlat_zh8")
    ap.add_argument("--ref", default="data/ami_multispk_si_v1/e2e_gstar_zh2en_v1")
    ap.add_argument("--splits", default="validation,train")
    ap.add_argument("--inside", type=float, default=0.9)
    ap.add_argument("--limit-meetings", type=int, default=0)
    ap.add_argument("--gap", type=float, default=GAP, help="window-closing silence (s); C2 shortens it (dev only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    split_of = json.load(open(data / "meetings.json"))
    skip = set(json.load(open(data / args.mix / "audit_diar.json"))["summary"]["failing_meetings"])
    dev_meetings = {json.loads(l)["meeting"] for l in open(Path(args.ref) / "windows_validation_ref.jsonl")}
    asr = {}
    for f in sorted(Path(args.recheck).glob("asr_s*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            asr[r["id"]] = r.get("asr_text", "")
    if not asr:
        print(f"warning: no recheck ASR under {args.recheck}; the CER gate is skipped (zh_mix_v8's audit lists no bad "
              "utterance and e2e_gstar_zh2en_v1/summary.json only utt_ok/utt_removed)", flush=True)
    en_norm = {r["record_id"]: r["en"] for r in map(json.loads, open(args.en_norm, encoding="utf-8"))}
    sess_map = json.loads((Path(args.ref) / "slot_map.json").read_text())["lowlat"]
    splits = args.splits.split(",")
    out.mkdir(parents=True)
    fh = {}
    for sp in splits:
        name = "dev" if sp == "validation" else sp
        fh[(sp, "online")] = open(out / f"windows_{name}_online.jsonl", "x", encoding="utf-8")
        if sp == "train":
            fh[(sp, "diar")] = open(out / "windows_train_diar.jsonl", "x", encoding="utf-8")
    stats = defaultdict(lambda: defaultdict(float))
    meetings = sorted(m for m, s in split_of.items() if s in splits)
    if args.limit_meetings:
        meetings = meetings[: args.limit_meetings]
    for m in meetings:
        sp = split_of[m]
        if m in skip or (sp == "validation" and m not in dev_meetings):
            stats[sp]["meetings_skipped"] += 1
            continue
        stats[sp]["meetings"] += 1
        post = torch.load(data / "diar" / args.diar / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
        x, sr = sf.read(str(data / "session_audio" / args.mix / f"{m}.wav"), dtype="float32", always_2d=True)
        x = x.mean(1)
        sess_s = len(x) / sr
        talk = []
        for u in map(json.loads, open(data / args.mix / "utts" / f"{m}.jsonl", encoding="utf-8")):
            if u["hes"] or removed(u) or not u["placed"]:
                continue
            if u["record_id"] in asr:
                u["cer2"] = round(cer_of(asr[u["record_id"]], u["zh"]), 4)
            elif not asr:   # recheck ASR absent on this server: the v8 build passed every non-removed utterance
                u["cer2"] = 0.0
            u["ref"] = en_norm.get(u["record_id"], "")
            u["_bad"] = bad_reason(u) or (None if u["ref"].strip() else "no_reference")
            talk.append(u)
        sess = sess_map.get(m)
        if sess is None and sp == "train":   # train meetings all carry a session map in the v8 builder
            raise SystemExit(f"{m}: no session slot map")
        for k, (st, en, j) in enumerate(online_windows(post, args.gap)):
            ws, we = max(0.0, st * FRAME - 0.2), min(sess_s, en * FRAME + 0.3)
            close = min(sess_s, j * FRAME + LOOKAHEAD) if j < len(post) else sess_s
            wid = f"{m}_on_{k:04d}"
            wav = out / "audio" / sp / f"{wid}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
            f0, nf = round(ws / FRAME), int(np.ceil((we - ws) / FRAME))
            s = post[f0: f0 + nf]
            if len(s) < nf:
                s = np.concatenate([s, np.zeros((nf - len(s), post.shape[1]), dtype=s.dtype)])
            pp = out / "post" / sp / f"{wid}.npy"
            pp.parent.mkdir(parents=True, exist_ok=True)
            np.save(pp, s.astype(np.float16))
            rec = {"id": wid, "meeting": m, "split": sp, "wtype": "online", "dir": "zh2en", "start_s": round(ws, 3),
                   "dur_ms": round((we - ws) * 1000), "audio": str(wav.resolve()),
                   "post": {"lowlat": str(pp.resolve())},
                   "open_detect_ms": round((st * FRAME + LOOKAHEAD - ws) * 1000),
                   "close_detect_ms": round((close - ws) * 1000), "utts": [], "strict_review": False}
            fh[(sp, "online")].write(json.dumps(rec) + "\n")
            stats[sp]["windows"] += 1
            stats[sp]["hours"] += (we - ws) / 3600
            if sp != "train":
                continue
            inside, ok = [], True
            for u in talk:
                ov = min(we, u["zh_end"]) - max(ws, u["zh_begin"])
                if ov <= 0:
                    continue
                if ov < args.inside * (u["zh_end"] - u["zh_begin"]):
                    ok = False
                    stats[sp]["drop_straddle"] += 1
                    break
                inside.append(u)
            if not ok:
                continue
            if not inside:
                stats[sp]["drop_no_speech"] += 1
                continue
            if any(u["_bad"] for u in inside):
                stats[sp]["drop_gate"] += 1
                continue
            members = [{"record_id": u["record_id"], "speaker_id": u["speaker_id"],
                        "begin_s": round(max(0.0, u["zh_begin"] - ws), 3), "end_s": round(min(we, u["zh_end"]) - ws, 3),
                        "zh": u["zh"], "en": u["ref"], "ref": u["ref"], "hesitation": False}
                       for u in sorted(inside, key=lambda u: (u["zh_begin"], u["zh_end"]))]
            wmap = window_slot_map(members, s.astype(np.float32), sess)
            for u in members:
                u["slot_sess"] = int(sess[u["speaker_id"]])
                u["slot_win"] = int(wmap[u["speaker_id"]])
                stats[sp]["utts"] += 1
                stats[sp]["utts_slot_changed"] += u["slot_sess"] != u["slot_win"]
            fh[(sp, "diar")].write(json.dumps({**rec, "wtype": "diar_online", "utts": members}, ensure_ascii=False) + "\n")
            stats[sp]["kept"] += 1
        print(f"{m} {sp}: {dict(stats[sp])}", flush=True)
    for f in fh.values():
        f.close()
    summ = {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in stats.items()}
    summ["args"] = vars(args)
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
