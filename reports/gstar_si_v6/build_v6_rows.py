"""Training rows for paper v6 (prereg_v6 §4, §5.3). Every output directory is new; nothing existing is touched.

  dlabc   (review M04, target control) deadline-label read rows on exactly the windows and read times of the
          stable-prefix ABC rows (data/ami_multispk_si_v1/e2e_gstar_abc_v1: 4,000 training windows per direction, each
          read every 0.5 / 1 / 1 / 2 s as its oracle dump did), with the deadline rules of the deployed models
          (En->Zh: e2e_gstar_stream_v1's chunks timed by the AMI words; Zh->En: the duration rule of v5) and the same
          keep-empty / offline / single-speaker mixture as the ABC rows (0.5 / 0.2 / 0.15, O2's rows)
  zhchar  (review M08) Zh->En deadline rows like data_clean/build_ami_gstar_stream_rows_zh2en.py, except that each
          chunk's deadline uses the forced-aligned character times of its source utterance (reports/gstar_si_v6/
          zh_char_times.py) exactly as En->Zh uses the AMI words; an utterance without character times keeps the
          duration rule (counted)
  turns   (review M05) the cascade's own training units: every whole-window offline row of O2 / O2-nosd's training
          rows (e2e_gstar_bidir_v2) cut into the diarizer turns the cascade uses at test time (gstar_cascade.turns on
          the row's offline posteriors: > 0.5, gaps < 0.4 s closed, >= 0.24 s kept, 0.1 s padding); each turn's target
          is the row's segments of the turn's slot lying >= 50 % inside it, re-timed to the turn; single-speaker rows
          kept as they are. Turn audio and posterior slices under <out>/audio, <out>/post

    PYTHONPATH=. python reports/gstar_si_v6/build_v6_rows.py dlabc --out data/ami_multispk_si_v1/e2e_gstar_dlabc_v6
    PYTHONPATH=. python reports/gstar_si_v6/build_v6_rows.py zhchar --out data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_char_v6
    PYTHONPATH=. python reports/gstar_si_v6/build_v6_rows.py turns --out data/ami_multispk_si_v1/e2e_gstar_turns_v6
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import q
from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import utt_chunks

D = Path("data/ami_multispk_si_v1")
ABC = D / "e2e_gstar_abc_v1"
LOOKAHEAD_MS, LAG_MS = 1040, 1000


def zh2en_chunks(w, smap, char_times=None, stats=None):
    out = []
    for u in w["utts"]:
        if u["hesitation"] or not u["ref"].strip():
            continue
        uu = {"begin_time": w["start_s"] + u["begin_s"], "end_time": w["start_s"] + u["end_s"], "zh": u["ref"]}
        ws = []
        if char_times is not None:
            ws = [tuple(c) for c in char_times.get(u["record_id"], [])]
            if stats is not None:
                stats["utts_char_timed" if ws else "utts_duration_rule"] += 1
        for c in utt_chunks(uu, ws, LAG_MS, w["start_s"]):
            c["commit_ms"] = min(c["commit_ms"], w["dur_ms"])
            c["piece"] = (f"<|spk{smap[w['meeting']][u['speaker_id']]}|><|t{q(c['src_start_ms'] / 1000)}|>{c['text'].strip()}"
                          f"<|t{q(c['src_end_ms'] / 1000)}|>")
            out.append(c)
    out.sort(key=lambda c: (c["commit_ms"], c["src_start_ms"]))
    return out


def read_rows(w, chunks, reads, rng, keep_empty, tag, stats):
    rows, prev = [], 0
    for t in reads:
        last = t >= w["dur_ms"]
        done = [c for c in chunks if c["commit_ms"] <= prev]
        new = [c for c in chunks if c["commit_ms"] > prev and (last or c["commit_ms"] <= t)]
        target = "".join(c["piece"] for c in new)
        stats["reads"] += 1
        if target or last or rng.random() < keep_empty:
            rows.append({"id": f"{w['id']}|{tag}|{t}", "audio": w["audio"], "dir": w.get("dir", "en2zh"), "t_ms": t,
                         "dur_ms": w["dur_ms"], "post": w["post"]["lowlat"], "post_until_ms": max(0, t - LOOKAHEAD_MS),
                         "prefix": "".join(c["piece"] for c in done), "target": target,
                         "end": "eos" if last else "wait", "kind": "read", "inst": "gstar", "preset": "lowlat"})
            stats["kept"] += 1
            stats["empty_kept"] += not target
        prev = t
    return rows


def mix_and_write(out, rows, rng, args, stats):
    off = [json.loads(l) for l in open(D / "e2e_gstar_bidir_v2/rows_train.jsonl", encoding="utf-8")]
    gstar_off = [r for r in off if r.get("inst") == "gstar"]
    sd = [r for r in off if r.get("inst") == "sd"]
    n = len(rows)
    extra = rng.sample(gstar_off, min(len(gstar_off), round(args.offline_share * n)))
    extra += rng.sample(sd, min(len(sd), round(args.sd_share * n)))
    allr = rows + extra
    rng.shuffle(allr)
    out.mkdir(parents=True)
    with open(out / "rows_train.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in allr)
    stats.update({"read_rows": n, "extra_rows": len(extra), "rows": len(allr), "args": vars(args)})
    (out / "summary.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))


def dlabc(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    stats = defaultdict(int)
    reads_of = {}
    for f in sorted(glob.glob(str(ABC / "dump_*_s*.jsonl"))):
        for h in map(json.loads, open(f, encoding="utf-8")):
            reads_of[h["id"]] = [r["t"] for r in h["read_log"]]
    smap_z = json.load(open(D / "e2e_gstar_zh2en_v1/slot_map.json"))["lowlat"]
    rows = []
    for d in ("en2zh", "zh2en"):
        for w in map(json.loads, open(ABC / f"windows_{d}.jsonl", encoding="utf-8")):
            if w["id"] not in reads_of:
                stats["no_dump"] += 1
                continue
            w["dir"] = d
            if d == "en2zh":
                chunks = [dict(c) for c in w["stream_target"]["chunks"]]
            else:
                chunks = zh2en_chunks(w, smap_z)
            stats[f"windows_{d}"] += 1
            rows += read_rows(w, chunks, reads_of[w["id"]], rng, args.keep_empty, "dlabc", stats)
    mix_and_write(out, rows, rng, args, stats)


def zhchar(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    stats = defaultdict(int)
    ct = {}
    for f in glob.glob(str(Path(args.char_times) / "*.jsonl")):
        for r in map(json.loads, open(f, encoding="utf-8")):
            ct[r["record_id"]] = r["chars"]
    src = D / "e2e_gstar_zh2en_v1"
    smap = json.load(open(src / "slot_map.json"))["lowlat"]
    rows = []
    for w in map(json.loads, open(src / "windows_train_ref.jsonl", encoding="utf-8")):
        w["dir"] = "zh2en"
        chunks = zh2en_chunks(w, smap, ct, stats)
        stats["windows"] += 1
        stats["chunks"] += len(chunks)
        win_ms = 1000
        reads = list(range(win_ms, w["dur_ms"], win_ms)) + [w["dur_ms"]]
        rows += read_rows(w, chunks, reads, rng, args.keep_empty, "lowlat", stats)
    # the v5 Zh->En rows also mixed in Zh->En offline rows (e2e_gstar_zh2en_v1) and the En->Zh streaming single-speaker
    # rows; keep that mixture here so that only the deadlines differ
    off_rows = [r for r in map(json.loads, open(src / "rows_train.jsonl", encoding="utf-8"))
                if r.get("inst") == "gstar" and r.get("preset") == "offline"]
    sd_rows = [r for r in map(json.loads, open(D / "e2e_gstar_stream_v1/rows_train.jsonl", encoding="utf-8"))
               if r.get("inst") == "sd"]
    n = len(rows)
    rng2 = random.Random(20261030)   # the v5 builder's seed for its mixture draw
    extra = rng2.sample(off_rows, min(len(off_rows), round(args.offline_share * n)))
    extra += rng2.sample(sd_rows, min(len(sd_rows), round(args.sd_share * n)))
    allr = rows + extra
    rng.shuffle(allr)
    out.mkdir(parents=True)
    with open(out / "rows_train.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in allr)
    stats.update({"read_rows": n, "extra_rows": len(extra), "rows": len(allr), "args": vars(args)})
    (out / "summary.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))


def turns_rows(args):
    import re
    import numpy as np
    import soundfile as sf
    from experiments.uniss_qwen3_omni_v1.evaluation.gstar_cascade import FRAME, turns
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "audio").mkdir(parents=True)
    (out / "post").mkdir()
    seg_re = re.compile(r"<\|spk(\d+)\|><\|t(\d+)\|>(.*?)<\|t(\d+)\|>", re.S)
    stats = defaultdict(int)
    rows = []
    for r in map(json.loads, open(D / "e2e_gstar_bidir_v2/rows_train.jsonl", encoding="utf-8")):
        if r.get("inst") != "gstar":
            rows.append(r)
            stats["sd_rows"] += 1
            continue
        stats["window_rows"] += 1
        post = np.load(r["post"]).astype(np.float32)
        x, sr = sf.read(r["audio"], dtype="float32")
        dur = len(x) / sr
        segs = [(int(k), int(a) * FRAME, int(b) * FRAME, t) for k, a, t, b in seg_re.findall(r["target"])]
        for j, (k, a, b) in enumerate(turns(post, 0.4, 0.24)):
            s0, s1 = max(0.0, a * FRAME - 0.1), min(dur, b * FRAME + 0.1)
            if s1 - s0 < 0.2:
                continue
            mine = [(sa, sb, t) for kk, sa, sb, t in segs
                    if kk == k and min(s1, max(sb, sa + 0.08)) - max(s0, sa) >= 0.5 * max(sb - sa, 0.08)]
            stats["turns"] += 1
            stats["turns_empty"] += not mine
            if not mine:   # a turn of no annotated speech of its slot: kept with an empty reply (as at test time)
                if random.Random(f"{r['id']}{j}").random() >= 0.3:
                    continue
            sid = f"{r['id'].split('|')[0]}#t{j}"
            wav = out / "audio" / f"{sid}.wav"
            sf.write(str(wav), x[int(s0 * sr): int(s1 * sr)], sr)
            f0, f1 = int(s0 / FRAME), int(np.ceil(s1 / FRAME))
            sl = post[f0:f1]
            if len(sl) < f1 - f0:
                sl = np.concatenate([sl, np.zeros((f1 - f0 - len(sl), post.shape[1]), dtype=sl.dtype)])
            pp = out / "post" / f"{sid}.npy"
            np.save(pp, sl.astype(np.float16))
            tgt = "".join(f"<|spk{k}|><|t{q(max(0.0, sa - s0))}|>{t}<|t{q(max(0.0, min(s1, sb) - s0))}|>"
                          for sa, sb, t in sorted(mine))
            rows.append({"id": f"{sid}|turn", "audio": str(wav.resolve()), "dir": r["dir"], "t_ms": round((s1 - s0) * 1000),
                         "dur_ms": round((s1 - s0) * 1000), "post": str(pp.resolve()), "prefix": "", "target": tgt,
                         "end": "eos", "kind": "offline", "inst": "gstar", "wtype": "turn", "preset": "offline"})
            stats["turn_rows"] += 1
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    with open(out / "rows_train.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    stats.update({"rows": len(rows), "args": vars(args)})
    (out / "summary.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dlabc", "zhchar", "turns"):
        a = sub.add_parser(name)
        a.add_argument("--out", required=True)
        a.add_argument("--keep-empty", type=float, default=0.5)
        a.add_argument("--offline-share", type=float, default=0.2)
        a.add_argument("--sd-share", type=float, default=0.15)
        a.add_argument("--seed", type=int, default=20261031)
        if name == "zhchar":
            a.add_argument("--char-times", default="data/ami_multispk_si_v1/zh_fa_v6")
    args = ap.parse_args()
    {"dlabc": dlabc, "zhchar": zhchar, "turns": turns_rows}[args.cmd](args)


if __name__ == "__main__":
    main()
