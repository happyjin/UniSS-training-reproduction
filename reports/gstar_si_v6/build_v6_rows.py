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

    PYTHONPATH=. python reports/gstar_si_v6/build_v6_rows.py dlabc --out data/ami_multispk_si_v1/e2e_gstar_dlabc_v6
    PYTHONPATH=. python reports/gstar_si_v6/build_v6_rows.py zhchar --out data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_char_v6
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dlabc", "zhchar"):
        a = sub.add_parser(name)
        a.add_argument("--out", required=True)
        a.add_argument("--keep-empty", type=float, default=0.5)
        a.add_argument("--offline-share", type=float, default=0.2)
        a.add_argument("--sd-share", type=float, default=0.15)
        a.add_argument("--seed", type=int, default=20261031)
        if name == "zhchar":
            a.add_argument("--char-times", default="data/ami_multispk_si_v1/zh_fa_v6")
    args = ap.parse_args()
    {"dlabc": dlabc, "zhchar": zhchar}[args.cmd](args)


if __name__ == "__main__":
    main()
