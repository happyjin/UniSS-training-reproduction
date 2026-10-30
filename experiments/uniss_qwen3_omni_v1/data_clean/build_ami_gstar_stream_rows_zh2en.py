"""Deadline-label streaming rows for Zh->En (paper v4): build_ami_gstar_stream_rows.py's recipe on the Chinese AMI
windows of e2e_gstar_zh2en_v1, for continuing the bidirectional offline model O2.

The Chinese source is resynthesised speech, so there are no source word timings: each English reference is cut at its
punctuation into chunks, chunk j covering the fraction phi_j of the utterance's characters ends at the same fraction of
the utterance's duration (utt_chunks without words), and is committed --lag s later. Everything else is as for En->Zh:
speaker labels are the slots of the 1.04 s streaming preset (slot_map.json "lowlat" of the windows' builder), a row per
read every --window s and at the window's end with the posteriors finalised by then (post_until_ms = t - 1040), the
chunks committed before as the prefix, the new ones then <|box_end|> (wait) -- or the rest then <|im_end|>; reads with
nothing new are kept with probability --keep-empty; plus whole-window offline Zh->En rows (--offline-share of the read
rows) and the same single-speaker rows as the En->Zh streaming rows (--sd-share). Train windows only; the test windows
of e2e_gstar_zh2en_v1 are used as they are.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_stream_rows_zh2en \
        --out data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import q
from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import utt_chunks

LOOKAHEAD_MS = 1040


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/ami_multispk_si_v1/e2e_gstar_zh2en_v1")
    ap.add_argument("--sd-from", default="data/ami_multispk_si_v1/e2e_gstar_stream_v1/rows_train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="lowlat")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--lag", type=float, default=1.0)
    ap.add_argument("--keep-empty", type=float, default=0.5)
    ap.add_argument("--offline-share", type=float, default=0.2)
    ap.add_argument("--sd-share", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    win_ms, lag_ms = round(args.window * 1000), round(args.lag * 1000)
    smap = json.loads((src / "slot_map.json").read_text(encoding="utf-8"))[args.preset]
    stats = defaultdict(lambda: defaultdict(int))
    rows = {"train": [], "validation": []}
    for split in rows:
        for w in map(json.loads, open(src / f"windows_{split}_ref.jsonl", encoding="utf-8")):
            m, dur = w["meeting"], w["dur_ms"]
            chunks = []
            for u in w["utts"]:
                if u["hesitation"] or not u["ref"].strip():
                    continue
                uu = {"begin_time": w["start_s"] + u["begin_s"], "end_time": w["start_s"] + u["end_s"], "zh": u["ref"]}
                for c in utt_chunks(uu, [], lag_ms, w["start_s"]):
                    c["commit_ms"] = min(c["commit_ms"], dur)
                    c["piece"] = (f"<|spk{smap[m][u['speaker_id']]}|><|t{q(c['src_start_ms'] / 1000)}|>{c['text'].strip()}"
                                  f"<|t{q(c['src_end_ms'] / 1000)}|>")
                    chunks.append(c)
            chunks.sort(key=lambda c: (c["commit_ms"], c["src_start_ms"]))
            stats[split]["windows"] += 1
            stats[split]["chunks"] += len(chunks)
            reads = list(range(win_ms, dur, win_ms)) + [dur]
            prev = 0
            for t in reads:
                last = t >= dur
                done = [c for c in chunks if c["commit_ms"] <= prev]
                new = [c for c in chunks if c["commit_ms"] > prev and (last or c["commit_ms"] <= t)]
                target = "".join(c["piece"] for c in new)
                if target or last or rng.random() < args.keep_empty:
                    rows[split].append({"id": f"{w['id']}|{args.preset}|{t}", "audio": w["audio"], "dir": "zh2en",
                                        "t_ms": t, "dur_ms": dur, "post": w["post"][args.preset],
                                        "post_until_ms": max(0, t - LOOKAHEAD_MS),
                                        "prefix": "".join(c["piece"] for c in done), "target": target,
                                        "end": "eos" if last else "wait", "kind": "read", "inst": "gstar",
                                        "preset": args.preset})
                    stats[split]["reads"] += 1
                    stats[split]["empty_reads"] += not target
                prev = t
    off_rows = [r for r in map(json.loads, open(src / "rows_train.jsonl", encoding="utf-8"))
                if r.get("inst") == "gstar" and r.get("preset") == "offline"]
    sd_rows = [r for r in map(json.loads, open(args.sd_from, encoding="utf-8")) if r.get("inst") == "sd"]
    n = len(rows["train"])
    rows["train"] += rng.sample(off_rows, min(len(off_rows), round(args.offline_share * n)))
    rows["train"] += rng.sample(sd_rows, min(len(sd_rows), round(args.sd_share * n)))
    out.mkdir(parents=True)
    for split, rs in rows.items():
        rng.shuffle(rs)
        with open(out / f"rows_{split}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    summary = {sp: dict(v) for sp, v in stats.items()}
    summary["rows"] = {sp: len(rs) for sp, rs in rows.items()}
    summary["rows"]["train_read"] = n
    summary["args"] = vars(args)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
