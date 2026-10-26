"""Plan 4A's rows: the Thinker taught to say, at each read, only what its own sampled futures agree on, then wait.

From the vote runs on the 40k training items (si_thinker_stream --policy vote --items-file, under
data/unist_simul_thinker_v1/p1/shard*/): at every read the state is (the audio heard, the translation committed
before it) and the target is what the vote committed at that read followed by the wait marker <|box_end|> -- or,
at the read that hears the end, the rest of the translation followed by <|im_end|>. A share of rows are the whole
audio with the whole translation (the run's own), so the Thinker keeps translating whole sentences.

Rows: {"id", "audio", "dir", "t_ms", "dur_ms", "prefix", "target", "end": "wait" | "eos", "kind": "read" | "offline"}

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_thinker_selfdistill --out data/si_thinker_selfdistill_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1", default="data/unist_simul_thinker_v1/p1")
    ap.add_argument("--items", default="data/unist_simul_thinker_v1/shards")
    ap.add_argument("--configs", default="w1_vote8g0.8,w0.5_vote8g0.8")
    ap.add_argument("--offline-share", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=240000)
    ap.add_argument("--val-items", type=int, default=400, help="items held out per direction")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    audio = {}
    for f in glob.glob(f"{args.items}/items_*_*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            audio[r["id"]] = (r["audio"], int(r["dur_ms"]))
    reads, offline = [], {}
    for cfg in args.configs.split(","):
        for f in sorted(glob.glob(f"{args.p1}/shard*/blocks_*_{cfg}.jsonl")):
            d = Path(f).name.split("_")[1]
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if r["id"] not in audio or not r["blocks"]:
                    continue
                path, dur = audio[r["id"]]
                at = {}
                for b in r["blocks"]:
                    at[b["write_ms"]] = at.get(b["write_ms"], "") + b["text"]
                done = ""
                for dr in r["drafts"]:
                    t = dr["t"]
                    new = at.get(t, "")
                    final = t >= dur
                    reads.append({"id": r["id"], "audio": path, "dir": d, "t_ms": t, "dur_ms": dur, "prefix": done,
                                  "target": new, "end": "eos" if final else "wait", "kind": "read"})
                    done += new
                offline[(r["id"], d)] = {"id": r["id"], "audio": path, "dir": d, "t_ms": dur, "dur_ms": dur, "prefix": "",
                                         "target": r["translation"], "end": "eos", "kind": "offline"}
    ids = sorted({(x["id"], x["dir"]) for x in reads})
    val = set()
    for d in ("zh2en", "en2zh"):
        pool = [k for k in ids if k[1] == d]
        val |= set(rng.sample(pool, min(args.val_items, len(pool))))
    train_reads = [x for x in reads if (x["id"], x["dir"]) not in val]
    rng.shuffle(train_reads)
    n_off = int(args.max_rows * args.offline_share)
    train = train_reads[: args.max_rows - n_off]
    off = [v for k, v in offline.items() if k not in val]
    rng.shuffle(off)
    train += off[:n_off]
    rng.shuffle(train)
    val_rows = [x for x in reads if (x["id"], x["dir"]) in val] + [offline[k] for k in sorted(val) if k in offline]
    out.mkdir(parents=True)
    for name, rows in (("train", train), ("val", val_rows)):
        with open(out / f"rows_{name}.jsonl", "w", encoding="utf-8") as fh:
            for x in rows:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    stats = {"reads_all": len(reads), "train": len(train), "val": len(val_rows), "offline_train": min(n_off, len(off)),
             "wait_share": round(sum(x["end"] == "wait" for x in train) / max(len(train), 1), 3),
             "empty_wait_share": round(sum(x["end"] == "wait" and not x["target"].strip() for x in train) / max(len(train), 1), 3),
             "zh2en": sum(x["dir"] == "zh2en" for x in train), "en2zh": sum(x["dir"] == "en2zh" for x in train)}
    (out / "README.md").write_text("Plan 4A rows (see data_clean/build_thinker_selfdistill.py).\n\n" + json.dumps(stats, indent=1) + "\n",
                                   encoding="utf-8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
