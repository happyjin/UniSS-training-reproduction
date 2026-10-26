"""Speakable blocks: hold a commit back until it is long enough to be said on its own.

The vote and the commit head commit sooner than LocalAgreement-2, and half of
what they commit comes one English word (or one or two Chinese characters) at
a time; spoken alone, such a block is a word in silence -- the ASR drops it or
hears something else in the pause after it, and RealSI's speech loses what the
text had. Here commits are joined until a block holds at least --en English
words or --zh Chinese characters; it is handed to the talker when the commit
that completes it is made. The last block takes what is left. The text is
unchanged; only when the talker gets it.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_block_merge --files 'blocks_*_headhead_cons_s0t0.4.jsonl' --en 2 --zh 3
writes blocks_..._m2c3.jsonl beside each.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

DIR = "eval_outputs/uniss_qwen3_omni_v1/realsi_thinker_stream_v1"


def size(text: str, zh: bool) -> int:
    return len([c for c in text if not c.isspace()]) if zh else len(text.split())


def merge(blocks: list[dict], k: int, zh: bool) -> list[dict]:
    out, held = [], None
    for b in blocks:
        held = dict(b) if held is None else {**b, "text": held["text"] + b["text"]}
        if b["final"] or size(held["text"], zh) >= k:
            out.append(held)
            held = None
    if held is not None:   # nothing final came after (should not happen): the held text is the last block
        out.append({**held, "final": True})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", required=True, help=f"glob of block files (relative to {DIR} unless absolute)")
    ap.add_argument("--en", type=int, default=2, help="words an English block waits for")
    ap.add_argument("--zh", type=int, default=3, help="characters a Chinese block waits for")
    args = ap.parse_args()
    pattern = args.files if args.files.startswith("/") else str(Path(DIR) / args.files)
    for f in sorted(glob.glob(pattern)):
        f = Path(f)
        out = f.with_name(f"{f.stem}_m{args.en}c{args.zh}.jsonl")
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        zh = f.name.split("_")[1] == "en2zh"
        with open(out, "w", encoding="utf-8") as fh:
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                fh.write(json.dumps({**r, "blocks": merge(r["blocks"], args.zh if zh else args.en, zh)}, ensure_ascii=False) + "\n")
        print(f"{f.name} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
