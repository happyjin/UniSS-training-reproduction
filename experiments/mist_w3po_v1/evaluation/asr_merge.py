"""Merge sharded ASR outputs of a talker dir (2026-11-07: one whisper process per eval took ~3 h; the single process
was stopped and the remaining items transcribed by N shard processes, evaluation.asr_transcribe --num-shards N
--shard-index k --completed-input asr_speech.snapshot.jsonl --output asr_speech.p<k>.jsonl). Writes
asr_speech_merged.jsonl (new file, never overwritten) = the valid rows of asr_speech.jsonl (a truncated last line of
the stopped process is skipped) + the part rows, one per (id, mode); exit 1 unless every row of
results_speechonly.jsonl is covered.

    python -m experiments.mist_w3po_v1.evaluation.asr_merge <talker dir> <n parts>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def rows(p: Path):
    if not p.exists():
        return
    for l in open(p, encoding="utf-8"):
        try:
            yield json.loads(l)
        except json.JSONDecodeError:
            continue


def main():
    d, n = Path(sys.argv[1]), int(sys.argv[2])
    have = set()
    added = 0
    with open(d / "asr_speech_merged.jsonl", "x", encoding="utf-8") as fh:
        for src in [d / "asr_speech.jsonl"] + [d / f"asr_speech.p{k}.jsonl" for k in range(n)]:
            for r in rows(src):
                key = (str(r["id"]), str(r["mode"]))
                if key not in have:
                    have.add(key)
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    added += 1
    need = {(str(r["id"]), str(r["mode"])) for r in rows(d / "results_speechonly.jsonl")}
    miss = need - have
    print(f"added {added}, total {len(have)}, needed {len(need)}, missing {len(miss)}", flush=True)
    sys.exit(1 if miss else 0)


if __name__ == "__main__":
    main()
