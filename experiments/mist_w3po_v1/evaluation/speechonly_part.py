"""Speech-only ASR input of ONE talker shard (2026-11-08, per-shard pipelining: ASR of talker shard k starts when
that shard is done instead of waiting for all 8). Row by row the same transformation as
scripts/gpu_queue/p80_speechonly.py (every row points at its speech-only wav), applied to <talker dir>/part_<k>.jsonl;
writes <talker dir>/results_speechonly.t<k>.jsonl (new file; kept if it exists, like p80_speechonly).

    python experiments/mist_w3po_v1/evaluation/speechonly_part.py <talker dir> <k>
"""
import json
import os
import sys

import soundfile as sf


def convert(r: dict) -> dict:
    if r.get("speech_path") and os.path.exists(r["speech_path"]):
        r["audio_path"] = r["speech_path"]
        r["audio_duration_seconds"] = round(sf.info(r["speech_path"]).duration, 3)
    return r


def main():
    base, k = sys.argv[1], int(sys.argv[2])
    src, out = f"{base}/part_{k}.jsonl", f"{base}/results_speechonly.t{k}.jsonl"
    if not os.path.exists(out):
        rows = [convert(r) for r in map(json.loads, open(src, encoding="utf-8"))]
        with open(out, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(out)


if __name__ == "__main__":
    main()
