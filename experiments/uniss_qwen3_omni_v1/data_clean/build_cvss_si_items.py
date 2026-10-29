"""CVSS-T test as si_thinker_stream / si_infer items ({id, audio, dur_ms, translation}): the 4,897 pairs per direction of
the Qwen3 cascade's zero-shot run (phaseA_zeroshot/<dir>/merged.json: source audio and reference), so the offline
anchor of the simultaneous paper is scored on the same sentences, with durations read from the audio.
Also writes blocks_<dir>_offline_zeroshot.jsonl: the zero-shot translation as one block at the end of the audio.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_cvss_si_items --out data/cvss_t/si_items_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "eval_outputs/uniss_qwen3_omni_v1/phaseA_zeroshot"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    for d in ("en2zh", "zh2en"):
        rows = json.load(open(SRC / d / "merged.json", encoding="utf-8"))
        with open(out / f"items_{d}.jsonl", "x", encoding="utf-8") as fi, \
                open(out / f"blocks_{d}_offline_zeroshot.jsonl", "x", encoding="utf-8") as fb:
            for r in rows:
                dur = int(round(sf.info(r["audio"]).duration * 1000))
                fi.write(json.dumps({"id": r["id"], "audio": r["audio"], "dur_ms": dur, "translation": r["reference"]},
                                    ensure_ascii=False) + "\n")
                fb.write(json.dumps({"id": r["id"], "translation": r["hypothesis"],
                                     "blocks": [{"write_ms": dur, "text": r["hypothesis"], "final": True}]},
                                    ensure_ascii=False) + "\n")
        print(d, len(rows), flush=True)


if __name__ == "__main__":
    main()
