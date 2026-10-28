"""One voice per speaker for the Chinese AMI meetings (docs/multi_speaker_si_plan_v9_bidir_abc.zh-CN.md §11.2 A8).

The chain cloned every utterance's Chinese speech from that utterance alone, so one person sounds different from clip to
clip and the diarizer splits them (audit A8: 128 of 171 meetings get extra slots, DER 40-44% against 28% in English).
Here every utterance of a speaker is spoken by the UniSS talker with the same voice prompt: that speaker's longest
close-talk (IHM) English utterances of the meeting, up to --max-s seconds.

  prep   <out>/voices/<meeting>_<speaker>.wav, and si_infer inputs <out>/items.jsonl / blocks.jsonl: one stream per
         utterance with a Chinese translation (hesitations and utterances without translation have none), the whole
         translation as one block. Then run si_infer (--text-policy file) and point build_ami_zh_mix --clips-from at
         its results.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_zh_revoice prep --out data/ami_multispk_si_v1/zh_revoice_v1
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

DATA = Path("data/ami_multispk_si_v1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("--out", required=True)
    a.add_argument("--max-s", type=float, default=8.0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "voices").mkdir(parents=True)
    import pyarrow.dataset as pads

    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import is_hesitation

    rows = pads.dataset(DATA / "index/chain_en_zh_ihm", format="parquet", partitioning="hive").to_table(
        columns=["record_id", "meeting_id", "speaker_id", "begin_time", "end_time", "transcript", "translation_text",
                 "source_audio_shard_uri", "source_audio_byte_offset", "source_audio_byte_length"]).to_pylist()

    def clean(r):
        parts = r["source_audio_shard_uri"].split("/")
        path = DATA / "shards" / "source_ihm" / Path(*parts[parts.index("source_subset=ihm") + 1:])
        with open(path, "rb") as fh:
            fh.seek(r["source_audio_byte_offset"])
            x, sr = sf.read(io.BytesIO(fh.read(r["source_audio_byte_length"])), dtype="float32", always_2d=True)
        assert sr == 16000
        return x.mean(1)

    by_spk = defaultdict(list)
    for r in rows:
        if not is_hesitation(r["transcript"]):
            by_spk[(r["meeting_id"], r["speaker_id"])].append(r)
    voices = {}
    for (m, s), rs in sorted(by_spk.items()):
        xs, tot = [], 0.0
        for r in sorted(rs, key=lambda r: r["end_time"] - r["begin_time"], reverse=True):
            if r["end_time"] - r["begin_time"] < 1.0:
                break
            x = clean(r)
            xs.append(x)
            tot += len(x) / 16000
            if tot >= args.max_s:
                break
        if tot < 1.0:   # nobody speaks a second at once: their own longest bits, however short
            xs = [clean(r) for r in sorted(rs, key=lambda r: r["end_time"] - r["begin_time"], reverse=True)[:6]]
        v = out / "voices" / f"{m}_{s}.wav"
        sf.write(str(v), np.concatenate(xs)[: int(args.max_s * 16000)], 16000)
        voices[(m, s)] = str(v.resolve())
    n = 0
    with open(out / "items.jsonl", "x", encoding="utf-8") as fi, open(out / "blocks.jsonl", "x", encoding="utf-8") as fb:
        for (m, s), rs in sorted(by_spk.items()):
            for r in rs:
                zh = (r["translation_text"] or "").strip()
                if not zh:
                    continue
                fi.write(json.dumps({"id": r["record_id"], "audio": voices[(m, s)], "dur_ms": 1000, "translation": zh,
                                     "voice_audio": voices[(m, s)]}, ensure_ascii=False) + "\n")
                fb.write(json.dumps({"id": r["record_id"], "blocks": [
                    {"write_ms": 0, "text": zh, "final": True, "start_s": 0.0, "end_s": 0.0}]}, ensure_ascii=False) + "\n")
                n += 1
    print(json.dumps({"voices": len(voices), "streams": n, "out": str(out)}))


if __name__ == "__main__":
    main()
