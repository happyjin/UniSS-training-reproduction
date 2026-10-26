"""P2 step zero, decoding half: BiCodec codes from p2_uniss_generate.py to speech.

Runs in the uniss-eval environment, where UniSS's BiCodec decoder lives. Each
utterance is decoded with the source speaker's 32 global tokens, and written
as a record in p1_native_talker's shape so the metric harness reads it
unchanged. Shards split the codes files' lines; --merge then joins the
shards' records into each direction's results.jsonl.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

COLUMNS = ["id", "translation", "transcription", "src_lang", "tgt_lang", "bicodec_global", "target_bicodec",
           "source_audio_path", "reference_audio_path", "synthetic_source", "synthetic_reference",
           "source_audio_duration_seconds", "reference_audio_duration_seconds"]


def record(row, index, translation, audio_path, codes, extra):
    """p1_native_talker.record, field for field."""
    return {
        "index": index, "name": f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}", "id": row["id"],
        "mode": "tts", "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"], "dataset_name": "CVSS-T",
        "transcription_ref": row["transcription"], "translation_ref": row["translation"],
        "generated_text_raw": translation, "generated_text_clean": translation,
        "translated_text": translation, "reference_text": row["translation"],
        "source_audio_path": row["source_audio_path"], "reference_audio_path": row["reference_audio_path"],
        "source_audio_duration_seconds": row["source_audio_duration_seconds"],
        "reference_audio_duration_seconds": row["reference_audio_duration_seconds"],
        "synthetic_source": bool(row["synthetic_source"]), "synthetic_reference": bool(row["synthetic_reference"]),
        "codes": codes, "gold_codes": len(row["target_bicodec"]),
        "audio_path": audio_path, "error": None if audio_path else "no audio", **extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", required=True, help="the directory p2_uniss_generate.py wrote codes_*.jsonl into")
    ap.add_argument("--parquet", action="append", required=True, help="name:parquet")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="join the shards' records into results.jsonl and stop")
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    args = ap.parse_args()

    out = Path(args.output).resolve()
    names = [spec.split(":")[0] for spec in args.parquet]
    if args.merge:
        for name in names:
            recs = [json.loads(l) for p in sorted(glob.glob(str(out / name / "part_*.jsonl"))) for l in open(p)]
            recs = sorted((r for r in recs if r["repeat"] == 0), key=lambda r: r["index"])
            with open(out / name / "results.jsonl", "w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
            print(f"{name}: {len(recs)} 句，无音频 {sum(not r['audio_path'] for r in recs)}，"
                  f"未正常结束 {sum(not r['ended'] for r in recs)}")
        return

    import pyarrow.parquet as pq
    import torch
    from uniss import UniSSTokenizer

    lines = [json.loads(l) for p in sorted(glob.glob(f"{args.codes}/codes_*.jsonl")) for l in open(p, encoding="utf-8")]
    lines = lines[args.shard :: args.num_shards]
    rows = {}
    for spec in args.parquet:
        name, parquet = spec.split(":")
        want = {x["id"] for x in lines if x["name"] == name}
        rows[name] = {r["id"]: r for r in pq.read_table(parquet, columns=COLUMNS).to_pylist() if r["id"] in want}
    device = torch.device("cuda:0")
    vocoder = UniSSTokenizer.from_pretrained(args.speech_tokenizer, device=device)
    files = {}
    for x in lines:
        row = rows[x["name"]][x["id"]]
        (out / x["name"] / "wav").mkdir(parents=True, exist_ok=True)
        path = None
        if x["codes"]:
            suffix = "" if x["repeat"] == 0 else f"_r{x['repeat']}"
            path = str(out / x["name"] / "wav" / f"{x['index']:05d}_{row['src_lang']}2{row['tgt_lang']}{suffix}.wav")
            tokens = torch.tensor([*row["bicodec_global"], *x["codes"]], dtype=torch.long, device=device)
            vocoder.save_audio(vocoder.decode(tokens), path, sample_rate=16_000)
        rec = record(row, x["index"], x["translation"], path, len(x["codes"]), {
            "system": "p2_step0_uniss_reads_qwen3" + ("_guard" if x["guard"] else ""), "repeat": x["repeat"],
            "ended": x["ended"], "finish_reason": x["finish_reason"], "expected_codes": x["expected"],
            "gen_seconds": x["gen_seconds"]})
        files.setdefault(x["name"], []).append(rec)
    for name, recs in files.items():
        with open(out / name / f"part_{args.shard}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    print(f"shard {args.shard}: {sum(len(v) for v in files.values())} decoded", flush=True)


if __name__ == "__main__":
    main()
