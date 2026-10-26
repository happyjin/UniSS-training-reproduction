"""Write the read/write timing data: one row per utterance, timestamps and indices only.

Joins the extracted meta, the label scores, the alignments and one rule's
segments into parquet parts with a README. No audio is copied: the source audio
and the decoded target audio are referenced by path, the target codes and the 32
global tokens by UniST parquet and row; segments address the codes by index
(20 ms each), so any piece of the target is a slice of its code list.

    python write_simul_dataset.py --rule hybrid --segments seg.parquet --supports sup.parquet \
        --omni data/unist_clean_v1/omni_links_v1 [--ctx ctx_dir] --out data/unist_simul_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

META_COLS = ["key", "id", "dataset", "src_lang", "tgt_lang", "transcription", "translation", "source_audio",
             "source_duration_ms", "target_audio", "target_duration_ms", "source_parquet", "source_row_index",
             "source_words", "target_words"]
SCORE_COLS = ["key", "tgt_err", "src_err", "latin_words", "latin_missed", "rate", "dur_ratio"]
SEG_COLS = ["key", "segments", "n_segments", "first_write_ms", "end_lag_ms", "silence_ratio", "long_gaps", "nir",
            "mean_word_delay_ms", "cuts", "abrupt_cuts", "class1_cuts", "class2_cuts"]

README = """# UniST read/write timing data ({version})

Simultaneous speech-to-speech translation data, zh<->en, built automatically from the
1.5M UniST rows whose target speech was decoded from its BiCodec codes (UniST
train-00000..00014, first 100k rows each). Timestamps and indices only: no audio is cut or copied.

## Rows kept
Qwen3-ASR-1.7B transcribed both sides. A row is kept when the target says the translation
(CER/WER <= 5%), every Latin word of a code-switched Chinese target is heard, the transcript is
in the target's script, the target codes do not loop, the source matches the UniST transcript
(error <= 5%: the word times and the alignment rest on it), and the a68 word grouping exists.

## Timing ({rule})
- support: each target word's moment of evidence. {rule_text}
- tau = max(previous tau, support + {delta} ms), monotonic.
- cut points only in real pauses of the target speech (energy {quiet} dB under the peak for
  >= {min_pause} ms between two words), at the pause's middle; never inside a word or across
  connected speech. Segments are closed at the first cut point once >= {min_seg} ms long.
- source chunks of {chunk} ms: a segment is written in the first chunk whose end reaches its tau;
  chunks before it are reads.

## Columns
key, id, dataset, src_lang, tgt_lang, transcription, translation
source_audio, source_duration_ms        the source speech (16 kHz FLAC)
target_audio, target_duration_ms        the target speech decoded from the label codes
source_parquet, source_row_index        data/raw/UniST: target_bicodec (labels), bicodec_global (voice)
source_words, target_words              JSON word timings (Qwen3-ForcedAligner); targets grouped (a68)
word_links                              JSON OmniAlign links (target_index, source_index, confidence)
context_align                           JSON contextual alignment (source unit per target word), if computed
support_ms                              JSON per target word, the rule's support time
support_<rule>_ms                       the same under every rule computed (omni, argmax, suff, hybrid), to
                                        switch rule or train several latency tiers without recomputing
segments                                JSON list: w0,w1 (target words), tau_ms, code0,code1 (target code
                                        slice), cut_ms, pause_ms, cut_class (1: pause >= 150 ms), chunk,
                                        pause_start_ms, fall_ms
first_write_ms, end_lag_ms, silence_ratio, long_gaps, mean_word_delay_ms, nir   timeline statistics
                                        (first sound = first_write_ms + the target's lead silence,
                                        target_words[0].start_ms: about 0.8 s on English targets)
tgt_err, src_err, latin_words, latin_missed, rate, dur_ratio                    label scores
"""

RULE_TEXT = {
    "omni": "The latest end among its OmniAlign links (zh-en AER 8.5).",
    "argmax": "Hibiki's contextual alignment with our Thinker: the source unit whose arrival raises the word's log-probability most.",
    "suff": "The earliest source prefix after which the Thinker's log-probability of the word stays within log 2 of its final value.",
    "hybrid": "The earliest prefix after which the Thinker's log-probability of the word stays within log 2 of its final value, "
              "and never before its strong OmniAlign links (confidence >= 0.9) have been heard.",
}


def load_dir(path: str, cols: list[str] | None = None) -> dict:
    files = sorted(Path(path).glob("*.parquet")) if Path(path).is_dir() else [Path(path)]
    out = {}
    for f in files:
        for r in pq.read_table(f, columns=cols).to_pylist():
            out[r["key"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="data/unist_clean_v1/meta")
    ap.add_argument("--scores", default="data/unist_clean_v1/scores.parquet")
    ap.add_argument("--rule", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--supports", required=True)
    ap.add_argument("--omni", required=True)
    ap.add_argument("--ctx", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--rows-per-part", type=int, default=50_000)
    ap.add_argument("--delta-ms", type=int, default=200)
    ap.add_argument("--chunk-ms", type=int, default=500)
    ap.add_argument("--min-seg-ms", type=int, default=800)
    ap.add_argument("--min-pause-ms", type=int, default=80)
    ap.add_argument("--quiet-db", type=float, default=40.0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")

    seg = load_dir(args.segments, SEG_COLS)
    # every rule's support is kept as its own column, so a later run can switch rule or train
    # several latency tiers without recomputing; support_ms is the chosen rule's
    sup_cols = [c for c in pq.read_schema(args.supports).names if c.startswith("support_")]
    sup = load_dir(args.supports, ["key"] + sup_cols)
    links = load_dir(args.omni)
    ctx = load_dir(args.ctx, ["key", "context_align"]) if args.ctx else {}
    scores = load_dir(args.scores, SCORE_COLS)
    (out / "rows").mkdir(parents=True)
    buf, part, total = [], 0, 0

    def flush():
        nonlocal buf, part
        if buf:
            pq.write_table(pa.Table.from_pylist(buf), out / "rows" / f"part-{part:05d}.parquet", compression="zstd")
            part += 1
            buf = []

    for f in sorted(Path(args.meta).glob("*.parquet")):
        for r in pq.read_table(f, columns=META_COLS).to_pylist():
            s = seg.get(r["key"])
            if s is None:
                continue
            row = dict(r)
            row["word_links"] = links.get(r["key"], {}).get("omni_links")
            row["context_align"] = ctx.get(r["key"], {}).get("context_align")
            row["support_ms"] = sup[r["key"]][f"support_{args.rule}"]
            for c in sup_cols:
                row[f"{c}_ms"] = sup[r["key"]][c]
            row["rule"] = args.rule
            row.update({k: v for k, v in s.items() if k != "key"})
            row.update({k: v for k, v in scores.get(r["key"], {}).items() if k != "key"})
            buf.append(row)
            total += 1
            if len(buf) >= args.rows_per_part:
                flush()
    flush()
    info = {"version": args.version, "rows": total, "parts": part, "rule": args.rule, "delta_ms": args.delta_ms,
            "chunk_ms": args.chunk_ms, "min_seg_ms": args.min_seg_ms, "min_pause_ms": args.min_pause_ms,
            "quiet_db": args.quiet_db, "segments": args.segments, "supports": args.supports}
    (out / "meta.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    (out / "README.md").write_text(README.format(version=args.version, rule=args.rule, rule_text=RULE_TEXT[args.rule],
                                                 delta=args.delta_ms, quiet=args.quiet_db, min_pause=args.min_pause_ms,
                                                 min_seg=args.min_seg_ms, chunk=args.chunk_ms), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
