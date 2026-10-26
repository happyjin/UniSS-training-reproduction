"""Per-row label scores from the Qwen3-ASR transcripts, for both cleaning plans.

For each row: how well the target speech says the translation (CER for
Chinese targets, WER for English), how well the source speech matches the
UniST transcript, whether every Latin word of a code-switched Chinese target
was heard, whether the transcript is in the target's script at all, whether
the target codes loop (an exact 8-code run repeated), speaking rate and the
target/source duration ratio. Scores only; the selection rules read them.

Normalisation, both sides alike: NFKC, traditional to simplified, lower case,
punctuation dropped, and every run of digits or number words (Arabic or
Chinese) collapsed to one '#' -- "2010" read as "二零一零" is not an error.

    python score_rows.py --meta data/unist_clean_v1/meta --asr data/unist_clean_v1/asr \
        --out data/unist_clean_v1/scores.parquet [--keys pilot_rows.parquet]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

SILENCE = 7645
CJK = re.compile(r"[一-鿿]")
LATIN_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
ZH_NUM = re.compile(r"[0-9零〇一二两三四五六七八九十百千万亿点几]+(?:[.,][0-9]+)?%?")
EN_NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                   "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
                   "hundred thousand million billion percent point first second third").split()
EN_NUM = re.compile(r"\b(?:\d+(?:[.,]\d+)*(?:st|nd|rd|th|s)?%?|" + "|".join(EN_NUMBER_WORDS) + r")\b")
_t2s = None


def t2s(s: str) -> str:
    global _t2s
    if _t2s is None:
        import opencc
        _t2s = opencc.OpenCC("t2s")
    return _t2s.convert(s)


def zh_tokens(s: str) -> list[str]:
    s = t2s(unicodedata.normalize("NFKC", s or "")).lower()
    s = ZH_NUM.sub("#", s)
    out = []
    for piece in re.findall(r"[一-鿿]|[a-z]+(?:'[a-z]+)?|#", s):
        out.append(piece)
    return out


def en_tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s or "").lower().replace("-", " ")
    s = EN_NUM.sub(" # ", s)
    s = re.sub(r"(?:\s*#\s*)+", " # ", s)
    return re.findall(r"[a-z]+(?:'[a-z]+)?|#|[一-鿿]", s)


def errors(hyp: list[str], ref: list[str]) -> tuple[int, int, int]:
    s = d = i = 0
    for op in Levenshtein.editops(ref, hyp):
        if op.tag == "replace": s += 1
        elif op.tag == "delete": d += 1
        else: i += 1
    return s, d, i


def loops(codes, n: int = 8) -> bool:
    seen = set()
    for k in range(len(codes) - n + 1):
        g = tuple(codes[k : k + n])
        if SILENCE in g:
            continue
        if g in seen:
            return True
        seen.add(g)
    return False


def score_chunk(job: tuple[str, dict, set | None]) -> list[dict]:
    path, asr, keys = job
    cols = ["key", "id", "dataset", "src_lang", "tgt_lang", "transcription", "translation", "target_bicodec",
            "source_duration_ms", "target_duration_ms", "a45_pass", "a68_present", "a68_pass"]
    out = []
    for r in pq.read_table(path, columns=cols).to_pylist():
        if keys is not None and r["key"] not in keys:
            continue
        t, s = asr.get((r["key"], "tgt")), asr.get((r["key"], "src"))
        if t is None or s is None:
            continue
        zh_tgt = r["tgt_lang"] in ("cmn", "zh")
        tok_t, tok_s = (zh_tokens, en_tokens) if zh_tgt else (en_tokens, zh_tokens)
        ref_t, hyp_t = tok_t(r["translation"]), tok_t(t["text"])
        ref_s, hyp_s = tok_s(r["transcription"]), tok_s(s["text"])
        et, es = errors(hyp_t, ref_t), errors(hyp_s, ref_s)
        latin_ref = LATIN_WORD.findall(unicodedata.normalize("NFKC", r["translation"]).lower()) if zh_tgt else []
        latin_hyp = set(LATIN_WORD.findall(unicodedata.normalize("NFKC", t["text"]).lower()))
        cjk_share = len(CJK.findall(t["text"])) / max(1, len(re.sub(r"\W", "", t["text"])))
        units = len([x for x in ref_t if x != "#"])
        tgt_s = max(r["target_duration_ms"], 1) / 1000
        out.append({
            "key": r["key"], "id": r["id"], "dataset": r["dataset"], "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"],
            "tgt_ref_units": len(ref_t), "tgt_err": sum(et) / max(1, len(ref_t)),
            "tgt_sub": et[0], "tgt_del": et[1], "tgt_ins": et[2],
            "src_ref_units": len(ref_s), "src_err": sum(es) / max(1, len(ref_s)),
            "latin_words": len(latin_ref), "latin_missed": sum(w not in latin_hyp for w in latin_ref),
            "tgt_script_ok": (cjk_share >= 0.5) if zh_tgt else (cjk_share <= 0.1),
            "tgt_asr_lang": t["lang"], "src_asr_lang": s["lang"],
            "loop8": loops(r["target_bicodec"]),
            "rate": units / tgt_s, "dur_ratio": r["target_duration_ms"] / max(r["source_duration_ms"], 1),
            "tgt_seconds": tgt_s, "a45_pass": r["a45_pass"], "a68_present": r["a68_present"], "a68_pass": r["a68_pass"],
            "tgt_asr": t["text"], "src_asr": s["text"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", default="", help="a parquet with a key column; only those rows")
    ap.add_argument("--jobs", type=int, default=30)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {args.out}")
    keys = set(pq.read_table(args.keys, columns=["key"]).column("key").to_pylist()) if args.keys else None
    asr_by_chunk: dict[str, dict] = {}
    for f in sorted(Path(args.asr).glob("shard*_part*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if keys is not None and r["key"] not in keys:
                continue
            shard, chunk, _ = r["key"].split("/")
            asr_by_chunk.setdefault(f"{shard}_{chunk}", {})[(r["key"], r["side"])] = r
    jobs = [(str(p), asr_by_chunk.get(p.stem, {}), keys) for p in sorted(Path(args.meta).glob("*.parquet"))
            if asr_by_chunk.get(p.stem)]
    rows = []
    with mp.Pool(min(args.jobs, max(1, len(jobs)))) as pool:
        for part in pool.imap_unordered(score_chunk, jobs):
            rows += part
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"{len(rows)} rows scored -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
