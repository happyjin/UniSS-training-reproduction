"""Talker fidelity of a Chinese (en2zh) talker run (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §5.4):
per stream, the talker's input text (results.jsonl translated_text) against the evaluation ASR of its speech-only wav
(asr_speech_merged.jsonl asr_text, Paraformer-zh), both reduced to zh_units (CJK characters / Latin words, punctuation
dropped); edit distance -> CER = sum(S + D + I) / sum(reference units), with S / D / I and the share of streams whose
ASR misses the last 30 % of the input (truncation proxy: the input's last 30 % of units is not matched in order by an
LCS against the ASR). English speech_gap_decomp's fidelity WER is English-only; this is its character analogue.
CPU; writes <out> (json; refuses to overwrite).

    python -m experiments.mist_w3po_v1.evaluation.talker_cer_zh --talker <talker dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.evaluation.zh_units import units


def align(ref: list[str], hyp: list[str]) -> tuple[int, int, int, list[bool]]:
    """(S, D, I, matched flags of ref) of a minimum edit-distance alignment."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    S = D = I = 0
    hit = [False] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] == hyp[j - 1]:
                hit[i - 1] = True
            else:
                S += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I, hit


def stream_stats(text: str, asr: str) -> dict:
    r, h = units(text), units(asr)
    S, D, I, hit = align(r, h)
    k = int(len(r) * 0.7)
    tail = r[k:]
    return {"n": len(r), "S": S, "D": D, "I": I, "trunc": bool(tail) and not any(hit[k:])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--talker", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    T = Path(a.talker)
    asr = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(T / "asr_speech_merged.jsonl", encoding="utf-8"))}
    rows = []
    for r in map(json.loads, open(T / "results.jsonl", encoding="utf-8")):
        if r.get("error") or not units(r.get("translated_text", "")):
            continue
        st = stream_stats(r.get("translated_text", ""), asr.get(r["id"], ""))
        st["id"] = r["id"]
        rows.append(st)
    n = sum(x["n"] for x in rows)
    rep = {"talker": str(T), "streams": len(rows), "units": n,
           "cer": round(sum(x["S"] + x["D"] + x["I"] for x in rows) / max(n, 1), 4),
           "sub": round(sum(x["S"] for x in rows) / max(n, 1), 4), "del": round(sum(x["D"] for x in rows) / max(n, 1), 4),
           "ins": round(sum(x["I"] for x in rows) / max(n, 1), 4),
           "trunc_share": round(sum(x["trunc"] for x in rows) / max(len(rows), 1), 4),
           "streams_missing_asr": sum(1 for x in rows if x["id"] not in asr)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
