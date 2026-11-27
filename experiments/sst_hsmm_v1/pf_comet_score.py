"""portfolio 2026-11-27 (change 55, COMET auxiliary): score a pf_comet_dump file with Unbabel/wmt22-comet-da (venvs/comet, one GPU).
Scores ASR transcripts (ASR-COMET) and text translations (text-COMET) against src + ref; unique triples scored once. Writes per-row
scores (<out>.rows.jsonl) and a summary {sys: {dir: {asr, text, cells}}} = mean over items per cell, then mean over the cells (as
collect.py), plus the difference to B_b.
    HF_HOME=... venvs/comet/bin/python experiments/sst_hsmm_v1/pf_comet_score.py --dump <jsonl> --out <json>
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows only")
    a = ap.parse_args()
    out, rows_out = Path(a.out), Path(a.out + ".rows.jsonl")
    for p in (out, rows_out):
        if p.exists():
            raise SystemExit(f"refuse to overwrite {p}")
    rows = [json.loads(l) for l in open(a.dump, encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    trip = sorted({(r["src"], r[k], r["ref"]) for r in rows for k in ("asr", "text")})
    pred = model.predict([{"src": s, "mt": m, "ref": t} for s, m, t in trip], batch_size=a.batch, gpus=1, progress_bar=False)
    sc = dict(zip(trip, pred.scores))
    acc = defaultdict(lambda: defaultdict(list))
    with open(rows_out, "w", encoding="utf-8") as fo:
        for r in rows:
            r["comet_asr"], r["comet_text"] = sc[(r["src"], r["asr"], r["ref"])], sc[(r["src"], r["text"], r["ref"])]
            fo.write(json.dumps({k: r[k] for k in ("sys", "dir", "cell", "id", "comet_asr", "comet_text")}) + "\n")
            acc[(r["sys"], r["dir"])][r["cell"]].append((r["comet_asr"], r["comet_text"]))
    res = defaultdict(dict)
    for (s, d), cells in acc.items():
        cm = {c: [sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v), len(v)] for c, v in cells.items()}
        res[s][d] = {"asr": sum(v[0] for v in cm.values()) / len(cm), "text": sum(v[1] for v in cm.values()) / len(cm), "cells": cm}
    for s in res:
        for d in res[s]:
            if "B_b" in res and d in res["B_b"]:
                res[s][d]["asr_vs_B_b"] = res[s][d]["asr"] - res["B_b"][d]["asr"]
                res[s][d]["text_vs_B_b"] = res[s][d]["text"] - res["B_b"][d]["text"]
    out.write_text(json.dumps(res, indent=1))
    for s in res:
        print(s, {d: (round(v["asr"], 4), round(v.get("asr_vs_B_b", 0), 4), round(v["text"], 4)) for d, v in res[s].items()})


if __name__ == "__main__":
    main()
