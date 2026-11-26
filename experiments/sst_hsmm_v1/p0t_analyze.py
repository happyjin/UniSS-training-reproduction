"""sst_hsmm_v1 prereg change 43 (P0-T): pure LA2 at 1.0 s vs 0.5 s reads (si_thinker_stream --policy la2 --windows 1,0.5).
<root>/<d>/blocks_<d>_w1.jsonl and blocks_<d>_w0.5.jsonl, items (with "translation") in --data. Per direction: corpus BLEU
(sacrebleu, zh / 13a), mean proxy speech LAAL D2 (s; p0rl_analyze.playback), mean gap / pauses, blocks per sentence.
pass (per direction) = D2(0.5) - D2(1.0) <= -0.15 s and BLEU(0.5) - BLEU(1.0) >= -0.3. Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1.p0rl_analyze import playback  # noqa: E402
from experiments.voice_onset_v1.playout import si_units  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    import sacrebleu
    res = {}
    for d in ("en2zh", "zh2en"):
        zh = d.endswith("zh")
        lang = "cmn" if zh else "eng"
        items = {r["id"]: r for r in map(json.loads, open(Path(a.data) / f"items_{d}.jsonl", encoding="utf-8"))}
        r_ = {}
        for w in ("1", "0.5"):
            rows = {r["id"]: r for r in map(json.loads, open(Path(a.root) / d / f"blocks_{d}_w{w}.jsonl", encoding="utf-8"))}
            ids = [i for i in items if i in rows]
            hyp = [rows[i]["translation"] for i in ids]
            ref = [items[i]["translation"] for i in ids]
            st = [playback(rows[i]["blocks"], lang, items[i]["dur_ms"], len(si_units(items[i]["translation"], lang))) for i in ids]
            r_[w] = {"n": len(ids), "bleu": float(sacrebleu.corpus_bleu(hyp, [ref], tokenize="zh" if zh else "13a").score),
                     "d2": float(np.nanmean([s[0] for s in st])), "gap": float(np.mean([s[1] for s in st])),
                     "pauses": float(np.mean([s[2] for s in st])), "blocks": float(np.mean([len(rows[i]["blocks"]) for i in ids]))}
        dd2, dbl = r_["0.5"]["d2"] - r_["1"]["d2"], r_["0.5"]["bleu"] - r_["1"]["bleu"]
        res[d] = {**r_, "dd2": dd2, "dbleu": dbl, "pass": bool(dd2 <= -0.15 and dbl >= -0.3)}
        print(d, json.dumps(res[d]), flush=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
