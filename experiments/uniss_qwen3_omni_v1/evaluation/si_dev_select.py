"""Choose a checkpoint on held-out training items, never on RealSI: text BLEU against the items' translations and
text LAAL (each unit at its block's commit time) of si_thinker_stream --items-file runs, then, per candidate, the
BLEU its curve reaches at fixed text-LAAL budgets (linear between its points), averaged over budgets both candidates
reach and over directions; the higher mean wins. With --offline, a candidate whose whole-sentence translation of the
same items (one read of the whole audio) falls more than --max-drop BLEU under --base-offline's, in either direction,
is out (forgetting, Plan 4A's failure); the winner being the last candidate is flagged (it may want more training).

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_dev_select --items-dir data/... \
        --cand iter200=<blocks dir>:<tag> --cand iter400=<blocks dir>:<tag> --out choice.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from evaluation.text_metrics import corpus_bleu
from experiments.uniss_qwen3_omni_v1.evaluation.si_text_eval import laal

BUDGETS = (1.5, 2.0, 2.5, 3.0, 3.5)


def units(text: str, zh: bool) -> list[str]:
    return [c for c in text if not c.isspace()] if zh else text.split()


def score(path: str, items: dict, zh: bool) -> tuple[float, float]:
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    lang = "cmn" if zh else "eng"
    bleu = corpus_bleu([r["translation"] for r in recs], [items[r["id"]]["translation"] for r in recs], language=lang)["score"]
    lags = []
    for r in recs:
        delays = [b["write_ms"] for b in r["blocks"] for _ in units(b["text"], zh)]
        lags.append(laal(delays, items[r["id"]]["dur_ms"], len(units(items[r["id"]]["translation"], zh))))
    return bleu, float(np.mean(lags)) / 1000


def at(points: list[tuple[float, float]], x: float) -> float | None:
    pts = sorted(points)
    for (l0, b0), (l1, b1) in zip(pts, pts[1:]):
        if l0 <= x <= l1:
            return b0 + (b1 - b0) * (x - l0) / max(l1 - l0, 1e-9)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items-dir", required=True, help="holds dev_items_<direction>.jsonl")
    ap.add_argument("--cand", action="append", required=True, help="name=<blocks dir>:<file tag>")
    ap.add_argument("--offline", action="append", default=[], help="name=<blocks dir>:<file tag>, whole-audio runs")
    ap.add_argument("--base-offline", default="", help="<blocks dir>:<file tag> of the untrained Thinker, whole audio")
    ap.add_argument("--max-drop", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    curves = {}
    for spec in args.cand:
        name, rest = spec.split("=", 1)
        bdir, tag = rest.split(":", 1)
        for d in ("en2zh", "zh2en"):
            items = {json.loads(l)["id"]: json.loads(l) for l in open(f"{args.items_dir}/dev_items_{d}.jsonl", encoding="utf-8")}
            for f in sorted(glob.glob(f"{bdir}/blocks_{d}_w*{tag}*.jsonl")):
                b, l = score(f, items, d == "en2zh")
                curves.setdefault(name, {}).setdefault(d, []).append((round(l, 3), round(b, 2), Path(f).stem))
    names = list(curves)
    res = {"curves": curves, "at_budget": {}}
    for d in ("en2zh", "zh2en"):
        for x in BUDGETS:
            vals = {n: at([(l, b) for l, b, _ in curves[n].get(d, [])], x) for n in names}
            if all(v is not None for v in vals.values()):
                res["at_budget"][f"{d}@{x}"] = {n: round(v, 3) for n, v in vals.items()}
    mean = {n: float(np.mean([v[n] for v in res["at_budget"].values()])) if res["at_budget"] else None for n in names}
    res["mean"] = mean
    allowed = list(names)
    if args.offline and args.base_offline:   # whole-sentence quality, against the untrained Thinker's
        def offline_bleu(spec):
            bdir, tag = spec.split(":", 1)
            got = {}
            for d in ("en2zh", "zh2en"):
                items = {json.loads(l)["id"]: json.loads(l) for l in open(f"{args.items_dir}/dev_items_{d}.jsonl", encoding="utf-8")}
                f = sorted(glob.glob(f"{bdir}/blocks_{d}_w*{tag}*.jsonl"))
                if f:
                    got[d] = round(score(f[0], items, d == "en2zh")[0], 2)
            return got
        base = offline_bleu(args.base_offline)
        res["offline"] = {"base": base}
        for spec in args.offline:
            n, rest = spec.split("=", 1)
            res["offline"][n] = got = offline_bleu(rest)
            if n in allowed and any(base.get(d, 0) - got.get(d, 0) > args.max_drop for d in base):
                allowed.remove(n)
        res["allowed"] = allowed
    res["choice"] = max(allowed or names, key=lambda n: mean[n] if mean[n] is not None else -1)
    res["choice_is_last"] = res["choice"] == names[-1]
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"mean": mean, "choice": res["choice"], "choice_is_last": res["choice_is_last"],
                      "offline": res.get("offline"), "allowed": res.get("allowed"), "budgets": list(res["at_budget"])}), flush=True)


if __name__ == "__main__":
    main()
