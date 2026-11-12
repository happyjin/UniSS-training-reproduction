"""Synthesis jobs for synth.py (registration paper/baselines_run_plan_qabc_omnitalker_v1.md §2).

realsi: the committed blocks of the 8 tier points (common.POINTS), read from the paper's seed-0 run and checked to be
  identical (text, write_ms) in its seed-1 / seed-2 runs; written to <OUT>/blocks/<dir>_<tier>.jsonl (id,
  source_duration_ms, translation_ref, blocks[write_ms, text, final]). One request per block with speakable text
  (common.tts_text), per Omni talker seed 0 / 1 / 2; shard_<seed*8 + i>.jsonl = that seed's requests [i::8].
smoke: 60 + 60 written pieces from the Thinker's TRAINING rows (data/si_thinker_oracle_prefix_v1/rows_train.jsonl;
  not RealSI), random.Random(0), one shard, seed 0 -- a pipeline check only.

Refuses an existing jobs folder.

    python experiments/s2st_baselines_v1/qabc_omnitalker/make_jobs.py realsi
    python experiments/s2st_baselines_v1/qabc_omnitalker/make_jobs.py smoke --root <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.s2st_baselines_v1.qabc_omnitalker.common import (  # noqa: E402
    LANGS, OUT, PAPER_SEEDS, POINTS, REPO, SEEDS, paper_run, point_name, tts_text)

SHARDS_PER_SEED = 8


def realsi() -> None:
    (OUT / "jobs").mkdir(parents=True, exist_ok=False)
    (OUT / "blocks").mkdir(exist_ok=False)
    per_seed = {s: [] for s in SEEDS}
    stats = {}
    for p in POINTS:
        d, tier = p[0], p[1]
        runs = [[json.loads(l) for l in open(paper_run(p, s) / "results.jsonl", encoding="utf-8")] for s in PAPER_SEEDS]
        for other in runs[1:]:   # the talker seeds share the committed text: check it
            assert [(r["id"], [(b["write_ms"], b["text"], b["final"]) for b in r["blocks"]]) for r in other] == \
                   [(r["id"], [(b["write_ms"], b["text"], b["final"]) for b in r["blocks"]]) for r in runs[0]], p
        name = point_name(d, tier)
        n_req = n_skip = 0
        with open(OUT / "blocks" / f"{name}.jsonl", "x", encoding="utf-8") as fh:
            for r in runs[0]:
                assert r["tgt_lang"] == LANGS[d][1]
                fh.write(json.dumps({"id": r["id"], "source_duration_ms": r["source_duration_ms"],
                                     "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"],
                                     "transcription_ref": r["transcription_ref"], "translation_ref": r["translation_ref"],
                                     "blocks": [{"write_ms": b["write_ms"], "text": b["text"], "final": b["final"]}
                                                for b in r["blocks"]]}, ensure_ascii=False) + "\n")
                for k, b in enumerate(r["blocks"]):
                    t = tts_text(b["text"])
                    if t is None:
                        n_skip += 1
                        continue
                    n_req += 1
                    for s in SEEDS:
                        per_seed[s].append({"key": f"{name}_s{s}__{r['id']}__{k}", "seed": s, "point": name,
                                            "id": r["id"], "k": k, "text": t, "lang": r["tgt_lang"]})
        stats[name] = {"sentences": len(runs[0]), "blocks": sum(len(r["blocks"]) for r in runs[0]),
                       "requests": n_req, "punct_only_blocks": n_skip, "paper_run": str(paper_run(p).relative_to(REPO))}
    n = 0
    for s in SEEDS:
        for i in range(SHARDS_PER_SEED):
            with open(OUT / "jobs" / f"shard_{s * SHARDS_PER_SEED + i:03d}.jsonl", "x", encoding="utf-8") as fh:
                for j in per_seed[s][i::SHARDS_PER_SEED]:
                    fh.write(json.dumps(j, ensure_ascii=False) + "\n")
                    n += 1
    with open(OUT / "jobs" / "stats.json", "x", encoding="utf-8") as fh:
        json.dump({"points": stats, "requests_total": n, "shards": SHARDS_PER_SEED * len(SEEDS)}, fh, indent=1)
    print(json.dumps(stats, indent=1), n)


def smoke(root: Path) -> None:
    (root / "jobs").mkdir(parents=True, exist_ok=False)
    rows = [json.loads(l) for l in open(REPO / "data/si_thinker_oracle_prefix_v1/rows_train.jsonl", encoding="utf-8")]
    rng = random.Random(0)
    out = []
    for d in ("en2zh", "zh2en"):
        pool = [r for r in rows if r["dir"] == d and tts_text(r["target"])]
        for i, r in enumerate(rng.sample(pool, 60)):
            out.append({"key": f"smoke_{d}_{i:02d}", "seed": 0, "point": f"smoke_{d}", "id": r["id"], "k": 0,
                        "text": tts_text(r["target"]), "lang": LANGS[d][1]})
    with open(root / "jobs" / "shard_000.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(j, ensure_ascii=False) + "\n" for j in out)
    print(len(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("realsi", "smoke"))
    ap.add_argument("--root")
    a = ap.parse_args()
    realsi() if a.mode == "realsi" else smoke(Path(a.root))
