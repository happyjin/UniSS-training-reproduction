"""sst_hsmm_v1 prereg change 22 (CR-LA): training-data commit events and counterfactual commit regret.
items:  p4 samples (s < S_MAX) -> R/data/items_<d>_<k>.jsonl ({id, audio, dur_ms, translation = reference}), 4 shards per dir
events: wave17 LA2 rollouts (R/roll_<d>_<k>/blocks_<d>_w1.jsonl) -> R/events.jsonl: per rollout with >= 1 non-final commit,
        P_0 = "" (key id|0) and P_k after each non-final commit k (key id|k)
regret: R/forced_*.jsonl (prefix_forced) -> R/regret.jsonl: {id, dir, wav, dur_ms, translation (rollout), events: [{t, pre, r}]}
        S(P) = sentence BLEU vs the reference (sacrebleu, zh / 13a), r_k = S(P_k) - S(P_{k-1})
    <qwen3omni python> -m experiments.sst_hsmm_v1.crla_regret {items|events|regret} --root R
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
P4 = REPO / "eval_outputs/sst_hsmm_v1/p4"
S_MAX = 600
DIRS = ("en2zh", "zh2en")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("items", "events", "regret"))
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    R = Path(a.root)
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import join, units
    if a.mode == "items":
        refs = {d: {r["id"]: r["ref"] for r in jl(DATA / f"train_{d}.jsonl")} for d in DIRS}
        rows = {d: [] for d in DIRS}
        for x in jl(P4 / "samples.jsonl"):
            if x["s"] < S_MAX:
                rows[x["dir"]].append({"id": f"s{x['s']:04d}_j{x['j']:02d}", "audio": str(REPO / x["wav"]), "dur_ms": int(x["dur_ms"]),
                                       "translation": refs[x["dir"]][x["row"]]})
        (R / "data").mkdir(parents=True, exist_ok=True)
        for d in DIRS:
            for k in range(4):
                with open(R / "data" / f"items_{d}_{k}.jsonl", "x", encoding="utf-8") as fh:
                    for r in rows[d][k::4]:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(d, len(rows[d]))
        return
    items = {r["id"]: (d, r) for d in DIRS for p in sorted(glob.glob(str(R / "data" / f"items_{d}_*.jsonl"))) for r in jl(p)}
    roll = {r["id"]: r for d in DIRS for p in sorted(glob.glob(str(R / f"roll_{d}_*" / f"blocks_{d}_w1.jsonl"))) for r in jl(p)}
    ev = {}
    for i, r in roll.items():
        d, it = items[i]
        zh = d == "en2zh"
        u, n, E = units(r["translation"], zh), 0, [(0, "")]
        for b in r["blocks"]:
            if b["final"]:
                break
            n += len(units(b["text"], zh))
            E.append((int(b["write_ms"]), join(u[:n], zh)))
        if len(E) > 1:
            ev[i] = (d, it, r, E)
    if a.mode == "events":
        with open(R / "events.jsonl", "x", encoding="utf-8") as fh:
            for i, (d, it, r, E) in ev.items():
                for k, (t, pre) in enumerate(E):
                    fh.write(json.dumps({"key": f"{i}|{k}", "id": i, "dir": d, "audio": it["audio"], "prefix": pre}, ensure_ascii=False) + "\n")
        print("rollouts", len(roll), "with commits", len(ev), "events", sum(len(v[3]) for v in ev.values()))
        return
    import sacrebleu
    F = {r["key"]: r["translation"] for p in sorted(glob.glob(str(R / "forced_*.jsonl"))) for r in jl(p)}
    nr, neg, pos = 0, 0, 0
    with open(R / "regret.jsonl", "x", encoding="utf-8") as fh:
        for i, (d, it, r, E) in ev.items():
            if any(f"{i}|{k}" not in F for k in range(len(E))):
                continue
            tok = "zh" if d == "en2zh" else "13a"
            S = [sacrebleu.sentence_bleu(F[f"{i}|{k}"], [it["translation"]], tokenize=tok).score for k in range(len(E))]
            evs = [{"t": E[k][0], "pre": E[k][1], "r": S[k] - S[k - 1]} for k in range(1, len(E))]
            neg += sum(e["r"] < 0 for e in evs); pos += sum(e["r"] > 0 for e in evs); nr += len(evs)
            fh.write(json.dumps({"id": i, "dir": d, "wav": it["audio"], "dur_ms": it["dur_ms"], "translation": r["translation"],
                                 "S0": S[0], "events": evs}, ensure_ascii=False) + "\n")
    print("events", nr, "r<0", neg, "r>0", pos)


if __name__ == "__main__":
    main()
