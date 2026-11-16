"""Paper v14 Table 1 (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §2): multi-speaker S2ST systems
on the three test sets, speaker-attributed speech quality and speech latency.

Per system and set: cpASR-BLEU (cp_asr_bleu: meeting-level concatenated minimum-permutation BLEU of the ASR
transcripts) and tcpASR-BLEU (v11_metrics.score_ext, the paper's tcpBLEU, on the same transcripts), per-meeting
statistics averaged over the system's TTS runs; chunk-protocol speech LAAL_block / LAAL_spread / StartOffset /
EndOffset (lat_units_dump files, mean over units, averaged over runs). Paired tests MIST - system: exact sign-flip and
meeting bootstrap (v11_factorial_analysis.paired_bleu / post_review_v10_analysis.paired_ratio), Holm within each BLEU
family (cp, tcp) over all sets x baselines; latency differences uncorrected. Systems whose runs do not exist yet are
skipped. Writes one JSON (refuses to overwrite).

    python -m experiments.mist_w3po_v1.evaluation.table1_analysis --out <json> [--extra name=label:set=run[,run]...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import cp_asr_bleu as CP
from experiments.mist_w3po_v1.evaluation import post_review_v10_analysis as PRA
from experiments.mist_w3po_v1.evaluation import v11_factorial_analysis as FA
from experiments.mist_w3po_v1.evaluation import v11_metrics as V

REPO = V.REPO
FL = REPO / "eval_outputs/mist_w3po_v1/joint/v11f_analysis/lat"      # B / MIST latency units (rows R0 / R5)
TL = REPO / "eval_outputs/mist_w3po_v1/joint/table1_v14/lat"         # other systems
SETS = ("ami", "en", "ali")
LK = {"speech_laal_online": None, "speech_end_online": "endoff", "speech_start_online": "startoff"}


def systems(extra: list[str]) -> list[dict]:
    T = FA.TEXT
    out = [{"key": "seamless", "label": "SeamlessStreaming", "runs": {s: [f"jtest_{s}_seamless_default"] for s in SETS}},
           {"key": "xtranslator", "label": "X-Translator", "runs": {s: [f"jtest_{s}_xtranslator_default"] for s in SETS}}]
    for e in extra:   # key=label:set=run,run;set=run,run
        k, rest = e.split("=", 1)
        lab, sets = rest.split(":", 1)
        out.append({"key": k, "label": lab, "runs": {x.split("=")[0]: x.split("=")[1].split(",") for x in sets.split(";")}})
    out += [{"key": "B", "label": "B (deadline baseline)", "runs": {s: [T[s] + "__t1c", T[s] + "__t1c_tts2"] for s in SETS},
             "lat_rows": "R0"},
            {"key": "MIST", "label": "MIST", "runs": {s: [T[s] + "__rlplem", T[s] + "__rlplem_tts2"] for s in SETS},
             "lat_rows": "R5"}]
    return out


def tcp_pm(s: str, run: str) -> dict:
    H = {h["id"]: h for h in map(json.loads, open(REPO / V.DEC / run / "hyp_asr.jsonl", encoding="utf-8"))}
    W = [w for w in map(json.loads, open(REPO / V.WIN[s], encoding="utf-8")) if w["id"] in H]
    RW = list(map(json.loads, open(REPO / V.REF[s], encoding="utf-8")))
    U = V.universe(V.DR[s], {w["meeting"] for w in W})
    r = V.score_ext(V.DR[s], W, H, RW, U, online=True)
    return r


def lat_files(sy: dict, s: str):
    if "lat_rows" in sy:
        return [(FL / mode / f"{s}_{sy['lat_rows']}_s{k}.json.units.jsonl") for k in (1, 2) for mode in ("perword", "blockstart")]
    return [(TL / mode / f"{n}.json.units.jsonl") for n in sy["runs"][s] for mode in ("perword", "blockstart")]


def lat_stats(files: list[Path]):
    """name -> (mean over runs of unit means, meeting -> (sum, count) pooled over runs)."""
    per = defaultdict(list)
    for f in files:
        mode = f.parent.name
        acc = defaultdict(lambda: {"vals": [], "pm": defaultdict(lambda: [0.0, 0])})
        for l in open(f):
            m, uid, reg, key, v = json.loads(l)
            if key not in LK or reg not in ("ov", "1spk"):
                continue
            acc[key]["vals"].append(v)
            acc[key]["pm"][m][0] += v
            acc[key]["pm"][m][1] += 1
        for key, d in acc.items():
            name = LK[key] or ("laal_spread" if mode == "perword" else "laal_block")
            if mode == "blockstart" and LK[key]:
                continue
            per[name].append((float(np.mean(d["vals"])), {m: tuple(v) for m, v in d["pm"].items()}))
    return {k: (float(np.mean([x[0] for x in v])), PRA.pool_seeds([x[1] for x in v])) for k, v in per.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", action="append", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rng = np.random.default_rng(20261116)
    SY = systems(a.extra)
    res = {"sets": {}, "systems": [{k: v for k, v in sy.items()} for sy in SY]}
    tests = {"cp": {}, "tcp": {}}
    for s in SETS:
        eng = V.DR[s] != "en2zh"
        rows, cpm, tpm, lats = {}, {}, {}, {}
        for sy in SY:
            runs = sy["runs"].get(s, [])
            if not runs or not all((REPO / V.DEC / n / "hyp_asr.jsonl").exists() for n in runs):
                continue
            lf = lat_files(sy, s)
            if not all(f.exists() for f in lf):
                print(f"{s} {sy['key']}: latency units missing, skipped")
                continue
            cps = [CP.cp_stats(s, n) for n in runs]
            ms = set.intersection(*[set(k for k, v in st.items() if v is not None) for st in cps])
            cpm[sy["key"]] = {m: np.mean([st[m] for st in cps], 0) for m in ms}
            tr = [tcp_pm(s, n) for n in runs]
            tpm[sy["key"]] = FA.pooled(tr, "all")
            lats[sy["key"]] = lat_stats(lf)
            x = {"runs": runs, "cpASR": round(float(FA.bleu_vec(sum(cpm[sy["key"]].values()))[0]), 2),
                 "tcpASR": round(float(FA.bleu_vec(sum(tpm[sy["key"]].values()))[0]), 2),
                 "tcpASR_mean_of_runs": round(float(np.mean([r["tcpBLEU_all"] for r in tr])), 2)}
            for name, (mu, _) in lats[sy["key"]].items():
                x[name] = round(mu, 3)
            rows[sy["key"]] = x
            print(s, sy["key"], json.dumps(x), flush=True)
        P = {}
        if "MIST" in rows:
            for k in rows:
                if k == "MIST":
                    continue
                d = {"cp": FA.paired_bleu(cpm[k], cpm["MIST"], rng), "tcp": FA.paired_bleu(tpm[k], tpm["MIST"], rng)}
                for name in ("laal_block", "laal_spread", "startoff", "endoff"):
                    if name in lats[k] and name in lats["MIST"]:
                        d[name] = PRA.paired_ratio(lats[k][name][1], lats["MIST"][name][1], rng)
                P[f"MIST-{k}"] = d
                tests["cp"][f"{s}:{k}"] = d["cp"]
                tests["tcp"][f"{s}:{k}"] = d["tcp"]
        res["sets"][s] = {"rows": rows, "paired": P}
    for fam in tests.values():
        FA.holm(fam)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
