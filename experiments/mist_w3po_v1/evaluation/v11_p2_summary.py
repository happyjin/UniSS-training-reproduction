"""v11 P2 summary: PA-F (fluency-constrained PA Talker) vs B+PA on the same B text (reports/mist_w3po_v1/
v11_p2_fluency_pa_prereg.zh-CN.md). CPU, read-only; writes one JSON (refuses to overwrite).

Per set (dev: devz / deve; test: ami / ali / en), systems "pa" = <N>__rlplem (E2, B+PA) and "paf" = <N>__rlplemf
(PA-F), two TTS seeds each:
  asr      ASR-BLEU (tcpBLEU of the ASR text) per seed and mean; the gate_2seed verdict (speech item) PA-F vs B+PA
  latency  chunk protocol, region all: speech LAAL perword (LAAL_spread) and blockstart (LAAL_block), speech start,
           EndOffset, text LAAL; paired meeting bootstrap + sign flip (post_review_v10_analysis.paired_ratio,
           seeds pooled as there)
  speech   units per second of speech (post_review_v10_analysis' definition), paired rate difference
  pauses   talker_block_pauses (>= 0.5 s gaps per minute of speech, mid-sentence ones), mean over seeds
  quality  UTMOS / SIM seed 1 (talker_quality), paired UTMOS difference

    python -m experiments.mist_w3po_v1.evaluation.v11_p2_summary --split dev|test --out <json>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import post_review_v10_analysis as PR
from experiments.mist_w3po_v1.evaluation.talker_block_pauses import n_words
from experiments.mist_w3po_v1.evaluation.zh_units import n_units

EO = Path("eval_outputs/mist_w3po_v1")
DEC, SPD, TE = EO / "decode", EO / "speech", EO / "crf/t_eval"
P2 = Path("logs/mist_v11/p2")
RT1 = "joint_v8h6k_20261113T073658Z_s06000"
NAMES = {"devz": f"jdev_bz_{RT1}", "deve": f"jdev_be_{RT1}", "ami": f"jtest_bami_{RT1}", "ali": f"jtest_bali_{RT1}",
         "en": f"jtest_ben_{RT1}"}
SUFFIX = {"pa": "__rlplem", "paf": "__rlplemf"}
LKEYS = PR.LKEYS


def run(s, sy, k):
    return NAMES[s] + SUFFIX[sy] + ("_tts2" if k == 2 else "")


def lat_units(path: Path) -> dict:
    out = defaultdict(lambda: {"vals": [], "pm": defaultdict(lambda: [0.0, 0])})
    for l in open(path):
        m, uid, reg, key, v = json.loads(l)
        if key not in LKEYS or reg not in ("ov", "1spk"):
            continue
        out[key]["vals"].append(v)
        out[key]["pm"][m][0] += v
        out[key]["pm"][m][1] += 1
    return out


def speech_pm(s, sy, k):
    cnt = n_units if s in ("deve", "en") else n_words
    sec = units = 0.0
    pm = defaultdict(lambda: [0.0, 0.0])
    for l in open(SPD / run(s, sy, k) / "talker/results.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("error"):
            continue
        m = PR.meeting_of(r["id"].split("__S")[0])
        for b in r.get("blocks") or []:
            d, u = b["dur_ms"] / 1000, cnt(b["text"])
            sec += d
            units += u
            pm[m][0] += u
            pm[m][1] += d
    return units / sec, {m: tuple(v) for m, v in pm.items()}


def summarise(s: str, rng) -> dict:
    R = {"names": {sy: [run(s, sy, k) for k in (1, 2)] for sy in SUFFIX}, "systems": {}, "paired": {}}
    D = {}
    for sy in SUFFIX:
        S = R["systems"].setdefault(sy, {})
        asr = [json.load(open(DEC / run(s, sy, k) / "score_asr.json"))["tcpBLEU_all"] for k in (1, 2)]
        S["asr_bleu"] = {"mean": round(float(np.mean(asr)), 3), "seeds": asr}
        for mode in ("perword", "blockstart"):
            us = []
            for k in (1, 2):
                p = P2 / "lat" / f"{mode}_{run(s, sy, k)}.json.units.jsonl"
                if p.exists():
                    us.append(lat_units(p))
            if len(us) < 2:
                continue
            for key, short in LKEYS.items():
                if mode == "blockstart" and key != "speech_laal_online":
                    continue
                name = short if key != "speech_laal_online" else f"slaal_{mode}"
                S[name] = round(float(np.mean([np.mean(u[key]["vals"]) for u in us])), 4)
                D[(sy, name)] = PR.pool_seeds([{m: tuple(v) for m, v in u[key]["pm"].items()} for u in us])
        rs = [speech_pm(s, sy, k) for k in (1, 2)]
        S["units_per_s"] = round(float(np.mean([r[0] for r in rs])), 4)
        D[(sy, "rate")] = PR.pool_seeds([r[1] for r in rs])
        ps = []
        for k in (1, 2):
            p = P2 / "pauses" / f"{run(s, sy, k)}.json"
            if p.exists() and p.stat().st_size:
                ps.append(json.loads(open(p).readline()))
        if ps:
            S["pauses_per_min"] = round(float(np.mean([x["pauses_all"]["per_min"] for x in ps])), 2)
            S["mid_pauses_per_min"] = round(float(np.mean([x["pauses_mid_sent"]["per_min"] for x in ps])), 2)
            S["pause_sec_per_min"] = round(float(np.mean([x["pauses_all"]["sec_per_min"] for x in ps])), 2)
            S["speech_min"] = round(float(np.mean([x["speech_min"] for x in ps])), 1)
        q = P2 / "quality" / run(s, sy, 1) / "quality.json"
        if q.exists():
            qq = json.load(open(q))
            S["utmos"], S["sim"] = qq.get("utmos_mean"), qq.get("sim_mean")
    names = sorted({n for (_, n) in D})
    for name in names:
        if ("pa", name) in D and ("paf", name) in D:
            res = PR.paired_ratio(D[("pa", name)], D[("paf", name)], rng)
            R["paired"][name] = res
    qi = {}
    for sy in SUFFIX:
        p = P2 / "quality" / run(s, sy, 1) / "quality.json.items.jsonl"
        if p.exists():
            qi[sy] = {json.loads(l)["id"]: json.loads(l)["utmos"] for l in open(p)}
    if len(qi) == 2:
        A, B = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
        for k in sorted(set(qi["pa"]) & set(qi["paf"])):
            m = PR.meeting_of(k.split("__S")[0])
            A[m][0] += qi["pa"][k]
            A[m][1] += 1
            B[m][0] += qi["paf"][k]
            B[m][1] += 1
        R["paired"]["utmos"] = PR.paired_ratio({m: tuple(v) for m, v in A.items()}, {m: tuple(v) for m, v in B.items()}, rng)
    g = sorted((TE / run(s, "paf", 1)).glob("gate_*.json"))
    if g:
        R["gate"] = json.load(open(g[0]))
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sets = a.sets or (["devz", "deve"] if a.split == "dev" else ["ami", "ali", "en"])
    rng = np.random.default_rng(20261114)
    R = {"doc": __doc__.split("\n\n")[0], "sets": {}}
    for s in sets:
        try:
            R["sets"][s] = summarise(s, rng)
        except FileNotFoundError as e:
            R["sets"][s] = {"missing": str(e)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1) + "\n")
    print(json.dumps({s: {sy: v for sy, v in r.get("systems", {}).items()} for s, r in R["sets"].items()}, indent=1))


if __name__ == "__main__":
    main()
