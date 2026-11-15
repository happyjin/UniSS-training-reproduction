"""v11 P0-2 analysis (reports/mist_w3po_v1/v11_p02_alieval_prereg.zh-CN.md §3): the AliMeeting Eval clean set.
CPU, read-only on finished runs; writes one new JSON (refuses to overwrite). Statistics are those of the v10 paper,
reused, not re-implemented:
  BLEU (text tcpBLEU, ASR-BLEU pooled over the two TTS seeds) make_tables_v10.paired / pooled / holm: corpus BLEU of
      summed meeting statistics, exact two-sided sign-flip p over all 2^M flips (M = 8: minimum p = 2/256), 95% interval
      from 10,000 meeting-bootstrap resamples (rng 20261114);
  latency / speech rate / UTMOS / pauses: post_review_v10_analysis (paired_ratio over meetings, two seeds pooled) on
      lat_units_dump / talker_quality / talker_block_pauses outputs under eval_outputs/mist_w3po_v1/joint/alieval_analysis
      (alieval_postproc.sh).
Confirmatory family (Holm over 3): B+PA - B on ASR-BLEU (H1a), LAAL_block (H1b), EndOffset (H2); confirmed when
p_Holm < 0.05 AND the 95% interval excludes 0 in the predicted direction. Everything else exploratory.

    python -m experiments.mist_w3po_v1.evaluation.alieval_analysis --out <new json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "paper/gstar_si_v10/scripts"))
import make_tables_v10 as MT  # noqa: E402
from experiments.mist_w3po_v1.evaluation import post_review_v10_analysis as PRA  # noqa: E402

DEC = REPO / "eval_outputs/mist_w3po_v1/decode"
TE = REPO / "eval_outputs/mist_w3po_v1/crf/t_eval"
NB = "jaev_b_joint_v8h6k_20261113T073658Z_s06000"
NM = "jaev_m_joint_v2p_20261113T120415Z_s03500_wbzm1"
NM4 = "jaev_m4_joint_v2p_20261113T120415Z_s04000_wbzm1"
TEXT = {"b": NB, "e2": NB, "m": NM, "m4": NM4}
SPEECH = {"b": NB + "__t1c", "e2": NB + "__rlplem", "m": NM + "__rlplem", "m4": NM4 + "__rlplem"}
PAIRS = (("e2", "b"), ("m", "e2"), ("m4", "e2"), ("m", "b"), ("m4", "m"))


def j(p: Path):
    return json.load(open(p))


def bleu(rng) -> dict:
    MT.RNG = rng
    R = {"systems": {}, "paired": {}}
    T, S = {}, {}
    for sy in TEXT:
        t = j(DEC / TEXT[sy] / "score_text.json")
        a = [j(DEC / (SPEECH[sy] + sfx) / "score_asr.json") for sfx in ("", "_tts2")]
        R["systems"][sy] = {"text": {r: t.get(f"tcpBLEU_{r}") for r in ("all", "ov", "1spk")},
                            "text_CR": t.get("CR"), "text_collapse": t.get("collapse"),
                            "asr_all_seeds": [x.get("tcpBLEU_all") for x in a],
                            "asr_all": round(float(np.mean([x["tcpBLEU_all"] for x in a])), 2),
                            "asr_ov": round(float(np.mean([x["tcpBLEU_ov"] for x in a])), 2),
                            "asr_1spk": round(float(np.mean([x["tcpBLEU_1spk"] for x in a])), 2)}
        T[sy] = {r: MT.per_meeting(t, r) for r in ("all", "ov", "1spk")}
        S[sy] = {r: MT.pooled(a, r) for r in ("all", "ov", "1spk")}
    for c, r in PAIRS:
        k = f"{c}-{r}"
        R["paired"][k] = {}
        if TEXT[c] != TEXT[r]:
            for reg in ("all", "ov", "1spk"):
                R["paired"][k][f"text_{reg}"] = MT.paired(T[r][reg], T[c][reg])
        for reg in ("all", "ov", "1spk"):
            R["paired"][k][f"asr_{reg}"] = MT.paired(S[r][reg], S[c][reg])
    return R


def text_laal() -> dict:
    out = {}
    for sy in ("b", "m", "m4"):
        f = TE / TEXT[sy] / f"textlat_chunk_{TEXT[sy]}.json"
        out[sy] = j(f)["unit"]["text_laal_online"]["all"]["mean"] if f.exists() else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rng = np.random.default_rng(20261114)
    # post_review_v10_analysis on the Eval rows: set "aev", systems b / e2 / m / m4
    PRA.PR = REPO / "eval_outputs/mist_w3po_v1/joint/alieval_analysis"
    PRA.SPD = REPO / "eval_outputs/mist_w3po_v1/speech"
    PRA.SETS = ("aev",)
    PRA.SYSS = ("b", "e2", "m", "m4")
    PRA.PAIRS = PAIRS
    PRA.run_name = lambda s, sy, k: SPEECH[sy] + ("_tts2" if k == 2 else "")
    res = {"doc": __doc__, "systems": {"text": TEXT, "speech": SPEECH}, "bleu": bleu(rng), "text_laal": text_laal(),
           "latency": PRA.latency(rng)["aev"], "speech": PRA.speech(rng)["aev"], "quality": PRA.quality(rng)["aev"],
           "block_pauses": PRA.pauses().get("aev")}
    L = res["latency"]["paired"]["e2-b"]
    conf = {"H1a_asr_bleu": dict(res["bleu"]["paired"]["e2-b"]["asr_all"]),
            "H1b_laal_block": {"diff": L["slaal_blockstart"]["diff_pooled"], "p": L["slaal_blockstart"]["p_signflip"],
                               "ci": L["slaal_blockstart"]["ci95"]},
            "H2_endoffset": {"diff": L["endoff"]["diff_pooled"], "p": L["endoff"]["p_signflip"], "ci": L["endoff"]["ci95"]}}
    MT.holm(conf)
    sign = {"H1a_asr_bleu": 1, "H1b_laal_block": -1, "H2_endoffset": -1}
    for k, v in conf.items():
        lo, hi = v["ci"]
        v["confirmed"] = bool(v["p_holm"] < 0.05 and (lo > 0 if sign[k] > 0 else hi < 0))
    res["confirmatory"] = conf
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(conf, indent=1))


if __name__ == "__main__":
    main()
