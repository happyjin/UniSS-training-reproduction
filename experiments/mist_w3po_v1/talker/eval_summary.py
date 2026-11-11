"""The dev table of the two talker schemes against A' (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md §4). CPU.

Per system (A' and every scheme x setting), both TTS seeds averaged: speech ASR-BLEU official (score_asr.json) and
normalised segment-level (speech_gap_decomp asr_seg, repo protocol), the text -> speech gap under both protocols
(text_seg raw - asr raw; text_seg repo - asr_seg repo), speech LAAL and EndOffset (speech_latency_lit, units, online),
talker WER (speech_gap_decomp fidelity), forced blocks (talker results), UTMOS / WavLM-SV (talker_quality), and the
2-seed gate vs A' (gate_2seed). Writes <out>.json and <out>.md (refuses to overwrite).

    python -m experiments.mist_w3po_v1.talker.eval_summary --out eval_outputs/mist_w3po_v1/talker_crf/eval/summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

A = "dev_crf_v2p_s3000_finalpost_s8_c2g06"
DEC = Path("eval_outputs/mist_w3po_v1/decode")
SPD = Path("eval_outputs/mist_w3po_v1/speech")
EV = Path("eval_outputs/mist_w3po_v1/talker_crf/eval")
QUAL = Path("eval_outputs/mist_w3po_v1/talker_crf/quality")
A_LAT = {"": "eval_outputs/mist_w3po_v1/talker_crf/eval/aprime/lat_aprime.json",   # current speech_latency_lit (+1ch LAAL)
         "_tts2": "eval_outputs/mist_w3po_v1/talker_crf/eval/aprime/lat_aprime_tts2.json"}
A_GAP = {"": "eval_outputs/mist_w3po_v1/speech_gap_v1/v2p_c2g06_s1.json", "_tts2": "eval_outputs/mist_w3po_v1/speech_gap_v1/v2p_c2g06_s2.json"}
SYSTEMS = [("A'", None, None), ("PL-EM k=1 自选语速", "plem", "k1"), ("PL-EM c1+b1+s2", "plem", "c1b1s2"),
           ("对齐CRF k=1 覆盖门", "align", "k1gate"), ("对齐CRF c1+b1+s2", "align", "c1b1s2")]


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def forced(talker: Path):
    n = f = 0
    p = talker / "results.jsonl"
    if not p.exists():
        return None
    for l in open(p):
        r = json.loads(l)
        for b in r.get("blocks", []):
            n += 1
            f += bool(b.get("forced"))
    return f


def one(scheme, tag, sfx):
    if scheme is None:
        name, lat, gap = A + sfx, jl(A_LAT[sfx]), jl(A_GAP[sfx])
    else:
        name = f"{A}__{scheme}_{tag}{sfx}"
        lat, gap = jl(EV / scheme / f"lat_{name}.json"), jl(EV / scheme / f"gap_{name}.json")
    sa = jl(DEC / name / "score_asr.json")
    q = jl(QUAL / name / "quality.json")
    r = {"speech_official": sa["tcpBLEU_all"] if sa else None}
    if gap:
        s = gap["scores"]
        r.update(speech_norm_seg=s["asr_seg"]["repo"]["all"], text_official=s["text_seg"]["raw"]["all"],
                 text_norm_seg=s["text_seg"]["repo"]["all"], talker_wer=gap["fidelity"]["wer"])
        r["gap_official"] = round(r["text_official"] - s["asr"]["raw"]["all"], 2)
        r["gap_norm_seg"] = round(r["text_norm_seg"] - r["speech_norm_seg"], 2)
    if lat:
        r.update(speech_laal=lat["unit"]["speech_laal_online"]["all"]["mean"],
                 end_offset=lat["unit"]["speech_end_online"]["all"]["mean"])
        if "speech_laal_online_1ch" in lat["unit"]:   # one output channel per meeting (supplementary, not registered)
            r.update(speech_laal_1ch=lat["unit"]["speech_laal_online_1ch"]["all"]["mean"],
                     end_offset_1ch=lat["unit"]["speech_end_online_1ch"]["all"]["mean"])
    r["forced_blocks"] = forced(SPD / name / "talker")
    if q:
        r.update(utmos=q["utmos_mean"], sim=q["sim_mean"])
    return r


def mean2(a, b):
    out = {}
    for k in set(a) | set(b):
        v = [x[k] for x in (a, b) if x.get(k) is not None]
        out[k] = round(float(np.mean(v)), 3) if len(v) == 2 else (v[0] if v else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out + ".json").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    res = {}
    for label, scheme, tag in SYSTEMS:
        s1, s2 = one(scheme, tag, ""), one(scheme, tag, "_tts2")
        g = jl(EV / scheme / f"gate_{tag}.json") if scheme else None
        res[label] = {"seed1": s1, "seed2": s2, "mean": mean2(s1, s2),
                      "gate": {k: {"diff": v["diff"], "p_le0": v["p_le0"], "pass": v["pass"]} for k, v in g["items"].items()}
                      if g else None}
    cols = [("speech_official", "语音官方"), ("speech_norm_seg", "语音归一按段"), ("gap_official", "差距官方"),
            ("gap_norm_seg", "差距归一按段"), ("speech_laal", "语音LAAL"), ("end_offset", "EndOffset"), ("speech_laal_1ch", "单通道LAAL"),
            ("talker_wer", "talker WER"), ("forced_blocks", "强制截断块"), ("utmos", "UTMOS"), ("sim", "SIM")]
    md = ["| 系统 | " + " | ".join(c for _, c in cols) + " | 门槛 |", "|---" * (len(cols) + 2) + "|"]
    for label, v in res.items():
        m = v["mean"]
        gate = "—" if not v["gate"] else ("通过" if all(x["pass"] for x in v["gate"].values()) else "未通过") + \
            " (" + ", ".join(f"{k} {x['diff']:+.2f}/P{x['p_le0']:.2f}" for k, x in v["gate"].items()) + ")"
        md.append(f"| {label} | " + " | ".join("—" if m.get(k) is None else str(m[k]) for k, _ in cols) + f" | {gate} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
