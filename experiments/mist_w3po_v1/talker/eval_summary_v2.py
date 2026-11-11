"""The dev table of talker_crf_prereg_v2 (reports/mist_w3po_v1/talker_crf_prereg_v2.zh-CN.md §4). CPU.

Systems (A''s text, both TTS seeds averaged with eval_summary.one / mean2): A', PL-EM c1+b1+s2 (v1), BoN-SFT c1+b1+s2
(experiment 1, eval dirs <A'>__bon_c1b1s2{,_tts2}), single-channel PL-EM c1+b1+s2 with the shared-channel backlog
token (experiment 2, <A'>__plemm_c1b1s2{,_tts2}). Per system: the gate vs A' (talker_crf/eval/<scheme>/gate_c1b1s2.json)
and the paired comparison vs PL-EM c1+b1+s2 (gate_vs_plem_c1b1s2.json: speech official, paired meeting bootstrap over
the two seeds; every other column: the difference of the 2-seed means, per seed in the json). 1-channel speech LAAL
(FIFO) is experiment 2's primary latency. Writes <out>.json and <out>.md (refuses to overwrite).

    python -m experiments.mist_w3po_v1.talker.eval_summary_v2 --out eval_outputs/mist_w3po_v1/talker_crf/eval/summary_v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.talker import eval_summary as ES

SYSTEMS = [("A'", None, None), ("PL-EM c1+b1+s2", "plem", "c1b1s2"), ("实验一 BoN-SFT c1+b1+s2", "bon", "c1b1s2"),
           ("实验二 单通道 PL-EM c1+b1+s2", "plemm", "c1b1s2")]
COLS = [("speech_official", "语音官方"), ("speech_norm_seg", "语音归一按段"), ("gap_official", "差距官方"),
        ("gap_norm_seg", "差距归一按段"), ("speech_laal", "语音LAAL(每条流)"), ("speech_laal_1ch", "单通道LAAL"),
        ("end_offset", "EndOffset"), ("end_offset_1ch", "单通道EndOffset"), ("talker_wer", "talker WER"),
        ("forced_blocks", "强制截断块"), ("utmos", "UTMOS"), ("sim", "SIM")]
VS = ("speech_official", "speech_norm_seg", "gap_norm_seg", "speech_laal", "speech_laal_1ch", "end_offset",
      "end_offset_1ch", "talker_wer", "forced_blocks", "utmos", "sim")


def diff(a: dict, b: dict) -> dict:
    return {k: (None if a.get(k) is None or b.get(k) is None else round(a[k] - b[k], 4)) for k in VS}


def build() -> dict:
    res = {}
    for label, scheme, tag in SYSTEMS:
        s1, s2 = ES.one(scheme, tag, ""), ES.one(scheme, tag, "_tts2")
        g = ES.jl(ES.EV / scheme / f"gate_{tag}.json") if scheme else None
        gp = ES.jl(ES.EV / scheme / f"gate_vs_plem_{tag}.json") if scheme in ("bon", "plemm") else None
        res[label] = {"scheme": scheme, "seed1": s1, "seed2": s2, "mean": ES.mean2(s1, s2),
                      "gate": {k: {"diff": v["diff"], "p_le0": v["p_le0"], "pass": v["pass"]} for k, v in g["items"].items()}
                      if g else None,
                      "gate_vs_plem": {k: {"diff": v["diff"], "p_le0": v["p_le0"], "pass": v["pass"], "A": v["A"], "B": v["B"]}
                                       for k, v in gp["items"].items()} if gp else None}
    pl = res["PL-EM c1+b1+s2"]
    for label, v in res.items():
        if v["scheme"] in ("bon", "plemm"):
            v["vs_plem"] = {"mean": diff(v["mean"], pl["mean"]), "seed1": diff(v["seed1"], pl["seed1"]),
                            "seed2": diff(v["seed2"], pl["seed2"])}
    return res


def markdown(res: dict) -> list[str]:
    md = ["| 系统 | " + " | ".join(c for _, c in COLS) + " | 门槛（对 A′） |", "|---" * (len(COLS) + 2) + "|"]
    for label, v in res.items():
        m = v["mean"]
        gate = "—" if not v["gate"] else ES.gate_label(v["gate"])
        md.append(f"| {label} | " + " | ".join("—" if m.get(k) is None else str(m[k]) for k, _ in COLS) + f" | {gate} |")
    md += ["", "对 PL-EM c1+b1+s2 的配对差（两种子均值；括号内为种子 1 / 种子 2；语音官方附按会议配对 bootstrap 的 P(差 ≤ 0)）：", "",
           "| 系统 | " + " | ".join(dict(COLS)[k] for k in VS) + " |", "|---" * (len(VS) + 1) + "|"]
    for label, v in res.items():
        if "vs_plem" not in v:
            continue
        cells = []
        for k in VS:
            d, d1, d2 = v["vs_plem"]["mean"][k], v["vs_plem"]["seed1"][k], v["vs_plem"]["seed2"][k]
            c = "—" if d is None else f"{d:+.3f} ({'—' if d1 is None else f'{d1:+.3f}'} / {'—' if d2 is None else f'{d2:+.3f}'})"
            if k == "speech_official" and v["gate_vs_plem"]:
                sp = v["gate_vs_plem"]["speech_all"]
                c += f"；配对 {sp['diff']:+.2f}，P(≤0)={sp['p_le0']:.3f}"
            cells.append(c)
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out + ".json").exists() or Path(a.out + ".md").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    res = build()
    md = markdown(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
