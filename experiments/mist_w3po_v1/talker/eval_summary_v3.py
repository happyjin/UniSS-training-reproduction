"""The dev table of talker_crf_prereg_v3 (E2', reports/mist_w3po_v1/talker_crf_prereg_v3.zh-CN.md §3-§4). CPU.

eval_summary_v2's table (A', PL-EM c1+b1+s2, E1 BoN-SFT, E2 single-channel PL-EM) plus E2' (single-channel PL-EM with
the corrected budget, eval dirs <A'>__plemm2_c1b1s2{,_tts2}): its gate vs A' (eval/plemm2/gate_c1b1s2.json), its
paired comparison vs PL-EM c1+b1+s2 (gate_vs_plem_c1b1s2.json + 2-seed mean differences), and -- descriptive -- vs E2
(gate_vs_plemm_c1b1s2.json + differences). Writes <out>.json and <out>.md (refuses to overwrite).

    python -m experiments.mist_w3po_v1.talker.eval_summary_v3 --out eval_outputs/mist_w3po_v1/talker_crf/eval/summary_v4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.talker import eval_summary as ES
from experiments.mist_w3po_v1.talker import eval_summary_v2 as V2

E2P = ("E2′ 单通道 PL-EM 修正预算 c1+b1+s2", "plemm2", "c1b1s2")
E2 = "实验二 单通道 PL-EM c1+b1+s2"
SYSTEMS = V2.SYSTEMS + [E2P]


def _gate(p, ab: bool = True):
    """eval_summary_v2's shapes: the gate vs A' without A / B, the paired comparisons with them."""
    g = ES.jl(p)
    return ({k: {"diff": v["diff"], "p_le0": v["p_le0"], "pass": v["pass"], **({"A": v["A"], "B": v["B"]} if ab else {})}
             for k, v in g["items"].items()} if g else None)


def build() -> dict:
    res = {}
    for label, scheme, tag in SYSTEMS:
        s1, s2 = ES.one(scheme, tag, ""), ES.one(scheme, tag, "_tts2")
        res[label] = {"scheme": scheme, "seed1": s1, "seed2": s2, "mean": ES.mean2(s1, s2),
                      "gate": _gate(ES.EV / scheme / f"gate_{tag}.json", ab=False) if scheme else None,
                      "gate_vs_plem": _gate(ES.EV / scheme / f"gate_vs_plem_{tag}.json")
                      if scheme in ("bon", "plemm", "plemm2") else None}
    pl, e2 = res["PL-EM c1+b1+s2"], res[E2]
    for label, v in res.items():
        if v["scheme"] in ("bon", "plemm", "plemm2"):
            v["vs_plem"] = {k: V2.diff(v[k], pl[k]) for k in ("mean", "seed1", "seed2")}
    v = res[E2P[0]]
    v["gate_vs_plemm"] = _gate(ES.EV / "plemm2" / "gate_vs_plemm_c1b1s2.json")
    v["vs_plemm"] = {k: V2.diff(v[k], e2[k]) for k in ("mean", "seed1", "seed2")}
    return res


def markdown(res: dict) -> list[str]:
    md = V2.markdown(res)
    v = res[E2P[0]]
    md += ["", "E2′ 对 E2（实验二）的配对差（描述，不判定；两种子均值，括号内为种子 1 / 种子 2）：", "",
           "| 系统 | " + " | ".join(dict(V2.COLS)[k] for k in V2.VS) + " |", "|---" * (len(V2.VS) + 1) + "|"]
    cells = []
    for k in V2.VS:
        d, d1, d2 = v["vs_plemm"]["mean"][k], v["vs_plemm"]["seed1"][k], v["vs_plemm"]["seed2"][k]
        c = "—" if d is None else f"{d:+.3f} ({'—' if d1 is None else f'{d1:+.3f}'} / {'—' if d2 is None else f'{d2:+.3f}'})"
        if k == "speech_official" and v["gate_vs_plemm"]:
            sp = v["gate_vs_plemm"]["speech_all"]
            c += f"；配对 {sp['diff']:+.2f}，P(≤0)={sp['p_le0']:.3f}"
        cells.append(c)
    md.append(f"| {E2P[0]} | " + " | ".join(cells) + " |")
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
