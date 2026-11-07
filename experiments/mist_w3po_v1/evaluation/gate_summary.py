"""One markdown table for gate reports (user 2026-11-07): per candidate decode dir, the formal gate (gate.json: text all /
ov / speech, pass), the target metrics that barely moved so far (ov tcpBLEU, <=0.5 s interjection recall 1spk/ov,
collapse, duplicates, window speaker mislabel, commit-before-window-end, ov latency) and the two low-noise direction
signals (low_noise_signals.json: dev STAMP R vs v8 ctx0 with CI/P, training-R slope). Reference row: v8 ctx0.

    python -m experiments.mist_w3po_v1.evaluation.gate_summary <decode dir> [<decode dir> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REF = Path("eval_outputs/mist_w3po_v1/decode/dev_v8_ctx0")


def load(p: Path, name: str):
    f = p / name
    return json.loads(f.read_text()) if f.exists() else None


def row(d: Path) -> str:
    t, s, g, sig = load(d, "score_text.json"), load(d, "score_asr.json"), load(d, "gate.json"), load(d, "low_noise_signals.json")
    if t is None:
        return f"| {d.name} | (no score) |"
    sr = t.get("short_recall", {})
    cells = [d.name, f"{t['tcpBLEU_all']:.2f}", f"{t['tcpBLEU_ov']:.2f}", f"{t['tcpBLEU_1spk']:.2f}",
             f"{s['tcpBLEU_all']:.2f}" if s else "—",
             f"{100 * sr.get('le0.5_1spk', {}).get('recall', float('nan')):.1f}/{100 * sr.get('le0.5_ov', {}).get('recall', float('nan')):.1f}",
             f"{t['collapse']:.3f}", f"{100 * t['duplicate_rate']:.2f}%", f"{t['pre_end_share']:.3f}",
             f"{t['lag_ov']['mean']:.2f}"]
    if sig:
        mis = sig.get("window_speaker_mislabel", {}).get("cand")
        cells.append(f"{100 * mis:.2f}%" if mis is not None else "—")
        dr = sig["dev_R"]
        cells.append(f"{dr['diff']:+.4f} [{dr['window_boot_ci95'][0]:+.4f}, {dr['window_boot_ci95'][1]:+.4f}] P≤0 {dr['window_boot_p_le0']:.2f}")
        tr = sig.get("train_R_trend")
        cells.append(f"{tr['slope_per_10_steps']:+.4f} [{tr['slope_ci95_per_10'][0]:+.4f}, {tr['slope_ci95_per_10'][1]:+.4f}]" if tr else "—")
    else:
        cells += ["—", "—", "—"]
    if g:
        it = g["items"]
        cells.append(" / ".join(f"{k}: {v['diff']:+.2f} (P≤0 {v['p_le0']:.2f}) {'✓' if v['pass'] else '✗'}" for k, v in it.items()))
        cells.append("**通过**" if g["pass"] else "**不通过**")
    else:
        cells += ["—", "—"]
    return "| " + " | ".join(cells) + " |"


def main():
    hdr = ("| 系统 | all | ov | 1spk | 语音 all | ≤0.5 s 召回 1spk/ov | collapse | 重复率 | 窗口末前提交 | ov 延迟均值 | "
           "窗口说话人错标 | dev R − v8 [95% CI] | 训练 R 斜率/10 步 [95% CI] | 闸门各项（Δ vs 参考） | 闸门 |")
    print(hdr)
    print("|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
    for d in [REF] + [Path(x) for x in sys.argv[1:]]:
        print(row(d))


if __name__ == "__main__":
    main()
