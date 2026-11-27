"""portfolio 2026-11-27 (change 55): final comparison table, 5-cell means vs B_b on every metric + COMET (pf_comet2), all 9 methods.
    python3 experiments/sst_hsmm_v1/pf_table.py <out.md>   (refuses to overwrite)"""
import json, subprocess, sys
from pathlib import Path
D0 = Path("eval_outputs/sst_hsmm_v1")
SYS = [("LA-2", "p3x", ""), ("p3hk（质量点）", "p3hk", ""), ("p3hk2（低时延点）", "p3hk2", ""),
       ("A1 偏置投机重写", "p3pfa1", "无效"), ("A2 BID 草稿选择", "p3pfa2", "无效"), ("B1 反事实音频证书", "p3pfb1", "无效"),
       ("B2 边界对齐读", "p3pfb2", "无效"), ("C1 分组 conformal θ", "p3pfc1", "无效"), ("C2 MAP 域校准", "p3pfc2", "无效"),
       ("D1 可提交性 MBR", "p3pfd1", "zh2en 有效"), ("D2 D1 + 断句候选", "p3pfd2", "无效"), ("E1 P2F-T 标签 + 匹配 rollout", "p3pfe1", "zh2en 有效")]
MET = ["asr", "text_bleu", "d2", "d1", "EndOffset", "NP_in", "PT_in", "NC", "sim", "pesq"]
HDR = ["ASR-BLEU", "文本 BLEU", "D2 LAAL", "D1", "EndOffset", "NP_in", "PT_in", "NC", "sim", "PESQ", "ASR-COMET", "文本 COMET", "判定"]
out = Path(sys.argv[1])
if out.exists():
    sys.exit(f"refuse to overwrite {out}")
tmp = out.with_suffix(".json")
subprocess.run(["python3", "paper/sst_hsmm_iclr_v1/scripts/collect.py", str(tmp), *[r for _, r, _ in SYS]], check=True, capture_output=True)
R = json.load(open(tmp))
C = json.load(open(D0 / "pf_comet2/comet.json"))
L = []
for d in ("en2zh", "zh2en"):
    b = R["p3hk"]["NEW_vs_B_b"][d]; n = R["p3hk"]["N_vs_B_b"][d]
    cb = C["B_b"][d]
    L += [f"### {d}（相对 B_b 的 5 cell 平均；B_b 绝对值：ASR-BLEU {b['asr']['ref']:.2f}，D2 {b['d2']['ref']:.3f} s，"
          f"ASR-COMET {cb['asr']:.4f}，文本 COMET {cb['text']:.4f}）", "",
          "| 系统 | " + " | ".join(HDR) + " |", "|---|" + "---|" * len(HDR)]
    cm = lambda s: f"{C[s][d]['asr_vs_B_b']:+.4f} | {C[s][d]['text_vs_B_b']:+.4f}"
    L.append("| N | " + " | ".join(f"{n[m]['diff']:+.3f}" for m in MET) + f" | {cm('N')} | |")
    for name, r, v in SYS:
        a = R[r]["NEW_vs_B_b"][d]
        L.append(f"| {name} | " + " | ".join(f"{a[m]['diff']:+.3f}" if m in a else "—" for m in MET) + f" | {cm(r)} | {v} |")
    L.append("")
out.write_text("\n".join(L), encoding="utf-8")
print(out.read_text())
