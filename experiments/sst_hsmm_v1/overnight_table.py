"""overnight 2026-11-27: one comparison table (5-cell means vs B_b, every metric) of the key systems and schemes S1-S8.
python3 experiments/sst_hsmm_v1/overnight_table.py <out.md>   (refuses to overwrite; reads eval_outputs/sst_hsmm_v1/<root>/summary.json)"""
import json, subprocess, sys
from pathlib import Path
SYS = [("LA-2", "p3x", "key"), ("p3hk（质量点）", "p3hk", "key"), ("p3hk2（低时延点）", "p3hk2", "key"), ("D3", "p3d3", "key"),
       ("S1 D3×0.5", "p3d3s05", "S1"), ("S2 忠实收尾", "p3d3s05ff", "S2"), ("S3 P2F+忠实收尾", "p3p2fxff", "S3"), ("S4 CMA", "p3d3cma", "S4"),
       ("S5 S1@θ0.75", "p3d3s05q", "S5"), ("S6 最后一读 MBR", "p3hkm", "S6"), ("S7 每读 MBR", "p3hkc", "S7"), ("S8 预判一致性证书", "p3aac", "S8")]
MET = ["asr", "text_bleu", "d2", "d1", "EndOffset", "NP_in", "PT_in", "NC", "sim", "pesq"]
HDR = ["ASR-BLEU", "文本 BLEU", "D2 LAAL", "D1", "EndOffset", "NP_in", "PT_in", "NC", "sim", "PESQ"]
out = Path(sys.argv[1])
if out.exists():
    sys.exit(f"refuse to overwrite {out}")
tmp = out.with_suffix(".json")
roots = [r for _, r, _ in SYS if (Path("eval_outputs/sst_hsmm_v1") / r / "summary.json").exists()]
subprocess.run(["python3", "paper/sst_hsmm_iclr_v1/scripts/collect.py", str(tmp), *roots], check=True, capture_output=True)
R = json.load(open(tmp))
L = []
for d in ("en2zh", "zh2en"):
    b = R["p3hk"]["NEW_vs_B_b"][d]; n = R["p3hk"]["N_vs_B_b"][d]          # B_b / N as in the paper (p3hk summary)
    L += [f"### {d}（相对 B_b 的 5 cell 平均；B_b 绝对值：ASR {b['asr']['ref']:.2f}，D2 {b['d2']['ref']:.3f} s）", "",
          "| 系统 | " + " | ".join(HDR) + " |", "|---|" + "---|" * len(HDR),
          "| N | " + " | ".join(f"{n[m]['diff']:+.3f}" for m in MET) + " |"]
    for name, r, _ in SYS:
        if r not in R:
            L.append(f"| {name} | " + " | ".join("—" for _ in MET) + " |"); continue
        a = R[r]["NEW_vs_B_b"][d]
        L.append(f"| {name} | " + " | ".join(f"{a[m]['diff']:+.3f}" if m in a else "—" for m in MET) + " |")
    L.append("")
out.write_text("\n".join(L), encoding="utf-8")
print(out.read_text())
