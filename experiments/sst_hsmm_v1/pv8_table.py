"""overnight 2026-11-27 rounds 1-8: copy of pv4_table.py (not modified) with every round's systems (EXTRA_SYS "name=p3tag;..." adds more);
COMET from pv8/comet. Was: change 57 (B1 / B2): copy of pv3_table.py; systems that have a test summary only, COMET from pv4/comet.
Was: plan v3 (change 56): final comparison table, 5-cell means vs B_b on every metric + COMET (pv3/comet), new methods next to the key
earlier systems; verdict per method computed with the change 55 rule (pf_verdict).
    python3 experiments/sst_hsmm_v1/pv3_table.py <out.md>   (refuses to overwrite)"""
import json, subprocess, sys
from pathlib import Path
D0 = Path("eval_outputs/sst_hsmm_v1")
SYS = [("LA-2", "p3x", 0), ("p3hk（质量点）", "p3hk", 0), ("p3hk2（低时延点）", "p3hk2", 0),
       ("S1 源未来 LoRA×0.5", "p3d3s05", 1), ("S3 P2F 预判", "p3p2fxff", 1), ("D1 可提交度 MBR", "p3pfd1", 1),
       ("E1 P2F-T", "p3pfe1", 1), ("H3 句法门控", "p3pfh3", 1), ("E1 + D1", "p3pfe1d1", 1),
       ("**B1 保留英文术语**", "p3pfn1", 1), ("**B2 只学预判**", "p3pff1", 1), ("**C1 完整输入 KL 锚定**", "p3pfi1", 1),
       ("**C2' HCG v2 视野对比引导**", "p3pfg2", 1), ("**SAA 按支持分配目标**", "p3pfs1", 1),
       ("**EGA 耗尽门控预判**", "p3pfq1", 1), ("**CCA 续说条件化预判**", "p3pfr1", 1)]
import os
SYS += [tuple(x.split("=", 1)) + (1,) for x in os.environ.get("EXTRA_SYS", "").split(";") if x]   # "name=p3tag;..."
SYS = [x for x in SYS if (D0 / x[1] / "summary.json").exists()]
MET = ["asr", "text_bleu", "d2", "d1", "EndOffset", "NP_in", "PT_in", "NC", "sim", "pesq"]
HDR = ["ASR-BLEU", "文本 BLEU", "D2 LAAL", "D1", "EndOffset", "NP_in", "PT_in", "NC", "sim", "PESQ", "ASR-COMET", "文本 COMET",
       "相对 p3hk ΔASR / ΔD2 / 连线", "判定"]


def verdict(R, s):
    ok, txt = {}, {}
    for dr in ("en2zh", "zh2en"):
        g = lambda n, k: R[n]["NEW_vs_B_b"][dr][k]["diff"]
        a, x = g(s, "asr"), g(s, "d2")
        ha, hx, ka, kx = g("p3hk", "asr"), g("p3hk", "d2"), g("p3hk2", "asr"), g("p3hk2", "d2")
        line = ha + (ka - ha) / (kx - hx) * (x - hx)
        ok[dr] = (a >= ha - 0.2 and x <= hx - 0.10) or a - line >= 0.2
        txt[dr] = f"{a - ha:+.2f} / {x - hx:+.3f} / {a - line:+.2f}"
    n = sum(ok.values())
    v = "有效" if n == 2 else (f"{[k for k, t in ok.items() if t][0]} 有效" if n == 1 else "无效")
    return txt, v


out = Path(sys.argv[1])
if out.exists():
    sys.exit(f"refuse to overwrite {out}")
tmp = out.with_suffix(".json")
subprocess.run(["python3", "paper/sst_hsmm_iclr_v1/scripts/collect.py", str(tmp), *[r for _, r, _ in SYS]], check=True, capture_output=True)
R = json.load(open(tmp))
C = json.load(open(D0 / "pv8/comet/comet.json"))
L = []
for d in ("en2zh", "zh2en"):
    b = R["p3hk"]["NEW_vs_B_b"][d]; n = R["p3hk"]["N_vs_B_b"][d]
    cb = C["B_b"][d]
    L += [f"### {d}（相对 B_b 的 5 cell 平均；B_b 绝对值：ASR-BLEU {b['asr']['ref']:.2f}，D2 {b['d2']['ref']:.3f} s，"
          f"ASR-COMET {cb['asr']:.4f}，文本 COMET {cb['text']:.4f}；连线 = 同 D2 处比 p3hk–p3hk2 连线高多少 BLEU）", "",
          "| 系统 | " + " | ".join(HDR) + " |", "|---|" + "---|" * len(HDR)]
    cm = lambda s: f"{C[s][d]['asr_vs_B_b']:+.4f} | {C[s][d]['text_vs_B_b']:+.4f}"
    L.append("| N | " + " | ".join(f"{n[m]['diff']:+.3f}" for m in MET) + f" | {cm('N')} | | |")
    for name, r, isv in SYS:
        a = R[r]["NEW_vs_B_b"][d]
        t, v = verdict(R, r) if isv else ({d: ""}, "")
        L.append(f"| {name} | " + " | ".join(f"{a[m]['diff']:+.3f}" if m in a else "—" for m in MET) + f" | {cm(r)} | {t[d]} | {v} |")
    L.append("")
out.write_text("\n".join(L), encoding="utf-8")
print(out.read_text())
