"""Fill the external baselines run with the paper's protocol into paper/bayes_simul_s2st_v12_baselines (Step 3 of
paper/baselines_run_plan_v1.md). Inputs:
  paper/baselines_results/seamless/<point>_<dir>.json      (R1: SeamlessStreaming, its own speech, one run a point)
  paper/baselines_results/hysimulmt/<point>_<dir>.json     (R2: Hy-SimulMT + our Talker, three talker seeds)
  eval_outputs/s2st_baselines_v1/qwen3omni_native_cvss/summary.json   (Qwen3-Omni native, CVSS-T)
Tables are rebuilt from the pre-baseline copies in eval_outputs/s2st_baselines_v1/paper_base_v12/ (the v12 files after
Step 1), so the script can be re-run as more points arrive. Tier rule = the paper's: per direction and budget, the point
with the highest ASR-BLEU whose segment LAAL (D1) fits. tab_realsi_main uses three-seed means (as its other columns);
tab_tiers / tab_tiers_main use talker seed 0 (their "default talker seed"). Bold = best ASR-BLEU in the row among the
systems we ran (our policies and these baselines); reported systems are never bold.

    /opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python experiments/s2st_baselines_v1/fill_paper_v12.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parents[2]
PAPER = R / "paper/bayes_simul_s2st_v12_baselines"
BASE = R / "eval_outputs/s2st_baselines_v1/paper_base_v12"
RES = R / "paper/baselines_results"
NATIVE = R / "eval_outputs/s2st_baselines_v1/qwen3omni_native_cvss/summary.json"
BUDGETS = {"en2zh": [1.18, 1.98, 2.46, 2.98, 3.48, 4.04], "zh2en": [1.18, 1.92, 2.40, 2.98, 3.62, 4.24]}
DIRS = ("en2zh", "zh2en")


# ------------------------------------------------------------------ points
def seamless_points():
    out = {d: [] for d in DIRS}
    for f in sorted(glob.glob(str(RES / "seamless/*_*.json"))):
        j = json.load(open(f, encoding="utf-8"))
        name = Path(f).stem.rsplit("_", 1)[0]
        d = j.get("config", {}).get("direction") or Path(f).stem.rsplit("_", 1)[1]
        if j.get("asr_bleu") is None or j.get("laal_d1") is None:
            continue
        c = j["config"]
        p = {"name": name, "setting": f"$\\tau{{=}}{c['decision_threshold']}$, $W{{=}}{c['min_starting_wait_w2vbert']}$, "
                                      f"$C{{=}}{c['min_unit_chunk_size']}$",
             "key": (c["min_starting_wait_w2vbert"], c["min_unit_chunk_size"], c["decision_threshold"]),
             "asr": j["asr_bleu"], "asr0": j["asr_bleu"], "text": j["text_bleu"], "d1": j["laal_d1"],
             "d1_0": j["laal_d1"], "d0w": j["laal_d0w"], "sd": None}
        out[d].append(p)
    for d in DIRS:
        out[d].sort(key=lambda p: (-p["key"][0], -p["key"][1], p["key"][2]))
    return out


def hy_points():
    out = {d: [] for d in DIRS}
    label = {"hy-std-w4": "standard (1.6\\,s chunks)", "hy-sc-w4": "short\\_chunk (1.6\\,s, then 0.8\\,s)"}
    for f in sorted(glob.glob(str(RES / "hysimulmt/hy-*_*.json"))):
        j = json.load(open(f, encoding="utf-8"))
        s0 = j["seeds"]["0"]
        out[j["direction"]].append({"name": j["point"], "setting": label.get(j["point"], j["point"]),
                                    "asr": j["asr_bleu"], "sd": j.get("asr_bleu_sd"), "asr0": s0["asr_bleu"],
                                    "text": j["text_bleu"], "d1": j["laal_d1"], "d1_0": s0["laal_d1"],
                                    "d0w": j["laal_d0w"]})
    for d in DIRS:
        out[d].sort(key=lambda p: p["d1"])
    return out


SYSTEMS = {"seam": seamless_points(), "hy": hy_points()}


def best(pts, budget, seed0):
    ok = [p for p in pts if (p["d1_0"] if seed0 else p["d1"]) <= budget + 1e-9]
    return max(ok, key=lambda p: p["asr0"] if seed0 else p["asr"]) if ok else None


# ------------------------------------------------------------------ table surgery
NUM = re.compile(r"^(?:\\textbf\{)?(-?[\d.]+)\}?")


def cell_value(c):
    m = NUM.match(c.strip())
    return float(m.group(1)) if m else None


def unbold(c):
    return re.sub(r"^\\textbf\{([\d.]+)\}", r"\1", c.strip())


def embolden(c):
    c = unbold(c)
    return re.sub(r"^([\d.]+)", r"\\textbf{\1}", c)


def rebold(cells, idx):
    vals = {i: cell_value(cells[i]) for i in idx}
    vals = {i: v for i, v in vals.items() if v is not None}
    for i in idx:
        cells[i] = unbold(cells[i])
    if vals:
        top = max(vals.values())
        for i, v in vals.items():
            if abs(v - top) < 1e-9:
                cells[i] = embolden(cells[i])
    return cells


def split_row(line):
    body = line.rstrip()
    assert body.endswith("\\\\"), line
    return [c.strip() for c in body[:-2].split(" & ")]


def join_row(cells):
    return " & ".join(cells) + " \\\\"


def fmt_tiers(p, seed0, text=True):
    if p is None:
        return "--"
    a, d1 = (p["asr0"], p["d1_0"]) if seed0 else (p["asr"], p["d1"])
    t = f"/{p['text']:.1f}" if text else ""
    return f"{a:.2f}{t}\\,{{\\scriptsize({d1:.2f})}}"


def section_of(line, cur):
    if "multicolumn" in line or line.startswith("En$") or line.startswith("Zh$"):
        if "En$\\rightarrow$Zh" in line and "ACL" not in line:
            return "en2zh"
        if "Zh$\\rightarrow$En" in line:
            return "zh2en"
        if "ACL" in line:
            return "acl"
    return cur


def fill_tab_tiers():
    s = (BASE / "tab_tiers.tex").read_text(encoding="utf-8")
    s = s.replace(r"\begin{tabular}{llccccccc}", r"\begin{tabular}{llccccccccc}", 1)
    s = s.replace(r"LiveInterpret~2.0$^\S$ & LA2", r"LiveInterpret~2.0$^\S$ & SeamlessStreaming$^\P$ & Hy-SimulMT$^\P$ & LA2", 1)
    s = s.replace(r"\multicolumn{9}{l}", r"\multicolumn{11}{l}")
    out, sec = [], None
    for line in s.split("\n"):
        sec = section_of(line, sec)
        m = re.match(r"^m(\d) & ", line)
        if m and sec:
            cells = split_row(line)
            k = int(m.group(1)) - 1
            if sec == "acl":
                new = ["n/a", "n/a"]
                cells = cells[:4] + new + cells[4:]
            else:
                new = [fmt_tiers(best(SYSTEMS[x][sec], BUDGETS[sec][k], True), True) for x in ("seam", "hy")]
                cells = cells[:4] + new + cells[4:]
                cells = rebold(cells, range(4, 10))
            line = join_row(cells)
        out.append(line)
    s = "\n".join(out)
    old = r"""All policies share the Thinker (Qwen3-Omni) and the Talker;"""
    new = r"""All policies share the Thinker (Qwen3-Omni) and the Talker; $^\P$external systems we ran with our protocol, inference
only (Appendix~\ref{app:baselines}, Table~\ref{tab:baselines}); bold: best ASR-BLEU (ACL: BLEU) in the row among the
systems we ran;"""
    assert s.count(old) == 1
    s = s.replace(old, new)
    (PAPER / "generated/tab_tiers.tex").write_text(s, encoding="utf-8")


def fill_tab_tiers_main():
    s = (BASE / "tab_tiers_main.tex").read_text(encoding="utf-8")
    s = s.replace(r"\begin{tabular}{lccccc}", r"\begin{tabular}{lccccccc}", 1)
    s = s.replace(r"Omni$^\dagger$ & LI$^\S$ & LA2", r"Omni$^\dagger$ & LI$^\S$ & SS$^\P$ & Hy$^\P$ & LA2", 1)
    s = s.replace(r"\multicolumn{6}{l}", r"\multicolumn{8}{l}")
    out, sec = [], None
    for line in s.split("\n"):
        sec = section_of(line, sec)
        m = re.match(r"^m(\d) \(", line)
        if m and sec:
            cells = split_row(line)
            k = int(m.group(1)) - 1
            if sec == "acl":
                cells = cells[:3] + ["n/a", "n/a"] + cells[3:]
            else:
                new = [fmt_tiers(best(SYSTEMS[x][sec], BUDGETS[sec][k], True), True, text=False) for x in ("seam", "hy")]
                cells = cells[:3] + new + cells[3:]
                cells = rebold(cells, range(3, 7))
            line = join_row(cells)
        out.append(line)
    s = "\n".join(out)
    old = r"""Text BLEU,"""
    new = r"""$^\P$SeamlessStreaming (SS) and Hy-SimulMT with our Talker (Hy), run by us with our protocol, inference only
(Table~\ref{tab:baselines}); bold: best in the row among the systems we ran. Text BLEU,"""
    assert s.count(old) == 1
    s = s.replace(old, new)
    (PAPER / "generated/tab_tiers_main.tex").write_text(s, encoding="utf-8")


def fill_tab_realsi_main():
    s = (BASE / "tab_realsi_main.tex").read_text(encoding="utf-8")
    s = s.replace(r"\begin{tabular}{llcccccccc}", r"\begin{tabular}{llcccccccccc}", 1)
    s = s.replace(r"SimulS2ST-Omni$^\dagger$ & LA2", r"SimulS2ST-Omni$^\dagger$ & SeamlessStr.$^\P$ & Hy-SimulMT$^\P$ & LA2", 1)
    out, sec = [], None
    for line in s.split("\n"):
        if line.startswith("En$\\rightarrow$Zh &"):
            sec = "en2zh"
        elif line.startswith("Zh$\\rightarrow$En &"):
            sec = "zh2en"
        m = re.match(r"^(?:En\$\\rightarrow\$Zh|Zh\$\\rightarrow\$En|) ?& m(\d) & ", line)
        if m and sec:
            cells = split_row(line)
            k = int(m.group(1)) - 1
            new = [fmt_tiers(best(SYSTEMS[x][sec], BUDGETS[sec][k], False), False, text=False) for x in ("seam", "hy")]
            cells = cells[:4] + new + cells[4:]
            cells = rebold(cells, range(4, len(cells)))
            line = join_row(cells)
        out.append(line)
    s = "\n".join(out)
    old = r"""LiveInterpret~2.0 (29.09/22.19 at its own LAAL of 3.12/3.79\,s; Table~\ref{tab:tiers}) fits no tier up to 3\,s.}"""
    new = r"""LiveInterpret~2.0 (29.09/22.19 at its own LAAL of 3.12/3.79\,s; Table~\ref{tab:tiers}) fits no tier up to 3\,s.
$^\P$Run by us with our protocol, inference only: SeamlessStreaming (its official agent and voice, single run) and
Hy-SimulMT speaking through our Talker (mean of three talker seeds); all their points in Table~\ref{tab:baselines}.
Bold: best in the row among the systems we ran.}"""
    assert s.count(old) == 1
    s = s.replace(old, new)
    (PAPER / "generated/tab_realsi_main.tex").write_text(s, encoding="utf-8")


# QABC's tier points (tab_realsi_main) with D0w, from eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112
# (reports/mist_w3po_v1/laal_definitions_compare_v1.zh-CN.md, section 4): (tier, ASR-BLEU, D1, D0w)
QABC_REF = {"en2zh": [("m1", 25.33, 1.13, 1.00), ("m2", 28.26, 1.77, 1.67), ("m3", 30.33, 2.25, 2.17), ("m4", 30.51, 2.75, 2.67)],
            "zh2en": [("m1", 17.83, 1.14, 1.04), ("m2", 21.42, 1.70, 1.61), ("m3", 22.46, 2.18, 2.08), ("m4", 22.65, 2.67, 2.57)]}


def fill_tab_baselines():
    rows = []
    for d, dn in (("en2zh", r"En$\rightarrow$Zh"), ("zh2en", r"Zh$\rightarrow$En")):
        rows.append(r"\midrule" if rows else "")
        rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{RealSI {dn}}}}} \\")
        for sysname, key in (("SeamlessStreaming", "seam"), ("Hy-SimulMT + Talker", "hy")):
            pts = SYSTEMS[key][d]
            for i, p in enumerate(pts):
                sd = f"$\\pm${p['sd']:.2f}" if p.get("sd") is not None else ""
                rows.append(f"{sysname if i == 0 else ''} & {p['setting']} & {p['asr']:.2f}{sd} & {p['text']:.2f} & "
                            f"{p['d1']:.2f} & {p['d0w']:.2f} \\\\")
            if not pts:
                rows.append(f"{sysname} & (pending) & & & & \\\\")
        for t, a, d1, dw in QABC_REF[d]:
            rows.append(f"{'QABC' if t == 'm1' else ''} & tier {t} point & {a:.2f} & -- & {d1:.2f} & {dw:.2f} \\\\")
    body = "\n".join(r for r in rows if r)
    tex = r"""\begin{table}[!htbp]
\centering
\small
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
System & Setting & ASR-BLEU & Text & D1 & D0w \\
""" + body + r"""
\bottomrule
\end{tabular}}
\caption{External systems run by us on RealSI (sentence level) with the paper's protocol, inference only, every operating
point of the grids registered before the runs. ASR-BLEU: speech only, Whisper-large-v3/Paraformer, sacreBLEU (Hy-SimulMT:
mean $\pm$ s.d.\ over three talker seeds); Text: BLEU of the committed text; D1: segment LAAL as in all our tables (s);
D0w: SimulEval word LAAL, units timed at emission, no queue (s). QABC rows: its tier points for reference.}
\label{tab:baselines}
\end{table}
"""
    (PAPER / "generated/tab_baselines.tex").write_text(tex, encoding="utf-8")


def fill_tab_cvss():
    s = (BASE / "tab_cvss.tex").read_text(encoding="utf-8")
    if not NATIVE.exists():
        (PAPER / "generated/tab_cvss.tex").write_text(s, encoding="utf-8")
        return
    n = json.load(open(NATIVE, encoding="utf-8"))
    e, z = n["en2zh"], n["zh2en"]
    row = (rf" & Qwen3-Omni (own Talker)$^\P$ & 30B-A3B & {e['asr_bleu']:.2f} & {z['asr_bleu']:.2f} & {e['text_bleu']:.2f} & "
           rf"{z['text_bleu']:.2f} & {e['apcp']:.2f} & {z['apcp']:.2f} & ({e['sim_o_baseplus']:.2f})$^\S$ & "
           rf"({z['sim_o_baseplus']:.2f})$^\S$ \\")
    old = r" & Step-Audio 2 mini & 7B & 32.81 & 25.35 & -- & -- & -- & -- & -- & -- \\"
    assert s.count(old) == 1
    s = s.replace(old, old + "\n" + row)
    oldc = r"""``--'': not reported. Ours:"""
    newc = r"""``--'': not reported. $^\P$Run by us with our protocol: Qwen3-Omni-30B-A3B-Instruct with its own Talker and
code2wav (vLLM-Omni, greedy Thinker, official talker sampling, preset voice; inference only). Ours:"""
    assert s.count(oldc) == 1
    s = s.replace(oldc, newc)
    (PAPER / "generated/tab_cvss.tex").write_text(s, encoding="utf-8")


def fill_curves():
    s = (BASE / "curves.tex").read_text(encoding="utf-8")
    s = s.replace(r"""  li/.style={black, only marks, mark=pentagon*, mark size=2.2pt},
}""", r"""  li/.style={black, only marks, mark=pentagon*, mark size=2.2pt},
  hy/.style={violet, mark=diamond, mark size=2.2pt, thick, dashdotted},
  ss/.style={brown, mark=x, mark size=2.2pt, thick, loosely dashed},
}""")
    ylim = {"en2zh": (18, 32), "zh2en": (13, 25.5)}
    for d in DIRS:
        add = []
        for key, nm, sty in (("hy", "HySimulMT", "hy"), ("seam", "SeamlessStreaming", "ss")):
            pts = sorted(SYSTEMS[key][d], key=lambda p: p["d1"])
            if not pts:
                continue
            env, top = [], -1.0          # upper envelope, as the ACL60/60 curves
            for p in pts:
                if p["asr"] > top:
                    env.append(p)
                    top = p["asr"]
            (PAPER / f"data/realsi_{d}_{nm}.dat").write_text(
                "laal bleu\n" + "".join(f"{p['d1']:.3f} {p['asr']:.2f}\n" for p in env), encoding="utf-8")
            add.append(f"\\addplot[{sty}] table[x=laal,y=bleu] {{data/realsi_{d}_{nm}.dat}};")
            off = [p for p in pts if not (ylim[d][0] <= p["asr"] <= ylim[d][1] and p["d1"] <= 3.2)]
            if off:
                a, b = min(p["asr"] for p in pts), max(p["asr"] for p in pts)
                x0, x1 = min(p["d1"] for p in pts), max(p["d1"] for p in pts)
                name = "SeamlessStreaming" if key == "seam" else "Hy-SimulMT"
                add.append(f"\\node[anchor=north west, font=\\tiny, inner sep=1.5pt, align=left] at (axis cs:1.0,{ylim[d][1]}) "
                           f"{{{name} (below axis):\\\\{a:.1f}--{b:.1f} at {x0:.2f}--{x1:.2f}\\,s}};")
        tag = f"\\addplot[omni] table[x=laal,y=bleu] {{data/realsi_{d}_Omni.dat}};"
        assert s.count(tag) == 1
        s = s.replace(tag, tag + "".join("\n" + a for a in add), 1)
    s = s.replace(r"""\addplot[li] coordinates {(10,0)}; \addlegendentry{LiveInterpret~2.0}""",
                  r"""\addplot[li] coordinates {(10,0)}; \addlegendentry{LiveInterpret~2.0}
\addplot[hy] coordinates {(10,0)}; \addlegendentry{Hy-SimulMT + Talker}
\addplot[ss] coordinates {(10,0)}; \addlegendentry{SeamlessStreaming}""")
    s = s.replace("legend columns=8,", "legend columns=5,")
    old = r"""LiveInterpret~2.0: ASR-BLEU reported by \citet{he2026simuls2st}, placed at the LAAL of its own paper
(own definition and test sets; Table~\ref{tab:tiers})."""
    assert s.count(old) == 1
    s = s.replace(old, old[:-1] + r"""; Hy-SimulMT (through our Talker, three talker seeds) and SeamlessStreaming: run by
us with our protocol, upper envelopes of their registered points (Table~\ref{tab:baselines}).""")
    (PAPER / "figures/curves.tex").write_text(s, encoding="utf-8")


def main():
    for k, v in SYSTEMS.items():
        print(k, {d: [(p["name"], p["asr"], round(p["d1"], 3)) for p in v[d]] for d in DIRS})
    fill_tab_tiers()
    fill_tab_tiers_main()
    fill_tab_realsi_main()
    fill_tab_baselines()
    fill_tab_cvss()
    fill_curves()
    print("filled", PAPER)


if __name__ == "__main__":
    main()
