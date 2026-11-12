"""Fill three more external results into paper/bayes_simul_s2st_v13_baselines (a copy of v12; v12 is not touched):
  paper/baselines_results/xtranslator/<point>_<dir>.json          (R4: X-Translator, its own TTS, one run a point)
  paper/baselines_results/hysimulmt_qwen3tts/<point>_<dir>.json   (R3: Hy-SimulMT blocks + Qwen3-TTS, three seeds)
  paper/baselines_results/qabc_omnitalker/summary.json            (R5: QABC blocks + Qwen3-Omni native Talker)
Inputs are the v12 files as they were when v13 was cut, snapshotted in eval_outputs/s2st_baselines_v1/paper_base_v13/
(generated/, sections/, references.bib), so the script can be re-run. Conventions as fill_paper_v12.py: tier rule = the
paper's (per direction and budget, the highest ASR-BLEU whose segment LAAL D1 fits); tab_realsi_main uses three-seed
means, tab_tiers / tab_tiers_main talker seed 0; bold = best ASR-BLEU in the row among the systems we ran; ASR-BLEU and
LAAL with two decimals, text BLEU with one decimal in tab_tiers and two in tab_baselines. tab_tiers_main (column width)
gets X-Translator only; the Qwen3-TTS reference cascade is in tab_tiers, tab_realsi_main and tab_baselines.

    /opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python experiments/s2st_baselines_v1/fill_paper_v13.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parents[2]
PAPER = R / "paper/bayes_simul_s2st_v13_baselines"
BASE = R / "eval_outputs/s2st_baselines_v1/paper_base_v13"
RES = R / "paper/baselines_results"
BUDGETS = {"en2zh": [1.18, 1.98, 2.46, 2.98, 3.48, 4.04], "zh2en": [1.18, 1.92, 2.40, 2.98, 3.62, 4.24]}
DIRS = ("en2zh", "zh2en")
assert PAPER.name.endswith("v13_baselines")

XT_LABEL = {"xt-default": "release default", "xt-c1280": "request chunk 1280 (0.08\\,s)",
            "xt-c5120": "request chunk 5120 (0.32\\,s)", "xt-a1.6": "ASR chunk 1.6\\,s"}
XT_ORDER = ["xt-default", "xt-c1280", "xt-c5120", "xt-a1.6"]
HY_LABEL = {"hy-std-w4": "standard (1.6\\,s chunks)", "hy-sc-w4": "short\\_chunk (1.6\\,s, then 0.8\\,s)"}


# ------------------------------------------------------------------ points
def xt_points():
    out = {d: [] for d in DIRS}
    for f in sorted(glob.glob(str(RES / "xtranslator/xt-*_*.json"))):
        j = json.load(open(f, encoding="utf-8"))
        assert j["asr_bleu"] is not None and j["laal_d1"] is not None, f
        out[j["direction"]].append({"name": j["point"], "setting": XT_LABEL[j["point"]], "asr": j["asr_bleu"],
                                    "asr0": j["asr_bleu"], "sd": None, "text": j["text_bleu"], "d1": j["laal_d1"],
                                    "d1_0": j["laal_d1"], "d0w": j["laal_d0w"]})
    for d in DIRS:
        out[d].sort(key=lambda p: XT_ORDER.index(p["name"]))
    return out


def hyq_points():
    out = {d: [] for d in DIRS}
    for f in sorted(glob.glob(str(RES / "hysimulmt_qwen3tts/hy-*_*.json"))):
        j = json.load(open(f, encoding="utf-8"))
        s0 = j["seeds"]["0"]
        out[j["direction"]].append({"name": j["point"], "setting": HY_LABEL[j["point"]], "asr": j["asr_bleu"],
                                    "sd": j["asr_bleu_sd"], "asr0": s0["asr_bleu"], "text": j["text_bleu"],
                                    "d1": j["laal_d1"], "d1_0": s0["laal_d1"], "d0w": j["laal_d0w"],
                                    "cap": j["n_hit_cap_3seeds"],
                                    "blocks": sum(s["n_tts_blocks"] for s in j["seeds"].values())})
    for d in DIRS:
        out[d].sort(key=lambda p: p["d1"])
    return out


SYSTEMS = {"xt": xt_points(), "hyq": hyq_points()}
OMNI = json.load(open(RES / "qabc_omnitalker/summary.json", encoding="utf-8"))["points"]


def best(pts, budget, seed0):
    ok = [p for p in pts if (p["d1_0"] if seed0 else p["d1"]) <= budget + 1e-9]
    return max(ok, key=lambda p: p["asr0"] if seed0 else p["asr"]) if ok else None


# ------------------------------------------------------------------ table surgery (as fill_paper_v12.py)
NUM = re.compile(r"^(?:\\textbf\{)?(-?[\d.]+)\}?")


def cell_value(c):
    m = NUM.match(c.strip())
    return float(m.group(1)) if m else None


def unbold(c):
    return re.sub(r"^\\textbf\{([\d.]+)\}", r"\1", c.strip())


def embolden(c):
    return re.sub(r"^([\d.]+)", r"\\textbf{\1}", unbold(c))


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


def replace_once(s, old, new):
    assert s.count(old) == 1, old[:80]
    return s.replace(old, new)


def fill_tab_tiers():
    s = (BASE / "generated/tab_tiers.tex").read_text(encoding="utf-8")
    s = replace_once(s, r"\begin{tabular}{llccccccccc}", r"\begin{tabular}{llccccccccccc}")
    s = replace_once(s, r"Hy-SimulMT$^\P$ & LA2", r"Hy-SimulMT$^\P$ & X-Translator$^\P$ & Hy+Qwen3-TTS$^\P$ & LA2")
    s = s.replace(r"\multicolumn{11}{l}", r"\multicolumn{13}{l}")
    out, sec = [], None
    for line in s.split("\n"):
        sec = section_of(line, sec)
        m = re.match(r"^m(\d) & ", line)
        if m and sec:
            cells = split_row(line)
            assert len(cells) == 11, line
            k = int(m.group(1)) - 1
            if sec == "acl":
                cells = cells[:6] + ["n/a", "n/a"] + cells[6:]
            else:
                new = [fmt_tiers(best(SYSTEMS[x][sec], BUDGETS[sec][k], True), True) for x in ("xt", "hyq")]
                cells = rebold(cells[:6] + new + cells[6:], range(4, 12))
            line = join_row(cells)
        out.append(line)
    s = "\n".join(out)
    s = replace_once(s, r"""only (Appendix~\ref{app:baselines}, Table~\ref{tab:baselines}); bold:""",
                     r"""only (Appendix~\ref{app:baselines}, Table~\ref{tab:baselines}): X-Translator speaks through its own TTS
(single run), Hy+Qwen3-TTS is the reference cascade of Hy-SimulMT's blocks spoken by Qwen3-TTS; bold:""")
    (PAPER / "generated/tab_tiers.tex").write_text(s, encoding="utf-8")


def fill_tab_tiers_main():
    s = (BASE / "generated/tab_tiers_main.tex").read_text(encoding="utf-8")
    s = replace_once(s, r"\begin{tabular}{lccccccc}", r"\begin{tabular}{lcccccccc}")
    s = replace_once(s, r"SS$^\P$ & Hy$^\P$ & LA2", r"SS$^\P$ & Hy$^\P$ & XT$^\P$ & LA2")
    s = s.replace(r"\multicolumn{8}{l}", r"\multicolumn{9}{l}")
    out, sec = [], None
    for line in s.split("\n"):
        sec = section_of(line, sec)
        m = re.match(r"^m(\d) \(", line)
        if m and sec:
            cells = split_row(line)
            assert len(cells) == 8, line
            k = int(m.group(1)) - 1
            if sec == "acl":
                cells = cells[:5] + ["n/a"] + cells[5:]
            else:
                new = [fmt_tiers(best(SYSTEMS["xt"][sec], BUDGETS[sec][k], True), True, text=False)]
                cells = rebold(cells[:5] + new + cells[5:], range(3, 8))
            line = join_row(cells)
        out.append(line)
    s = "\n".join(out)
    s = replace_once(s, r"""$^\P$SeamlessStreaming (SS) and Hy-SimulMT with our Talker (Hy), run by us""",
                     r"""$^\P$SeamlessStreaming (SS), Hy-SimulMT with our Talker (Hy) and X-Translator (XT), run by us""")
    (PAPER / "generated/tab_tiers_main.tex").write_text(s, encoding="utf-8")


def fill_tab_realsi_main():
    s = (BASE / "generated/tab_realsi_main.tex").read_text(encoding="utf-8")
    s = replace_once(s, r"\begin{tabular}{llcccccccccc}", r"\begin{tabular}{llcccccccccccc}")
    s = replace_once(s, r"Hy-SimulMT$^\P$ & LA2", r"Hy-SimulMT$^\P$ & X-Translator$^\P$ & Hy+Qwen3-TTS$^\P$ & LA2")
    out, sec = [], None
    for line in s.split("\n"):
        if line.startswith("En$\\rightarrow$Zh &"):
            sec = "en2zh"
        elif line.startswith("Zh$\\rightarrow$En &"):
            sec = "zh2en"
        m = re.match(r"^(?:En\$\\rightarrow\$Zh|Zh\$\\rightarrow\$En|) ?& m(\d) & ", line)
        if m and sec:
            cells = split_row(line)
            assert len(cells) == 12, line
            k = int(m.group(1)) - 1
            new = [fmt_tiers(best(SYSTEMS[x][sec], BUDGETS[sec][k], False), False, text=False) for x in ("xt", "hyq")]
            cells = rebold(cells[:6] + new + cells[6:], range(4, 14))
            line = join_row(cells)
        out.append(line)
    s = "\n".join(out)
    s = replace_once(s, r"""Hy-SimulMT speaking through our Talker (mean of three talker seeds); all their points in Table~\ref{tab:baselines}.""",
                     r"""Hy-SimulMT speaking through our Talker (mean of three talker seeds), X-Translator (its released stack and
own TTS, single run) and, as a reference cascade, Hy-SimulMT's blocks spoken by Qwen3-TTS (Hy+Qwen3-TTS, mean of three
seeds); all their points in Table~\ref{tab:baselines}.""")
    (PAPER / "generated/tab_realsi_main.tex").write_text(s, encoding="utf-8")


def fill_tab_baselines():
    s = (BASE / "generated/tab_baselines.tex").read_text(encoding="utf-8")
    for d in DIRS:
        rows = []
        for sysname, key in (("X-Translator", "xt"), ("Hy-SimulMT + Qwen3-TTS", "hyq")):
            for i, p in enumerate(SYSTEMS[key][d]):
                sd = f"$\\pm${p['sd']:.2f}" if p.get("sd") is not None else ""
                rows.append(f"{sysname if i == 0 else ''} & {p['setting']} & {p['asr']:.2f}{sd} & {p['text']:.2f} & "
                            f"{p['d1']:.2f} & {p['d0w']:.2f} \\\\")
        # insert before this direction's QABC reference rows (the first "QABC & tier m1 point" after its heading)
        head = r"\multicolumn{6}{l}{\emph{RealSI " + (r"En$\rightarrow$Zh" if d == "en2zh" else r"Zh$\rightarrow$En") + "}} \\\\"
        i0 = s.index(head)
        j0 = s.index("QABC & tier m1 point", i0)
        s = s[:j0] + "\n".join(rows) + "\n" + s[j0:]
    s = replace_once(s, r"""sacreBLEU (Hy-SimulMT:
mean $\pm$ s.d.\ over three talker seeds);""", r"""sacreBLEU (Hy-SimulMT rows:
mean $\pm$ s.d.\ over three talker seeds; X-Translator: one run, its own TTS; Qwen3-TTS: the same Hy-SimulMT blocks
spoken by Qwen3-TTS instead of our Talker, a reference cascade);""")
    (PAPER / "generated/tab_baselines.tex").write_text(s, encoding="utf-8")


def omni_rows():
    rows = []
    for d in DIRS:
        for t in range(1, 5):
            p, o = OMNI[f"{d}_m{t}"]["paper"], OMNI[f"{d}_m{t}"]["omni"]
            assert p["laal_d0w"] == o["laal_d0w"] and p["text_bleu"] == o["text_bleu"]
            rows.append({"dir": d, "tier": f"m{t}", "p": p, "o": o, "delta": p["asr_bleu"] - o["asr_bleu"]})
    return rows


def fill_tab_omnitalker():
    lines = []
    for d, dn in (("en2zh", r"En$\rightarrow$Zh"), ("zh2en", r"Zh$\rightarrow$En")):
        lines.append(r"\midrule")
        unit = "CER" if d == "en2zh" else "WER"
        lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{RealSI {dn}}} (talker {unit})}} \\")
        for r in (x for x in omni_rows() if x["dir"] == d):
            p, o = r["p"], r["o"]
            a, b = f"{p['asr_bleu']:.2f}", f"{o['asr_bleu']:.2f}"
            a = rf"\textbf{{{a}}}" if p["asr_bleu"] >= o["asr_bleu"] else a
            b = rf"\textbf{{{b}}}" if o["asr_bleu"] > p["asr_bleu"] else b
            lines.append(f"{r['tier']} & {a}$\\pm${p['asr_bleu_sd']:.2f} & {b}$\\pm${o['asr_bleu_sd']:.2f} & "
                         f"{r['delta']:+.2f} & {p['laal_d1']:.2f} & {o['laal_d1']:.2f} & {p['talker_wer']:.2f} & "
                         f"{o['talker_wer']:.2f} \\\\")
    ex = [x["o"]["thinker_exact_share_s0"] for x in omni_rows()]
    tex = r"""\begin{table}[!htbp]
\centering
\small
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{2}{c}{ASR-BLEU} & & \multicolumn{2}{c}{D1 (s)} & \multicolumn{2}{c}{Talker WER/CER} \\
Tier & Ours & Omni & $\Delta$ & Ours & Omni & Ours & Omni \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\caption{Speech generator ablation on RealSI (sentence level): \qabc's committed blocks at its four tier points of
Table~\ref{tab:realsi-main}, spoken by our Talker (Ours) or by Qwen3-Omni-30B-A3B-Instruct's native Thinker, Talker and
code2wav asked to read each block verbatim (Omni; preset voice, official talker sampling, run by us, inference only).
Mean $\pm$ s.d.\ over three seeds; $\Delta$: Ours minus Omni; D1: segment LAAL as in all our tables (D0w is the same for
both, as are the blocks); talker WER/CER: per sentence, the speech-only ASR transcript against the committed text (\%). Omni's Thinker
reproduces the block text exactly for """ + f"{100 * min(ex):.1f}--{100 * max(ex):.1f}" + r"""\% of blocks (seed 0).}
\label{tab:omnitalker}
\end{table}
"""
    (PAPER / "generated/tab_omnitalker.tex").write_text(tex, encoding="utf-8")


def numbers_for_text():
    """Ranges quoted in the prose, from the same sources."""
    rows = omni_rows()
    deltas = [r["delta"] for r in rows]
    wer_better = sum(r["p"]["talker_wer"] < r["o"]["talker_wer"] for r in rows)
    d1_lower = sum(round(r["p"]["laal_d1"], 2) < round(r["o"]["laal_d1"], 2) for r in rows)
    d1_equal = sum(round(r["p"]["laal_d1"], 2) == round(r["o"]["laal_d1"], 2) for r in rows)
    assert min(deltas) > 0 and len(rows) == 8
    # QABC's tier column of tab_realsi_main (= the paper-talker side of the R5 summary, reproduces_paper) against
    # X-Translator's tier points (D1 rule), as displayed
    qabc = {d: [round(OMNI[f"{d}_m{t}"]["paper"]["asr_bleu"], 2) for t in range(1, 5)] for d in DIRS}
    assert all(OMNI[k]["paper"]["reproduces_paper"] for k in OMNI)
    gaps, fits = [], {}
    for d in DIRS:
        fits[d] = []
        for k in range(4):
            p = best(SYSTEMS["xt"][d], BUDGETS[d][k], False)
            if p:
                fits[d].append(k + 1)
                gaps.append(qabc[d][k] - round(p["asr"], 2))
    caps = [p["cap"] / p["blocks"] for d in DIRS for p in SYSTEMS["hyq"][d]]
    return {"delta": (min(deltas), max(deltas)), "wer_better": wer_better, "d1_lower": d1_lower, "d1_equal": d1_equal,
            "xt_gap": (min(gaps), max(gaps)), "xt_fits": fits, "cap": (min(caps), max(caps))}


def fill_text():
    n = numbers_for_text()
    assert n["xt_fits"] == {"en2zh": [4], "zh2en": [3, 4]}, n["xt_fits"]
    # setup.tex: one sentence per baseline
    s = (BASE / "sections/setup.tex").read_text(encoding="utf-8")
    s = replace_once(s, r"""SeamlessStreaming~\citep{seamless2023} and Hy-SimulMT~\citep{hy2026simulmt}, its text spoken by our Talker, on RealSI,
and Qwen3-Omni with its own Talker on CVSS-T.""", r"""SeamlessStreaming~\citep{seamless2023}, X-Translator~\citep{zhao2026xtranslator} with its own TTS, and
Hy-SimulMT~\citep{hy2026simulmt}, its text spoken by our Talker (and, as a reference cascade, by
Qwen3-TTS~\citep{qwen2026qwen3tts}), on RealSI; Qwen3-Omni with its own Talker on CVSS-T; and, as a talker ablation,
Qwen3-Omni's native Talker reading \qabc's committed blocks on RealSI.""")
    (PAPER / "sections/setup.tex").write_text(s, encoding="utf-8")
    # results.tex: one sentence for X-Translator, one for the talker ablation
    s = (BASE / "sections/results.tex").read_text(encoding="utf-8")
    g0, g1 = n["xt_gap"]
    d0, d1 = n["delta"]
    s = replace_once(s, r"""it by 0.46 at Zh$\rightarrow$En m2; SeamlessStreaming stays near 13--15.""",
                     rf"""it by 0.46 at Zh$\rightarrow$En m2; SeamlessStreaming stays near 13--15; X-Translator fits only
En$\rightarrow$Zh m4 and Zh$\rightarrow$En m3--m4, {g0:.1f}--{g1:.1f} below \qabc. On \qabc's own committed blocks our
Talker is above Qwen3-Omni's native Talker at all eight tier points (+{d0:.1f}--{d1:.1f} ASR-BLEU;
Table~\ref{{tab:omnitalker}}).""")
    (PAPER / "sections/results.tex").write_text(s, encoding="utf-8")
    # appendix C: setup of the three new runs + the ablation table
    s = (BASE / "sections/appendix.tex").read_text(encoding="utf-8")
    s = replace_once(s, r"""We ran three external systems with released weights and their own inference code, inference only (no training or
adaptation); their settings were registered before the runs.""", r"""We ran external systems with released weights and their own inference code, inference only (no training or
adaptation); their settings were registered before the runs.""")
    c0, c1 = n["cap"]
    s = replace_once(s, r"""wavlm-base-plus-sv scale.

\input{generated/tab_baselines}
""", rf"""wavlm-base-plus-sv scale. \emph{{X-Translator}}~\citep{{zhao2026xtranslator}} runs its released stack (Qwen3-ASR-1.7B,
LMT-60-8B and its own voice-cloning TTS, X-Voice 0.4B) with its server's own segmentation and commit logic on a simulated
source clock, one run per point: the release default (main point), the two other request chunk sizes its configuration
exposes, and the 1.6\,s ASR chunk of its paper. Its speech segments are laid on the timeline as SeamlessStreaming's;
segments X-Voice cannot synthesise (at most two per point, interjections) stay silent. \emph{{Hy-SimulMT + Qwen3-TTS}} is
a reference cascade: the same Hy-SimulMT blocks, spoken by Qwen3-TTS-12Hz-0.6B-Base~\citep{{qwen2026qwen3tts}} (x-vector
clone of the source speaker, default sampling, three seeds, no rate control) instead of our Talker. Some very short
blocks never emit the end token, so generation is capped at $\min(1800,\max(180,18\times\text{{characters}}))$ codec
frames per block, the value of Hy-SimulMT's released inference code; {100 * c0:.1f}--{100 * c1:.1f}\% of blocks reach the cap.

\input{{generated/tab_baselines}}

\paragraph{{Talker ablation.}} To compare speech generators on the same committed text, \qabc's committed blocks (text
and commit times) at its four tier points per direction are also spoken by Qwen3-Omni's native Thinker, Talker and
code2wav, asked to read each block verbatim, and scored with the same protocol (Table~\ref{{tab:omnitalker}}). Our Talker
is higher at all eight points (+{d0:.2f} to +{d1:.2f} ASR-BLEU), with a lower talker error rate at {n['wer_better']} of eight
and a lower D1 at {n['d1_lower']} (equal at {n['d1_equal']}).

\input{{generated/tab_omnitalker}}
""")
    (PAPER / "sections/appendix.tex").write_text(s, encoding="utf-8")


BIB_NEW = r"""
@misc{zhao2026xtranslator,
  title = {{X-Translator}: A Real-Time Multilingual Speaker-Aware Speech-to-Speech Translation System},
  author = {Zhao, Yuxiang and Zhang, Yichi and An, Yanjie and Zhu, Yanqiao and Liu, Zhanxun and Chen, Yushen and Zheng, Qixi and Zhu, Haina and Xiao, Yunchong and Deng, Keqi and Fan, Shuai and Yu, Kai and Chen, Xie},
  year = {2026},
  eprint = {2607.17544},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2607.17544},
  note = {arXiv preprint, July 2026}
}
@misc{qwen2026qwen3tts,
  title = {{Qwen3-TTS-12Hz-0.6B-Base}},
  author = {{Qwen Team}},
  year = {2026},
  howpublished = {\url{https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base}}
}
"""


def fill_bib():
    s = (BASE / "references.bib").read_text(encoding="utf-8")
    assert "zhao2026xtranslator" not in s and "qwen2026qwen3tts" not in s
    (PAPER / "references.bib").write_text(s.rstrip("\n") + "\n" + BIB_NEW, encoding="utf-8")


def main():
    for k, v in SYSTEMS.items():
        print(k, {d: [(p["name"], p["asr"], round(p["d1"], 3)) for p in v[d]] for d in DIRS})
    print(numbers_for_text())
    fill_tab_tiers()
    fill_tab_tiers_main()
    fill_tab_realsi_main()
    fill_tab_baselines()
    fill_tab_omnitalker()
    fill_text()
    fill_bib()
    print("filled", PAPER)


if __name__ == "__main__":
    main()
