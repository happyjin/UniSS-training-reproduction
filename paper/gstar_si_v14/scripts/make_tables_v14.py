"""Paper v14 tables (Table 1 = multi-speaker S2ST systems; ablation separate) (baseline = B only; rows = MIST's cumulative components) (the playback-alignment CRF in the Talker is the method; no write-alignment rows).

Inputs (read-only): eval_outputs/mist_w3po_v1/joint/v11f_analysis/factorial_test_v1.json and the v11 generated tables.
Outputs (derived LaTeX, regenerated on every run): generated/v12/*.tex
  tab_main      factorial ablation on B's text = the main result table (B, rule layers, PA Talker, MIST)
  tab_sig       paired BLEU tests, MIST - B rows only
  tab_speech    speech latency / rate / voice / pauses and paired latency differences, B and MIST rows only
  tab_decomp    delay decomposition, B and MIST rows only
  tab_alieval   locked test, B and MIST rows only
  tab_e7        SeamlessStreaming against MIST (B's text + playback layer)
  tab_metricsv  second recogniser and COMET (unchanged)

    cd paper/gstar_si_v12 && python scripts/make_tables_v12.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
G = HERE / "generated"
OUT = G / "v14"
T1 = REPO / "eval_outputs/mist_w3po_v1/joint/table1_v14/table1_v1.json"   # table1_analysis output
FA = REPO / "eval_outputs/mist_w3po_v1/joint/v11f_analysis/factorial_test_v1.json"
SETS = ("ami", "en", "ali")


def w(name: str, s: str) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.tex").write_text(s)


def drop_wa(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if not ("+WA" in l and "&" in l)) + "\n"


def main_table() -> str:
    """ablation (single column): B and MIST's playback-layer components added one at a time."""
    F = json.loads(FA.read_text())
    S = F["sets"]
    rows = [("R0", "R0 B (base Talker)"), ("R1", "R1 + read merging"), ("R2", "R2 + backlog rate"),
            ("R4", "R3 + PA Talker"), ("R5", "R4 + best-of-2 = \\sys{}")]
    out = []
    for key, lab in rows:
        cells = []
        for m in ("asr_all", "endoff"):
            for s_ in SETS:
                v = "{:.2f}".format(S[s_]["rows"][key][m])
                if key == "R5":
                    v = "\\textbf{" + v + "}$^{*}$"
                cells.append(v)
        if key == "R5":
            out.append("\\midrule")
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return ("\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{2.2pt}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{tcpASR-BLEU $\\uparrow$} & \\multicolumn{3}{c}{EndOffset (s) $\\downarrow$} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Speech layer & Z$\\rightarrow$E & E$\\rightarrow$Z & Ali & Z$\\rightarrow$E & E$\\rightarrow$Z & Ali \\\\\n"
            "\\midrule\n" + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}}\n"
            "\\caption{Ablation on the deadline Thinker's text (text tcpBLEU 25.02 / 41.15 / 21.43 in every row): the "
            "components of \\sys{}'s playback layer added to B one at a time (R2: base rate rising with the backlog; "
            "R3: PA Talker, one sample; R4: two samples, the higher CTC fidelity kept). Speaker labels come from the "
            "Thinker, so every row attributes speech identically. $^{*}$: \\sys{} $-$ B significant (Tables~"
            "\\ref{tab:sig},~\\ref{tab:speech2}); LAAL$_\\text{block}$: Table~\\ref{tab:speech}.}\n"
            "\\label{tab:ablation}\n\\end{table}\n")


def macros(R: dict) -> str:
    """\\num{v14:<set>:<system>:<metric>} and \\num{v14:<set>:MIST-<system>:<cp|tcp|laal_block|endoff>}."""
    out = []
    d = lambda k, v: out.append("\\expandafter\\def\\csname n@v14:%s\\endcsname{%s}" % (k, v))  # noqa: E731
    sd = lambda x: ("+%.2f" % x) if x >= 0 else ("$-$%.2f" % -x)  # noqa: E731
    for s_ in SETS:
        for k, row in R["sets"][s_]["rows"].items():
            for m in ("cpASR", "tcpASR", "laal_block", "endoff", "startoff", "laal_spread"):
                if m in row:
                    d(f"{s_}:{k}:{m}", "%.2f" % row[m])
        for k, P in R["sets"][s_]["paired"].items():
            for m in ("cp", "tcp"):
                d(f"{s_}:{k}:{m}", sd(P[m]["diff"]))
            for m in ("laal_block", "endoff"):
                if m in P:
                    d(f"{s_}:{k}:{m}", sd(P[m]["diff_pooled"]))
    return "\n".join(out) + "\n"


DE = REPO / "eval_outputs/mist_w3po_v1/diarist_eval/diarist_eval_v1.json"
# DiariST (Yang et al., arXiv 2309.08007v2), IHM-MIX, checked against the PDF: (dev SAg, dev SAt, test SAg, test SAt)
PUBLISHED = [("Whisper-small, SD$\\rightarrow$ST", "text", "offline", (10.71, 9.21, 10.40, 9.31)),
             ("Whisper-small, ST$\\rightarrow$SD", "text", "offline", (14.48, 11.96, 13.68, 12.10)),
             ("DiariST (t-SOT + t-vector)", "text", "streaming", (16.05, 13.91, 16.62, 14.76))]


def diarist_table(D: dict) -> str:
    f = lambda x: "--" if x is None else "%.2f" % x  # noqa: E731
    rows = [f"{n}$^\\dagger$ & {o} & {m} & " + " & ".join(f(x) for x in v) + " \\\\" for n, o, m, v in PUBLISHED]
    rows.append("\\midrule")
    ours = [("B_MIST:text", "\\sys{} Thinker (= B)", "text"), ("MIST:asr", "\\sys{}", "speech"), ("B:asr", "B", "speech"),
            ("SeamlessStreaming:asr", "SeamlessStreaming", "speech"), ("X-Translator:asr", "X-Translator", "speech")]
    for k, n, o in ours:
        v = []
        for st in ("dev", "test"):
            x = D["sets"].get(st, {}).get("systems", {}).get(k)
            v += [x["later"]["SAgBLEU"], x["later"]["SAtBLEU"]] if x else [None, None]
        if all(x is None for x in v):
            continue
        rows.append(f"{n} & {o} & streaming & " + " & ".join(f(x) for x in v) + " \\\\")
    return ("\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{2.5pt}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lllrrrr}\n\\toprule\n"
            " & & & \\multicolumn{2}{c}{dev} & \\multicolumn{2}{c}{test} \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "System & output & mode & SAg & SAt & SAg & SAt \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}}\n"
            "\\caption{DiariST-AliMeeting (IHM-MIX, human English references) with its official evaluation: "
            "speaker-agnostic (SAg) and speaker-attributed (SAt) BLEU over 76 dev / 195 test mini-sessions of 3--6 min. "
            "$^\\dagger$Published by \\citet{yang2024diarist}; these systems are trained on AliMeeting, ours are not. "
            "Ours decode whole meetings online and are cut into the mini-sessions afterwards; speech rows score the ASR "
            "transcripts (two TTS seeds averaged for B and \\sys{}).}\n\\label{tab:diarist}\n\\end{table}\n")


def systems_table(R: dict) -> str:
    """Table 1: multi-speaker S2ST systems x test sets: cpASR-BLEU, tcpASR-BLEU, LAAL_block, EndOffset."""
    order = [sy["key"] for sy in R["systems"]]
    label = {"seamless": "SeamlessStreaming", "xtranslator": "X-Translator", "cascade": "Cascade (blind ST + relabel)",
             "B": "B (deadline baseline)", "MIST": "\\sys{} (ours)"}
    note = {"seamless": "$^a$", "xtranslator": "$^b$", "cascade": "$^c$", "B": "", "MIST": ""}
    lines = []
    present = [k for k in order if any(k in R["sets"][s_]["rows"] for s_ in SETS)]
    best = {}
    for s_ in SETS:
        rows = R["sets"][s_]["rows"]
        for m, hi in (("cpASR", True), ("tcpASR", True), ("laal_block", False), ("endoff", False)):
            vals = [rows[k][m] for k in present if k in rows]
            best[(s_, m)] = (max if hi else min)(vals) if vals else None
    for k in present:
        if k == "MIST" or (k == "B" and len(lines) and not lines[-1].startswith("\\midrule")):
            lines.append("\\midrule")
        cells = []
        for s_ in SETS:
            rows, P = R["sets"][s_]["rows"], R["sets"][s_]["paired"]
            for m in ("cpASR", "tcpASR", "laal_block", "endoff"):
                if k not in rows:
                    cells.append("--")
                    continue
                v = rows[k][m]
                t = f"{v:.2f}"
                if v == best[(s_, m)]:
                    t = "\\textbf{" + t + "}"
                if k != "MIST":
                    d = P.get(f"MIST-{k}", {})
                    if m in ("cpASR", "tcpASR"):
                        sig = d.get(m[:-3].lower() if False else ("cp" if m == "cpASR" else "tcp"), {}).get("p_holm", 1) < 0.05
                    else:
                        x = d.get(m, {})
                        sig = x.get("p_signflip", x.get("p", 1)) < 0.05
                    if sig:
                        t += "$^{*}$"
                cells.append(t)
        lines.append(f"{label.get(k, k)}{note.get(k, '')} & " + " & ".join(cells) + " \\\\")
    head = ("\\begin{table*}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{2.6pt}\n"
            "\\resizebox{\\textwidth}{!}{\\begin{tabular}{l" + "rrrr" * 3 + "}\n\\toprule\n"
            " & \\multicolumn{4}{c}{AMI Zh$\\rightarrow$En} & \\multicolumn{4}{c}{AMI En$\\rightarrow$Zh} & "
            "\\multicolumn{4}{c}{AliMeeting Zh$\\rightarrow$En} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n"
            "System & cpASR & tcpASR & LAAL & EndOff. & cpASR & tcpASR & LAAL & EndOff. & cpASR & tcpASR & LAAL & "
            "EndOff. \\\\\n\\midrule\n")
    cap = ("\\caption{Multi-speaker speech-to-speech interpretation of whole test meetings, run online on the same "
           "detector windows. cpASR / tcpASR: speaker-attributed ASR-BLEU of the synthesised speech, without (cp, "
           "meeting-level minimum-permutation) and with (tcp) time constraints ($\\uparrow$); LAAL: block-start speech "
           "LAAL; EndOff.: EndOffset (s, $\\downarrow$; chunk protocol, without computation). $^{a}$zero-shot "
           "single-speaker SimulS2ST; $^{b}$open speaker-aware cascade S2ST (streaming ASR, MT, voice-cloning TTS), "
           "zero-shot; $^{c}$speaker-blind streaming model trained on our data, base Talker. Systems without speaker "
           "labels of their own are attributed by causal relabelling with the same diarizer (\\S\\ref{sec:setup}). "
           "$^{*}$: \\sys{} $-$ system significant (BLEU: Holm over all sets and systems; latency: sign-flip "
           "$p<0.05$). Best per column in bold.}\n\\label{tab:main}\n\\end{table*}\n")
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n" + cap


def main() -> None:
    w("tab_ablation", main_table())
    import os
    t1 = Path(os.environ.get("T1_JSON", str(T1)))
    R = json.loads(t1.read_text()) if t1.exists() else None
    if R is not None:
        w("tab_systems", systems_table(R))
        w("macros_v14", macros(R))
    if DE.exists():
        w("tab_diarist", diarist_table(json.loads(DE.read_text())))
    have = lambda k: R is not None and any(k in R["sets"][s_]["rows"] for s_ in SETS)  # noqa: E731
    w("switches", "\\XT%s\n\\Cascade%s\n" % ("true" if have("xtranslator") else "false",
                                               "true" if have("cascade") else "false"))
    sig = drop_wa((G / "tab_v11_sig.tex").read_text())
    sig = sig.replace("Holm over the 12 tests", "Holm over the twelve BLEU tests of the analysis plan")
    sig = sig.replace("; \\sys{}+WA: $\\bias=-1$. Text of \\sys{} equals B by construction",
                      ". The text of \\sys{} equals B's by construction")
    sig = sig.replace(" (Appendix~\\ref{app:noise})", "").replace("\\label{tab:sig10}", "\\label{tab:sig}")
    sig = sig.replace("speech: \\sys{} $-$ B", "ASR-BLEU: \\sys{} $-$ B")
    w("tab_sig", sig)
    sp = drop_wa((G / "tab_v11_postreview.tex").read_text())
    sp = sp.replace("\\label{tab:postreview}", "\\label{tab:speech}").replace("\\label{tab:postreview2}",
                                                                          "\\label{tab:speech2}")
    w("tab_speech", sp)
    dc = drop_wa((G / "tab_v11_decomp.tex").read_text())
    dc = dc.replace("\\sys{}: deadline baseline text with rl + PA Talker.",
                    "\\sys{}: the deadline Thinker's text with read-level merging and the PA Talker.")
    dc = dc.replace("\\label{tab:decomp10}", "\\label{tab:decomp}")
    w("tab_decomp", dc)
    ae = drop_wa((G / "tab_v11_alieval.tex").read_text())
    ae = ae.replace(", each system decoded once (+WA: $\\bias=-1$)", ", each system decoded once")
    ae = ae.replace("\\sys{} (B + PA)", "\\sys{}")
    w("tab_alieval", ae)
    # SeamlessStreaming against MIST: MIST's values from the v11 main table (B + playback alignment rows)
    mt = (G / "tab_v11_main.tex").read_text().splitlines()
    mist = {}
    cur = None
    for l in mt:
        m = re.match(r"\\multirow\{3\}\{\*\}\{(.+?)\}", l)
        if m:
            cur = m.group(1)
        if "B + playback alignment" in l and cur:
            c = [x.strip() for x in l.rstrip("\\ ").split("&")]
            nums = [re.sub(r"\$.*?\$", "", x) for x in c[2:]]
            mist[cur] = {"all": nums[0], "ov": nums[1], "asr": nums[3], "spread": nums[7], "end": nums[8]}
    e7 = (G / "tab_v11_e7.tex").read_text().splitlines()
    names = {"AMI Zh$\\rightarrow$En": "AMI Zh$\\rightarrow$En", "AliMeeting": "AliMeeting Zh$\\rightarrow$En",
             "AMI En$\\rightarrow$Zh": "AMI En$\\rightarrow$Zh"}
    new = []
    for l in e7:
        if "\\sys{}+WA" in l:
            setn = l.split("&")[0].strip()
            v = mist[names[setn]]
            l = f"{setn} & \\sys{{}} & {v['all']} & {v['ov']} & {v['asr']} & {v['spread']} & {v['end']} \\\\"
        new.append(l)
    w("tab_e7", "\n".join(new) + "\n")
    w("tab_metricsv", (G / "tab_v11_metricsv.tex").read_text())
    ov = "\n".join(l for l in (G / "clean" / "tab_v8_ovmain.tex").read_text().splitlines() if "LA2" not in l) + "\n"
    ov = re.sub(r"\s*All rows and\s+metrics: Table~\\ref\{tab:overlap8\}\.", "", ov)
    ov = ov.replace("(Appendix~\\ref{app:qe7})", "(Appendix~\\ref{app:data})")
    w("tab_ovmain", ov)
    print("written:", sorted(p.name for p in OUT.glob("*.tex")))
    print(mist)


if __name__ == "__main__":
    main()
