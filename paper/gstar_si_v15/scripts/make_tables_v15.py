"""Paper v15 tables and Figure 3 (body: Table 1 systems, Figure 3 quality-LAAL plane, overlap, delay
decomposition + speech properties, ablation, robustness; appendix: MIST-vs-system tests and fixed copies of v14 tables).

Inputs (read-only):
  eval_outputs/mist_w3po_v1/joint/table1_v14/table1_v1.json        Table 1 rows and paired tests
  eval_outputs/mist_w3po_v1/joint/v11f_analysis/factorial_test_v1.json  ablation rows (mean of the two TTS runs)
  generated/v14/*.tex                                                numbers of the v14 appendix tables
Outputs: generated/v15/*.tex (refuses to overwrite a file whose content differs unless --force is given).
Rounding: half-up on the stored decimal value (31.645 -> 31.65), which v14's float formatting printed as 31.64.

    cd paper/gstar_si_v15 && python3 scripts/make_tables_v15.py
"""
from __future__ import annotations

import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
G14 = HERE / "generated" / "v14"
OUT = HERE / "generated" / "v15"
T1 = REPO / "eval_outputs/mist_w3po_v1/joint/table1_v14/table1_v1.json"
FA = REPO / "eval_outputs/mist_w3po_v1/joint/v11f_analysis/factorial_test_v1.json"
SETS = ("ami", "en", "ali")                       # body order: AMI Zh->En, AMI En->Zh, AliMeeting
SETNAME = {"ami": "AMI Zh$\\rightarrow$En", "en": "AMI En$\\rightarrow$Zh", "ali": "AliMeeting Zh$\\rightarrow$En"}
FORCE = "--force" in sys.argv


def r2(x: float, nd: int = 2) -> str:
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def sgn(x: float, nd: int = 2) -> str:
    s = r2(x, nd)
    return ("+" + s) if not s.startswith("-") else "$-$" + s[1:]


def w(name: str, s: str) -> None:
    OUT.mkdir(exist_ok=True)
    p = OUT / f"{name}.tex"
    if p.exists() and p.read_text() != s and not FORCE:
        sys.exit(f"refusing to overwrite {p} (content differs); rerun with --force after checking")
    p.write_text(s)
    print("wrote", p.relative_to(HERE))


def check_in(src: str, values: list[str]) -> None:
    """every number must appear verbatim in the v14 source table it was copied from"""
    text = (G14 / f"{src}.tex").read_text()
    miss = [v for v in values if v.replace("$-$", "-") not in text.replace("$-$", "-")]
    if miss:
        sys.exit(f"numbers not found in generated/v14/{src}.tex: {miss}")


T = json.loads(T1.read_text())
F = json.loads(FA.read_text())["sets"]
SYS = [("seamless", "SeamlessStreaming$^{a}$"), ("xtranslator", "X-Translator$^{b}$"),
       ("cascade", "Cascade$^{c}$"), ("B", "B (deadline baseline)"), ("MIST", "\\sys{} (ours)")]


# ---------------------------------------------------------------- Table 1
def tab_systems() -> str:
    cols = [(s, m) for s in SETS for m in ("cpASR", "tcpASR", "laal_block")]
    best = {}
    for s, m in cols:
        vals = [T["sets"][s]["rows"][k][m] for k, _ in SYS]
        best[(s, m)] = r2(min(vals) if m == "laal_block" else max(vals))
    lines = []
    for k, lab in SYS:
        cells = []
        for s, m in cols:
            v = r2(T["sets"][s]["rows"][k][m])
            cells.append("\\textbf{" + v + "}" if v == best[(s, m)] else v)
        if k in ("B", "MIST"):
            lines.append("\\midrule")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(["cpASR-BLEU", "tcpASR-BLEU", "LAAL"] * 3)
    return ("\\begin{table*}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{3.2pt}\n"
            "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{AMI Zh$\\rightarrow$En} & \\multicolumn{3}{c}{AMI En$\\rightarrow$Zh}"
            " & \\multicolumn{3}{c}{AliMeeting Zh$\\rightarrow$En (out-of-domain)} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            f"System & {head} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n"
            "\\caption{Multi-speaker speech-to-speech translation of whole test meetings (same online windows and "
            "recognisers for all systems). cpASR-BLEU / tcpASR-BLEU ($\\uparrow$): speaker-attributed BLEU of the speech "
            "without / with time constraints; LAAL ($\\downarrow$): speech LAAL in s. Every BLEU difference between "
            "\\sys{} and another system is significant (Holm-corrected paired test over meetings, $p<0.001$; "
            "Table~\\ref{tab:syssig}). $^{a}$Public single-speaker SimulS2ST and $^{b}$open speaker-aware cascade, both "
            "zero-shot; $^{c}$speaker-blind model trained on our data; $^{a\\text{--}c}$attributed by causal relabelling "
            "(Appendix~\\ref{app:baselines}). Bold: best.}\n\\label{tab:main}\n\\end{table*}\n")


# ---------------------------------------------------------------- Figure 3 (quality-LAAL plane)
def fig_tradeoff() -> str:
    marks = {"seamless": ("triangle*", "gray!70!black", "SeamlessStreaming"),
             "xtranslator": ("diamond*", "frozen", "X-Translator"),
             "cascade": ("square*", "spkc", "Cascade"),
             "B": ("*", "black", "B"),
             "MIST": ("pentagon*", "trained", "\\protect\\sys{}")}
    panels = []
    for n, s in enumerate(SETS):
        rows = T["sets"][s]["rows"]
        fa = F[s]["rows"]
        pts = []
        for k, (mk, col, lab) in marks.items():
            x, y = r2(rows[k]["laal_block"]), r2(rows[k]["tcpASR"])
            leg = f"\\addlegendentry{{{lab}}}" if n == 0 else ""
            pts.append(f"\\addplot[only marks, mark={mk}, mark size=2.6pt, color={col}] coordinates {{({x},{y})}};{leg}")
        path = " ".join(f"({r2(fa[k]['laal_block'])},{r2(fa[k]['asr_all'])})" for k in ("R0", "R1", "R2", "R4", "R5"))
        leg = "\\addlegendentry{B, R1--R3, \\protect\\sys{} (ablation path)}" if n == 0 else ""
        pts.append(f"\\addplot[thin, color=trained, mark=o, mark size=1.2pt, -{{Stealth[length=1.6mm]}}] coordinates {{{path}}};{leg}")
        opts = f"title={{{SETNAME[s]}{' (out-of-domain)' if s == 'ali' else ''}}}"
        if n == 0:
            opts += ", ylabel={tcpASR-BLEU $\\uparrow$}, legend to name=leg:tradeoff, legend columns=6"
        panels.append(f"\\nextgroupplot[{opts}]\n" + "\n".join(pts))
    return ("\\begin{figure*}[t]\n\\centering\n\\begin{tikzpicture}\n"
            "\\begin{groupplot}[group style={group size=3 by 1, horizontal sep=1.1cm}, width=0.36\\textwidth, "
            "height=3.2cm, xlabel={LAAL (s) $\\downarrow$}, title style={font=\\small, yshift=-1mm}, "
            "tick label style={font=\\scriptsize}, label style={font=\\small}, grid=major, grid style={gray!20}, "
            "legend style={font=\\scriptsize, draw=none, column sep=4pt}]\n"
            + "\n".join(panels) + "\n\\end{groupplot}\n\\end{tikzpicture}\\\\[-1mm]\n"
            "\\pgfplotslegendfromname{leg:tradeoff}\n"
            "\\caption{Speech quality against speech LAAL for the systems of Table~\\ref{tab:main} (upper left is "
            "better). The line traces B through the playback-layer components of Table~\\ref{tab:ablation} to \\sys{} "
            "(points: mean of the two TTS runs).}\n\\label{fig:tradeoff}\n\\end{figure*}\n")


# ---------------------------------------------------------------- Table 3 (overlap; v14 tab_ovmain online rows)
def tab_overlap() -> str:
    blind = {"ami": ("19.33", "9.06", "54.4"), "en": ("35.22", "27.17", "65.5"), "ali": ("12.67", "8.55", "63.0")}
    spk = {"ami": ("24.53", "17.73", "80.2"), "en": ("40.09", "35.71", "82.3"), "ali": ("11.47", "9.18", "80.2")}
    check_in("tab_ovmain", [v for d in (blind, spk) for t in d.values() for v in t])
    rows = []
    for lab, d, o in (("Speaker-blind + relabel", blind, spk), ("With speaker tokens", spk, blind)):
        cells = []
        for s in SETS:
            for a, b in zip(d[s], o[s]):
                cells.append("\\textbf{" + a + "}" if float(a) > float(b) else a)
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return ("\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{2.2pt}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{AMI Zh$\\rightarrow$En} & \\multicolumn{3}{c}{AMI En$\\rightarrow$Zh} & "
            "\\multicolumn{3}{c}{AliMeeting} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            "Thinker & all & ov. & CR & all & ov. & CR & all & ov. & CR \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}}\n"
            "\\caption{Speaker tokens under overlap: text tcpBLEU over all and overlapped (ov.) units and "
            "concurrent-speaker recall (CR, \\%), Thinkers trained alike with and without (zeroed) speaker tokens. "
            "Confirmatory tests (Table~\\ref{tab:reg8}): ov.\\ and CR gains hold on both AMI sets, only CR on AliMeeting "
            "(machine-translated references here; Appendix~\\ref{app:data}).}\n"
            "\\label{tab:overlap}\n\\end{table}\n")


# ---------------------------------------------------------------- Table 4 (delay decomposition + speech properties)
def tab_delay() -> str:
    dec = {  # model, detector, queue, last block, EndOffset, load  (v14 tab_decomp, test rows)
        ("ami", "B"): ("0.67", "0.66", "0.65", "2.10", "4.08", "1.37"), ("ami", "M"): ("0.68", "0.66", "0.14", "1.64", "3.13", "0.94"),
        ("en", "B"): ("0.69", "0.64", "0.54", "1.61", "3.48", "1.28"), ("en", "M"): ("0.71", "0.64", "0.14", "1.42", "2.91", "0.99"),
        ("ali", "B"): ("0.88", "0.40", "1.33", "1.96", "4.57", "1.56"), ("ali", "M"): ("0.95", "0.40", "0.33", "1.55", "3.23", "1.05")}
    sp = {  # rate, UTMOS, SIM, pauses >= 0.5 s per min  (v14 tab_speech)
        ("ami", "B"): ("2.40", "1.80", "0.827", "3.3"), ("ami", "M"): ("3.49", "1.94", "0.827", "8.3"),
        ("en", "B"): ("3.90", "3.25", "0.833", "4.6"), ("en", "M"): ("5.04", "3.40", "0.831", "9.4"),
        ("ali", "B"): ("2.40", "3.48", "0.873", "4.4"), ("ali", "M"): ("3.56", "3.72", "0.873", "11.8")}
    check_in("tab_decomp", [v for t in dec.values() for v in t])
    check_in("tab_speech", [v for t in sp.values() for v in t])
    short = {"ami": "Z$\\rightarrow$E", "en": "E$\\rightarrow$Z", "ali": "Ali"}
    rows = []
    for s in SETS:
        for k, lab in (("B", "B"), ("M", "\\sys{}")):
            first = short[s] if k == "B" else ""
            rows.append(f"{first} & {lab} & " + " & ".join(dec[(s, k)]) + " & " + " & ".join(sp[(s, k)][1:]) + " \\\\")
        if s != SETS[-1]:
            rows.append("\\midrule")
    return ("\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{2.2pt}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llrrrrrrrrr}\n\\toprule\n"
            " & & \\multicolumn{5}{c}{EndOffset and its parts (s)} & \\multicolumn{4}{c}{speech} \\\\\n"
            "\\cmidrule(lr){3-7}\\cmidrule(lr){8-11}\n"
            "Set & Sys. & lag & det. & queue & last & total & load & UTMOS & SIM & pauses \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n"
            "\\caption{Where B's delay is removed (test sets, identical text): EndOffset (total) = model lag + "
            "detector wait + queue wait (Eq.~\\eqref{eq:backlog}) + last block. Load: speech per second of source; "
            "UTMOS: automatic MOS predictor; SIM: similarity to the voice prompt; pauses $\\ge$0.5\\,s per minute.}\n"
            "\\label{tab:decomp}\n\\end{table}\n")


# ---------------------------------------------------------------- Table 5 (ablation)
def tab_ablation() -> str:
    rows = [("R0", "R0 B (base Talker)"), ("R1", "R1 + read merging"), ("R2", "R2 + backlog-aware rate"),
            ("R4", "R3 + PA Talker"), ("R5", "R4 + best of two = \\sys{}")]
    out = []
    for key, lab in rows:
        cells = [r2(F[s]["rows"][key]["asr_all"]) for s in SETS] + [r2(F[s]["rows"][key]["endoff"]) for s in SETS]
        if key == "R5":
            out.append("\\midrule")
            cells = ["\\textbf{" + c + "}" for c in cells]
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    for s in SETS:   # R0 / R4 must equal Table 1's B / MIST
        for key, sysk in (("R0", "B"), ("R5", "MIST")):
            assert r2(F[s]["rows"][key]["asr_all"]) == r2(T["sets"][s]["rows"][sysk]["tcpASR"]), (s, key)
    return ("\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{2.4pt}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{tcpASR-BLEU $\\uparrow$} & \\multicolumn{3}{c}{EndOffset (s) $\\downarrow$} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Speech layer & Z$\\rightarrow$E & E$\\rightarrow$Z & Ali & Z$\\rightarrow$E & E$\\rightarrow$Z & Ali \\\\\n"
            "\\midrule\n" + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}}\n"
            "\\caption{Ablation on the deadline Thinker's text (text tcpBLEU 25.02 / 41.15 / 21.43 in all rows). R2: "
            "rate of Eq.~\\eqref{eq:rate}; R3: PA Talker; R4: best of two by $f$. \\sys{} $-$ B is significant "
            "(Table~\\ref{tab:robust}); other rows are descriptive.}\n"
            "\\label{tab:ablation}\n\\end{table}\n")


# ---------------------------------------------------------------- Table 6 (robustness; v14 tab_metricsv, tab_speech2, tab_alieval)
def tab_robust() -> str:
    rows = [("tcpASR-BLEU, main recognisers", ("+2.28", "+0.58", "+1.35", "+2.23")),
            ("tcpASR-BLEU, second recogniser", ("+1.32", "+3.74", "+0.93", "--")),
            ("COMET of the speech", ("+2.58", "+0.34", "+1.47", "--")),
            ("LAAL (s)", ("$-$0.09", "$-$0.11", "$-$0.33", "$-$0.37")),
            ("EndOffset (s)", ("$-$0.96", "$-$0.58", "$-$1.34", "$-$1.43"))]
    check_in("tab_metricsv", ["+1.32", "+3.74", "+0.93", "+2.58", "+0.34", "+1.47"])
    check_in("tab_speech", ["$-$0.09", "$-$0.11", "$-$0.33", "$-$0.96", "$-$0.58", "$-$1.34"])
    check_in("tab_alieval", ["+2.23", "$-$0.37", "$-$1.43"])
    check_in("tab_sig", ["+2.28", "+0.58", "+1.35"])
    body = "\n".join(f"{lab} & " + " & ".join(v) + " \\\\" for lab, v in rows)
    return ("\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lrrrr}\n\\toprule\n"
            " & AMI & AMI & \\multicolumn{2}{c}{AliMeeting (out-of-domain)} \\\\\n\\cmidrule(lr){4-5}\n"
            "\\sys{} $-$ B & Zh$\\rightarrow$En & En$\\rightarrow$Zh & test & locked \\\\\n\\midrule\n" + body +
            "\n\\bottomrule\n\\end{tabular}}\n"
            "\\caption{\\sys{} $-$ B with a second recogniser (not used for selection), COMET, and on eight locked "
            "AliMeeting meetings never decoded before. All differences are significant (Appendix~\\ref{app:more}).}\n"
            "\\label{tab:robust}\n\\end{table}\n")


# ---------------------------------------------------------------- appendix: MIST vs each system
def tab_syssig() -> str:
    names = {"seamless": "SeamlessStreaming", "xtranslator": "X-Translator", "cascade": "Cascade", "B": "B"}
    lines = []
    for s in SETS:
        P = T["sets"][s]["paired"]
        for n, (k, lab) in enumerate(names.items()):
            v = P[f"MIST-{k}"]
            cp, tcp, la = v["cp"], v["tcp"], v["laal_block"]
            ci = lambda c: f"[{r2(c[0])}, {r2(c[1])}]".replace("[-", "[$-$").replace(", -", ", $-$")
            ph = lambda p: "$<$0.001" if p < 0.001 else r2(p, 3)
            lines.append(f"{SETNAME[s] if n == 0 else ''} & {lab} & {sgn(cp['diff'])} {ci(cp['ci95'])} & {ph(cp['p_holm'])} & "
                         f"{sgn(tcp['diff'])} {ci(tcp['ci95'])} & {ph(tcp['p_holm'])} & "
                         f"{sgn(la['diff_pooled'])} {ci(la['ci95'])} & {ph(la['p_signflip'])} \\\\")
        if s != SETS[-1]:
            lines.append("\\midrule")
    return ("\\begin{table*}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{llrrrrrr}\n\\toprule\n"
            "Set & \\sys{} $-$ & cpASR-BLEU [95\\%] & $p_\\text{Holm}$ & tcpASR-BLEU [95\\%] & $p_\\text{Holm}$ & "
            "LAAL (s) [95\\%] & $p$ \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Paired tests of Table~\\ref{tab:main}: \\sys{} minus each system over meetings (exact sign-flip "
            "test, 95\\% meeting-bootstrap interval; BLEU contrasts Holm-corrected per score as fixed in the analysis "
            "plan; LAAL uncorrected). A positive LAAL difference means \\sys{} is heard later.}\n"
            "\\label{tab:syssig}\n\\end{table*}\n")


# ---------------------------------------------------------------- appendix: fixed copies of v14 tables
def fix_sig() -> str:
    s = (G14 / "tab_sig.tex").read_text()
    order = iter(["AMI Zh$\\rightarrow$En", "AliMeeting", "AMI En$\\rightarrow$Zh"])   # v14 row order (diffs 2.28 / 1.35 / 0.58)
    s = re.sub(r"^ & ASR-BLEU", lambda m: next(order) + " & tcpASR-BLEU", s, flags=re.M)
    return s.replace("Speech pools", "tcpASR-BLEU pools")


def fix_speech() -> str:
    s = (G14 / "tab_speech.tex").read_text()
    s = "\n".join(l for l in s.splitlines() if "\\bias" not in l) + "\n"
    order = iter(["AMI Zh$\\rightarrow$En", "AliMeeting", "AMI En$\\rightarrow$Zh"])
    s = re.sub(r"^ & \\sys\{\} \$-\$ B & \$-\$0", lambda m: next(order) + " & \\sys{} $-$ B & $-$0", s, flags=re.M)
    return s.replace("Speech of the streaming systems", "Speech of B and \\sys{}")


def fix_decomp_full() -> str:
    s = (G14 / "tab_decomp.tex").read_text()
    return s.replace("\\label{tab:decomp}", "\\label{tab:decompfull}").replace(
        "\\caption{Where", "\\caption{Development and test sets: where")


def rename_asr(name: str) -> str:
    """v14 appendix tables call tcpASR-BLEU 'ASR-BLEU'; same numbers, v15 name"""
    s = (G14 / f"{name}.tex").read_text()
    s = s.replace("text & ASR- & text", "text & tcpASR- & text")
    s = s.replace("\\multicolumn{2}{c}{ASR-BLEU} & \\multicolumn{3}{c}{ASR-BLEU, second recogniser}",
                  "\\multicolumn{2}{c}{tcpASR-BLEU} & \\multicolumn{3}{c}{tcpASR-BLEU, second recogniser}")
    s = s.replace("& ASR-BLEU & LAAL", "& tcpASR-BLEU & LAAL").replace("ASR-BLEU: the paper's", "tcpASR-BLEU: the paper's")
    s = s.replace("ASR-BLEU and speech metrics", "tcpASR-BLEU and speech metrics").replace("ASR-BLEU +2.23", "tcpASR-BLEU +2.23")
    return s


def fix_diarist() -> str:
    """mode column: streaming -> simultaneous (v15 wording); numbers unchanged"""
    return (G14 / "tab_diarist.tex").read_text().replace("& streaming &", "& simultaneous &").replace(
        "decode whole meetings online", "decode whole meetings simultaneously")


def main() -> None:
    w("tab_systems", tab_systems())
    w("fig_tradeoff", fig_tradeoff())
    w("tab_overlap", tab_overlap())
    w("tab_delay", tab_delay())
    w("tab_ablation", tab_ablation())
    w("tab_robust", tab_robust())
    w("tab_syssig", tab_syssig())
    w("tab_sig", fix_sig())
    w("tab_speech", fix_speech())
    w("tab_decompfull", fix_decomp_full())
    w("tab_diarist", fix_diarist())
    for n in ("tab_alieval", "tab_metricsv", "tab_e7"):
        w(n, rename_asr(n))


if __name__ == "__main__":
    main()
