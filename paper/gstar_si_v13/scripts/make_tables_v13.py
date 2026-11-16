"""Paper v13 tables (baseline = B only; rows = MIST's cumulative components) (the playback-alignment CRF in the Talker is the method; no write-alignment rows).

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
OUT = G / "v13"
FA = REPO / "eval_outputs/mist_w3po_v1/joint/v11f_analysis/factorial_test_v1.json"
SETS = ("ami", "en", "ali")


def w(name: str, s: str) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.tex").write_text(s)


def drop_wa(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if not ("+WA" in l and "&" in l)) + "\n"


def main_table() -> str:
    F = json.loads(FA.read_text())
    S = F["sets"]
    rows = [("R0", "R0\\ \\ B: base Talker, fixed rate"), ("R1", "R1\\ \\ + read-level merging"),
            ("R2", "R2\\ \\ + backlog-dependent base rate"), ("R4", "R3\\ \\ + PA Talker (CRF-distilled)"),
            ("R5", "R4\\ \\ + best-of-2 = \\sys{}")]
    out = []
    for key, lab in rows:
        cells = []
        for m in ("asr_all", "laal_block", "endoff"):
            for s_ in SETS:
                v = "{:.2f}".format(S[s_]["rows"][key][m])
                if key == "R5":
                    v = "\\textbf{" + v + "}$^{*}$"
                cells.append(v)
        if key == "R5":
            out.append("\\midrule")
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return ("\\begin{table*}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{ASR-BLEU $\\uparrow$} & \\multicolumn{3}{c}{LAAL$_\\text{block}$ (s) $\\downarrow$} & "
            "\\multicolumn{3}{c}{EndOffset (s) $\\downarrow$} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            "Speech layer (B's text throughout) & Z$\\rightarrow$E & E$\\rightarrow$Z & Ali & Z$\\rightarrow$E & "
            "E$\\rightarrow$Z & Ali & Z$\\rightarrow$E & E$\\rightarrow$Z & Ali \\\\\n\\midrule\n"
            + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Whole test meetings run online: the deadline baseline B and the components of \\sys{}'s "
            "playback layer added one at a time, all on one fixed text, that of the deadline Thinker (text tcpBLEU "
            "25.02 / 41.15 / 21.43 in every row). Z$\\rightarrow$E, E$\\rightarrow$Z: AMI test (16 meetings each); "
            "Ali: AliMeeting test (20 meetings). Each row decoded once; ASR-BLEU and latency averaged over two TTS "
            "seeds; latency per speaker stream under the chunk protocol, without computation. R2: base rate rising "
            "with the backlog (\\S\\ref{sec:method-voice}); R3: the PA Talker of \\S\\ref{sec:method-plem} with one "
            "sample; R4: two samples, the one with the higher CTC fidelity kept. $^{*}$: \\sys{} $-$ B significant "
            "(Table~\\ref{tab:sig}; latency: Table~\\ref{tab:speech2}).}\n\\label{tab:main}\n\\end{table*}\n")


def main() -> None:
    w("tab_main", main_table())
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
