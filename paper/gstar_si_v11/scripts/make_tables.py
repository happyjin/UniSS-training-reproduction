"""Every number of the paper's tables, read from the evaluation outputs (nothing typed by hand). v4, both directions.

  systems   scripts/systems.py (where each system's hyp.jsonl / score.json / analysis.json and speech outputs live)
  speech    <speech dir>/{score,voice,listen}.json, talker/asr_speech.jsonl, utmos/per_sample_utmos.jsonl
  stats     generated/bootstrap.json (scripts/bootstrap.py)
  data      data/ami_multispk_si_v1/{e2e_gstar_v1,e2e_gstar_zh2en_v1}/windows_<split>_ref.jsonl

Writes generated/tab_*.tex, generated/fig_tradeoff.tex, generated/macros.tex and generated/numbers.json.
Run from the paper folder:
    python scripts/make_tables.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systems import ARROW, ASR_FILE, D, DIRS, HEAD, LABELS, NAMES, REAL, SPEECH_DIR, V4, V5, hyp_dir, windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import units  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated"
V6V = ROOT / "eval_outputs/uniss_qwen3_omni_v1/gstar_v6_20261031/voice"
DATA ={"en2zh": D / "e2e_gstar_v1", "zh2en": D / "e2e_gstar_zh2en_v1", "ali": ROOT / "data/alimeeting_si_v1"}
LANG = {"en2zh": "cmn", "zh2en": "eng", "ali": "eng"}
ALL = DIRS + ("ali",)
CLAUSE_END = tuple(",.;:!?，。；：！？、…")
# significance marks in the main table: system -> the strongest baseline of its block (paired, both bootstraps)
SIG = {"O2": "C-ft2", HEAD: "O2_LA2_d1"}


def load(p: Path | None):
    return json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else None


def f2(x) -> str:
    return "--" if x is None else f"{x:.2f}"


def f1(x) -> str:
    return "--" if x is None else f"{x:.1f}"


def pct(x) -> str:
    return "--" if x is None else f"{100 * x:.1f}"


def neg(s: str) -> str:
    return s.replace("-", "$-$") if s.startswith("-") else s


def ref_text(u: dict) -> str:
    return u.get("ref", u["zh"])


def ref_units(dr: str) -> int:
    return sum(len(units(ref_text(u), LANG[dr])) for l in open(DATA[dr] / "windows_test_ref.jsonl", encoding="utf-8")
               for u in json.loads(l)["utts"] if not u["hesitation"])


def len_ratio(dr: str, d: Path | None, ref: int) -> float | None:
    if d is None or not (d / "hyp.jsonl").exists():
        return None
    hyp = sum(len(units(x["text"], LANG[dr])) for l in open(d / "hyp.jsonl", encoding="utf-8")
              for x in json.loads(l)["segments"])
    return hyp / ref


def block_stats(dr: str, d: Path | None) -> dict:
    """Text-side shape of a streaming output: mean units per committed block, share of one-unit blocks, and share of
    blocks that stop mid-phrase (no clause-final punctuation) before the speaker's next block."""
    if d is None or not (d / "hyp.jsonl").exists():
        return {}
    n = one = mid = cnt = 0
    for h in map(json.loads, open(d / "hyp.jsonl", encoding="utf-8")):
        bl = [b for b in h.get("blocks", []) if b["text"].strip()]
        for i, b in enumerate(bl):
            k = len(units(b["text"], LANG[dr]))
            n, one, cnt = n + k, one + (k <= 1), cnt + 1
            nxt = next((c for c in bl[i + 1:] if c["spk"] == b["spk"]), None)
            mid += nxt is not None and not b["text"].strip().endswith(CLAUSE_END)
    return {"units_per_block": n / max(cnt, 1), "single_unit": one / max(cnt, 1), "mid_stop": mid / max(cnt, 1),
            "blocks": cnt} if cnt else {}


def edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def speech_extras(dr: str, sp: Path | None) -> dict:
    """Talker misread rate (CER for Chinese, WER for English, of the speech's ASR against the text the Talker was
    given), listening measures (gstar_listen) and UTMOS."""
    out = {}
    if sp is None or not ASR_FILE(sp).exists():
        return out
    if dr == "zh2en":
        norm = lambda s: re.findall(r"[a-z0-9']+", (s or "").lower())   # noqa: E731
    else:
        norm = lambda s: list(re.sub(r"[^\w]", "", s or ""))   # noqa: E731
    asr = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(ASR_FILE(sp), encoding="utf-8"))}
    E = N = 0
    for r in map(json.loads, open(sp / "talker_in/blocks.jsonl", encoding="utf-8")):
        t = norm(("" if dr == "en2zh" else " ").join(b["text"] for b in r["blocks"]))
        if len(t) > 400:   # a few degenerate repetition loops; the edit distance is quadratic
            continue
        E += edit(t, norm(asr.get(r["id"], "")))
        N += len(t)
    out["misread"] = E / max(N, 1)
    li = load(sp / "listen.json")
    if li:
        out.update({k: li[k] for k in ("abrupt_per_min", "single_unit", "lag_start_med", "lag_end_med")})
    ut = sp / "utmos/per_sample_utmos.jsonl"
    if ut.exists():
        v = [r["utmos_score"] for r in map(json.loads, open(ut, encoding="utf-8"))]
        out["utmos"] = sum(v) / len(v) if v else None
    return out


def rtf(d: Path | None, audio_s: float) -> float | None:
    """GPU seconds of decoding (sum over shards, batch 16, one H200 each) per second of test audio."""
    if d is None:
        return None
    logs = sorted(d.glob("infer_s*.log"))
    if not logs:
        return None
    tot = 0.0
    for f in logs:
        m = re.findall(r"windows, (\d+)s", f.read_text(errors="ignore"))
        if not m:
            return None
        tot += float(m[-1])
    return tot / audio_s


def data_stats(dr: str, splits=("train", "validation", "test")) -> dict:
    out = {}
    for sp in splits:
        W = [json.loads(l) for l in open(DATA[dr] / f"windows_{sp}_ref.jsonl", encoding="utf-8")]
        talk = lambda w: [u for u in w["utts"] if not u["hesitation"]]   # noqa: E731

        def ov(w):
            s = [(u["begin_s"], u["end_s"]) for u in talk(w)]
            return any(min(a[1], b[1]) > max(a[0], b[0]) for i, a in enumerate(s) for b in s[i + 1:])
        out[sp] = {"windows": len(W), "meetings": len({w["meeting"] for w in W}),
                   "hours": sum(w["dur_ms"] for w in W) / 3.6e6,
                   "mean_s": sum(w["dur_ms"] for w in W) / 1000 / len(W),
                   "multi": sum(len({u["speaker_id"] for u in talk(w)}) > 1 for w in W),
                   "overlap": sum(map(ov, W)), "utts": sum(len(talk(w)) for w in W)}
    return out


def table(path: Path, body: str) -> None:
    path.write_text(body.replace(r"\end{tabular}", r"\end{tabular}}"), encoding="utf-8")


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def main() -> None:
    GEN.mkdir(exist_ok=True)
    DS = {dr: data_stats(dr) for dr in DIRS}
    ali_ok = (DATA["ali"] / "windows_test_ref.jsonl").exists()
    if ali_ok:
        DS["ali"] = data_stats("ali", ("test",))
    AUDIO = {dr: DS[dr]["test"]["hours"] * 3600 for dr in DS}
    REF = {dr: ref_units(dr) for dr in DS}
    S, A, SP, VO, SX, LR, BL, RT = ({dr: {} for dr in ALL} for _ in range(8))
    for dr in DS:
        for k in NAMES:
            d = hyp_dir(dr, k)
            S[dr][k] = load(d / "score.json") if d else None
            A[dr][k] = load(V5 / "analysis" / f"{dr}_{k}.json")   # scripts/analyze.py: the right slot map per run
            LR[dr][k] = len_ratio(dr, d, REF[dr])
            BL[dr][k] = block_stats(dr, d)
            RT[dr][k] = rtf(d, AUDIO[dr])
            sd = SPEECH_DIR(dr, k)
            SP[dr][k] = load(sd / "score.json") if sd else None
            # v6 corrected matching (target-language text; Appendix O) supersedes the speech dir's voice.json,
            # so Table 3 and Table 21 come from the same files
            VO[dr][k] = load(V6V / f"{dr}_{k}_wavlm.json") or (load(sd / "voice.json") if sd else None)
            SX[dr][k] = speech_extras(dr, sd)
    BS = load(GEN / "bootstrap.json") or {}
    nums = {"score": S, "speech": SP, "voice": VO, "speech_extras": SX, "len_ratio": LR, "blocks": BL, "data": DS,
            "rtf": RT, "bootstrap": BS}

    def al(dr, k, g="all"):
        return (S[dr].get(k) or {}).get(g)

    def sig_mark(dr, k):
        b = SIG.get(k)
        r = BS.get(f"{dr}:{k}_vs_{dr}:{b}") if b else None
        return r"$^\dagger$" if r and r["meeting"]["p"] < 0.05 and r["window"]["p"] < 0.05 else ""

    # data (appendix)
    rows = []
    for dr in DIRS:
        rows.append(r"\multicolumn{8}{l}{\emph{" + ARROW[dr] + r"}} \\")
        for sp, name in (("train", "Train"), ("validation", "Dev"), ("test", "Test")):
            d = DS[dr][sp]
            rows.append(f"{name} & {d['meetings']} & {thousands(d['windows'])} & {d['hours']:.1f} & {d['mean_s']:.1f} & "
                        f"{thousands(d['utts'])} & {100 * d['multi'] / d['windows']:.1f} & "
                        f"{100 * d['overlap'] / d['windows']:.1f} \\\\")
    if ali_ok:
        d = DS["ali"]["test"]
        rows += [r"\multicolumn{8}{l}{\emph{Zh$\rightarrow$En, AliMeeting (real meetings)}} \\",
                 f"Test & {d['meetings']} & {thousands(d['windows'])} & {d['hours']:.1f} & {d['mean_s']:.1f} & "
                 f"{thousands(d['utts'])} & {100 * d['multi'] / d['windows']:.1f} & {100 * d['overlap'] / d['windows']:.1f} \\\\"]
    table(GEN / "tab_data.tex", r"""\begin{table}[t]
\centering\small
\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrrrrr}
\toprule
Split & Mtg. & Windows & Hours & Mean\,s & Utt. & Multi\,\% & Ovl.\,\% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Reference windows of the two directions. En$\rightarrow$Zh: AMI IHM-mix with Chinese references;
Zh$\rightarrow$En: the re-voiced Chinese meetings (Appendix~\ref{app:zhdata}) with the AMI English transcripts as
references. \emph{Multi}: windows with two or more speakers; \emph{Ovl.}: windows in which two utterances overlap.}
\label{tab:data}
\end{table}
""")

    # main table: both directions side by side
    def cells(dr, k, best=False):
        s = al(dr, k)
        if s is None:
            return [r"\multicolumn{5}{c}{--}"]
        cp = f2(s["cpBLEU_session"]) + sig_mark(dr, k)
        return [f2(s.get("LAAL_s")) if s.get("LAAL_s") else "--", rf"\textbf{{{cp}}}" if best else cp,
                f2(s["BLEU_speaker_blind"]), pct(s["count_exact"]), f1(s["DER_output"]["DER"])]

    def top(dr, k, rivals):   # bold only where the system really is the best of its block
        mine = (al(dr, k) or {}).get("cpBLEU_session")
        return mine is not None and all(mine >= ((al(dr, r) or {}).get("cpBLEU_session") or -1) for r in rivals)

    def row(k, best=False):
        rivals = {"O2": ["B-off", "C-van", "C-ft2", "O2-nosd"], HEAD: ["B-LA2", "O2_LA2_d1"]}.get(k, [])
        return " & ".join([NAMES[k]] + cells("en2zh", k, best and top("en2zh", k, rivals))
                          + cells("zh2en", k, best and top("zh2en", k, rivals))) + r" \\"
    sec = lambda t: r"\multicolumn{11}{l}{\emph{" + t + r"}} \\"   # noqa: E731
    body = "\n".join([sec("Offline"), row("B-off"), row("C-van"), row("C-ft2"), row("O2-nosd"), row("O2", True),
                      r"\addlinespace", sec(r"Simultaneous ($\Delta{=}1$\,s)"), row("B-LA2"), row("O2_LA2_d1"),
                      row(HEAD, True), r"\addlinespace",
                      sec("Oracle speaker activity instead of the diarizer"), row("O2-oracle")])
    n_test = {dr: thousands(DS[dr]["test"]["windows"]) for dr in DIRS}
    table(GEN / "tab_main.tex", r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{\begin{tabular}{lccccc@{\hskip 10pt}ccccc}
\toprule
 & \multicolumn{5}{c}{En$\rightarrow$Zh} & \multicolumn{5}{c}{Zh$\rightarrow$En} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
System & LAAL\,s & cpBLEU$_\text{sess}$ & BLEU$_\text{blind}$ & Spk & DER & LAAL\,s & cpBLEU$_\text{sess}$ & BLEU$_\text{blind}$ & Spk & DER \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Speaker-attributed translation (text) of the AMI test meetings in both directions (""" + n_test["en2zh"] + r"""
En$\rightarrow$Zh and """ + n_test["zh2en"] + r""" Zh$\rightarrow$En windows, 16 meetings). One bidirectional
\sys{} (offline) serves both directions; \sys{}-Simul is trained per direction (3{,}000 steps each). cpBLEU$_\text{sess}$ (primary) matches output speakers to reference speakers once per
meeting; BLEU$_\text{blind}$ ignores speakers (character BLEU for Chinese, 13a for English). Spk: windows with the
right number of speakers (\%); DER (\%) from the output segments. LAAL includes the diarizer's 1.04\,s look-ahead
and is computation-unaware; Zh$\rightarrow$En has human references (the AMI transcripts). $^\dagger$: better than the strongest baseline of its block (same-data cascade offline,
\sys{} + LA2 simultaneous) with $p<0.05$ under paired bootstrap over windows and over meetings
(Appendix~\ref{app:sig}). Prompted and LA2-on-prompted baselines write no time stamps, so their DER is near 100\%
by construction.}
\label{tab:main}
\end{table*}
""")

    # real Mandarin meetings (AliMeeting test, Zh->En)
    rr = []
    for k in REAL:
        s = al("ali", k)
        if k == "O2_LA2_d1":
            rr.append(r"\addlinespace")
        if s is None:
            rr.append(NAMES[k] + r" & \multicolumn{6}{c}{--} \\")
            continue
        rr.append(" & ".join([NAMES[k], f2(s.get("LAAL_s")) if s.get("LAAL_s") else "--",
                              f2(s["cpBLEU_session"]) + sig_mark("ali", k), f2(s["BLEU_speaker_blind"]),
                              pct(s["count_exact"]), f1(s["DER_output"]["DER"]), f1(s["DER_output"]["conf"])]) + r" \\")
    da = DS.get("ali", {}).get("test")
    table(GEN / "tab_real.tex", r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{lcccccc}
\toprule
System & LAAL & cpBLEU$_\text{sess}$ & BLEU$_\text{blind}$ & Spk & DER & Conf. \\
\midrule
""" + "\n".join(rr) + r"""
\bottomrule
\end{tabular}
\caption{Real Mandarin meetings: Zh$\rightarrow$En on the AliMeeting test set (""" + (thousands(da["windows"]) if da else "--")
        + r""" windows, 20 meetings, headset channels summed), never seen in training. References are Hy-MT2
translations of the human transcripts; speakers and times are human annotations. Simultaneous rows at
$\Delta=1$\,s; --: not completed. $^\dagger$ as in Table~\ref{tab:main}.}
\label{tab:real}
\end{table}
""")

    # full metrics per direction (appendix)
    for dr in DIRS:
        fr = []
        for k in ["B-off", "C-van", "C-ft2", "O2-nosd", "O2", "O2-oracle", "B-LA2", "O2_LA2_d1", HEAD]:
            s = al(dr, k)
            if s is None:
                continue
            fr.append(" & ".join([NAMES[k], f2(s["cpBLEU_session"]), f2(s["cpBLEU_window"]), f2(s.get("cpchrF_session")),
                                  f2(s["BLEU_speaker_blind"]), f2(LR[dr][k]), pct(s["count_exact"]),
                                  f1(s["DER_output"]["DER"]), f1(s["DER_output"]["conf"]), pct(s.get("empty"))]) + r" \\")
        table(GEN / f"tab_full_{dr}.tex", r"""\begin{table*}[t]
\centering\small
\resizebox{\textwidth}{!}{\begin{tabular}{lccccccccc}
\toprule
System & cpBLEU$_\text{sess}$ & cpBLEU$_\text{win}$ & cpchrF$_\text{sess}$ & BLEU$_\text{blind}$ & Len & Spk & DER & Conf. & Empty \\
\midrule
""" + "\n".join(fr) + r"""
\bottomrule
\end{tabular}
\caption{All text metrics, """ + ARROW[dr] + r""". cpBLEU$_\text{win}$ matches speakers once per window; Len: output
over reference length (""" + ("characters" if dr == "en2zh" else "words") + r"""); Conf.: speaker-confusion part of
DER; Empty: windows with no output.}
\label{tab:full-""" + dr + r"""}
\end{table*}
""")

    # latency-quality trade-off: table (appendix) and figure (main text)
    TR = [("B-LA2", None), ("O2_LA2_d1", None), ("O2_LA2_d2", None), ("S2dl_d0.5", None), (HEAD, None), ("S2dl_d2", None),
          ("S2dlm_d0.5", None), ("S2dlm_d1", None), ("S2dlm_d2", None), ("S2dl1k_d1", None)]
    trows = []
    for k, _ in TR:
        c = []
        for dr in DIRS:
            s = al(dr, k)
            c += ([f2(s.get("LAAL_s")), f2(s.get("first_commit_s")), f2(s["cpBLEU_session"]), f2(s["BLEU_speaker_blind"]),
                   f2(LR[dr][k])] if s else [r"\multicolumn{5}{c}{--}"])
        name = {"O2_LA2_d1": r"\sys{} (offline) + LA2, $\Delta{=}1$\,s", HEAD: r"\sys{}-Simul, $\Delta{=}1$\,s",
                "B-LA2": r"Qwen3-Omni + LA2, $\Delta{=}1$\,s"}.get(k, NAMES[k])
        trows.append(" & ".join([name] + c) + r" \\")
    table(GEN / "tab_tradeoff.tex", r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{\begin{tabular}{lccccc@{\hskip 10pt}ccccc}
\toprule
 & \multicolumn{5}{c}{En$\rightarrow$Zh} & \multicolumn{5}{c}{Zh$\rightarrow$En} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
System & LAAL & First & cpBLEU$_\text{sess}$ & BLEU$_\text{blind}$ & Len & LAAL & First & cpBLEU$_\text{sess}$ & BLEU$_\text{blind}$ & Len \\
\midrule
""" + "\n".join(trows) + r"""
\bottomrule
\end{tabular}
\caption{Latency and quality of the simultaneous systems at read intervals $\Delta$ (\sys{}-Simul is trained with
reads every 1\,s; \sys{}-Simul-M with reads every 0.5, 1 or 2\,s, drawn per training window, En$\rightarrow$Zh only;
the last row is the 1{,}000-step Zh$\rightarrow$En model of our previous version; Zh$\rightarrow$En at 0.5\,s: 1{,}419 of the 3{,}317 windows were decoded with batch 4 instead of 16 after a memory stop). LAAL and \emph{First} (delay of the first commit after the first
source utterance starts) in seconds. Len: output over reference length (characters for Chinese, words for
English).}
\label{tab:tradeoff}
\end{table*}
""")

    def coords(dr, keys):
        pts = []
        for k in keys:
            s = al(dr, k)
            if s and s.get("LAAL_s"):
                pts.append(f"({s['LAAL_s']:.3f},{s['cpBLEU_session']:.2f})")
        return " ".join(pts)
    series = [("S2dl", ["S2dl_d0.5", HEAD, "S2dl_d2"], "blue, mark=*", r"\sys{}-Simul ($\Delta$)"),
              ("S2dlm", ["S2dlm_d0.5", "S2dlm_d1", "S2dlm_d2"], "teal, mark=square*", r"\sys{}-Simul-M ($\Delta$)"),
              ("O2 LA2", ["O2_LA2_d1", "O2_LA2_d2"], "red, mark=triangle*", r"\sys{} + LA2 ($\Delta$)"),
              ("B LA2", ["B-LA2"], "gray, only marks, mark=x", r"Qwen3-Omni + LA2")]
    pics = []
    for i, dr in enumerate(DIRS):
        plots = []
        for _, keys, style, label in series:
            c = coords(dr, keys)
            if c:
                plots.append(rf"\addplot[{style}] coordinates {{{c}}};" + (rf" \addlegendentry{{{label}}}" if i == 0 else ""))
        off = al(dr, "O2")
        if off:
            plots.append(rf"\addplot[black, densely dotted, domain=0:4, samples=2] {{{off['cpBLEU_session']:.2f}}};"
                         + (r" \addlegendentry{\sys{} offline}" if i == 0 else ""))
        legend = (r", legend style={font=\scriptsize, at={(0.98,0.02)}, anchor=south east, draw=none, fill=white, "
                  r"fill opacity=0.8, text opacity=1}" if i == 0 else "")
        pics.append(r"\begin{tikzpicture}\begin{axis}[width=0.52\textwidth, height=4.6cm, xlabel={LAAL (s)}, "
                    r"ylabel={cpBLEU$_\text{sess}$}, title={" + ARROW[dr] + r"}, title style={font=\small}, "
                    r"tick label style={font=\scriptsize}, label style={font=\small}, xmin=0, xmax=4, enlarge y limits=0.12, "
                    r"grid=major, grid style={gray!20}" + legend + "]\n" + "\n".join(plots) + "\n" + r"\end{axis}\end{tikzpicture}")
    (GEN / "fig_tradeoff.tex").write_text(r"""\begin{figure*}[t]
\centering
""" + "\n\\hfill\n".join(pics) + r"""
\caption{Latency--quality trade-off (text) over the read interval $\Delta\in\{0.5,1,2\}$\,s (left to right).
\sys{}-Simul is trained at $\Delta=1$\,s, \sys{}-Simul-M at mixed intervals (En$\rightarrow$Zh). The dotted line is the offline \sys{}. Numbers in Table~\ref{tab:tradeoff}.}
\label{fig:tradeoff}
\end{figure*}
""", encoding="utf-8")

    # ablation: read targets (both directions), with the text-side shape of the commits
    ab = []
    for dr in DIRS:
        ab.append(r"\multicolumn{6}{l}{\emph{" + ARROW[dr] + r"}} \\")
        for k in (HEAD, LABELS):
            s, b = al(dr, k), BL[dr][k]
            if s is None:
                continue
            name = {HEAD: r"Deadline labels (\sys{}-Simul)", LABELS: r"Stable-prefix labels"}[k]
            ab.append(" & ".join([name, f2(s.get("LAAL_s")), f2(s["cpBLEU_session"]), f1(b.get("units_per_block")),
                                  pct(b.get("single_unit")), pct(b.get("mid_stop"))]) + r" \\")
    table(GEN / "tab_ablation.tex", r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{lccccc}
\toprule
 & LAAL & cpBLEU$_\text{sess}$ & Units/blk & 1-unit\,\% & Mid\,\% \\
\midrule
""" + "\n".join(ab) + r"""
\bottomrule
\end{tabular}
\caption{Read targets at $\Delta=1$\,s, both continued from the same offline model for 3{,}000 steps. Stable-prefix
labels: at each read, the prefix of the model's own draft that it also writes after hearing the whole window, decoded
with the same plain wait rule. Units/blk: mean characters (Zh) or words (En) per committed block; 1-unit: blocks of a
single unit; Mid: blocks that stop mid-phrase before the same speaker's next block.}
\label{tab:ablation}
\end{table}
""")

    # speech (one column: the main cpASR, voice and naturalness measures; En->Zh only where Zh->En is unfinished)
    SHORT = {"B-off": "Prompted", "C-ft2": "Cascade (same data)", "O2": r"\sys{} (offline)", "O2_LA2_d1": r"\sys{} + LA2",
             HEAD: r"\sys{}-Simul"}
    srows = []
    for dr in DIRS:
        if not any((SP[dr][k] or {}).get("all") for k in SHORT):
            srows.append(r"\multicolumn{7}{l}{\emph{" + ARROW[dr] + r": recognition not completed}} \\")
            continue
        srows.append(r"\multicolumn{7}{l}{\emph{" + ARROW[dr] + r"}} \\")
        for k in SHORT:
            sp = (SP[dr][k] or {}).get("all")
            if sp is None:
                continue
            vo, sx = VO[dr][k] or {}, SX[dr][k]
            stream = k in ("O2_LA2_d1", HEAD)
            srows.append(" & ".join([SHORT[k], f2(sp.get("cpASR_BLEU_session")), f2(vo.get("SIM_mean")),
                                     f1(100 * vo["confusion_rate"]) if vo.get("confusion_rate") is not None else "--",
                                     f1(100 * sx["misread"]) if sx.get("misread") is not None else "--",
                                     f2(sx.get("utmos")), f2(sx.get("abrupt_per_min")) if stream else "--"]) + r" \\")
    sep = {dr: (VO[dr]["O2"] or {}).get("ref_ref_cos_mean") for dr in DIRS}
    table(GEN / "tab_speech.tex", r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{lcccccc}
\toprule
System & cpASR$\uparrow$ & SIM$\uparrow$ & V-conf$\downarrow$ & Misr.$\downarrow$ & UTMOS$\uparrow$ & Abr.$\downarrow$ \\
\midrule
""" + "\n".join(srows) + r"""
\bottomrule
\end{tabular}
\caption{Speech of the UniSS Talker: offline systems dubbed, simultaneous ones spoken block by block on one channel with
causal voice prompts. cpASR: cpBLEU$_\text{sess}$ of the ASR transcripts; SIM: WavLM cosine to the attributed speaker
(two speakers of a meeting: """ + f2(sep["en2zh"]) + r"""); V-conf: output speakers closer to another speaker of the window
(\%); Misr.: CER of the transcript against the Talker's input (\%); Abr.: abrupt pauses per minute.}
\label{tab:speech}
\end{table}
""")

    # diarizer-defined En->Zh windows (appendix): v4 models; the prompted base model is the v2 run (same model, windows)
    from systems import EV
    V2B = EV / "gstar_v2b_20261028"
    v4diar = (V5 / "en2zh_D-O2" / "score.json").exists()   # the bidirectional models on these windows: run in v5 (make_v5_runs.py p5)
    DSYS = [("B-off", V2B / "D-B-off"), ("C-ft2", V5 / "en2zh_D-C-ft2" if v4diar else V2B / "D-C-ft"),
            ("O2-nosd", V5 / "en2zh_D-O2-nosd" if v4diar else V2B / "D-O-nosd"), ("O2", V5 / "en2zh_D-O2" if v4diar else V2B / "D-O-G1")]
    dr_rows, DIARW = [], {}
    for k, d in DSYS:
        sc = load(d / "score.json")
        if sc is None:
            continue
        a = sc["all"]
        DIARW[k] = a
        dr_rows.append(" & ".join([NAMES[k], f2(a["cpBLEU_session"]), f2(a["BLEU_speaker_blind"]), pct(a["count_exact"]),
                                   f1(a["DER_output"]["DER"])]) + r" \\")
    nwin = next(iter(DIARW.values()))["windows"] if DIARW else 0
    table(GEN / "tab_diar.tex", r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{lcccc}
\toprule
System & cpBLEU$_\text{sess}$ & BLEU$_\text{blind}$ & Spk-count & DER \\
\midrule
""" + "\n".join(dr_rows) + r"""
\bottomrule
\end{tabular}
\caption{En$\rightarrow$Zh test windows opened and closed by the diarizer itself (""" + thousands(nwin) + r"""
windows that contain whole utterances), offline.""" + ("" if v4diar else r""" Models of our earlier En$\rightarrow$Zh-only version
(the same recipe trained on En$\rightarrow$Zh alone); this evaluation was not rerun for the bidirectional models.""") + r"""}
\label{tab:diar}
\end{table}
""")
    nums["diar_windows"] = DIARW

    # single vs multi (appendix)
    g = []
    for dr in DIRS:
        g.append(r"\multicolumn{6}{l}{\emph{" + ARROW[dr] + r"}} \\")
        for k in ("B-off", "C-ft2", "O2-nosd", "O2", "O2-oracle", "O2_LA2_d1", HEAD):
            if not S[dr][k]:
                continue
            s1, sm = S[dr][k]["single_spk"], S[dr][k]["multi_spk"]
            g.append(" & ".join([NAMES[k], f2(s1["cpBLEU_session"]), f2(sm["cpBLEU_session"]), f1(s1["DER_output"]["DER"]),
                                 f1(sm["DER_output"]["DER"]), pct(sm["count_exact"])]) + r" \\")
    table(GEN / "tab_groups.tex", r"""\begin{table}[t]
\centering\small
\resizebox{\columnwidth}{!}{\begin{tabular}{lccccc}
\toprule
 & \multicolumn{2}{c}{cpBLEU$_\text{sess}$} & \multicolumn{2}{c}{DER} & Spk-count \\
System & single & multi & single & multi & multi \\
\midrule
""" + "\n".join(g) + r"""
\bottomrule
\end{tabular}
\caption{Single- and multi-speaker test windows.}
\label{tab:groups}
\end{table}
""")

    # error analysis (appendix)
    e = []
    for dr in DIRS:
        e.append(r"\multicolumn{7}{l}{\emph{" + ARROW[dr] + r"}} \\")
        for k in ("O2-nosd", "O2", "O2-oracle", "O2_LA2_d1", HEAD):
            a = (A[dr][k] or {}).get("multi_spk")
            if a is None:
                continue
            e.append(" & ".join([NAMES[k], pct(a["count_under"]), pct(a["count_over"]), pct(a["seg_slot_wrong"]),
                                 pct(a["ref_omitted"]), pct(a["seg_inserted"]), f2(a["start_err_s"])]) + r" \\")
    table(GEN / "tab_errors.tex", r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.4pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{lcccccc}
\toprule
System & Under & Over & Wrong & Omit & Insert & Start\,s \\
\midrule
""" + "\n".join(e) + r"""
\bottomrule
\end{tabular}
\caption{Errors on the multi-speaker test windows (\%). \emph{Under}/\emph{Over}: windows with too few / too many
speakers; \emph{Wrong}: segments whose speaker differs from that of the reference utterance they align to;
\emph{Omit}: reference utterances no segment aligns to; \emph{Insert}: segments aligned to no reference; \emph{Start}:
mean start-time error (s), per committed chunk for the simultaneous systems. Slots are those the targets were built
with (offline or low-latency preset). Without the speaker stream, \emph{Wrong} is measured against slots the model never sees.}
\label{tab:errors}
\end{table}
""")

    # significance (appendix)
    def pname(key):
        speech = key.startswith("sp:")
        dr, k = key.removeprefix("sp:").split(":")
        return NAMES[k] + (" (speech)" if speech else "")
    srow = []
    for dr in DIRS:
        srow.append(r"\multicolumn{6}{l}{\emph{" + ARROW[dr] + r"}} \\")
        for key, r in BS.items():
            if r["a"].removeprefix("sp:").split(":")[0] != dr:
                continue
            srow.append(" & ".join([pname(r["a"]) + " vs.\\ " + pname(r["b"]), neg(f2(r["diff"])),
                                    "[" + neg(f2(r["window"]["ci95"][0])) + ", " + neg(f2(r["window"]["ci95"][1])) + "]",
                                    f"{r['window']['p']:.3f}",
                                    "[" + neg(f2(r["meeting"]["ci95"][0])) + ", " + neg(f2(r["meeting"]["ci95"][1])) + "]",
                                    f"{r['meeting']['p']:.3f}"]) + r" \\")
    table(GEN / "tab_sig.tex", r"""\begin{table*}[t]
\centering\small
\resizebox{\textwidth}{!}{\begin{tabular}{lccccc}
\toprule
Comparison & $\Delta$cpBLEU$_\text{sess}$ & 95\% CI (windows) & $p$ & 95\% CI (meetings) & $p$ \\
\midrule
""" + "\n".join(srow) + r"""
\bottomrule
\end{tabular}
\caption{Paired bootstrap (1{,}000 resamples) of cpBLEU$_\text{sess}$ differences (cpASR-BLEU$_\text{sess}$ for
speech), resampling windows or whole meetings; $p$ is the share of resamples in which the first system is not better.}
\label{tab:sig}
\end{table*}
""")

    # macros for numbers quoted in the prose: \num{tx:en2zh:O2:cp} etc.
    mac = [r"\providecommand{\num}[1]{\csname n@#1\endcsname}"]

    def put(key, v, fmt="{:.2f}"):
        if v is not None:
            mac.append(rf"\expandafter\def\csname n@{key}\endcsname{{{fmt.format(v)}}}")
    for dr in DS:
        for sp in DS[dr]:
            put(f"data:{dr}:{sp}:windows", DS[dr][sp]["windows"], "{:,}")
            put(f"data:{dr}:{sp}:hours", DS[dr][sp]["hours"], "{:.1f}")
            put(f"data:{dr}:{sp}:multi", 100 * DS[dr][sp]["multi"] / DS[dr][sp]["windows"], "{:.1f}")
        for k in NAMES:
            p = f"{dr}:{k}"
            s = S[dr][k]
            if s:
                a = s["all"]
                put(f"tx:{p}:cp", a["cpBLEU_session"])
                put(f"tx:{p}:win", a["cpBLEU_window"])
                put(f"tx:{p}:blind", a["BLEU_speaker_blind"])
                put(f"tx:{p}:spk", 100 * a["count_exact"], "{:.1f}")
                put(f"tx:{p}:der", a["DER_output"]["DER"], "{:.1f}")
                put(f"tx:{p}:conf", a["DER_output"]["conf"], "{:.1f}")
                put(f"tx:{p}:laal", a.get("LAAL_s"))
                put(f"tx:{p}:first", a.get("first_commit_s"))
                put(f"tx:{p}:multi", s["multi_spk"]["cpBLEU_session"])
                put(f"tx:{p}:single", s["single_spk"]["cpBLEU_session"])
                put(f"tx:{p}:spkmulti", 100 * s["multi_spk"]["count_exact"], "{:.1f}")
            if SP[dr][k]:
                a = SP[dr][k]["all"]
                put(f"sp:{p}:cp", a.get("cpASR_BLEU_session"))
                put(f"sp:{p}:blind", a.get("ASR_BLEU_speaker_blind"))
                if s and a.get("cpASR_BLEU_session") is not None:
                    put(f"sp:{p}:gap", s["all"]["cpBLEU_session"] - a["cpASR_BLEU_session"])
            if VO[dr][k]:
                put(f"vo:{p}:sim", VO[dr][k]["SIM_mean"])
                put(f"vo:{p}:conf", 100 * VO[dr][k]["confusion_rate"], "{:.1f}")
            for m, v in SX[dr][k].items():
                put(f"sx:{p}:{m}", (100 * v if m in ("misread", "single_unit") else v) if v is not None else None)
            for m, v in BL[dr][k].items():
                put(f"bl:{p}:{m}", 100 * v if m in ("single_unit", "mid_stop") else v, "{:.1f}" if m != "blocks" else "{:d}")
            put(f"len:{p}", LR[dr][k])
            put(f"rtf:{p}", RT[dr][k])

        def diff(a, b, key, m="cpBLEU_session"):
            if al(dr, a) and al(dr, b) and al(dr, a).get(m) is not None and al(dr, b).get(m) is not None:
                put(key, al(dr, a)[m] - al(dr, b)[m])
        diff("O2", "C-ft2", f"gain:{dr}:cas")
        diff("O2", "C-van", f"gain:{dr}:casvan")
        diff("O2", "B-off", f"gain:{dr}:prompt")
        diff("O2", "O2-nosd", f"gain:{dr}:sd")
        diff("O2-oracle", "O2", f"gain:{dr}:oracle")
        diff(HEAD, "O2_LA2_d1", f"gain:{dr}:simul")
        diff("O2", HEAD, f"cost:{dr}:simul")
        diff(HEAD, "O2_LA2_d1", f"lat:{dr}:simul", "LAAL_s")
        diff(HEAD, LABELS, f"gain:{dr}:labels")
        diff("S2dl_d2", "O2_LA2_d2", f"gain:{dr}:simul2")
        diff(HEAD, "S2dl_d2", f"cost:{dr}:d2")
        diff("S2dl_d2", HEAD, f"lat:{dr}:d2", "LAAL_s")
        diff(HEAD, "S2dlm_d0.5", f"lat:{dr}:m05", "LAAL_s")   # MIST-Simul (1 s) minus MIST-Simul-M (0.5 s)
        diff("S2dlm_d0.5", HEAD, f"gain:{dr}:m05")
    for dr in DIRS:
        for k in NAMES:
            a = (A[dr].get(k) or {}).get("multi_spk")
            if a:
                put(f"er:{dr}:{k}:under", 100 * a["count_under"], "{:.1f}")
                put(f"er:{dr}:{k}:omit", 100 * a["ref_omitted"], "{:.1f}")
                put(f"er:{dr}:{k}:insert", 100 * a["seg_inserted"], "{:.1f}")
                put(f"er:{dr}:{k}:wrong", 100 * a["seg_slot_wrong"], "{:.1f}")
    for k, a in DIARW.items():
        put(f"dw:{k}:cp", a["cpBLEU_session"])
    for key, r in BS.items():
        put(f"bs:{key}:diff", r["diff"])
        put(f"bs:{key}:pw", "$<$0.001" if r["window"]["p"] < 0.001 else f'{r["window"]["p"]:.3f}', "{}")   # review M14
        put(f"bs:{key}:pm", "$<$0.001" if r["meeting"]["p"] < 0.001 else f'{r["meeting"]["p"]:.3f}', "{}")
    (GEN / "macros.tex").write_text("\n".join(mac) + "\n", encoding="utf-8")
    (GEN / "numbers.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False), encoding="utf-8")
    print("missing text:", {dr: [k for k in NAMES if not S[dr][k] and True]
                            for dr in ALL})
    print("missing speech:", {dr: [k for k in ("B-off", "C-ft2", "O2", "O2_LA2_d1", HEAD) if not SP[dr][k]] for dr in DIRS})


if __name__ == "__main__":
    main()
