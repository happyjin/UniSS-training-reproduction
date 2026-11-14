"""One overlapped test window as a worked example (Appendix "Example"): the reference and the outputs of the same-data
cascade and of MIST (offline), each as speaker / time / translation. The window is picked automatically: two to three
reference speakers with overlap, a short reference, MIST with the right speaker count and the cascade without it,
then the highest sentence chrF of MIST among those. Speaker labels are mapped to reference speakers per meeting as
in cpBLEU_sess, and printed as A, B, C in order of first appearance in the reference.

    python scripts/make_example.py   (writes generated/example.tex)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match  # noqa: E402

EV = ROOT / "eval_outputs/uniss_qwen3_omni_v1"
WIN = ROOT / "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl"
SYS = {"Cascade (same data)": EV / "gstar_cascade_20261028/C-ft/hyp.jsonl",
       r"\sys{} (offline)": EV / "gstar_offline_eval_o1_20261026T195403Z/O-G1/hyp.jsonl"}
GEN = Path(__file__).resolve().parents[1] / "generated"


def esc(s: str) -> str:
    return s.replace("\\", "").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


def zh(s: str) -> str:
    """Chinese with a break opportunity after every character (the engine has no Chinese line breaking)."""
    return r"\zh{" + r"\hspace{0pt}".join(esc(s)) + "}"


def main() -> None:
    from sacrebleu.metrics import CHRF

    W = {w["id"]: w for w in map(json.loads, open(WIN, encoding="utf-8"))}
    H = {n: {h["id"]: h for h in map(json.loads, open(p, encoding="utf-8"))} for n, p in SYS.items()}
    maps = {}
    for n, hs in H.items():   # session-level matching per system
        hm, rm = defaultdict(lambda: defaultdict(str)), defaultdict(lambda: defaultdict(str))
        for wid, w in W.items():
            for s in hs[wid]["segments"]:
                hm[w["meeting"]][s["spk"]] += s["text"]
            for u in w["utts"]:
                if not u["hesitation"]:
                    rm[w["meeting"]][u["speaker_id"]] += u["zh"]
        maps[n] = {m: match(dict(hm[m]), dict(rm[m])) for m in hm}
    chrf = CHRF()
    mist, cas = r"\sys{} (offline)", "Cascade (same data)"
    best = None
    for wid, w in W.items():
        talk = [u for u in w["utts"] if not u["hesitation"]]
        spk = {u["speaker_id"] for u in talk}
        ref = "".join(u["zh"] for u in sorted(talk, key=lambda u: u["begin_s"]))
        sp = [(u["begin_s"], u["end_s"]) for u in talk]
        ov = any(min(a[1], b[1]) > max(a[0], b[0]) for i, a in enumerate(sp) for b in sp[i + 1:])
        if not (2 <= len(spk) <= 3 and ov and 25 <= len(ref) <= 70 and len(talk) <= 4):
            continue
        n_m = len({s["spk"] for s in H[mist][wid]["segments"]})
        n_c = len({s["spk"] for s in H[cas][wid]["segments"]})
        if n_m != len(spk) or n_c == len(spk):
            continue
        # every MIST speaker must map to the reference speaker of its utterances
        mp = maps[mist][w["meeting"]]
        if not all(mp.get(s["spk"]) in spk for s in H[mist][wid]["segments"]):
            continue
        sc = chrf.sentence_score("".join(s["text"] for s in H[mist][wid]["segments"]), [ref]).score
        if best is None or sc > best[0]:
            best = (sc, wid)
    wid = best[1]
    w = W[wid]
    talk = sorted((u for u in w["utts"] if not u["hesitation"]), key=lambda u: u["begin_s"])
    letter = {}
    for u in talk:
        letter.setdefault(u["speaker_id"], "ABCDEFGH"[len(letter)])
    lines = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{lp{0.78\textwidth}}", r"\toprule"]
    lines.append(r"English source & " + " ".join(
        rf"\textbf{{{letter[u['speaker_id']]}}} [{u['begin_s']:.1f}--{u['end_s']:.1f}\,s] {esc(u['en'].lower())}"
        for u in talk) + r" \\")
    lines.append(r"Reference & " + " ".join(
        rf"\textbf{{{letter[u['speaker_id']]}}} [{u['begin_s']:.1f}] {zh(u['zh'])}" for u in talk) + r" \\")
    for n in SYS:
        mp = maps[n][w["meeting"]]
        segs = sorted(H[n][wid]["segments"], key=lambda s: s["start_s"])
        lines.append(r"\midrule" if n == cas else "")
        lines.append(n + " & " + " ".join(
            rf"\textbf{{{letter.get(mp.get(s['spk']), '?')}}} [{s['start_s']:.1f}] {zh(s['text'])}" for s in segs)
            + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Test window \texttt{" + esc(wid) + r"} with overlapping speech. Speakers are lettered by first "
              r"appearance in the reference; output speakers are mapped to reference speakers by the meeting-level "
              r"matching of cpBLEU$_\text{sess}$ (``?'' = unmatched). Times in seconds from the window start.}",
              r"\label{tab:example}", r"\end{table*}"]
    (GEN / "example.tex").write_text("\n".join(l for l in lines if l) + "\n", encoding="utf-8")
    print(wid, best[0])


if __name__ == "__main__":
    main()
