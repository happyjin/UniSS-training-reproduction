"""v11 P1-4 supplement: the second recogniser's known failure modes, removed identically for every system.
Whisper-large-v3 (Chinese second ASR) loops on short clips ("鳄鱼" -> "鳄鳄鳄鳄鳄...") and emits video-outro
hallucinations ("请不吝点赞 订阅 转发"); the loops hit B's base-Talker audio about four times as often as MIST's, which
inflates the raw En->Zh second-ASR gain (+6.46 vs +0.58 with Paraformer). Rule (fixed before scoring, all sets and
systems): (1) a unit of 1-6 characters (Chinese) or 1-3 words (English) repeated >= 4 times in a row is kept once;
(2) an utterance containing an outro phrase is emptied. Transcripts -> asr2_deloop/<run>/asr2_speech.jsonl ->
hyp_asr2.jsonl (asr_hyp rule) -> scores_deloop/ -> summary_v11_p14_deloop.json (same tests as v11_summarize, Holm
over the 9 second-ASR contrasts). Refuses to overwrite.

    python -m experiments.mist_w3po_v1.evaluation.v11_asr2_deloop
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.evaluation import v11_metrics as V  # noqa: E402
from experiments.mist_w3po_v1.evaluation import v11_summarize as S  # noqa: E402

EO = REPO / "eval_outputs/mist_w3po_v1/v11_metrics"
A2, D2, SD = EO / "asr2", EO / "asr2_deloop", EO / "scores_deloop"
OUTRO = re.compile(r"点赞|订阅|转发|打赏|明镜|栏目|谢谢观看|感谢观看|字幕由|请不吝")
ZH_LOOP = re.compile(r"(.{1,6}?)\1{3,}")
EN_LOOP = re.compile(r"\b((?:\S+\s+){0,2}?\S+)(?:\s+\1\b){3,}", re.I)


def deloop(t: str, eng: bool) -> tuple[str, int]:
    if OUTRO.search(t):
        return "", 1
    n = 0
    while True:
        u = (EN_LOOP if eng else ZH_LOOP).sub(r"\1", t)
        if u == t:
            return t, n
        t, n = u, n + 1


def main() -> None:
    D2.mkdir(exist_ok=True)
    SD.mkdir(exist_ok=True)
    out = EO / "summary_v11_p14_deloop.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    changed, jobs = {}, []
    for s in S.TEST:
        eng = S.LANG_EN[s]
        b = f"{V.TXT[s]}_{V.RT_B}"
        m = f"{V.TXT[s].replace('_b', '_m')}_{V.RT_W}"
        for sysk, base, sp in (("b", b, "__t1c"), ("e2", b, "__rlplem"), ("m", m, "__rlplem")):
            for seed in ("", "_tts2"):
                run = base + sp + seed
                d = D2 / run
                d.mkdir(exist_ok=True)
                a, h = d / "asr2_speech.jsonl", d / "hyp_asr2.jsonl"
                if not a.exists():
                    nc = 0
                    with open(a, "x", encoding="utf-8") as fh:
                        for r in map(json.loads, open(A2 / run / "asr2_speech.jsonl", encoding="utf-8")):
                            t, k = deloop(r["asr_text"], eng)
                            nc += k > 0
                            fh.write(json.dumps({**r, "asr_text": t, "deloop": k}, ensure_ascii=False) + "\n")
                    changed[run] = nc
                if not h.exists():
                    subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.asr_hyp",
                                    str(REPO / V.DEC / run / "hyp.jsonl"), str(a), str(h)], check=True, cwd=REPO)
                jobs.append({"key": f"{s}:{sysk}:asr2{seed or '_tts1'}", "set": s, "dr": V.DR[s], "hyp": str(h),
                             "windows": V.WIN[s], "ref": V.REF[s], "online": True, "relabel": False, "check": ""})
    S.SC = SD
    R2 = S.score_asr2(jobs)
    res, AT = {"rule": __doc__.split("Rule")[1].split("Transcripts")[0].strip(), "changed_utts": changed, "asr2": {}}, {}
    for s in S.TEST:
        eng = S.LANG_EN[s]
        sp = {k: [R2[f"{s}:{k}:asr2_tts1"], R2[f"{s}:{k}:asr2_tts2"]] for k in ("b", "e2", "m")}
        for k, v in sp.items():
            st = [S.pm_stats(x) for x in v]
            res["asr2"][f"{s}:{k}"] = round(S.bleu_of(sum(sum(x.values()) for x in st), eng), 2)
        for x, y in (("b", "e2"), ("e2", "m"), ("b", "m")):
            AT[f"{s}:{y}-{x}"] = S.paired_bleu([S.pm_stats(r) for r in sp[x]], [S.pm_stats(r) for r in sp[y]], eng)
    for k, hh in S.holm({k: v["p"] for k, v in AT.items()}).items():
        AT[k]["p_holm"] = hh
    res["tests"] = AT
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
