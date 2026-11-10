"""How much of the speech tcpBLEU loss comes from the hallucination tail of the TTS -> ASR channel (CPU; reports/mist_w3po_v1/
metrics_first_si_analysis.zh-CN.md §4.3).

Per window x speaker stream of a talker run (<speech dir>/talker/asr_speech.*.jsonl): tokens of the text sent to the
talker (translated_text) vs of the Whisper transcript (asr_text), 13a-like tokens; the talker's speech seconds (sum of
the blocks' dur_ms) per text token. The speech hyp is then rebuilt exactly as asr_hyp.py builds it and scored with the
metric_headroom re-implementation of overlap_score (checked against the run's score_asr.json):
  base             as scored
  fix_tail_asr     streams with ASR / text tokens > 1.5 get their text back (oracle; detector uses the ASR)
  fix_tail_dur     streams whose speech seconds per text token exceed the run's p97 get their text back (detector uses
                   only the talker output, i.e. deployable before any ASR; "text back" is the upper bound of a re-draw)
  text_all         every stream gets its text back (= the text hyp merged per stream; TTS/ASR-free)
Writes <out> (json; refuses to overwrite). Dev only.

    python -m experiments.mist_w3po_v1.evaluation.metric_speech_tail --run <decode dir> --speech <speech dir> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import metric_headroom as M


def tok(t):
    return re.findall(r"\w+|[^\w\s]", t or "")


def rebuild(H, A):
    out = {}
    for wid, h in H.items():
        by = defaultdict(list)
        for s in h["segments"]:
            by[s["spk"]].append(s)
        segs = []
        for k, ss in by.items():
            t = A.get(f"{wid}__S{k}", "").strip()
            if t:
                segs.append({"spk": k, "start_s": min(s["start_s"] for s in ss), "end_s": max(s["end_s"] for s in ss),
                             "text": t, "write_ms": max(s.get("write_ms", 0) for s in ss)})
        out[wid] = {"id": wid, "segments": segs}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--speech", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from meeting_score import universe
    import ast
    rows = {}
    for f in glob.glob(str(Path(args.speech) / "talker/asr_speech.*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            rows[r["id"]] = r
    W = [json.loads(l) for l in open(M.REPO / M.WIN, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(Path(args.run) / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
    U = universe(M.DR, {w["meeting"] for w in W})
    sc = M.scored_utts(W, RW)
    st = {}
    for k, r in rows.items():
        bl = r["blocks"]
        if isinstance(bl, str):
            bl = ast.literal_eval(bl)
        tw, aw = len(tok(r["translated_text"])), len(tok(r.get("asr_text")))
        sp = sum(b.get("dur_ms", 0) for b in bl) / 1000
        st[k] = {"tw": tw, "aw": aw, "ratio": aw / max(tw, 1), "spt": sp / max(tw, 1)}
    p97 = float(np.percentile([v["spt"] for v in st.values()], 97))
    flag_asr = {k for k, v in st.items() if v["ratio"] > 1.5}
    flag_dur = {k for k, v in st.items() if v["spt"] > p97}
    asr = {k: r.get("asr_text", "") or "" for k, r in rows.items()}
    txt = {k: r["translated_text"] for k, r in rows.items()}
    extra = {k: max(0, v["aw"] - v["tw"]) for k, v in st.items()}
    tot_extra = sum(extra.values())
    res = {"run": args.run, "speech": args.speech, "streams": len(rows), "p97_speech_s_per_token": round(p97, 4),
           "asr_over_text_tokens": round(sum(v["aw"] for v in st.values()) / max(sum(v["tw"] for v in st.values()), 1), 4),
           "flag_asr": {"n": len(flag_asr), "share_of_extra_tokens": round(sum(extra[k] for k in flag_asr) / max(tot_extra, 1), 4)},
           "flag_dur": {"n": len(flag_dur), "share_of_extra_tokens": round(sum(extra[k] for k in flag_dur) / max(tot_extra, 1), 4),
                        "overlap_with_flag_asr": len(flag_dur & flag_asr)},
           "scores": {}}
    variants = {
        "base": asr,
        "fix_tail_asr": {k: (txt[k] if k in flag_asr else v) for k, v in asr.items()},
        "fix_tail_dur": {k: (txt[k] if k in flag_dur else v) for k, v in asr.items()},
        "text_all": txt,
    }
    for name, A in variants.items():
        Hs = rebuild(H, A)
        res["scores"][name] = {k: v["bleu"] for k, v in M.corpus(M.assign_all(W, Hs, sc, U)).items()}
    chk = Path(args.speech.replace("/speech/", "/decode/")) / "score_asr.json"
    if chk.exists():
        c = json.load(open(chk, encoding="utf-8"))
        res["check_scorer"] = {r: (res["scores"]["base"][r], c[f"tcpBLEU_{r}"]) for r in ("all", "ov", "1spk")}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("run", "speech")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
