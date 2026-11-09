"""CPU diagnostic (read-only): where does text -> speech (TTS + speech-only ASR) lose BLEU, per system?

For each NAME=SPEECH_DIR it reads <speech>/talker/asr_speech_merged.jsonl (per window x output speaker stream:
translated_text = what the talker was asked to say, asr_text = Whisper on the speech-only wav, blocks = TTS blocks with
commit/start/dur) and measures TTS->ASR *fidelity* (asr_text vs translated_text, so translation quality cancels):
  - corpus BLEU(asr | text) and word error rate, overall and by stream bins (words, short blocks, final-read share);
  - paired per window vs the first system: d fidelity (1 - WER) against d features (Spearman / Pearson).

    python -m experiments.mist_w3po_v1.rl.diag_speech_drop --out X.json NAME=speech_dir [NAME=speech_dir ...]
"""
import argparse
import ast
import json
import re
from collections import defaultdict

import numpy as np
import sacrebleu

_P = re.compile(r"[^\w' ]+")


def norm(s):
    return _P.sub(" ", (s or "").lower()).split()


def edit(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def load(sp):
    out = []
    for r in map(json.loads, open(f"{sp}/talker/asr_speech_merged.jsonl", encoding="utf-8")):
        b = r.get("blocks") or []
        if isinstance(b, str):
            b = ast.literal_eval(b)
        ref, hyp = norm(r.get("translated_text")), norm(r.get("asr_text"))
        nw = max(len(ref), 1)
        fw = sum(len(norm(x.get("text"))) for x in b if x.get("final"))
        short = sum(1 for x in b if x.get("dur_ms", 0) < 1000)
        delay = sum(max(0, x.get("start_ms", 0) - x.get("write_ms", 0)) for x in b) / 1000.0
        wid, spk = r["id"].rsplit("__S", 1)
        out.append(dict(id=r["id"], win=wid, ref=" ".join(ref), hyp=" ".join(hyp), nw=len(ref), err=edit(ref, hyp),
                        nb=len(b), short=short, final_share=fw / nw, dur=r.get("audio_duration_seconds") or 0.0,
                        delay=delay, empty=int(not hyp)))
    return out


def summ(rows):
    if not rows:
        return {"n": 0}
    nw = sum(r["nw"] for r in rows)
    return {"n": len(rows), "words": nw,
            "bleu": round(sacrebleu.corpus_bleu([r["hyp"] for r in rows], [[r["ref"] for r in rows]]).score, 2),
            "wer": round(sum(r["err"] for r in rows) / max(nw, 1), 4),
            "empty_asr": round(np.mean([r["empty"] for r in rows]), 4)}


def bins(rows):
    B = {
        "words<=2": lambda r: r["nw"] <= 2, "words3-5": lambda r: 3 <= r["nw"] <= 5, "words6+": lambda r: r["nw"] >= 6,
        "dur<1s": lambda r: r["dur"] < 1.0, "dur>=1s": lambda r: r["dur"] >= 1.0,
        "has_short_block": lambda r: r["short"] > 0, "no_short_block": lambda r: r["short"] == 0,
        "final_share>=0.5": lambda r: r["final_share"] >= 0.5, "final_share<0.5": lambda r: r["final_share"] < 0.5,
        "nb1": lambda r: r["nb"] == 1, "nb2+": lambda r: r["nb"] >= 2,
    }
    return {k: summ([r for r in rows if f(r)]) for k, f in B.items()}


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return {"pearson": round(float(np.corrcoef(x, y)[0, 1]), 4),
            "spearman": round(float(np.corrcoef(rank(x), rank(y))[0, 1]), 4), "n": int(len(x))}


def per_window(rows):
    W = defaultdict(lambda: dict(nw=0, err=0, nb=0, short=0, fw=0.0, streams=0, short_streams=0, delay=0.0))
    for r in rows:
        w = W[r["win"]]
        w["nw"] += r["nw"]; w["err"] += r["err"]; w["nb"] += r["nb"]; w["short"] += r["short"]
        w["fw"] += r["final_share"] * r["nw"]; w["streams"] += 1; w["short_streams"] += int(r["dur"] < 1.0)
        w["delay"] += r["delay"]
    for w in W.values():
        w["fid"] = 1.0 - min(w["err"] / max(w["nw"], 1), 1.5)
        w["final_share"] = w["fw"] / max(w["nw"], 1)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("systems", nargs="+")
    a = ap.parse_args()
    S = {}
    for s in a.systems:
        k, v = s.split("=", 1)
        S[k] = load(v)
    res = {"systems": {}}
    for k, rows in S.items():
        res["systems"][k] = {"all": summ(rows), "bins": bins(rows),
                             "mean": {f: round(float(np.mean([r[f] for r in rows])), 4)
                                      for f in ("nw", "nb", "short", "final_share", "dur", "delay")},
                             "streams_dur<1s_share": round(float(np.mean([r["dur"] < 1.0 for r in rows])), 4)}
    names = list(S)
    base = per_window(S[names[0]])
    res["paired_vs_" + names[0]] = {}
    for k in names[1:]:
        W = per_window(S[k])
        ids = sorted(set(base) & set(W))
        d = {f: [W[i][f] - base[i][f] for i in ids] for f in ("fid", "final_share", "short", "nb", "streams",
                                                                   "short_streams", "nw", "delay")}
        res["paired_vs_" + names[0]][k] = {"n_windows": len(ids), "mean_dfid": round(float(np.mean(d["fid"])), 4),
                                            **{f"corr_dfid_d{f}": corr(d["fid"], d[f]) for f in d if f != "fid"}}
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v["all"] for k, v in res["systems"].items()}))


if __name__ == "__main__":
    main()
