"""Paper v6 statistics (prereg_v6 §0; review M12): paired comparisons of cpBLEU_sess (and of whole-meeting tcpBLEU).

For each pair (a better than b?) on the windows both systems decoded:
  diff            full-test difference
  meeting         cluster bootstrap over meetings (1,000 resamples). The slot-to-speaker matching is made per meeting,
                  so resampling whole meetings re-uses exactly the matching a re-match would find
  window          bootstrap over windows (1,000) with the matching fixed on the full test set (v5's statistic) and, for
                  --rematch pairs, 200 resamples that redo the per-meeting matching on the resampled windows
  perm            paired permutation test by meeting: each meeting's two outputs swapped at random (10,000 draws)
  lomo            the difference with each meeting left out (min, max)
  per_meeting     the difference per meeting
Two-sided p: twice the smaller tail (resampled differences <= 0 or >= 0), capped at 1. Speech pairs ("sp:") score the
ASR of the synthesised speech with the text's matching, as before. Online pairs ("on:<dir>:<a>|<b>") compare the
per-meeting tcpBLEU statistics of two meeting_score.py outputs. Holm over the pre-registered family (prereg_v6).

    python scripts/bootstrap_v6.py           (writes generated/bootstrap_v6.json; existing keys are kept)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systems import ASR_FILE, SPEECH_DIR  # noqa: E402
from systems_v6 import ROOT, SP6, V6, hyp_dir, windows  # noqa: E402

sys.path.insert(0, str(ROOT))
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match, pairs_of  # noqa: E402

B, BP, BR = 1000, 10000, 200
GEN = Path(__file__).resolve().parents[1] / "generated"


def tok(dr):
    return "13a" if dr in ("zh2en", "ali") else "zh"


def bleu_of(dr):
    from sacrebleu.metrics import BLEU
    return BLEU(tokenize=tok(dr))


def window_data(dr, d, win, asr_file=None):
    sep = " " if dr in ("zh2en", "ali") else ""
    W = [json.loads(l) for l in open(win, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(d / "hyp.jsonl", encoding="utf-8"))}
    W = [w for w in W if w["id"] in H]
    asr = ({r["id"]: r.get("asr_text", "") for r in map(json.loads, open(asr_file, encoding="utf-8"))}
           if asr_file else None)
    out = {}
    for w in W:
        h, r = defaultdict(str), defaultdict(str)
        for s in H[w["id"]]["segments"]:
            h[s["spk"]] += sep + s["text"]
        for u in sorted(w["utts"], key=lambda u: u["begin_s"]):
            if not u["hesitation"]:
                r[u["speaker_id"]] += sep + u.get("ref", u["zh"]).strip()
        hs = {k: asr.get(f"{w['id']}__S{k}", "") for k in h} if asr is not None else dict(h)
        out[w["id"]] = {"meeting": w["meeting"], "hyp": dict(h), "ref": dict(r), "say": hs}
    return out


def stats(dr, data, ids, smap):
    bleu = bleu_of(dr)
    X = []
    for i in ids:
        x = data[i]
        ps = pairs_of(x["say"], x["ref"], smap[x["meeting"]])
        st = bleu._extract_corpus_statistics([a for a, _ in ps], [[b for _, b in ps]])
        X.append(np.sum(np.array(st, dtype=np.float64), 0) if st else None)
    dim = next(s for s in X if s is not None).shape[0]
    return np.stack([s if s is not None else np.zeros(dim) for s in X])


def smap_of(data, ids):
    hm, rm = defaultdict(lambda: defaultdict(str)), defaultdict(lambda: defaultdict(str))
    for i in ids:
        x = data[i]
        for k, t in x["hyp"].items():
            hm[x["meeting"]][k] += t
        for k, t in x["ref"].items():
            rm[x["meeting"]][k] += t
    return {m: match(dict(hm[m]), dict(rm[m])) for m in rm}


def score(dr, s):
    return bleu_of(dr)._compute_score_from_stats(list(s)).score


def two_sided(ds):
    ds = np.asarray(ds)
    return round(float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))), 4)


def compare(dr, Xa, Xb, meet, rng, rematch=None):
    full = score(dr, Xa.sum(0)) - score(dr, Xb.sum(0))
    ms = np.unique(meet)
    idx = {m: np.flatnonzero(meet == m) for m in ms}
    r = {"diff": round(full, 2), "windows": int(len(meet)), "meetings": int(len(ms))}
    ds = []
    for _ in range(B):
        j = np.concatenate([idx[m] for m in rng.choice(ms, len(ms))])
        ds.append(score(dr, Xa[j].sum(0)) - score(dr, Xb[j].sum(0)))
    r["meeting"] = {"ci95": [round(float(np.percentile(ds, 2.5)), 2), round(float(np.percentile(ds, 97.5)), 2)],
                    "p_one_sided": round(float((np.array(ds) <= 0).mean()), 4), "p_two_sided": two_sided(ds)}
    ds = []
    for _ in range(B):
        j = rng.integers(0, len(meet), len(meet))
        ds.append(score(dr, Xa[j].sum(0)) - score(dr, Xb[j].sum(0)))
    r["window"] = {"ci95": [round(float(np.percentile(ds, 2.5)), 2), round(float(np.percentile(ds, 97.5)), 2)],
                   "p_one_sided": round(float((np.array(ds) <= 0).mean()), 4), "p_two_sided": two_sided(ds)}
    Sa = {m: Xa[idx[m]].sum(0) for m in ms}
    Sb = {m: Xb[idx[m]].sum(0) for m in ms}
    obs = abs(full)
    cnt = 0
    for _ in range(BP):
        flip = rng.random(len(ms)) < 0.5
        A = sum((Sb[m] if f else Sa[m]) for m, f in zip(ms, flip))
        Bm = sum((Sa[m] if f else Sb[m]) for m, f in zip(ms, flip))
        cnt += abs(score(dr, A) - score(dr, Bm)) >= obs - 1e-9
    r["perm_p_two_sided"] = round((cnt + 1) / (BP + 1), 4)
    lomo = []
    for m in ms:
        keep = meet != m
        lomo.append(score(dr, Xa[keep].sum(0)) - score(dr, Xb[keep].sum(0)))
    r["lomo"] = [round(min(lomo), 2), round(max(lomo), 2)]
    r["per_meeting"] = {str(m): round(score(dr, Sa[m]) - score(dr, Sb[m]), 2) for m in ms}
    if rematch is not None:   # window bootstrap that redoes the matching on every resample
        (da, db, ids) = rematch
        ds = []
        for _ in range(BR):
            j = rng.integers(0, len(ids), len(ids))
            sid = [ids[k] for k in j]
            ma, mb = smap_of(da, sid), smap_of(db, sid)
            ds.append(score(dr, stats(dr, da, sid, ma).sum(0)) - score(dr, stats(dr, db, sid, mb).sum(0)))
        r["window_rematch"] = {"ci95": [round(float(np.percentile(ds, 2.5)), 2), round(float(np.percentile(ds, 97.5)), 2)],
                               "p_two_sided": two_sided(ds), "resamples": BR}
    return r


def data_of(key):
    speech = key.startswith("sp:")
    dr, name = key.removeprefix("sp:").split(":")
    d = hyp_dir(dr, name)
    if d is None or not (d / "hyp.jsonl").exists():
        return None
    asr = None
    if speech:
        sp = SPEECH_DIR(dr, name)
        cand = SP6 / f"{dr}_{name}" / "talker/asr_speech.jsonl"
        asr = cand if cand.exists() else (ASR_FILE(sp) if sp else None)
        if asr is None or not asr.exists():
            return None
    return dr, window_data(dr, d, windows(dr, name), asr)


def online_pair(dr, fa, fb, rng):
    A, Bm = json.load(open(fa))["per_meeting"], json.load(open(fb))["per_meeting"]
    ms = np.array(sorted(set(A) & set(Bm)))
    Xa = np.stack([np.array(A[m]["tcp_stats"]) for m in ms])
    Xb = np.stack([np.array(Bm[m]["tcp_stats"]) for m in ms])
    return compare(dr, Xa, Xb, ms, rng)


def main() -> None:
    from pairs_v6 import ONLINE, PAIRS, REMATCH
    f = GEN / "bootstrap_v6.json"
    res = json.load(open(f)) if f.exists() else {}
    rng = np.random.default_rng(20261031)
    cache = {}
    for a, b in PAIRS:
        key = f"{a}_vs_{b}"
        if key in res:
            continue
        for k in (a, b):
            if k not in cache:
                cache[k] = data_of(k)
        if cache[a] is None or cache[b] is None:
            print("skip", key, flush=True)
            continue
        dr, da = cache[a]
        _, db = cache[b]
        ids = sorted(set(da) & set(db))
        Xa, Xb = stats(dr, da, ids, smap_of(da, ids)), stats(dr, db, ids, smap_of(db, ids))
        meet = np.array([da[i]["meeting"] for i in ids])
        res[key] = {"a": a, "b": b, **compare(dr, Xa, Xb, meet, rng, (da, db, ids) if (a, b) in REMATCH else None)}
        print(key, json.dumps({k: v for k, v in res[key].items() if k != "per_meeting"}), flush=True)
        f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for dr, na, nb, fa, fb in ONLINE:
        key = f"on:{dr}:{na}_vs_{nb}"
        if key in res or not (Path(fa).exists() and Path(fb).exists()):
            continue
        res[key] = {"a": na, "b": nb, **online_pair(dr, fa, fb, rng)}
        print(key, json.dumps({k: v for k, v in res[key].items() if k != "per_meeting"}), flush=True)
        f.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
