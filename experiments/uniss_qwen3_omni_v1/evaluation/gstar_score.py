"""Scores of G-STAR-SI offline output (docs/multi_speaker_si_plan_v7.zh-CN.md) on reviewed windows.

  cpBLEU / cpchrF, window level   per window, hypothesis slots matched to reference speakers (Hungarian, chrF)
  cpBLEU / cpchrF, session level  per meeting one matching of slots to speakers on the whole meeting's text --
                                  the labels are meant to hold all meeting long, so this is the score of who-said-it
  BLEU, speaker-blind             the window's whole output against its whole reference
  count_exact                     windows whose number of speakers is right
  DER                             from the output segments (10 ms frames, collar 0, speech of the translated
                                  utterances), slots mapped to speakers per meeting by overlap; beside it the same
                                  DER of Nemotron-3's own posteriors (> 0.5) on the same windows
  LAAL, first_ms                  streaming output ("blocks" with write_ms): LAAL of the committed characters
                                  against the window and its reference length; delay of the first commit after
                                  the first reference utterance starts
  --asr                           the same cp scores on what the speech says: each slot's text replaced by the ASR
                                  of its talker stream "<window>__S<k>" (gstar_talker.py, asr_speech.jsonl)
Groups: all / single-speaker / multi-speaker / overlapped / strict review.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_score --windows <windows_test_ref.jsonl> --hyp <hyp.jsonl>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HOP = 0.01


def chrf_matrix(hyp: dict, ref: dict):
    from sacrebleu.metrics import CHRF

    c = CHRF()
    hs, rs = [k for k in hyp if hyp[k].strip()], [k for k in ref if ref[k].strip()]
    M = np.array([[c.sentence_score(hyp[h], [ref[r]]).score for r in rs] for h in hs]) if hs and rs else np.zeros((0, 0))
    return hs, rs, M


def match(hyp: dict, ref: dict) -> dict:
    hs, rs, M = chrf_matrix(hyp, ref)
    if not hs or not rs:
        return {}
    i, j = linear_sum_assignment(-M)
    return {hs[a]: rs[b] for a, b in zip(i, j)}


def pairs_of(hyp: dict, ref: dict, m: dict) -> list[tuple[str, str]]:
    used = set()
    out = []
    for h, t in hyp.items():
        if not t.strip():
            continue
        r = m.get(h)
        if r is not None and r in ref and ref[r].strip():
            out.append((t, ref[r]))
            used.add(r)
        else:
            out.append((t, ""))
    out += [("", t) for r, t in ref.items() if t.strip() and r not in used]
    return out


def frames(segs, dur_s):
    n = int(np.ceil(dur_s / HOP))
    act = defaultdict(lambda: np.zeros(n, dtype=bool))
    for k, a, b in segs:
        act[k][int(a / HOP): int(np.ceil(b / HOP))] = True
    return dict(act)


def der(windows_by_meeting: dict, hyp_act_of) -> dict:
    """DER over the given windows; per meeting, hypothesis labels mapped to reference speakers by total overlap."""
    tot = miss = fa = conf = 0.0
    for m, ws in windows_by_meeting.items():
        refs = [frames([(u["speaker_id"], u["begin_s"], u["end_s"]) for u in w["utts"] if not u["hesitation"]],
                       w["dur_ms"] / 1000) for w in ws]
        hyps = [hyp_act_of(w) for w in ws]
        rk = sorted({k for r in refs for k in r})
        hk = sorted({k for h in hyps for k in h})
        if not rk:
            continue
        M = np.zeros((len(hk), len(rk)))
        for r, h in zip(refs, hyps):
            for a, x in enumerate(hk):
                for b, y in enumerate(rk):
                    if x in h and y in r:
                        n = min(len(h[x]), len(r[y]))
                        M[a, b] += (h[x][:n] & r[y][:n]).sum()
        mp = {}
        if hk:
            i, j = linear_sum_assignment(-M)
            mp = {hk[a]: rk[b] for a, b in zip(i, j)}
        for r, h in zip(refs, hyps):
            n = max([len(v) for v in r.values()] + [len(v) for v in h.values()] + [1])
            R = np.zeros((n, len(rk)), dtype=bool)
            for b, y in enumerate(rk):
                if y in r:
                    R[: len(r[y]), b] = r[y]
            H = np.zeros((n, len(rk) + len(hk)), dtype=bool)   # unmapped hypothesis labels get their own columns
            for a, x in enumerate(hk):
                if x in h:
                    col = rk.index(mp[x]) if x in mp else len(rk) + a
                    H[: len(h[x]), col] |= h[x]
            nr, nh = R.sum(1), H.sum(1)
            corr = (R & H[:, : len(rk)]).sum(1)
            tot += nr.sum()
            miss += np.maximum(0, nr - nh).sum()
            fa += np.maximum(0, nh - nr).sum()
            conf += (np.minimum(nr, nh) - corr).sum()
    return {"DER": round(100 * (miss + fa + conf) / max(tot, 1), 2), "miss": round(100 * miss / max(tot, 1), 2),
            "fa": round(100 * fa / max(tot, 1), 2), "conf": round(100 * conf / max(tot, 1), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--post-key", default="offline", help="Nemotron posteriors for the reference DER")
    ap.add_argument("--asr", default="", help="asr_speech.jsonl of the talker streams")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from sacrebleu.metrics import BLEU, CHRF

    W = [json.loads(l) for l in open(args.windows, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(args.hyp, encoding="utf-8"))}
    W = [w for w in W if w["id"] in H]
    asr = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(args.asr, encoding="utf-8"))} if args.asr else None
    from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units
    eng = any(w.get("dir") == "zh2en" for w in W)   # zh->en windows: English references in "ref", word units
    lang, sep = ("eng", " ") if eng else ("cmn", "")
    for w in W:
        talk = sorted((u for u in w["utts"] if not u["hesitation"]), key=lambda u: u["begin_s"])
        w["_ref"] = defaultdict(str)
        for u in talk:
            w["_ref"][u["speaker_id"]] += sep + u.get("ref", u["zh"]).strip()
        w["_ref_plain"] = sep.join(u.get("ref", u["zh"]).strip() for u in talk)
        w["_hyp"] = defaultdict(str)
        for s in H[w["id"]]["segments"]:
            w["_hyp"][s["spk"]] += sep + s["text"]
        w["_hyp_plain"] = sep.join(s["text"] for s in sorted(H[w["id"]]["segments"], key=lambda s: s["start_s"]))
        if asr is not None:
            w["_asr"] = {k: asr.get(f"{w['id']}__S{k}", "") for k in w["_hyp"]}
        bl = H[w["id"]].get("blocks")
        if bl:
            delays = [b["write_ms"] for b in bl for _ in units(b["text"], lang)]
            w["_laal"] = laal(delays, w["dur_ms"], len(units(w["_ref_plain"], lang))) if delays else None
            first_ref = min((u["begin_s"] for u in w["utts"] if not u["hesitation"]), default=0.0)
            w["_first"] = bl[0]["write_ms"] - 1000 * first_ref
        sp = [(u["begin_s"], u["end_s"]) for u in talk]
        w["_overlap"] = any(min(a[1], b[1]) > max(a[0], b[0]) for i, a in enumerate(sp) for b in sp[i + 1:])
        w["_nspk"] = len(w["_ref"])
    # session-level matching: one mapping of slots to speakers per meeting
    by_m = defaultdict(list)
    for w in W:
        by_m[w["meeting"]].append(w)
    sess_map = {}
    for m, ws in by_m.items():
        hyp, ref = defaultdict(str), defaultdict(str)
        for w in ws:
            for k, t in w["_hyp"].items():
                hyp[k] += t
            for k, t in w["_ref"].items():
                ref[k] += t
        sess_map[m] = match(hyp, ref)
    groups = {"all": W, "single_spk": [w for w in W if w["_nspk"] == 1], "multi_spk": [w for w in W if w["_nspk"] > 1],
              "overlap": [w for w in W if w["_overlap"]], "strict_review": [w for w in W if w.get("strict_review")]}
    bleu, chrf = BLEU(tokenize="13a" if eng else "zh"), CHRF()
    res = {}
    for g, ws in groups.items():
        if not ws:
            continue
        pw = [p for w in ws for p in pairs_of(w["_hyp"], w["_ref"], match(w["_hyp"], w["_ref"]))]
        ps = [p for w in ws for p in pairs_of(w["_hyp"], w["_ref"], sess_map[w["meeting"]])]
        bym = defaultdict(list)
        for w in ws:
            bym[w["meeting"]].append(w)

        def hyp_act(w):
            return frames([(s["spk"], s["start_s"], s["end_s"]) for s in H[w["id"]]["segments"]], w["dur_ms"] / 1000)

        def nemo_act(w):
            p = np.load(w["post"][args.post_key]).astype(np.float32) > 0.5
            return {k + 1: np.repeat(p[:, k], 8) for k in range(p.shape[1]) if p[:, k].any()}

        res[g] = {"windows": len(ws),
                  "cpBLEU_session": round(bleu.corpus_score([a for a, _ in ps], [[b for _, b in ps]]).score, 2),
                  "cpchrF_session": round(chrf.corpus_score([a for a, _ in ps], [[b for _, b in ps]]).score, 2),
                  "cpBLEU_window": round(bleu.corpus_score([a for a, _ in pw], [[b for _, b in pw]]).score, 2),
                  "BLEU_speaker_blind": round(bleu.corpus_score([w["_hyp_plain"] for w in ws],
                                                                [[w["_ref_plain"] for w in ws]]).score, 2),
                  "count_exact": round(np.mean([len([k for k in w["_hyp"] if w["_hyp"][k].strip()]) == w["_nspk"]
                                                for w in ws]), 4),
                  "empty": round(np.mean([not w["_hyp_plain"].strip() for w in ws]), 4),
                  "DER_output": der(bym, hyp_act), "DER_nemotron": der(bym, nemo_act)}
        if asr is not None:
            pa = [p for w in ws for p in pairs_of(w["_asr"], w["_ref"], sess_map[w["meeting"]])]
            res[g]["cpASR_BLEU_session"] = round(bleu.corpus_score([a for a, _ in pa], [[b for _, b in pa]]).score, 2)
            res[g]["ASR_BLEU_speaker_blind"] = round(bleu.corpus_score(
                [sep.join(w["_asr"][k] for k in sorted(w["_asr"])) for w in ws], [[w["_ref_plain"] for w in ws]]).score, 2)
        lags = [w["_laal"] for w in ws if w.get("_laal") is not None]
        if lags:
            res[g]["LAAL_s"] = round(float(np.mean(lags)) / 1000, 3)
            res[g]["first_commit_s"] = round(float(np.mean([w["_first"] for w in ws if "_first" in w])) / 1000, 3)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
