"""Where the text -> speech tcpBLEU gap of a MIST dev decode goes (CPU only; reports/mist_w3po_v1/
text_to_speech_gap_plan.zh-CN.md). One talker run (one TTS seed) per call.

Hypothesis variants, each scored with the metric_headroom re-implementation of overlap_score (online, split, collar 2 s,
merge 1 s; the raw protocol reproduces score_text.json / score_asr.json exactly) under three string protocols applied to
hyp AND ref after the assignment (the assignment itself is always the official one, on the raw strings):
  raw       the official one: sacreBLEU 13a, case-sensitive, punctuation kept
  repo      the single-speaker repo protocol (evaluation.text_metrics.normalize_english: NFKC, lowercase, punctuation
            stripped except apostrophes), then 13a -- the protocol of every single-speaker ASR-BLEU of this project
  whisper   Whisper's EnglishTextNormalizer (normalizer.json of whisper-large-v3; numbers, spellings, fillers), then 13a
            -- the Seamless / Hibiki protocol
Variants:
  text_seg        the decoded segments                                   (= tcpBLEU)
  text_merged     the text merged per window x speaker stream             (asr_hyp.py granularity, no TTS / ASR)
  text_tts        the talker's input text per stream (translated_text)    (= text_merged up to joins)
  asr             Whisper on the stream's speech-only wav                  (= tcpASR-BLEU)
  asr_seg         the ASR words projected back onto the stream's segments through a word alignment to the talker's own
                  input text (no reference used); the speech hyp at the text's granularity (protocol fix A1)
  asr_fix_*       word-level oracles on the stream alignment (ASR vs talker input): insertions dropped / deletions
                  restored / substitutions replaced / surface (case, punctuation of matched words) replaced; cumulative
                  chain asr -> +ins -> +del -> +sub -> +surface (= text_tts)
  asr_back_<cat>  streams of a category get their talker input text back (upper bound of fixing that category alone)
  asr_deloop      consecutive n-gram loops (n <= 6, >= 3 repeats) of the ASR collapsed to one (text-free guard)
Per-stream fidelity (repo-normalised words, talker input vs ASR): WER / S / D / I by stream length, segments per stream,
words per segment, streams per window, overlap of the stream's source span, unspeakable content, forced blocks, speech
rate, voice-prompt length; deletion runs at the stream end (truncation); loops; leakage of the other streams' words of the
same window into the ASR; top substitutions / insertions / deletions; Whisper-normaliser-equivalent substitutions.
--sem: tcpSemER-style semantic error (Tawara et al., arXiv 2603.22709, Eq. 2) on the same unit pairs with
all-MiniLM-L12-v2: sum over pairs of |H| (no ref) / |R| (no hyp) / (1 - cos) |R|, divided by the reference words.
Writes <out>.json and <out>.streams.jsonl (refuses to overwrite). Dev only.

    python -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run <decode dir> --speech <speech dir> \
        [--sem <MiniLM dir>] --out <prefix>
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import metric_headroom as M

WHISPER_NORM = ("/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub/models--openai--whisper-large-v3/snapshots/"
                "06f233fe06e710322aca913c1bc4249a0d71fce1/normalizer.json")
FILLERS = {"uh", "um", "hmm", "mm", "mhm", "mmm", "uh-huh", "mm-hmm", "hm", "er", "ah", "eh", "huh", "oh"}
UNSPEAK = re.compile(r"\d|[%$&/+=@#*_~^<>|\\\[\]{}]|\.\.|--|!!|\?\?")


def norms():
    import sys
    sys.path.insert(0, str(M.REPO))
    from evaluation.text_metrics import normalize_english
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    wn = EnglishTextNormalizer(json.load(open(WHISPER_NORM, encoding="utf-8")))
    return {"raw": lambda t: t, "repo": lru_cache(None)(normalize_english), "whisper": lru_cache(None)(wn)}


NORMS = None


def key(w):
    """word comparison key of the alignment: repo-normalised form."""
    return NORMS["repo"](w)


def align(ref_toks, hyp_toks):
    """Levenshtein alignment on comparison keys; returns ops [(op, ref_tok, hyp_tok, ref_index)] with op in
    eq / sub / del (ref word missing) / ins (hyp word extra)."""
    R = [key(t) for t in ref_toks]
    H = [key(t) for t in hyp_toks]
    n, m = len(R), len(H)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = R[i - 1]
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (ri != H[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (R[i - 1] != H[j - 1]):
            ops.append(("eq" if R[i - 1] == H[j - 1] else "sub", ref_toks[i - 1], hyp_toks[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", ref_toks[i - 1], None, i - 1))
            i -= 1
        else:
            ops.append(("ins", None, hyp_toks[j - 1], i - 1))
            j -= 1
    return ops[::-1]


def words(t):
    """surface tokens whose comparison key is not empty (pure punctuation tokens dropped from the alignment)."""
    return [w for w in (t or "").split() if key(w)]


def hybrid(ops, fix_ins=False, fix_del=False, fix_sub=False, fix_surface=False):
    out = []
    for op, r, h, _ in ops:
        if op == "eq":
            out.append(r if fix_surface else h)
        elif op == "sub":
            out.append(r if fix_sub else h)
        elif op == "del":
            if fix_del:
                out.append(r)
        elif not fix_ins:
            out.append(h)
    return " ".join(out)


def loops(toks, nmax=6, rep=3):
    """collapse consecutive repeats (>= rep times) of an n-gram (n <= nmax) to one copy; returns (tokens, collapsed?)"""
    k = [key(t) for t in toks]
    changed = False
    for n in range(1, nmax + 1):
        i, out_t, out_k = 0, [], []
        while i < len(k):
            g = k[i:i + n]
            r = 1
            while len(g) == n and k[i + r * n:i + (r + 1) * n] == g:
                r += 1
            if len(g) == n and r >= rep:
                out_t += toks[i:i + n]
                out_k += g
                i += r * n
                changed = True
            else:
                out_t.append(toks[i])
                out_k.append(k[i])
                i += 1
        toks, k = out_t, out_k
    return toks, changed


def stream_hyp(H, A):
    """asr_hyp.py: one segment per window x speaker, spanning that speaker's segments."""
    out = {}
    for wid, h in H.items():
        by = defaultdict(list)
        for s in h["segments"]:
            by[s["spk"]].append(s)
        segs = []
        for k, ss in by.items():
            t = (A.get(f"{wid}__S{k}") or "").strip()
            if t:
                segs.append({"spk": k, "start_s": min(s["start_s"] for s in ss), "end_s": max(s["end_s"] for s in ss),
                             "text": t, "write_ms": max(s.get("write_ms", 0) for s in ss)})
        out[wid] = {"id": wid, "segments": segs}
    return out


def seg_hyp(H, T):
    """segment-level hyp: T[(wid, spk)] = list of texts, one per segment of that speaker in order."""
    out = {}
    for wid, h in H.items():
        cnt = defaultdict(int)
        segs = []
        for s in h["segments"]:
            k = s["spk"]
            L = T.get((wid, k))
            t = L[cnt[k]] if L is not None and cnt[k] < len(L) else ""
            cnt[k] += 1
            if t.strip():
                segs.append({**s, "text": t})
        out[wid] = {"id": wid, "segments": segs}
    return out


def score(W, Hs, sc, U, want_pairs=False):
    As = M.assign_all(W, Hs, sc, U)
    P = {"all": [], "ov": [], "1spk": []}
    for A in As.values():
        p = M.pairs_of(A)
        for r in P:
            P[r] += p[r]
    res = {}
    for name, f in NORMS.items():
        res[name] = {}
        for r in P:
            st = M.bleu_stats([(f(h), f(y)) for h, y in P[r]])
            b = M.bleu_from(st)
            res[name][r] = b["bleu"]
            if r == "all":
                res[name]["bp_all"] = b["bp"]
    return (res, P) if want_pairs else res


def sem_err(P, model_dir):
    """tcpSemER-style score of the unit / insertion pairs (Tawara et al., arXiv 2603.22709, Eq. 2)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    mod = AutoModel.from_pretrained(model_dir).eval()
    cache = {}

    def emb(texts):
        need = [t for t in dict.fromkeys(texts) if t not in cache]
        for i in range(0, len(need), 256):
            b = need[i:i + 256]
            x = tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                o = mod(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            e = torch.nn.functional.normalize((o * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=-1)
            for t, v in zip(b, e.numpy()):
                cache[t] = v
        return [cache[t] for t in texts]
    out = {}
    for r, pairs in P.items():
        both = [(h, y) for h, y in pairs if h.strip() and y.strip()]
        E1, E2 = emb([h for h, _ in both]), emb([y for _, y in both])
        cos = {(h, y): float(np.dot(a, b)) for (h, y), a, b in zip(both, E1, E2)}
        err = nref = 0.0
        parts = defaultdict(float)
        for h, y in pairs:
            nr, nh = len(NORMS["repo"](y).split()), len(NORMS["repo"](h).split())
            nref += nr
            if not y.strip():
                parts["ins"] += nh
            elif not h.strip():
                parts["omit"] += nr
            else:
                parts["sem"] += (1 - cos[(h, y)]) * nr
        err = sum(parts.values())
        out[r] = {"tcpSemER": round(err / max(nref, 1), 4), **{k: round(v / max(nref, 1), 4) for k, v in parts.items()}}
    return out


def multi_share(a, b, us):
    """share of [a, b] during which two or more annotated speakers talk."""
    ev = []
    for u in us:
        x0, x1 = max(a, u[2]), min(b, u[3])
        if x1 > x0:
            ev += [(x0, 1, u[1]), (x1, -1, u[1])]
    act, last, tot = defaultdict(int), None, 0.0
    for t, d, s in sorted(ev, key=lambda e: (e[0], e[1])):
        if last is not None and sum(1 for v in act.values() if v > 0) >= 2:
            tot += t - last
        act[s] += d
        last = t
    return tot / max(b - a, 1e-6)


def binned(rows, keyf, bins):
    out = {}
    for name, test in bins:
        g = [r for r in rows if test(keyf(r))]
        n = sum(r["n"] for r in g)
        if not g:
            continue
        out[name] = {"streams": len(g), "words": n, "word_share": None,
                     "wer": round(sum(r["S"] + r["D"] + r["I"] for r in g) / max(n, 1), 4),
                     "sub": round(sum(r["S"] for r in g) / max(n, 1), 4), "del": round(sum(r["D"] for r in g) / max(n, 1), 4),
                     "ins": round(sum(r["I"] for r in g) / max(n, 1), 4)}
    tot = sum(r["n"] for r in rows)
    for v in out.values():
        v["word_share"] = round(v["words"] / max(tot, 1), 4)
    return out


def main() -> None:
    global NORMS
    import faulthandler
    faulthandler.enable()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--speech", required=True, help="speech dir holding talker/ (one TTS seed)")
    ap.add_argument("--sem", default="", help="all-MiniLM-L12-v2 directory: also compute the tcpSemER-style score")
    ap.add_argument("--out", required=True, help="output prefix (<out>.json, <out>.streams.jsonl)")
    ap.add_argument("--asr-override", default="", help="jsonl (id, asr_text[, asr_text_guarded]) replacing the run's Whisper "
                    "transcripts, e.g. whisper_b1 output; every stream must be present")
    ap.add_argument("--asr-override-field", default="asr_text",
                    help="field of --asr-override to use; a row without it keeps that file's asr_text")
    args = ap.parse_args()
    out_j, out_s = Path(args.out + ".json"), Path(args.out + ".streams.jsonl")
    for p in (out_j, out_s, Path(args.out + ".pairs.json"), Path(args.out + ".sem.json")):
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
    NORMS = norms()
    from meeting_score import universe
    import soundfile as sf
    W = [json.loads(l) for l in open(M.REPO / M.WIN, encoding="utf-8")]
    wins = {w["id"]: w for w in W}
    H = {h["id"]: h for h in map(json.loads, open(Path(args.run) / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
    U = universe(M.DR, {w["meeting"] for w in W})
    sc = M.scored_utts(W, RW)
    T = Path(args.speech) / "talker"
    src = T / "asr_speech_merged.jsonl"
    rows = {}
    for f in ([src] if src.exists() else sorted(T.glob("asr_speech.t*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if isinstance(r.get("blocks"), str):
                r["blocks"] = ast.literal_eval(r["blocks"])
            rows[r["id"]] = r
    items = {}
    for cand in (Path(args.speech) / "talker_in_all/items.jsonl", Path(args.speech) / "talker_in/items.jsonl"):
        if cand.exists():
            items = {json.loads(l)["id"]: json.loads(l) for l in open(cand, encoding="utf-8")}
            break
    res = {"run": args.run, "speech": args.speech, "streams": len(rows), "scores": {}}
    asr = {k: (r.get("asr_text") or "") for k, r in rows.items()}
    if args.asr_override:
        ov = {json.loads(l)["id"]: json.loads(l) for l in open(args.asr_override, encoding="utf-8")}
        missing = set(rows) - set(ov)
        if missing:
            raise SystemExit(f"--asr-override lacks {len(missing)} streams")
        asr = {k: (ov[k].get(args.asr_override_field, ov[k]["asr_text"]) or "") for k in rows}
        for k, r in rows.items():
            r["asr_text"] = asr[k]
        res["asr_override"] = [args.asr_override, args.asr_override_field]
    tts = {k: r["translated_text"] for k, r in rows.items()}

    # ---- per-stream alignment and features
    S = []
    seg_texts, seg_texts_self = {}, {}
    fixes = {n: {} for n in ("ins", "ins_del", "ins_del_sub", "ins_del_sub_surface", "ins_only", "del_only", "sub_only",
                             "surface_only")}
    deloop = {}
    mism = 0
    win_words = defaultdict(dict)
    for k, r in rows.items():
        wid, spk = k.split("__S")
        win_words[wid][int(spk)] = {key(w) for w in words(tts[k])}
    sub_pairs, ins_w, del_w = Counter(), Counter(), Counter()
    wn_equiv = 0
    for k, r in sorted(rows.items()):
        wid, spk = k.split("__S")
        spk = int(spk)
        segs = [s for s in H[wid]["segments"] if s["spk"] == spk]
        bl = r["blocks"]
        if len(bl) != len(segs) or any(b["text"].strip() != s["text"].strip() for b, s in zip(bl, segs)):
            mism += 1
        # talker input words labelled by segment
        rt, lab = [], []
        for i, s in enumerate(segs):
            ws = words(s["text"])
            rt += ws
            lab += [i] * len(ws)
        ht = words(asr[k])
        ops = align(rt, ht)
        # projection of the ASR words onto the segments
        proj = [[] for _ in segs]
        last = 0
        pend = []
        for op, rr, hh, ri in ops:
            if op in ("eq", "sub"):
                last = lab[ri]
                proj[last] += pend + [hh]
                pend = []
            elif op == "ins":
                if ri >= 0 and lab:
                    proj[lab[ri]].append(hh)
                else:
                    pend.append(hh)
            elif op == "del" and lab:
                last = lab[ri]
        if pend and segs:
            proj[0] = pend + proj[0]
        seg_texts[(wid, spk)] = [" ".join(p) for p in proj]
        sops = align(rt, rt)
        sp = [[] for _ in segs]
        for op, rr, hh, ri in sops:
            sp[lab[ri]].append(hh)
        seg_texts_self[(wid, spk)] = [" ".join(p) for p in sp]
        fixes["ins"][k] = hybrid(ops, fix_ins=True)
        fixes["ins_del"][k] = hybrid(ops, fix_ins=True, fix_del=True)
        fixes["ins_del_sub"][k] = hybrid(ops, fix_ins=True, fix_del=True, fix_sub=True)
        fixes["ins_del_sub_surface"][k] = hybrid(ops, True, True, True, True)
        fixes["ins_only"][k] = hybrid(ops, fix_ins=True)
        fixes["del_only"][k] = hybrid(ops, fix_del=True)
        fixes["sub_only"][k] = hybrid(ops, fix_sub=True)
        fixes["surface_only"][k] = hybrid(ops, fix_surface=True)
        dl, changed = loops(ht)
        deloop[k] = " ".join(dl) if changed else asr[k]
        c = Counter(op for op, *_ in ops)
        n = len(rt)
        # deletion run at the end / start
        tail = 0
        for op, *_ in reversed(ops):
            if op == "del":
                tail += 1
            elif op == "ins":
                continue
            else:
                break
        head = 0
        for op, *_ in ops:
            if op == "del":
                head += 1
            elif op == "ins":
                continue
            else:
                break
        for op, rr, hh, _ in ops:
            if op == "sub":
                sub_pairs[(key(rr), key(hh))] += 1
                if NORMS["whisper"](rr) == NORMS["whisper"](hh):
                    wn_equiv += 1
            elif op == "ins":
                ins_w[key(hh)] += 1
            elif op == "del":
                del_w[key(rr)] += 1
        # leakage: substituted / inserted ASR words found in another stream's input of the same window, not in own
        own = win_words[wid][spk]
        other = set().union(*[v for kk, v in win_words[wid].items() if kk != spk]) if len(win_words[wid]) > 1 else set()
        err_h = [key(hh) for op, rr, hh, _ in ops if op in ("sub", "ins")]
        leak = sum(1 for w in err_h if w in other and w not in own)
        w = wins[wid]
        a = w["start_s"] + min(s["start_s"] for s in segs) if segs else w["start_s"]
        b = w["start_s"] + max(max(s["end_s"], s["start_s"] + 0.08) for s in segs) if segs else a + 0.08
        spd = sum(x.get("dur_ms", 0) for x in bl) / 1000
        vp = None
        it = items.get(k)
        if it and it.get("voice_audio"):
            pth = Path(it["voice_audio"])
            pth = pth if pth.is_absolute() else M.REPO / pth
            try:
                vp = sf.info(str(pth)).duration
            except Exception:
                vp = None
        txt = tts[k]
        S.append({"id": k, "n": n, "S": c["sub"], "D": c["del"], "I": c["ins"], "asr_n": len(ht), "nseg": len(segs),
                  "wps": n / max(len(segs), 1), "nstreams": len(win_words[wid]), "multi_share": round(multi_share(a, b, U[w["meeting"]]), 3),
                  "unspeak": bool(UNSPEAK.search(txt)), "filler": any(key(x) in FILLERS for x in words(txt)),
                  "partial": bool(re.search(r"\w-(\s|$)", txt)), "acronym": bool(re.search(r"\b[A-Z]{2,}\b", txt)),
                  "forced": any(x.get("forced") for x in bl), "speech_s": round(spd, 3),
                  "spt": round(spd / max(n, 1), 4), "asr_wps": round(len(ht) / max(spd, 1e-3), 3),
                  "tail_del": tail, "head_del": head, "loop": changed, "leak": leak, "err_h": len(err_h),
                  "voice_s": vp, "ratio13a": len(re.findall(r"\w+|[^\w\s]", asr[k])) / max(len(re.findall(r"\w+|[^\w\s]", txt)), 1),
                  "asr_empty": not asr[k].strip()})
    res["block_segment_mismatch_streams"] = mism
    res["hyp_streams_without_talker_row"] = sum(1 for wid, h in H.items() for k in {s["spk"] for s in h["segments"] if s["text"].strip()} if f"{wid}__S{k}" not in rows)
    with open(out_s, "x", encoding="utf-8") as fh:
        for r in S:
            fh.write(json.dumps(r) + "\n")

    # ---- fidelity tables
    tw = sum(r["n"] for r in S)
    res["fidelity"] = {
        "words": tw, "wer": round(sum(r["S"] + r["D"] + r["I"] for r in S) / tw, 4),
        "sub": round(sum(r["S"] for r in S) / tw, 4), "del": round(sum(r["D"] for r in S) / tw, 4),
        "ins": round(sum(r["I"] for r in S) / tw, 4),
        "asr_empty_streams": sum(r["asr_empty"] for r in S),
        "by_len": binned(S, lambda r: r["n"], [("1-2", lambda x: x <= 2), ("3-4", lambda x: 3 <= x <= 4), ("5-8", lambda x: 5 <= x <= 8),
                                               ("9-16", lambda x: 9 <= x <= 16), ("17-32", lambda x: 17 <= x <= 32), (">32", lambda x: x > 32)]),
        "by_nseg": binned(S, lambda r: r["nseg"], [("1", lambda x: x == 1), ("2", lambda x: x == 2), ("3-4", lambda x: 3 <= x <= 4), ("5+", lambda x: x >= 5)]),
        "by_words_per_seg": binned(S, lambda r: r["wps"], [("<=2", lambda x: x <= 2), ("2-5", lambda x: 2 < x <= 5), ("5-10", lambda x: 5 < x <= 10), (">10", lambda x: x > 10)]),
        "by_nseg_len9plus": binned([r for r in S if r["n"] >= 9], lambda r: r["nseg"], [("1", lambda x: x == 1), ("2", lambda x: x == 2), ("3+", lambda x: x >= 3)]),
        "by_nstreams": binned(S, lambda r: r["nstreams"], [("1", lambda x: x == 1), ("2", lambda x: x == 2), ("3+", lambda x: x >= 3)]),
        "by_overlap": binned(S, lambda r: r["multi_share"], [("1spk(<0.2)", lambda x: x < 0.2), ("ov(>=0.2)", lambda x: x >= 0.2)]),
        "by_unspeak": binned(S, lambda r: r["unspeak"], [("yes", lambda x: x), ("no", lambda x: not x)]),
        "by_filler": binned(S, lambda r: r["filler"], [("yes", lambda x: x), ("no", lambda x: not x)]),
        "by_partial": binned(S, lambda r: r["partial"], [("yes", lambda x: x), ("no", lambda x: not x)]),
        "by_acronym": binned(S, lambda r: r["acronym"], [("yes", lambda x: x), ("no", lambda x: not x)]),
        "by_forced": binned(S, lambda r: r["forced"], [("yes", lambda x: x), ("no", lambda x: not x)]),
        "by_voice_s": binned([r for r in S if r["voice_s"] is not None], lambda r: r["voice_s"],
                             [("<1", lambda x: x < 1), ("1-3", lambda x: 1 <= x < 3), ("3-6", lambda x: 3 <= x < 6), (">=6", lambda x: x >= 6)]),
        "by_spt": binned(S, lambda r: r["spt"], [("<0.25", lambda x: x < 0.25), ("0.25-0.4", lambda x: 0.25 <= x < 0.4),
                                                 ("0.4-0.6", lambda x: 0.4 <= x < 0.6), (">=0.6", lambda x: x >= 0.6)]),
        "by_ratio13a": binned(S, lambda r: r["ratio13a"], [("<=1.5", lambda x: x <= 1.5), (">1.5", lambda x: x > 1.5)]),
        "truncation": {"streams_tail_del_ge2": sum(r["tail_del"] >= 2 for r in S), "streams_tail_del_ge3": sum(r["tail_del"] >= 3 for r in S),
                       "tail_del_words_share_of_del": round(sum(r["tail_del"] for r in S) / max(sum(r["D"] for r in S), 1), 4),
                       "head_del_words_share_of_del": round(sum(r["head_del"] for r in S) / max(sum(r["D"] for r in S), 1), 4),
                       "streams_all_deleted": sum(r["D"] == r["n"] and r["n"] > 0 for r in S)},
        "loops": {"streams": sum(r["loop"] for r in S)},
        "leak": {"err_words_multi_stream": sum(r["err_h"] for r in S if r["nstreams"] > 1),
                 "leaked_words": sum(r["leak"] for r in S),
                 "leak_share_of_err_words_multi": round(sum(r["leak"] for r in S) / max(sum(r["err_h"] for r in S if r["nstreams"] > 1), 1), 4)},
        "whisper_equiv_subs": {"n": wn_equiv, "share_of_subs": round(wn_equiv / max(sum(r["S"] for r in S), 1), 4)},
        "top_sub": [[a, b, c] for (a, b), c in sub_pairs.most_common(40)],
        "top_ins": ins_w.most_common(40), "top_del": del_w.most_common(40),
    }
    # tail streams (13a ratio > 1.5, as metric_speech_tail): what they are
    tl = [r for r in S if r["ratio13a"] > 1.5]
    nt = [r for r in S if r["ratio13a"] <= 1.5]
    res["tail"] = {"n": len(tl), "loop_share": round(np.mean([r["loop"] for r in tl]), 4) if tl else None,
                   "median_asr_wps": float(np.median([r["asr_wps"] for r in tl])) if tl else None,
                   "median_asr_wps_rest": float(np.median([r["asr_wps"] for r in nt])),
                   "median_spt": float(np.median([r["spt"] for r in tl])) if tl else None,
                   "median_spt_rest": float(np.median([r["spt"] for r in nt])),
                   "median_n": float(np.median([r["n"] for r in tl])) if tl else None}
    # ---- leakage baseline: the same count against a random other window's stream (shuffled)
    rng = np.random.default_rng(0)
    wl = list(win_words)
    base_leak = base_err = 0
    for r in S:
        if r["nstreams"] < 2:
            continue
        wid, spk = r["id"].split("__S")
        ow = wl[int(rng.integers(len(wl)))]
        other = set().union(*win_words[ow].values()) if win_words[ow] else set()
        own = win_words[wid][int(spk)]
        ops = align(words(tts[r["id"]]), words(asr[r["id"]]))
        eh = [key(hh) for op, rr, hh, _ in ops if op in ("sub", "ins")]
        base_err += len(eh)
        base_leak += sum(1 for w in eh if w in other and w not in own)
    res["fidelity"]["leak"]["baseline_random_window_share"] = round(base_leak / max(base_err, 1), 4)

    # ---- scores
    sc_ = res["scores"]
    sc_["text_seg"], P_text = score(W, H, sc, U, want_pairs=True)
    sc_["text_seg_selfproj"] = score(W, seg_hyp(H, seg_texts_self), sc, U)
    sc_["text_merged"], P_merged = score(W, M.merged_text_hyp(H), sc, U, want_pairs=True)
    sc_["text_tts"] = score(W, stream_hyp(H, tts), sc, U)
    sc_["asr"], P_asr = score(W, stream_hyp(H, asr), sc, U, want_pairs=True)
    sc_["asr_seg"], P_asr_seg = score(W, seg_hyp(H, seg_texts), sc, U, want_pairs=True)
    sc_["asr_deloop"] = score(W, stream_hyp(H, deloop), sc, U)
    for n, A in fixes.items():
        sc_[f"asr_fix_{n}"] = score(W, stream_hyp(H, {k: A.get(k, v) for k, v in asr.items()}), sc, U)
    cats = {
        "tail_ratio_gt1.5": lambda r: r["ratio13a"] > 1.5,
        "len_le4": lambda r: r["n"] <= 4,
        "len_5_16": lambda r: 5 <= r["n"] <= 16,
        "len_gt16": lambda r: r["n"] > 16,
        "nseg_1": lambda r: r["nseg"] == 1,
        "nseg_ge2": lambda r: r["nseg"] >= 2,
        "nstreams_1": lambda r: r["nstreams"] == 1,
        "nstreams_ge2": lambda r: r["nstreams"] >= 2,
        "ov_stream": lambda r: r["multi_share"] >= 0.2,
        "1spk_stream": lambda r: r["multi_share"] < 0.2,
        "unspeak": lambda r: r["unspeak"],
        "filler": lambda r: r["filler"],
        "forced": lambda r: r["forced"],
        "wer_gt0.5": lambda r: (r["S"] + r["D"] + r["I"]) / max(r["n"], 1) > 0.5,
        "wer_le0.5": lambda r: (r["S"] + r["D"] + r["I"]) / max(r["n"], 1) <= 0.5,
        "tail_del_ge2": lambda r: r["tail_del"] >= 2,
        "loop": lambda r: r["loop"],
    }
    res["category_streams"] = {}
    for n, f in cats.items():
        ids = {r["id"] for r in S if f(r)}
        res["category_streams"][n] = {"streams": len(ids), "word_share": round(sum(r["n"] for r in S if r["id"] in ids) / tw, 4)}
        sc_[f"asr_back_{n}"] = score(W, stream_hyp(H, {k: (tts[k] if k in ids else v) for k, v in asr.items()}), sc, U)
    # cumulative chain by category (order: tail, short, unspeakable, forced, rest)
    chain, done = [], set()
    for n in ("tail_ratio_gt1.5", "len_le4", "unspeak", "forced", "nstreams_ge2", "nstreams_1"):
        done |= {r["id"] for r in S if cats[n](r)}
        chain.append((n, score(W, stream_hyp(H, {k: (tts[k] if k in done else v) for k, v in asr.items()}), sc, U)))
    res["chain_by_category"] = chain
    chk = Path(args.run) / "score_text.json"
    if chk.exists():
        c = json.load(open(chk, encoding="utf-8"))
        res["check_text"] = {r: (sc_["text_seg"]["raw"][r], c[f"tcpBLEU_{r}"]) for r in ("all", "ov", "1spk")}
    with open(out_j, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    # the unit / insertion pairs, for the semantic score (computed apart: --sem, or sem_score on this file later)
    pairs = {"text_seg": P_text, "text_merged": P_merged, "asr": P_asr, "asr_seg": P_asr_seg}
    with open(Path(args.out + ".pairs.json"), "x", encoding="utf-8") as fh:
        json.dump(pairs, fh, ensure_ascii=False)
    if args.sem:
        sem = {k: sem_err(v, args.sem) for k, v in pairs.items()}
        with open(Path(args.out + ".sem.json"), "x", encoding="utf-8") as fh:
            json.dump(sem, fh, indent=1)
    print(json.dumps({"check_text": res.get("check_text"), "asr_raw": sc_["asr"]["raw"], "mismatch": mism}, ensure_ascii=False))


if __name__ == "__main__":
    main()
