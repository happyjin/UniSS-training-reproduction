"""STAMP reward (docs/mist_grpo_plan.zh-CN.md §3.2, C1) and the per-token-type credit of W3 (§3.3-3.5) for one window.

Inputs (window clock, seconds):
  units  reference units: same-speaker utterances closer than MERGE_S merged (as reports/gstar_si_v8/overlap_score.py),
         {"spk", "a", "b", "ref", "src"}; every unit weighs the same, whatever its length (P4)
  segs   committed segments {"spk" (slot), "a", "b", "text", "r" (read index of the commit), "t" (commit time, s)}

  pi*    window-level slot -> reference speaker map, Hungarian on chrF of per-speaker concatenations (as cpBLEU)
  assign each segment goes to the unit maximising spk-factor x the share of the segment inside the unit +- TAU
         (many-to-one: a unit's translation may come as several chunk segments); none -> spurious
  S(u)   = Q(concatenated text, ref) x mean speaker factor (eta + (1 - eta) 1[pi*(k) = rho_u], text-length weighted)
           x T(u) (recall x precision of the assigned span against the unit, TAU-second grace)
  R      = (sum_u S(u) - LAMBDA_SP n_spurious - LAMBDA_LANG n_lang - LAMBDA_FMT n_format) / |U|
  Q      = mean(chrF++, sentence BLEU) / 100 (COMET / MetricX are not on this server: see LOG; plug-in via q_fn)

Credit:
  Delta_j     leave-one-out R(C) - R(C minus sigma_j), pi* fixed (duplicates get <= 0, no dedup rule)
  Delta^Q_j   the same with the speaker factor fixed to 1 (quality and time only: the what credit)
  who_j       RBC: R(C) - R(C with sigma_j's slot set to its causal relabel k_rl), minus LAMBDA_LOC when the
              diarizer posterior of one slot is > 0.9 over the segment and the model chose another
  miss        a unit with no segment: -1/|U| spread over the non-final reads at or after its end + EPS_MISS whose
              block ended in <wait> (the reads that should have written it)
  latency     mean completion lag over the units that got text (commit of the last assigned segment - unit end)
  FLAD        r_lat = -L if Q >= max over anchors with L_a <= L + DELTA of Q_a - EPS_DOM, else -L_MAX
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

MERGE_S = 1.0
TAU = 2.0
ETA = 0.3
LAMBDA_SP = 0.5
LAMBDA_LANG = 1.0
LAMBDA_FMT = 0.1
LAMBDA_LOC = 0.2
EPS_MISS = 0.5
DELTA_LAT = 0.25
EPS_DOM = 0.02
L_MAX = 6.0
CJK = re.compile(r"[㐀-鿿豈-﫿]")


@lru_cache(maxsize=1)
def _metrics():
    from sacrebleu.metrics import BLEU, CHRF
    return CHRF(word_order=2), BLEU(effective_order=True, tokenize="13a")


@lru_cache(maxsize=200000)
def q_chrf_bleu(hyp: str, ref: str) -> float:
    if not hyp.strip() or not ref.strip():
        return 0.0
    chrf, bleu = _metrics()
    return 0.5 * (chrf.sentence_score(hyp, [ref]).score + bleu.sentence_score(hyp, [ref]).score) / 100.0


@lru_cache(maxsize=200000)
def chrf(hyp: str, ref: str) -> float:
    if not hyp.strip() or not ref.strip():
        return 0.0
    return _metrics()[0].sentence_score(hyp, [ref]).score


def units_of(utts: list[dict], merge_s: float = MERGE_S) -> list[dict]:
    """window utterances (begin_s, end_s, speaker_id, ref, zh) -> units, same-speaker gaps < merge_s merged."""
    out = []
    by = {}
    for u in sorted((u for u in utts if not u.get("hesitation") and u.get("ref", "").strip()),
                    key=lambda u: u["begin_s"]):
        cur = by.get(u["speaker_id"])
        if cur is not None and u["begin_s"] - cur["b"] < merge_s:
            cur["b"] = max(cur["b"], u["end_s"])
            cur["ref"] += " " + u["ref"].strip()
            cur["src"] += u.get("zh", "")
        else:
            cur = {"spk": u["speaker_id"], "a": u["begin_s"], "b": u["end_s"], "ref": u["ref"].strip(),
                   "src": u.get("zh", "")}
            by[u["speaker_id"]] = cur
            out.append(cur)
    return sorted(out, key=lambda v: v["a"])


def ov(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def speaker_map(segs: list[dict], units: list[dict]) -> dict:
    """slot -> reference speaker, Hungarian on chrF of the concatenations (cpBLEU's matching)."""
    hs = sorted({s["spk"] for s in segs if s["text"].strip()})
    rs = sorted({u["spk"] for u in units})
    if not hs or not rs:
        return {}
    ht = {k: " ".join(s["text"] for s in sorted(segs, key=lambda s: s["a"]) if s["spk"] == k) for k in hs}
    rt = {k: " ".join(u["ref"] for u in units if u["spk"] == k) for k in rs}
    M = np.array([[chrf(ht[h], rt[r]) for r in rs] for h in hs])
    i, j = linear_sum_assignment(-M)
    return {hs[a]: rs[b] for a, b in zip(i, j) if M[a, b] > 0}


def violations(s: dict, dur: float, src_texts: list[str], eng: bool = True) -> str | None:
    t = s["text"].strip()
    if not t:
        return "empty"
    if eng and len(CJK.findall(t)) > 0.2 * max(1, len(t.replace(" ", ""))):
        return "lang"
    if any(src and chrf(t, src) >= 60 for src in src_texts):
        return "copy"
    return None


def fmt_bad(s: dict, dur: float) -> bool:
    return s["b"] <= s["a"] or s["a"] > dur + 1.0 or s["b"] > dur + 2.0 or s["spk"] not in range(1, 9)


@dataclass
class Score:
    R: float
    S: list            # per unit
    assign: list       # per segment: unit index or -1
    spurious: int
    lang: int
    fmt: int


def score(segs: list[dict], units: list[dict], pi: dict, dur: float, *, use_spk: bool = True, q_fn=q_chrf_bleu,
          eng: bool = True) -> Score:
    n_u = max(len(units), 1)
    src = [u["src"] for u in units]
    assign = [-1] * len(segs)
    spurious = lang = fmt = 0
    for j, s in enumerate(segs):
        if fmt_bad(s, dur):
            fmt += 1
        v = violations(s, dur, src, eng)
        if v is not None:
            if v == "lang":
                lang += 1
            else:
                spurious += 1
            continue
        L = max(s["b"] - s["a"], 0.08)
        best, bi = 0.0, -1
        for i, u in enumerate(units):
            inside = ov(s["a"], max(s["b"], s["a"] + 0.08), u["a"] - TAU, u["b"] + TAU) / L
            if inside <= 0:
                continue
            f = 1.0 if (not use_spk or pi.get(s["spk"]) == u["spk"]) else ETA
            if f * inside > best:
                best, bi = f * inside, i
        assign[j] = bi
        if bi < 0:
            spurious += 1
    S = []
    for i, u in enumerate(units):
        js = sorted((j for j in range(len(segs)) if assign[j] == i), key=lambda j: (segs[j]["a"], segs[j].get("r", 0)))
        if not js:
            S.append(0.0)
            continue
        text = " ".join(segs[j]["text"].strip() for j in js)
        w = np.array([max(len(segs[j]["text"]), 1) for j in js], dtype=float)
        f = np.array([1.0 if (not use_spk or pi.get(segs[j]["spk"]) == u["spk"]) else ETA for j in js])
        A, B = min(segs[j]["a"] for j in js), max(max(segs[j]["b"], segs[j]["a"] + 0.08) for j in js)
        du, dh = max(u["b"] - u["a"], 0.08), B - A
        T = min(1.0, ov(A - TAU, B + TAU, u["a"], u["b"]) / du) * min(1.0, ov(u["a"] - TAU, u["b"] + TAU, A, B) / dh)
        S.append(q_fn(text, u["ref"]) * float((w * f).sum() / w.sum()) * T)
    R = (sum(S) - LAMBDA_SP * spurious - LAMBDA_LANG * lang - LAMBDA_FMT * fmt) / n_u
    return Score(R, S, assign, spurious, lang, fmt)


def relabel_slot(s: dict, post: np.ndarray | None, lookahead: float = 1.04, frame: float = 0.08) -> int:
    """overlap_score.relabel_causal on the window clock: the slot with the highest mean posterior over the segment's
    span among frames known at its commit (t - 1.04 s); else over [a - 2 s, commit - 1.04 s]; else 0 (unknown)."""
    if post is None:
        return 0
    lim = min(int((s["t"] - lookahead) / frame), len(post))
    if lim <= 0:
        return 0
    r0, r1 = int(s["a"] / frame), min(int(math.ceil(s["b"] / frame)), lim)
    if r1 > max(r0, 0):
        m = post[max(r0, 0):r1].astype(np.float32).mean(0)
        if m.max() > 0:
            return int(m.argmax()) + 1
    q0 = max(0, int((s["a"] - 2.0) / frame))
    if lim > q0:
        m = post[q0:lim].astype(np.float32).mean(0)
        if m.max() > 0:
            return int(m.argmax()) + 1
    return 0


def confident_slot(s: dict, post: np.ndarray | None, thr: float = 0.9, frame: float = 0.08) -> int:
    """a slot whose causal posterior is > thr on average over the segment's known frames (RBC locality), else 0."""
    if post is None:
        return 0
    lim = min(int((s["t"] - 1.04) / frame), len(post))
    r0, r1 = max(0, int(s["a"] / frame)), min(int(math.ceil(s["b"] / frame)), lim)
    if r1 <= r0:
        return 0
    m = post[r0:r1].astype(np.float32).mean(0)
    return int(m.argmax()) + 1 if m.max() > thr else 0


def latency(segs: list[dict], units: list[dict], sc: Score, dur: float) -> float:
    lags = []
    for i, u in enumerate(units):
        ts = [segs[j]["t"] for j in range(len(segs)) if sc.assign[j] == i]
        if ts:
            lags.append(max(ts) - u["b"])
    return float(np.mean(lags)) if lags else min(L_MAX, dur)


@dataclass
class Credit:
    R: float
    RQ: float
    lat: float
    delta: list       # per segment, LOO on R
    delta_q: list     # per segment, LOO on R^Q
    who: list         # per segment, RBC
    unit_of: list     # per segment: unit index or -1
    S: list           # per unit
    SQ: list          # per unit, speaker factor fixed to 1
    miss: dict        # read index -> penalty (<= 0)
    n_units: int
    spurious: int
    lang: int


def credit(segs: list[dict], units: list[dict], dur: float, wait_reads: list[int], read_t: list[float],
           post: np.ndarray | None = None, q_fn=q_chrf_bleu, eng: bool = True, rbc: bool = True) -> Credit:
    """Everything W3 needs for one rollout of one window. wait_reads: read indices (0-based) whose block ended in
    <wait>; read_t: the read times (s)."""
    pi = speaker_map(segs, units)
    full = score(segs, units, pi, dur, q_fn=q_fn, eng=eng)
    fq = score(segs, units, pi, dur, use_spk=False, q_fn=q_fn, eng=eng)
    delta, delta_q, who = [], [], []
    for j in range(len(segs)):
        rest = segs[:j] + segs[j + 1:]
        delta.append(full.R - score(rest, units, pi, dur, q_fn=q_fn, eng=eng).R)
        delta_q.append(fq.R - score(rest, units, pi, dur, use_spk=False, q_fn=q_fn, eng=eng).R)
        w = 0.0
        if rbc:
            k_rl = relabel_slot(segs[j], post)
            if k_rl and k_rl != segs[j]["spk"]:
                alt = segs[:j] + [{**segs[j], "spk": k_rl}] + segs[j + 1:]
                w = full.R - score(alt, units, pi, dur, q_fn=q_fn, eng=eng).R
            k_hi = confident_slot(segs[j], post)
            if k_hi and k_hi != segs[j]["spk"]:
                w -= LAMBDA_LOC / max(len(units), 1)
        who.append(w)
    miss = {}
    n_u = max(len(units), 1)
    for i, u in enumerate(units):
        if full.S[i] > 0 or any(a == i for a in full.assign):
            continue
        rs = [r for r in wait_reads if read_t[r] >= u["b"] + EPS_MISS]
        for r in rs:
            miss[r] = miss.get(r, 0.0) - 1.0 / n_u / len(rs)
    return Credit(full.R, fq.R, latency(segs, units, full, dur), delta, delta_q, who, full.assign, full.S, fq.S, miss,
                  len(units), full.spurious, full.lang)


def flad_reward(Q: float, L: float, anchors: list[tuple[float, float]]) -> float:
    """C3: the latency reward is earned only when no anchor at most DELTA_LAT slower is better by more than EPS_DOM."""
    ref = [qa for qa, la in anchors if la <= L + DELTA_LAT]
    if ref and Q < max(ref) - EPS_DOM:
        return -L_MAX
    return -min(L, L_MAX)


def hpo_gate_reward(Q: float, L: float, thr: float = 0.3) -> float:
    """C3 ablation (HPO's absolute gate): latency counts only above a fixed quality threshold."""
    return -min(L, L_MAX) if Q >= thr else -L_MAX


def corpus_bleu_pairs(pairs: list[tuple[str, str]]) -> float:
    """corpus BLEU (13a) over (hyp, ref) pairs, in [0, 1]; an unmatched side is paired with ""."""
    from sacrebleu.metrics import BLEU
    if not pairs or not any(h.strip() for h, _ in pairs):
        return 0.0
    return BLEU(tokenize="13a").corpus_score([h for h, _ in pairs], [[r for _, r in pairs]]).score / 100.0


def seq_cpbleu(segs: list[dict], units: list[dict]) -> float:
    """C1 ablation / baseline (d), DiaScriber-style: the window's speaker-attributed BLEU -- per-slot and
    per-speaker concatenations, Hungarian-matched on chrF (cpBLEU), unmatched sides against empty."""
    pi = speaker_map(segs, units)
    ht, rt = {}, {}
    for s in sorted(segs, key=lambda s: s["a"]):
        ht[s["spk"]] = (ht.get(s["spk"], "") + " " + s["text"]).strip()
    for u in units:
        rt[u["spk"]] = (rt.get(u["spk"], "") + " " + u["ref"]).strip()
    pairs, used = [], set()
    for k, h in ht.items():
        r = pi.get(k)
        pairs.append((h, rt.get(r, "") if r is not None else ""))
        if r is not None:
            used.add(r)
    pairs += [("", t) for r, t in rt.items() if r not in used]
    return corpus_bleu_pairs(pairs)


def window_quality(segs: list[dict], units: list[dict]) -> float:
    """baseline (a): speaker-agnostic quality of the whole window translation (Q of the time-ordered concatenations)."""
    hyp = " ".join(s["text"] for s in sorted(segs, key=lambda s: s["a"]))
    ref = " ".join(u["ref"] for u in sorted(units, key=lambda u: u["a"]))
    return q_chrf_bleu(hyp, ref)


def prefix_bleu(segs: list[dict], units: list[dict], r: int, t: float) -> float:
    """baseline (b), Hibiki-Zero's process reward at a checkpoint: sentence BLEU of the text committed by read r
    (speaker-agnostic, by start) against the references ending by t."""
    from sacrebleu.metrics import BLEU
    hyp = " ".join(s["text"] for s in sorted((s for s in segs if s["r"] <= r), key=lambda s: s["a"]))
    ref = " ".join(u["ref"] for u in units if u["b"] <= t)
    if not ref.strip():
        return 0.0 if hyp.strip() else 1.0
    if not hyp.strip():
        return 0.0
    return BLEU(effective_order=True, tokenize="13a").sentence_score(hyp, [ref]).score / 100.0
