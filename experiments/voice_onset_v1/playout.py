"""voice_onset_v1 playout (reports/voice_onset_v1/S0_S1_S2_prereg.zh-CN.md sections 1, 2, 4 + change 1).

An si_infer item = blocks i with arrival a_i (write_ms), an interval [s_i, e_i) of the one-pass decoded speech (ms), n_i
output units, a punctuation-end flag and a final flag. A policy turns it into timeline segments (x0, x1, t0): speech
[x0, x1) played from t0. Policies:
  none  the stored rendering (block i at max(a_i, previous end)); no cut move. Must reproduce laal_defs D2 exactly.
  r0    rule player: cut move (stop at the last quiet 20 ms frame, < ref - 25 dB, within the last 300 ms of the audio
        that has arrived when the next block has not arrived by then; the tail waits for the next block), and when idle
        with a non-final pending group of <= 3 units not ending in punctuation, hold until the next block (re-checked)
        for at most 1000 ms from the start of the hold.
  s2    learned player: the same cut move; the hold length j in {0..J} steps of 250 ms from a release hazard
        h_k = sigmoid(MLP([10 causal features])) (left-to-right semi-Markov duration, h_J = 1); deployed: first h_k >= 0.5.
Metrics: D1 / D2 (laal_defs conventions, units mapped through the segments), NP_in (in-phrase pauses > 300 ms, count),
PT_in / PT_all (s), NC (pauses > 300 ms after a voiced stop: last 40 ms > ref - 15 dB), EndOffset (s).
"""
from __future__ import annotations

import bisect
import math
import re
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import laal_defs  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal as si_laal, units as si_units  # noqa: E402

LD = laal_defs()
PUNCT = re.compile(r"[，。,.!?！？；;：:、…]\s*$")
FR, GAP, DIP_WIN, DIP_DB, VOICED_DB = 20, 300, 300, -25.0, -15.0
HOLD_U, HOLD_MS = 3, 1000
STEP, J = 250, 8
NF = 10


def read16(p) -> np.ndarray:
    with wave.open(str(p)) as w:
        n, sr, ch = w.getnframes(), w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768
    assert sr == 16000, (p, sr)
    return x.reshape(-1, ch).mean(1) if ch > 1 else x


def frames_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // 320
    return 10 * np.log10((x[: n * 320].reshape(n, 320).astype(np.float64) ** 2).mean(1) + 1e-10)


class Item:
    def __init__(self, r: dict, lang: str, speech=None, src=None):
        self.id, self.X, self.r = r["id"], int(r["source_duration_ms"]), r
        self.T = len(si_units(r["translation_ref"], lang))
        self.P = [dict(a=int(b["write_ms"]), s=int(b["speech_at_ms"]), e=int(b["speech_at_ms"]) + int(b["dur_ms"]),
                       n=len(si_units(b["text"], lang)), punct=bool(PUNCT.search(b["text"])), final=bool(b["final"]))
                  for b in r["blocks"]]
        self.A = [p["a"] for p in self.P]
        assert self.A == sorted(self.A), self.id
        for p, q in zip(self.P, self.P[1:]):
            assert p["e"] == q["s"], self.id
        self.E = self.P[-1]["e"] if self.P else 0
        self.Utot = sum(p["n"] for p in self.P)
        self.speech = read16(r["speech_path"]) if speech is None else speech
        self.db = frames_db(self.speech)
        self.ref = float(np.percentile(self.db, 90)) if len(self.db) else 0.0
        s = read16(r["source_audio_path"]) if src is None else src
        self.spow = 10 ** (frames_db(s) / 10)
        self.ends = {p["e"]: p["punct"] for p in self.P}   # block ends -> punctuation flag

    def arrived(self, t: float) -> int:
        return bisect.bisect_right(self.A, t)

    def voiced_before(self, x: int) -> bool:
        f = x // FR
        seg = self.db[max(0, f - 2): f]
        return len(seg) > 0 and float(seg.max()) > self.ref + VOICED_DB

    def dip(self, lo: int, hi: int):
        """center of the last quiet frame fully inside [max(lo, hi - 300), hi), else None"""
        start = max(lo, hi - DIP_WIN)
        f = hi // FR - 1
        while f >= 0 and FR * f >= start:
            if FR * f + FR <= hi and f < len(self.db) and self.db[f] < self.ref + DIP_DB:
                return FR * f + FR // 2
            f -= 1
        return None

    def src_db(self, tau: float) -> float:
        hi = int(tau) // FR
        lo = max(0, hi - 500 // FR)
        seg = self.spow[lo:hi]
        return float(10 * np.log10(seg.mean() + 1e-10)) if len(seg) else -100.0


# ---------------------------------------------------------------- simulation
def play(it: Item, x: int, t: float, dip: bool):
    """play from cursor x at time t until the arrived audio runs out; -> (x0, x1, t0)"""
    P, A = it.P, it.A
    k = it.arrived(t) - 1
    x0, t0, end = x, t, P[k]["e"]
    while True:
        tend = t0 + (end - x0)
        nxt = A[k + 1] if k + 1 < len(P) else math.inf
        if dip and not P[k]["final"]:
            xd = it.dip(x0, end)
            if xd is not None and xd > x0 and not (nxt <= t0 + (xd - x0)):
                return (x0, xd, t0)
        if nxt <= tend:
            k += 1
            end = P[k]["e"]
            continue
        return (x0, end, t0)


def piece_of(it: Item, x: int) -> int:
    for m, p in enumerate(it.P):
        if p["s"] <= x < p["e"]:
            return m
    return len(it.P) - 1


def simulate(it: Item, policy=None, dip: bool = False, x: int = 0, t=None, until=None, segs=None):
    """policy(it, t, x, segs) -> release time >= t (None: release at once). Returns the segment list."""
    P, A = it.P, it.A
    segs = [] if segs is None else segs
    if not P:
        return segs
    t = A[0] if t is None else t
    while x < it.E and (until is None or x < until):
        k = it.arrived(t) - 1
        if k < 0 or P[k]["e"] <= x:
            nxt = next((A[m] for m in range(max(k + 1, 0), len(P)) if P[m]["e"] > x), None)
            if nxt is None:
                break
            t = max(t, nxt)
            continue
        m = piece_of(it, x)
        if x > P[m]["s"] and m + 1 < len(P) and A[m + 1] > t:   # a cut-move tail waits for the next block
            t = A[m + 1]
            continue
        if policy is not None:
            t = policy(it, t, x, segs)
        x0, x1, t0 = play(it, x, t, dip)
        segs.append((x0, x1, t0))
        t, x = t0 + (x1 - x0), x1
    return segs


# ---------------------------------------------------------------- metrics
def unit_times(it: Item, segs: list, pieces=None):
    """D1 / D2 delays (ms) of every unit (of `pieces`, default all) through the segments"""
    xs0 = [s[0] for s in segs]

    def t_end(x):   # segment with x0 < x <= x1
        g = bisect.bisect_left(xs0, x) - 1
        g = max(g, 0)
        x0, x1, t0 = segs[g]
        return t0 + (min(x, x1) - x0)

    def t_start(x):   # segment with x0 <= x < x1
        g = max(bisect.bisect_right(xs0, x) - 1, 0)
        x0, x1, t0 = segs[g]
        if x >= x1 and g + 1 < len(segs):
            return segs[g + 1][2]
        return t0 + (x - x0)

    d1, d2 = [], []
    for i, p in enumerate(it.P):
        if pieces is not None and i not in pieces:
            continue
        n = p["n"]
        if n == 0:
            continue
        if p["e"] > p["s"]:
            ts = t_start(p["s"])
            d1 += [ts] * n
            d2 += [t_end(p["s"] + (k + 1) / n * (p["e"] - p["s"])) for k in range(n)]
        else:   # a block without audio: at its arrival or when the cursor reaches it
            ts = max(p["a"], t_end(p["s"]) if p["s"] > 0 else segs[0][2])
            d1 += [ts] * n
            d2 += [ts] * n
    return d1, d2


def pauses(it: Item, segs: list, prev=None):
    """[(gap_ms, in_phrase, voiced_cut)] between consecutive segments (prev = (stop_time, stop_x) before segs[0])"""
    out = []
    last_end, last_x = None, None
    if prev:
        last_end, last_x = prev
    for (x0, x1, t0) in segs:
        if last_end is not None:
            gap = t0 - last_end
            if gap > GAP:
                out.append((gap, not it.ends.get(last_x, False), it.voiced_before(last_x)))
        last_end, last_x = t0 + (x1 - x0), x1
    return out


def metrics(it: Item, segs: list) -> dict | None:
    d1, d2 = unit_times(it, segs)
    if not d1:
        return None
    ps = pauses(it, segs)
    end = segs[-1][2] + (segs[-1][1] - segs[-1][0])
    return {"D1": si_laal(d1, it.X, it.T), "D2": LD.laal_unitwise(d2, it.X, it.T),
            "NP_in": sum(1 for g, ip, _ in ps if ip), "PT_in": sum(g for g, ip, _ in ps if ip) / 1000,
            "PT_all": sum(g for g, _, _ in ps) / 1000, "NC": sum(1 for _, _, v in ps if v), "n_pause": len(ps),
            "EndOffset": (end - it.X) / 1000, "n_seg": len(segs)}


# ---------------------------------------------------------------- policies
def r0(it: Item, t: float, x: int, segs) -> float:
    h0 = t
    while True:
        k = it.arrived(t) - 1
        pend = [p for p in it.P[: k + 1] if p["e"] > x]
        if not pend or any(p["final"] for p in pend):
            return t
        if sum(p["n"] for p in pend) > HOLD_U or it.P[k]["punct"]:
            return t
        if k + 1 < len(it.P) and it.P[k + 1]["a"] <= h0 + HOLD_MS:
            t = max(t, it.P[k + 1]["a"])
            continue
        return h0 + HOLD_MS


def feats(it: Item, tau: float, tau0: float, x: int) -> list:
    k = it.arrived(tau) - 1
    k0 = it.arrived(tau0) - 1
    pend = [p for p in it.P[: k + 1] if p["e"] > x]
    U = sum(p["n"] for p in pend)
    a_last = it.P[k]["a"]
    e = it.P[k]["e"]
    f = e // FR
    seg = it.db[max(0, f - 2): f]
    endv = (float(seg.max()) - it.ref) if len(seg) else -60.0
    mean_int = (a_last - it.P[0]["a"]) / k if k > 0 else 1000.0
    return [math.log1p(U), float(it.P[k]["punct"]), float(k > k0), (tau - tau0) / 1000, (tau - a_last) / 1000,
            max(it.src_db(tau), -100.0) / 10, float(tau >= it.X), (e - x) / 1000, max(endv, -60.0) / 10, mean_int / 1000]


def release_grid(it: Item, tau0: float, x: int):
    """release times r_0..r_jmax (r_j = tau0 + 250 j; a final block arriving inside the hold releases at its arrival,
    which ends the grid) -> (times, forced_last)"""
    times = []
    for j in range(J + 1):
        tau = tau0 + STEP * j
        k = it.arrived(tau) - 1
        fin = [p["a"] for p in it.P[: k + 1] if p["final"] and p["e"] > x]
        if fin:
            times.append(max(tau0, min(fin)))
            return times, True
        times.append(tau)
    return times, True


class S2Policy:
    """deployed learned player; `rec` collects decision points (for training)"""
    def __init__(self, hz, rec=None):
        self.hz, self.rec = hz, rec

    def __call__(self, it: Item, t: float, x: int, segs) -> float:
        k = it.arrived(t) - 1
        pend = [p for p in it.P[: k + 1] if p["e"] > x]
        if not pend or any(p["final"] for p in pend):
            return t
        times, _ = release_grid(it, t, x)
        F = [feats(it, tau, t, x) for tau in times]
        h = self.hz(np.asarray(F, dtype=np.float32))
        jd = next((j for j in range(len(times) - 1) if h[j] >= 0.5), len(times) - 1)
        if self.rec is not None:
            prev = (segs[-1][2] + (segs[-1][1] - segs[-1][0]), segs[-1][1]) if segs else None
            self.rec.append((it, t, x, prev, F, times))
        return times[jd]


def costs(it: Item, x: int, times: list, prev, alpha: float) -> np.ndarray:
    """c(j) = sum of D2 times of U_set / U_total (s) + alpha * NP_in in the window, for every release time"""
    kmax = it.arrived(times[-1]) - 1
    U = {i for i in range(kmax + 1) if it.P[i]["e"] > x}
    until = max(it.P[i]["e"] for i in U)
    out = []
    for r in times:
        segs = simulate(it, None, dip=True, x=x, t=r, until=until)
        _, d2 = unit_times(it, segs, pieces=U)
        npi = sum(1 for g, ip, _ in pauses(it, segs, prev) if ip)
        out.append(sum(d2) / max(it.Utot, 1) / 1000 + alpha * npi)
    return np.asarray(out, dtype=np.float64)


def mlp_numpy(state: dict):
    """hazard function from a torch state dict (Linear 10->32, tanh, Linear 32->1)"""
    W1, b1 = state["net.0.weight"].numpy(), state["net.0.bias"].numpy()
    W2, b2 = state["net.2.weight"].numpy(), state["net.2.bias"].numpy()
    mu, sd = state["mu"].numpy(), state["sd"].numpy()

    def hz(F):
        z = np.tanh(((F - mu) / sd) @ W1.T + b1) @ W2.T + b2
        h = 1 / (1 + np.exp(-z[:, 0]))
        h[-1] = 1.0
        return h
    return hz


def render(it: Item, segs: list, path) -> None:
    """listening wav: segments on the timeline, 20 ms fade-out at voiced stops followed by silence, 10 ms fade-in"""
    import soundfile as sf
    end = segs[-1][2] + (segs[-1][1] - segs[-1][0])
    out = np.zeros(int(max(end, it.X) + 300) * 16, dtype=np.float32)
    for g, (x0, x1, t0) in enumerate(segs):
        piece = it.speech[x0 * 16: x1 * 16].copy()
        nxt_gap = (segs[g + 1][2] - (t0 + x1 - x0)) if g + 1 < len(segs) else 0
        prv_gap = (t0 - (segs[g - 1][2] + segs[g - 1][1] - segs[g - 1][0])) if g else 0
        if nxt_gap > 0 and it.voiced_before(x1) and len(piece) > 320:
            piece[-320:] *= np.cos(np.linspace(0, np.pi / 2, 320)) ** 2
        if prv_gap > 0 and len(piece) > 160:
            piece[:160] *= np.sin(np.linspace(0, np.pi / 2, 160)) ** 2
        a = int(t0 * 16)
        out[a: a + len(piece)] += piece
    sf.write(str(path), out, 16000)
