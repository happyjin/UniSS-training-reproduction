"""SpecClose: speculative window close for the online diarizer windows (reports/mist_w3po_v1/
thinker_detector_specclose_prereg_v1.zh-CN.md). CPU only. Additive: no existing tool is changed.

The C2 window rule (build_windows_zh2en.online_windows, gap 0.6 s = 8 frames of 80 ms, cap 20 s) on the low-latency
Nemotron posteriors P. The window jsonl stamps each detection with the preset's WORST-CASE delay (a frame known 1.04 s
after its start). The lowlat preset (dump_nemotron_posteriors.PRESETS["lowlat"]: chunk_len 9, chunk_right_context 4;
NeMo streaming_feat_loader chunks from frame 0) emits the 9 frames of chunk c at once, when the audio up to
(9c + 13) * 80 ms has arrived:  T(f) = (9 floor(f/9) + 13) * 80 ms  <=  f * 80 ms + 1.04 s.

Two layers, each switchable:
  chunk  (protocol)   open = T(st); a gap / cap close = T(j - 1) (the decision needs the frames up to j - 1);
                      session end as before. Same windows, same posteriors.
  judge  (method)     at each pause of an open window -- f = the first silent frame after an active one -- the gap's
                      audio (frames f .. f + 7) has arrived at (f + 8) * 80 ms and P of every frame <= chunk_end(f) =
                      9 floor(f/9) + 8 is out at T(f). At tau_f = max(T(f), (f + 8) * 80 ms):
                        - a P-known frame in (f, f + 7] is active -> nothing (the window goes on);
                        - all of (f, f + 7] P-known (f mod 9 <= 1)  -> nothing (the chunk close T(f + 7) = T(f) applies);
                        - else the judge (a small classifier on the log-mel audio of frames f-2 .. f+7, distilled from
                          P on train meetings) predicts whether any of the u = f + 7 - chunk_end(f) P-unknown frames is
                          active; p_hold < theta -> CLOSE at tau_f: the window is [st, f) (its final read at tau_f,
                          audio to f * 80 ms + 0.3 s as the builder's), and the next window starts at the next active
                          frame (a FALSE close splits the C2 window there).
                      Every active frame of a closed window has its P out at tau_f (T is monotone and f-1 < f), so the
                      final read's attribution input is that of the base final read; only silent pad frames can be
                      masked. The final read sees P of the frames out at its close time: final_post_cut_ms.

Subcommands
  train-judge  train meetings only (meetings.json split train, A8-passing): candidates from the C2 windows (every pause
               of every window, augmented over u = 1..7: label = any active frame among the last u frames of
               (f, f + 7] given the first 7 - u are silent), 10 % of the meetings held out (sorted, every 10th);
               MLP 807 -> 128 -> 1; epoch by hold-out BCE; theta = the largest threshold with false-close rate <= 1 %
               on the hold-out meetings' real candidates (u from f mod 9). Writes <out>/judge.pt + judge.json.
  build        dev meetings (the validation meetings of the reference windows): windows jsonl with the chunk times,
               the judge (or --judge off: chunk only), final_post_cut_ms, close_kind, orig_id; new audio / posterior
               files only for windows that differ from the C2 window; plus windows_dev_retime.jsonl (the C2 windows
               with the method's open / close where the C2 window is unchanged; a C2 window the judge split keeps its
               chunk times) for CPU re-timing of existing decodes; judge.jsonl (every dev judge call); summary.json.
Refuses existing outputs.

    python -m experiments.mist_w3po_v1.data.specclose train-judge --out <dir>
    python -m experiments.mist_w3po_v1.data.specclose build --judge <dir>|off --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

FR = 0.08                       # s per posterior frame
FR_MS = 80
LA = 1.04                       # worst-case look-ahead of the jsonl convention
CHUNK, RC = 9, 4                # lowlat chunk_len, chunk_right_context (80 ms frames)
GAP = round(0.6 / FR)           # 8 frames (C2 gap 0.6 s; build_windows_zh2en rounds 7.5 -> 8)
CAP = round(20.0 / FR)          # 250 frames
CTX0, CTX1 = 2, 7               # judge audio frames f - CTX0 .. f + CTX1
N_MEL = 40
FEAT = (CTX0 + CTX1 + 1) * 2 * N_MEL + GAP - 1   # 807
DATA = Path("data/ami_multispk_si_v1")
DIAR = "nemotron_lowlat_zh8"
MIX = "zh_mix_v8"
REF = Path("data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl")
BASE_WIN = Path("eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl")
SEED = 20261112
FCR_TARGET = 0.01


def t_chunk(f: int) -> float:
    """absolute time (s) at which the lowlat streaming preset emits posterior frame f"""
    return (CHUNK * (f // CHUNK) + CHUNK + RC) * FR


def chunk_end(f: int) -> int:
    return CHUNK * (f // CHUNK) + CHUNK - 1


def vis_end(tau: float) -> int:
    """the last frame whose posterior is out at absolute time tau (-1: none)"""
    c = tau / FR - CHUNK - RC
    if c < -1e-9:
        return -1
    return CHUNK * int(math.floor(c / CHUNK + 1e-9)) + CHUNK - 1


def simulate(act: np.ndarray, judge=None) -> list[dict]:
    """the C2 online window rule (build_windows_zh2en.online_windows, gap GAP, cap CAP) with optional speculative
    closes. judge(f, u) -> True = CLOSE at tau_f. Returns per window {st, en (last active + 1), j (the frame the rule
    stopped at; for a spec close j = f), kind (gap | cap | end | spec), tau (spec only), calls: [(f, u, closed)]}.
    With judge None the (st, en, j) are exactly online_windows(post, 0.6)'s."""
    n = len(act)
    out, i = [], 0
    while i < n:
        if not act[i]:
            i += 1
            continue
        st, last, j = i, i, i
        calls = []
        kind = None
        while True:
            if j >= n:
                kind = "end"
                break
            if j - st >= CAP:
                kind = "cap"
                break
            if j - last > GAP:
                kind = "gap"
                break
            if act[j]:
                last = j
            elif judge is not None and j - 1 == last and j + GAP - 1 < n and j + GAP - st <= CAP:
                f = j
                ce = chunk_end(f)
                known_hi = min(ce, f + GAP - 1)
                u = f + GAP - 1 - ce
                if u >= 1 and not act[f + 1: known_hi + 1].any():
                    closed = bool(judge(f, u))
                    calls.append((f, u, closed))
                    if closed:
                        kind = "spec"
                        break
            j += 1
        w = {"st": st, "en": last + 1, "j": j, "kind": kind, "calls": calls}
        if kind == "spec":
            w["tau"] = max(t_chunk(j), (j + GAP) * FR)
        out.append(w)
        i = last + 1
    return out


def close_time(w: dict, sess_s: float, chunk: bool = True) -> float:
    """absolute close detection time (s) of a simulated window"""
    if w["kind"] == "end":
        return sess_s
    if w["kind"] == "spec":
        return min(sess_s, w["tau"])
    if chunk:
        return min(sess_s, t_chunk(w["j"] - 1))
    return min(sess_s, w["j"] * FR + LA)


def window_row(w: dict, m: str, sess_s: float, chunk: bool = True) -> dict:
    """the builder's row fields (build_windows_zh2en) + the SpecClose fields; no audio / post paths."""
    st, en = w["st"], w["en"]
    ws, we = max(0.0, st * FR - 0.2), min(sess_s, en * FR + 0.3)
    ws_r = round(ws, 3)
    dur_ms = round((we - ws) * 1000)
    open_t = t_chunk(st) if chunk else st * FR + LA
    close = close_time(w, sess_s, chunk)
    f0 = round(ws / FR)
    if w["kind"] == "end":
        cut = dur_ms
    else:
        ve = vis_end(close) if chunk else int(math.floor((close - LA) / FR + 1e-9)) - 1
        cut = min(dur_ms, max(0, (ve - f0 + 1) * FR_MS))
    return {"meeting": m, "start_s": ws_r, "dur_ms": dur_ms,
            "open_detect_ms": round((open_t - ws) * 1000), "close_detect_ms": round((close - ws) * 1000),
            "final_post_cut_ms": int(cut), "close_kind": w["kind"], "st_frame": st, "en_frame": en, "j_frame": w["j"],
            "f0": f0}


# ---------------------------------------------------------------- judge features
def frame_feats(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    """[n80, 2 * N_MEL]: per 80 ms frame g, mean and max of the 40 log-mel sub-frames 8g-2 .. 8g+5 (kaldi fbank,
    25 ms / 10 ms, snip_edges: sub-frame k covers [10k, 10k + 25) ms, so every sub-frame of g ends by (g + 1) * 80 ms)."""
    import torch
    import torchaudio
    assert sr == 16000
    fb = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None] * 32768.0,
                                          num_mel_bins=N_MEL, frame_length=25.0, frame_shift=10.0, dither=0.0,
                                          energy_floor=0.0, sample_frequency=16000.0).numpy()
    ns = len(fb)
    n80 = int(math.ceil(len(x) / sr / FR))
    out = np.zeros((n80, 2 * N_MEL), dtype=np.float32)
    for g in range(n80):
        a, b = max(0, 8 * g - 2), min(ns, 8 * g + 6)
        if b > a:
            seg = fb[a:b]
            out[g, :N_MEL] = seg.mean(0)
            out[g, N_MEL:] = seg.max(0)
        else:
            out[g] = out[g - 1] if g else 0.0
    return out


def cand_vec(F: np.ndarray, f: int, u: int) -> np.ndarray:
    """judge input for a pause at frame f with u P-unknown frames: frames f - CTX0 .. f + CTX1 (edge-clipped) + onehot(u)"""
    idx = np.clip(np.arange(f - CTX0, f + CTX1 + 1), 0, len(F) - 1)
    oh = np.zeros(GAP - 1, dtype=np.float32)
    oh[u - 1] = 1.0
    return np.concatenate([F[idx].reshape(-1), oh])


def load_meeting(m: str):
    import soundfile as sf
    import torch
    post = torch.load(DATA / "diar" / DIAR / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
    x, sr = sf.read(str(DATA / "session_audio" / MIX / f"{m}.wav"), dtype="float32", always_2d=True)
    return post, x.mean(1), sr


def candidates(act: np.ndarray, augment: bool) -> list[tuple[int, int, int, bool]]:
    """(f, u, label hold, real) for every pause of every C2 window (no judge): real = u is the chunk phase's u;
    augment adds every other u in 1..7 whose first 7 - u gap frames are silent."""
    out = []
    n = len(act)
    for w in simulate(act):
        for f in range(w["st"] + 1, min(w["j"], n)):
            if not (act[f - 1] and not act[f]) or f + GAP - 1 >= n or f + GAP - w["st"] > CAP:
                continue
            gapv = act[f + 1: f + GAP]
            ur = f + GAP - 1 - chunk_end(f)
            for u in range(1, GAP):
                real = u == ur
                if not (real or augment):
                    continue
                if gapv[: GAP - 1 - u].any():
                    continue
                out.append((f, u, bool(gapv[GAP - 1 - u:].any()), real))
    return out


def meeting_cands(m: str):
    """(m, feature rows, hold labels, real flags) of one train meeting (augmented candidates)"""
    import torch
    torch.set_num_threads(1)
    post, x, sr = load_meeting(m)
    act = (post > 0.5).any(1)
    F = frame_feats(x, sr)
    X, Y, R = [], [], []
    for f, u, y, real in candidates(act, augment=True):
        if f + CTX1 >= len(F):
            continue
        X.append(cand_vec(F, f, u))
        Y.append(y)
        R.append(real)
    return m, X, Y, R


class Judge:
    def __init__(self, d: str):
        import torch
        z = torch.load(Path(d) / "judge.pt", weights_only=False)
        self.net = make_net()
        self.net.load_state_dict(z["state"])
        self.net.eval()
        self.mu, self.sd, self.theta = z["mu"], z["sd"], float(z["theta"])

    def p_hold(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.from_numpy((X - self.mu) / self.sd))).numpy()[:, 0]


def make_net():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(FEAT, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(128, 1))


def theta_for(p_hold: np.ndarray, y: np.ndarray, target: float = FCR_TARGET) -> float:
    """the largest theta with false-close rate P(p_hold < theta | hold) <= target (close iff p_hold < theta)"""
    ph = np.sort(p_hold[y == 1])
    k = int(math.floor(target * len(ph)))      # at most k holds strictly below theta
    return float(ph[k]) if k < len(ph) else 1.0


def train_meetings() -> list[str]:
    split = json.load(open(DATA / "meetings.json"))
    skip = set(json.load(open(DATA / MIX / "audit_diar.json"))["summary"]["failing_meetings"])
    return sorted(m for m, s in split.items() if s == "train" and m not in skip)


def cmd_train(a) -> None:
    import torch
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    ms = train_meetings()
    if a.limit_meetings:
        ms = ms[: a.limit_meetings]
    hold = set(ms[::10])
    Xs = {"tr": [], "ho": []}
    Ys = {"tr": [], "ho": []}
    R = {"tr": [], "ho": []}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        for m, X, y, real in ex.map(meeting_cands, ms):
            k = "ho" if m in hold else "tr"
            Xs[k].append(X)
            Ys[k].append(y)
            R[k].append(real)
            print(f"[judge] {m} ({k}) {len(y)} candidates", flush=True)
    Xs = {k: [x for X in v for x in X] for k, v in Xs.items()}
    Ys = {k: [x for X in v for x in X] for k, v in Ys.items()}
    R = {k: [x for X in v for x in X] for k, v in R.items()}
    Xtr, ytr = np.stack(Xs["tr"]).astype(np.float32), np.asarray(Ys["tr"], dtype=np.float32)
    Xho, yho, rho = np.stack(Xs["ho"]).astype(np.float32), np.asarray(Ys["ho"], dtype=np.float32), np.asarray(R["ho"])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(a.threads)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy((Xtr - mu) / sd), torch.from_numpy(ytr)[:, None]
    Xh, yh = torch.from_numpy((Xho - mu) / sd), torch.from_numpy(yho)[:, None]
    best, log = None, []
    g = torch.Generator().manual_seed(SEED)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        tl = 0.0
        for b in range(0, len(Xt), 512):
            ix = perm[b: b + 512]
            opt.zero_grad()
            loss = lossf(net(Xt[ix]), yt[ix])
            loss.backward()
            opt.step()
            tl += float(loss) * len(ix)
        net.eval()
        with torch.no_grad():
            hl = float(lossf(net(Xh), yh))
        log.append({"epoch": ep + 1, "train_bce": round(tl / len(Xt), 5), "holdout_bce": round(hl, 5)})
        print(f"[judge] epoch {ep + 1}: train {tl / len(Xt):.4f} holdout {hl:.4f}", flush=True)
        if best is None or hl < best[0]:
            best = (hl, ep + 1, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[2])
    net.eval()
    with torch.no_grad():
        ph = torch.sigmoid(net(Xh)).numpy()[:, 0]
    real = rho.astype(bool)
    theta = theta_for(ph[real], yho[real])
    close = ph < theta
    rep = {"meetings_train": len(ms) - len(hold), "meetings_holdout": sorted(hold), "n_train": int(len(ytr)),
           "n_holdout": int(len(yho)), "holdout_real": int(real.sum()), "epochs": log, "best_epoch": best[1],
           "theta": theta, "fcr_target": FCR_TARGET,
           "holdout_real_fcr": float(close[real & (yho == 1)].mean()) if (real & (yho == 1)).any() else None,
           "holdout_real_recall": float(close[real & (yho == 0)].mean()) if (real & (yho == 0)).any() else None,
           "holdout_real_n_hold": int((real & (yho == 1)).sum()), "holdout_real_n_close": int((real & (yho == 0)).sum()),
           "feat": FEAT, "ctx": [CTX0, CTX1], "n_mel": N_MEL, "seed": SEED}
    out.mkdir(parents=True)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "theta": theta, "report": rep}, out / "judge.pt")
    with open(out / "judge.json", "x") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "epochs"}), flush=True)


# ---------------------------------------------------------------- build
def dev_meetings() -> list[str]:
    return sorted({json.loads(l)["meeting"] for l in open(REF)})


def cmd_build(a) -> None:
    import soundfile as sf
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    base = defaultdict(list)
    for w in map(json.loads, open(a.base_win, encoding="utf-8")):
        base[w["meeting"]].append(w)
    J = None if a.judge == "off" else Judge(a.judge)
    out.mkdir(parents=True)
    fo = open(out / "windows_dev_online.jsonl", "x", encoding="utf-8")
    fr = open(out / "windows_dev_retime.jsonl", "x", encoding="utf-8")
    fj = open(out / "judge.jsonl", "x", encoding="utf-8")
    S = defaultdict(int)
    for m in sorted(base):
        post, x, sr = load_meeting(m)
        sess_s = len(x) / sr
        act = (post > 0.5).any(1)
        F = frame_feats(x, sr) if J is not None else None
        decided = {}

        def judge(f, u):
            p = float(J.p_hold(cand_vec(F, f, u)[None].astype(np.float32))[0])
            c = p < J.theta
            res = act[f + 1: f + GAP].any()
            decided[f] = (u, p, c, bool(res))
            return c

        b0 = simulate(act)                         # the C2 windows
        bw = base[m]
        assert len(b0) == len(bw), (m, len(b0), len(bw))
        for w, r in zip(b0, bw):
            k = int(r["id"].rsplit("_", 1)[1])
            assert r["id"] == f"{m}_on_{k:04d}" and abs(round(max(0.0, w['st'] * FR - 0.2), 3) - r["start_s"]) < 1e-6, r["id"]
            assert abs(round((min(sess_s, w["en"] * FR + 0.3) - max(0.0, w["st"] * FR - 0.2)) * 1000) - r["dur_ms"]) <= 1, r["id"]
        sims = simulate(act, judge if J is not None else None)
        key0 = {(w["st"], w["en"]): (w, r) for w, r in zip(b0, bw)}
        # orig id: the C2 window whose [st, j) holds the new window's st
        def orig_of(st):
            for w, r in zip(b0, bw):
                if w["st"] <= st < max(w["j"], w["en"]):
                    return r
            raise AssertionError((m, st))
        pieces = defaultdict(int)
        unchanged = set()
        for w in sims:
            row = window_row(w, m, sess_s, chunk=True)
            hit = key0.get((w["st"], w["en"]))
            if hit is not None:
                r0 = hit[1]
                row = {"id": r0["id"], **{k: r0[k] for k in ("split", "wtype", "dir", "audio", "post")}, **row,
                       "orig_id": r0["id"], "utts": [], "strict_review": False}
                assert row["start_s"] == r0["start_s"] and abs(row["dur_ms"] - r0["dur_ms"]) <= 1
                row["dur_ms"] = r0["dur_ms"]
                row["final_post_cut_ms"] = min(row["final_post_cut_ms"], r0["dur_ms"])
                unchanged.add(r0["id"])
                S["unchanged"] += 1
            else:
                r0 = orig_of(w["st"])
                pieces[r0["id"]] += 1
                wid = f"{r0['id']}p{pieces[r0['id']]}"
                ws, we = row["start_s"], row["start_s"] + row["dur_ms"] / 1000
                wav = out / "audio" / "validation" / f"{wid}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
                nf = int(np.ceil((we - ws) / FR))
                s = post[row["f0"]: row["f0"] + nf]
                if len(s) < nf:
                    s = np.concatenate([s, np.zeros((nf - len(s), post.shape[1]), dtype=s.dtype)])
                pp = out / "post" / "validation" / f"{wid}.npy"
                pp.parent.mkdir(parents=True, exist_ok=True)
                np.save(pp, s.astype(np.float16))
                row = {"id": wid, "split": "validation", "wtype": "online", "dir": "zh2en", "audio": str(wav.resolve()),
                       "post": {"lowlat": str(pp.resolve())}, **row, "orig_id": r0["id"], "utts": [],
                       "strict_review": False}
                S["new_piece"] += 1
            S[f"kind_{w['kind']}"] += 1
            fo.write(json.dumps(row) + "\n")
        # retime file: every C2 window; the method's times where the C2 window is unchanged, else chunk times
        tm = {(w["st"], w["en"]): w for w in sims}
        for w, r in zip(b0, bw):
            ww = tm.get((w["st"], w["en"])) if r["id"] in unchanged else None
            row = window_row(ww if ww is not None else w, m, sess_s, chunk=True)
            rr = dict(r)
            rr.update({k: row[k] for k in ("open_detect_ms", "close_detect_ms", "final_post_cut_ms", "close_kind")})
            rr["final_post_cut_ms"] = min(rr["final_post_cut_ms"], r["dur_ms"])
            rr["split_by_judge"] = r["id"] not in unchanged
            S["base_split"] += int(r["id"] not in unchanged)
            fr.write(json.dumps(rr) + "\n")
        for f, (u, p, c, res) in sorted(decided.items()):
            fj.write(json.dumps({"meeting": m, "f": f, "u": u, "p_hold": round(p, 5), "close": c, "resumes": res}) + "\n")
            S["calls"] += 1
            S["calls_hold_truth"] += res
            S["false_close"] += int(c and res)
            S["true_close"] += int(c and not res)
            S["missed_close"] += int((not c) and not res)
        print(f"[build] {m}: {len(sims)} windows (C2 {len(b0)})", flush=True)
    for fh in (fo, fr, fj):
        fh.close()
    summ = dict(S)
    summ["judge"] = a.judge
    summ["fcr"] = round(S["false_close"] / max(S["calls_hold_truth"], 1), 4)
    summ["recall"] = round(S["true_close"] / max(S["true_close"] + S["missed_close"], 1), 4)
    with open(out / "summary.json", "x") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train-judge")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--threads", type=int, default=16)
    t.add_argument("--workers", type=int, default=16)
    t.add_argument("--limit-meetings", type=int, default=0, help="smoke runs only")
    b = sp.add_parser("build")
    b.add_argument("--judge", required=True, help="judge dir, or off (chunk protocol only)")
    b.add_argument("--base-win", default=str(BASE_WIN))
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    {"train-judge": cmd_train, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
