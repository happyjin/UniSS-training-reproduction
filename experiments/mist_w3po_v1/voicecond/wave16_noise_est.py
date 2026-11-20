"""Robust wave 16 NCWI (reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md, section 1): the noise estimate b-hat of
every S1 dev item from its first 1 s only, with an open pretrained VAD (silero-vad 6.2.3, the JIT weights shipped in the
pip package; no training, no tuning).

The audio is read exactly as si_thinker_stream --items-file reads it (soundfile float32, channel mean, cut to dur_ms); the
first 16000 samples are cut into 31 frames of 512 samples (silero's 16 kHz window, no padding); the VAD state is reset per
item and every frame with speech probability >= 0.5 (silero's default) is a speech frame. b-hat = mean frame energy of the
non-speech frames / mean frame energy of the speech frames. No speech frame: the most energetic frame is the speech
reference, the others are non-speech; no non-speech frame: the least energetic frame is the non-speech reference, the
others are speech (both flagged). `key` = sha1 of the first 16000 float32 samples, the same key wave9_common.head_key gives
si_thinker_stream's prompt audio (the inference wrapper finds the item by it).

Auxiliary (no selection uses it): correlation of b-hat (dB) with the true BVR of the noisy items -- (a) the wave-1 manifest
bvr_db per sentence; (b) bg_track / (mix - bg_track) energy over the item's cut (id2 mix = limiter x clean twin + bg_track, checked) --
and the AUC of b-hat for clean vs noisy.
    <simuls2s python> experiments/mist_w3po_v1/voicecond/wave16_noise_est.py --out eval_outputs/mist_w3po_v1/wave16/noise
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
WAVE1 = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5/data/manifest.jsonl"
BENCH = REPO / "data/open_voicecond_v1/bench_dev"
HEAD, FR, THR = 16000, 512, 0.5


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def load(path: str, dur_ms: int) -> np.ndarray:
    """si_thinker_stream's items-file loading (float32, channel mean, [0, dur_ms))"""
    x, sr = sf.read(path if Path(path).is_absolute() else str(REPO / path), dtype="float32", always_2d=True)
    assert sr == 16000, path
    return x.mean(axis=1)[: int(dur_ms) * 16]


def bhat(head: np.ndarray, model, torch) -> dict:
    n = len(head) // FR
    fr = head[: n * FR].reshape(n, FR).astype(np.float64)
    e = (fr ** 2).mean(1)
    model.reset_states()
    p = np.array([float(model(torch.from_numpy(head[k * FR:(k + 1) * FR].copy()), 16000).item()) for k in range(n)])
    sp = p >= THR
    flag = ""
    if n == 0:
        return {"bhat": float("nan"), "n_frames": 0, "n_speech": 0, "flag": "empty"}
    if n == 1:
        return {"bhat": float("nan"), "n_frames": 1, "n_speech": int(sp.sum()), "flag": "one_frame"}
    if not sp.any():
        k = int(e.argmax()); es, en = e[k], np.delete(e, k).mean(); flag = "no_speech"
    elif sp.all():
        k = int(e.argmin()); en, es = e[k], np.delete(e, k).mean(); flag = "no_nonspeech"
    else:
        es, en = e[sp].mean(), e[~sp].mean()
    return {"bhat": float(en / max(es, 1e-20)), "n_frames": n, "n_speech": int(sp.sum()), "flag": flag}


def true_bvr(d: str) -> dict:
    """true BVR (dB) of the noisy items; clean items get -inf"""
    out = {}
    for r in jl(WAVE1):
        if r["dir"] == d:
            out[f"a_noisy__{r['id']}"] = float(r["bvr_db"])
            out[f"a_clean__{r['id']}"] = float("-inf")
    meet = json.load(open(BENCH / "id2" / d / "meetings.json"))
    twin_audio = {}
    for w in jl(BENCH / "clean" / d / "refs_windows.jsonl"):
        twin_audio[w["meeting"]] = w["audio"]
        out[f"b_clean__{w['id']}"] = float("-inf")
    cache, worst = {}, 0.0
    for w in jl(BENCH / "id2" / d / "refs_windows.jsonl"):
        m = w["meeting"]
        if m not in cache:
            rd = lambda p: sf.read(str(p), dtype="float32", always_2d=True)[0].mean(1)
            cache = {m: (rd(BENCH / "id2" / d / "bg_track" / f"{m}.wav"), rd(twin_audio[meet[m]["twins"]["clean"]]),
                         rd(w["audio"]))}
        bg, cl, mx = cache[m]
        s0 = int(round(w["start_s"] * 16000))
        sl = slice(s0, s0 + w["dur_ms"] * 16)
        b = bg[sl].astype(np.float64)
        c = mx[sl].astype(np.float64) - b      # the voice as it is in the mix (= limiter x clean twin)
        worst = max(worst, float(np.abs(c - meet[m]["limiter"] * cl[sl]).max()))
        assert worst < 1e-3, f"{w['id']}: id2 mix != limiter x clean twin + bg_track (max residual {worst})"
        out[f"b_noisy__{w['id']}"] = float(10 * np.log10((b ** 2).mean() / max((c ** 2).mean(), 1e-20)))
    print(f"{d}: id2 mix = limiter x clean twin + bg_track, max residual {worst:.2e}", flush=True)
    return out


def auc(pos, neg) -> float:
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N items per direction (no correlation report)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import torch
    from silero_vad import load_silero_vad
    from scipy.stats import pearsonr, spearmanr
    torch.set_num_threads(4)
    model = load_silero_vad(onnx=False)
    rows, rep = {}, {"vad": "silero-vad 6.2.3 (jit), threshold 0.5, 31 x 512-sample frames of the first 1 s"}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(S1 / f"meta_{d}.jsonl")}
        its = jl(S1 / f"items_{d}.jsonl")
        if a.limit:
            its = its[: a.limit]
        rs = []
        for it in its:
            x = load(it["audio"], it["dur_ms"])
            h = x[:HEAD]
            r = {"id": it["id"], "key": hashlib.sha1(np.ascontiguousarray(h, dtype=np.float32).tobytes()).hexdigest(),
                 **bhat(h, model, torch)}
            r["bhat_db"] = float(10 * np.log10(r["bhat"] + 1e-12)) if np.isfinite(r["bhat"]) else float("nan")
            rs.append(r)
        assert all(np.isfinite(r["bhat"]) for r in rs), [r["id"] for r in rs if not np.isfinite(r["bhat"])][:5]
        keys = {}
        for r in rs:
            assert keys.setdefault(r["key"], r["id"]) == r["id"], f"{d}: two items share the first-1 s key: {r['id']}"
        rows[d] = rs
        st = {"n": len(rs), "flags": {f: sum(r["flag"] == f for r in rs) for f in ("no_speech", "no_nonspeech")}}
        if not a.limit:
            tb = true_bvr(d)
            assert set(tb) == set(meta), f"{d}: true BVR ids differ from the S1 meta"
            by = {r["id"]: r for r in rs}
            for set_ in ("a", "b"):
                ids = [i for i, m in meta.items() if m["set"] == set_ and m["cond"] == "noisy"]
                x_ = np.array([by[i]["bhat_db"] for i in ids]); y_ = np.array([tb[i] for i in ids])
                st[f"{set_}_noisy"] = {"n": len(ids), "spearman": round(float(spearmanr(x_, y_).statistic), 4),
                                       "pearson": round(float(pearsonr(x_, y_).statistic), 4)}
                cl = [by[i]["bhat_db"] for i, m in meta.items() if m["set"] == set_ and m["cond"] == "clean"]
                st[f"{set_}_auc_noisy_vs_clean"] = round(auc(list(x_), cl), 4)
            noisy = [by[i]["bhat_db"] for i, m in meta.items() if m["cond"] == "noisy"]
            clean = [by[i]["bhat_db"] for i, m in meta.items() if m["cond"] == "clean"]
            st["all_auc_noisy_vs_clean"] = round(auc(noisy, clean), 4)
            for r in rs:
                r["true_bvr_db"] = tb[r["id"]] if np.isfinite(tb[r["id"]]) else None
        rep[d] = st
        print(d, json.dumps(st), flush=True)
    out.mkdir(parents=True)
    for d, rs in rows.items():
        with open(out / f"bhat_{d}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    with open(out / "report.json", "x", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
