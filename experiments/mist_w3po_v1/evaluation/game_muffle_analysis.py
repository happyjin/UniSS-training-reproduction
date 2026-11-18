"""Why does baseline B's translated speech sound muffled on the game (HOK esports) excerpts?
(reports/mist_w3po_v1/game_muffled_speech_analysis.zh-CN.md). Per talker stream of a run (all streams, or a seeded
random sample), for its voice prompt (items.jsonl voice_audio: what si_infer turns into BiCodec's 32 global tokens) and
its generated speech (results.jsonl speech_path):
  spectral   long-term average spectrum over speech frames (STFT 512 / hop 160 at 16 kHz; frames within 25 dB of the
             clip's 95th-percentile frame energy): centroid (Hz), 95 % roll-off (Hz), band shares (dB of total power)
             0-0.5 / 0.5-1 / 1-2 / 2-4 / 4-8 kHz, hf_db = 2-8 kHz share
  squim      torchaudio SQUIM objective (reference-free estimates of STOI / PESQ / SI-SDR)
  bvr_db     prompt only: HDemucs (MUSDB-HQ+) background-to-vocals energy ratio, 10 log10 (drums+bass+other) / vocals
  fallback   prompt only: True when the slot had < 1 s of solo frames before its first commit, so gstar_talker_v6 prep
             used the WINDOW audio up to the first commit (all speakers + background) as the prompt
  dur_s      prompt / speech duration
--save-vocals <dir>: also writes each prompt's HDemucs vocals stem (16 kHz) as <dir>/<stream id>.wav (game runs, for
the prompt-cleaning intervention). CPU only (multiprocessing). Refuses an existing --out.

    python -m experiments.mist_w3po_v1.evaluation.game_muffle_analysis --run <speech run name> --data <GSTAR_DATA dir>
        --session <sess> --diar <diar> --windows <online windows> [--sample N] --out <jsonl> [--save-vocals <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
CK = Path("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/checkpoints")
SR = 16000
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
_M = {}


def models():
    if not _M:
        import torch
        from torchaudio.models import hdemucs_high, squim_objective_base
        torch.set_num_threads(2)
        q = squim_objective_base()
        q.load_state_dict(torch.load(CK / "squim_objective_dns2020.pth", map_location="cpu"))
        h = hdemucs_high(sources=["drums", "bass", "other", "vocals"])
        h.load_state_dict(torch.load(CK / "hdemucs_high_trained.pt", map_location="cpu"))
        _M["q"], _M["h"] = q.eval(), h.eval()
    return _M["q"], _M["h"]


def read16(p):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def spectral(x):
    if len(x) < 2048:
        return None
    w = np.hanning(512).astype(np.float32)
    n = 1 + (len(x) - 512) // 160
    fr = np.stack([x[i * 160: i * 160 + 512] * w for i in range(n)])
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    e = 10 * np.log10(P.sum(1) + 1e-12)
    sp = e > np.percentile(e, 95) - 25
    if sp.sum() < 10:
        return None
    L = P[sp].mean(0)
    f = np.fft.rfftfreq(512, 1 / SR)
    tot = L[f > 50].sum()
    c = np.cumsum(L) / L.sum()
    out = {"centroid": float((f * L).sum() / L.sum()), "rolloff95": float(f[np.searchsorted(c, 0.95)])}
    for a, b in BANDS:
        out[f"b{a}_{b}"] = float(10 * np.log10(L[(f >= max(a, 50)) & (f < b)].sum() / tot + 1e-12))
    out["hf_db"] = float(10 * np.log10(L[(f >= 2000)].sum() / tot + 1e-12))
    out["ltas"] = [round(float(v), 2) for v in 10 * np.log10(L / tot + 1e-12)]
    return out


def squim(x):
    import torch
    q, _ = models()
    x = x[: SR * 20]
    if len(x) < SR // 2:
        return None
    with torch.no_grad():
        stoi, pesq, sisdr = q(torch.from_numpy(x)[None])
    return {"stoi": float(stoi), "pesq": float(pesq), "sisdr": float(sisdr)}


def demucs(x):
    import torch
    import torchaudio.functional as AF
    _, h = models()
    y = AF.resample(torch.from_numpy(x), SR, 44100)
    y = torch.stack([y, y])[None]
    ref = y.mean(1)
    m, s = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        st = h((y - m) / s)[0] * s + m          # [4, 2, T]: drums, bass, other, vocals
    voc = st[3].mean(0)
    bg = st[:3].sum(0).mean(0)
    bvr = float(10 * np.log10((bg ** 2).sum().item() + 1e-12) - 10 * np.log10((voc ** 2).sum().item() + 1e-12))
    return bvr, AF.resample(voc, 44100, SR).numpy().astype(np.float32)


def solo_total(post, k, until):
    alone = (post[:, k] > 0.5) & (np.delete(post, k, axis=1).max(1) < 0.5)
    alone = alone[: max(0, until)]
    tot, i = 0.0, 0
    while i < len(alone):
        if alone[i]:
            j = i
            while j < len(alone) and alone[j]:
                j += 1
            if (j - i) * 0.08 >= 0.4:
                tot += (j - i) * 0.08
            i = j
        else:
            i += 1
    return tot


def job(a):
    it, res, first_ms, win, post_path, save = a
    out = {"id": it["id"], "meeting": win["meeting"]}
    try:
        p = read16(it["voice_audio"])
        out["prompt"] = {"dur_s": round(len(p) / SR, 2), **(spectral(p) or {}), **(squim(p) or {})}
        out["prompt"]["bvr_db"], voc = demucs(p)
        if save:
            sf.write(str(Path(save) / f"{it['id']}.wav"), voc, SR)
        if post_path:
            import torch
            post = torch.load(post_path, weights_only=False)["post80"].float().numpy()
            k = int(it["id"].rsplit("__S", 1)[1]) - 1
            cut = win["start_s"] + first_ms / 1000
            out["prompt"]["fallback"] = solo_total(post, k, int(cut / 0.08)) < 1.0
        y = read16(res["speech_path"])
        out["speech"] = {"dur_s": round(len(y) / SR, 2), **(spectral(y) or {}), **(squim(y) or {})}
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:300]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--diar", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-vocals", default="")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    if a.save_vocals:
        Path(a.save_vocals).mkdir(parents=True, exist_ok=True)
    S = REPO / "eval_outputs/mist_w3po_v1/speech" / a.run
    items = {r["id"]: r for r in map(json.loads, open(S / "talker_in/items.jsonl", encoding="utf-8"))}
    blocks = {r["id"]: r["blocks"] for r in map(json.loads, open(S / "talker_in/blocks.jsonl", encoding="utf-8"))}
    res = {r["id"]: r for r in map(json.loads, open(S / "talker/results.jsonl", encoding="utf-8"))
           if r.get("error") is None and r.get("speech_path")}
    W = {w["id"]: w for w in map(json.loads, open(REPO / a.windows, encoding="utf-8"))}
    ids = sorted(i for i in items if i in res)
    if a.sample and len(ids) > a.sample:
        ids = sorted(random.Random(0).sample(ids, a.sample))
    jobs = []
    for i in ids:
        w = W[i.rsplit("__S", 1)[0]]
        pp = REPO / a.data / "diar" / a.diar / f"{w['meeting']}.post.pt"
        first = min(b["write_ms"] for b in blocks[i])
        jobs.append((items[i], res[i], first, w, str(pp) if pp.exists() else "", a.save_vocals))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.procs) as pool, open(out, "x", encoding="utf-8") as fh:
        for n, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2), 1):
            fh.write(json.dumps(r) + "\n")
            if n % 50 == 0:
                print(n, "/", len(jobs), flush=True)
    print("done", len(jobs), out)


if __name__ == "__main__":
    main()
