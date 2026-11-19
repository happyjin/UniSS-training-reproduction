"""Idea 2 frozen method shrink+Vfb (prereg v3): per-stream voice tokens for ANY baseline T1c run whose data directory
has the pipeline layout (session_audio/{mix/<m>.wav | ihm_mix/<m>.Mix-Headset.wav}, diar/nemotron_lowlat/<m>.post.pt).
Same computation as idea2_game_tokens.py (prereg v2 §D/§G): HDemucs on the whole session, window codes (L = 2 s,
hop 0.5 s), v~, slot / pi from the meeting-level posteriors; per item <win>__S<k>: posterior of slot k-1 at
t_commit = window start_s + first block write_ms (windows usable from end + 1.04 s), u = s + b_M, FSQ tokens.
Fallback (prereg v2.2): items without any usable window (n_win == 0) get V tokens = the tokenizer's encoding of the
HDemucs vocals of the baseline prompt (game_muffle_analysis.demucs on the prompt, written as 16 kHz wav, phi_path).

    python -I experiments/mist_w3po_v1/voicecond/idea2_stream_tokens.py --data <data dir> --dir zh2en|en2zh \
        --run <baseline speech run> --windows <windows jsonl> --fit <fit dir> --tag <name> [--device cuda:0] [--meetings m ...]
        [--session zh_mix_v8|ihm_mix|near_mix|mix] [--diar nemotron_lowlat|nemotron_lowlat_zh8]
Writes eval_outputs/mist_w3po_v1/voicecond_idea2/tokens/<tag>.jsonl (rows as idea2_game_tokens + fallback flag) and
<tag>.sidecar.json ({item id: [32 ids]}, the OpenBroadcast-SI hook format); caches features in
data/open_voicecond_v1/idea2/streams_cache/<tag-data-key>/<m>.npz. Refuses to overwrite outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import DIAR_DELAY, I2  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_feats2 import window_vt  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_game_tokens import L, SR, HOP, posterior, slot_pi  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams  # noqa: E402

TOK = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/tokens"


SESS = {"sess": "", "diar": "nemotron_lowlat"}


def session_audio(data: Path, d: str, m: str) -> Path:
    sess = SESS["sess"] or ("mix" if d == "zh2en" else "ihm_mix")
    return data / "session_audio" / sess / (f"{m}.Mix-Headset.wav" if sess == "ihm_mix" else f"{m}.wav")


def features(data, d, m, cache, P, model, dev):
    cp = cache / f"{m}.npz"
    if cp.exists():
        z = np.load(cp)
        return {k: z[k] for k in z.files}
    from experiments.mist_w3po_v1.voicecond.idea2_demucs import separate
    y, sr = sf.read(str(session_audio(data, d, m)), dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    assert sr == SR, sr
    vh, bh = separate(model, y, dev)
    W = int(L * SR)
    K = int((len(y) - W) // (HOP * SR)) + 1
    st = (np.arange(K) * HOP * SR).astype(int)
    ends = (st + W) / SR
    z = np.concatenate([P.phi_wave([y[s:s + W] for s in st[i:i + 256]]) for i in range(0, K, 256)]).reshape(K, -1)
    vt = window_vt(vh, bh, ends, L)
    post = torch.load(data / "diar" / SESS["diar"] / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
    slot, pi = slot_pi(post, ends)
    out = {"end": ends, "z": z, "zv": np.zeros((0,)), "vt": vt, "slot": slot, "pi": pi}
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.parent / f".{m}.partial.npz"
    np.savez(tmp, **out)
    os.rename(tmp, cp)
    return out


def v_tokens(P, prompt: Path, vdir: Path):
    """Idea 1's V: HDemucs vocals of the prompt (game_muffle_analysis.demucs), saved at 16 kHz, tokenizer file path"""
    from experiments.mist_w3po_v1.evaluation.game_muffle_analysis import demucs, read16
    vp = vdir / f"{prompt.stem}.wav"
    if not vp.exists():
        _, voc = demucs(read16(prompt))
        vdir.mkdir(parents=True, exist_ok=True)
        tmp = vdir / f".{prompt.stem}.partial.wav"
        sf.write(str(tmp), voc, SR)
        os.rename(tmp, vp)
    return P.tokens(P.phi_path(str(vp))).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dir", required=True, choices=["zh2en", "en2zh"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lang", default="", help="source language for c (default: zh for zh2en, en for en2zh)")
    ap.add_argument("--meetings", nargs="*", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--session", default="", help="session_audio/<session> (default mix for zh2en, ihm_mix for en2zh)")
    ap.add_argument("--diar", default="nemotron_lowlat", help="diar/<diar>/<m>.post.pt")
    a = ap.parse_args()
    SESS["sess"], SESS["diar"] = a.session, a.diar
    out, side = TOK / f"{a.tag}.jsonl", TOK / f"{a.tag}.sidecar.json"
    if out.exists() or side.exists():
        sys.exit(f"refuse to overwrite {out}")
    data = Path(a.data) if Path(a.data).is_absolute() else REPO / a.data
    lang = a.lang or ("zh" if a.dir == "zh2en" else "en")
    pf = ShrinkParams.load(Path(a.fit) / "full.npz")
    fit_sha1 = hashlib.sha1((Path(a.fit) / "full.npz").read_bytes()).hexdigest()
    S = REPO / "eval_outputs/mist_w3po_v1/speech" / a.run
    items = [json.loads(l) for l in open(S / "talker_in/items.jsonl", encoding="utf-8")]
    blocks = {r["id"]: r["blocks"] for r in map(json.loads, open(S / "talker_in/blocks.jsonl", encoding="utf-8"))}
    wpath = Path(a.windows) if Path(a.windows).is_absolute() else REPO / a.windows
    W = {w["id"]: w for w in map(json.loads, open(wpath, encoding="utf-8"))}
    key = hashlib.sha1(f"{data.resolve()}|{a.session}|{a.diar}".encode()).hexdigest()[:12]
    cache = I2 / "streams_cache" / f"{a.dir}_{key}"
    vdir = I2 / "streams_cache" / f"{a.dir}_{key}_vocals_prompt"
    from torchaudio.models import hdemucs_high
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    dev = torch.device(a.device)
    model = hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    model.load_state_dict(torch.load("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/checkpoints/hdemucs_high_trained.pt",
                                     map_location="cpu"))
    model = model.eval().to(dev)
    P = BiCodecPhi(a.device)
    feats, rows, seen = {}, [], set()
    for it in items:
        p = Path(it["voice_audio"])
        p = p if p.is_absolute() else REPO / p
        if p.stem in seen:
            continue
        wid, k = it["id"].rsplit("__S", 1)
        w = W[wid]
        m = w["meeting"]
        if a.meetings and m not in a.meetings:
            continue
        seen.add(p.stem)
        if m not in feats:
            feats[m] = features(data, a.dir, m, cache, P, model, dev)
            print(f"[tokens] {a.tag} {m}: {len(feats[m]['end'])} windows", flush=True)
        f = feats[m]
        t = w["start_s"] + min(b["write_ms"] for b in blocks[it["id"]]) / 1000.0
        u, sel = posterior(pf, lang, f, int(k) - 1, t, "full")
        ids = P.tokens(u.reshape(32, 6).astype(np.float32)).tolist()
        r = {"key": p.stem, "id": it["id"], "mode": "shrink_vfb", "fit": str(a.fit), "fit_sha1": fit_sha1,
             "sha1": hashlib.sha1(p.read_bytes()).hexdigest(), "t_commit": round(t, 3), "n_win": int(len(sel)),
             "pi_sum": float(f["pi"][sel].sum()), "fallback": len(sel) == 0}
        if len(sel) == 0:
            r["ids_shrink"] = [int(v) for v in ids]
            ids = v_tokens(P, p, vdir)
        r["ids"] = [int(v) for v in ids]
        rows.append(r)
    TOK.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(side, "x", encoding="utf-8") as fh:
        json.dump({r["key"]: r["ids"] for r in rows}, fh)
    nw = np.array([r["n_win"] for r in rows])
    print(json.dumps({"out": str(out), "n": len(rows), "fallback": int((nw == 0).sum()), "n_win_median": float(np.median(nw)),
                      "fit_sha1": fit_sha1}))


if __name__ == "__main__":
    main()
