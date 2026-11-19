"""Robust wave 3, V group (reports/mist_w3po_v1/robust_wave3_single_prereg.zh-CN.md): voice-token tables of the arms
V1 (Idea 1 C), V2 (Idea 2 frozen shrink+Vfb) and V3 (Idea 6 C_post) for the single-speaker LA2 base (S1).

si_infer (--items-file, --text-policy file) writes each item's voice prompt tmp/<id>_voice.wav = the item audio up to
max(first write, 1 s) and encodes it with BiCodecTokenizer.encode_wav_to_tokens. This script writes the same prompt
bytes (same read / slice / sf.write), and for every item gives the 32 ids of each arm in the table format of
idea1_si_infer_vo.py ({key = prompt stem, id, sha1 of the prompt, ids}); that hook replaces the encoding at run time
and checks the sha1, so the tokens are tied to the very prompt the baseline B used.

  V1  C   : u = C encoder (voicecond_idea1/m2/ctrl/encoder.pt) on the prompt (idea1_tokens --mode ctrl computation).
  V2  I2  : Idea 2 shrink posterior, parameters shrink_fit_all/full.npz (sha1 25aaf6cd), one slot, pi = 1, no
            diarizer (so no diarizer delay: a window is usable when it has been heard). Windows L = 2 s, hop 0.5 s,
            per audio segment: (a) the prompt only; (b) the same stream's (meta `stream`, same condition) earlier turns
            as heard by this item's first write (start_s + max(first, 1 s)), each turn windowed on its own, plus the
            prompt. z = frozen encoder (phi_wave), v~ = window_vt of the segment's HDemucs split (idea2_demucs.separate).
            u = posterior mean + b_M -> FSQ. No usable window -> V fallback (HDemucs vocals of the prompt, phi_path), as
            the frozen method.
  V3  Cpost: the same windows encoded by C (ref_clip, tiled to 6 s), closed-form posterior of Idea 6 precheck A
            (prior N(mu0 + b_M, Sig0) of the same fit, R_C = idea6_precheckA/R_C.npz scaled by L / hop, alpha = 0,
            pi = 1). No usable window -> the same V fallback ids.

    python -I experiments/mist_w3po_v1/voicecond/wave3_v_tokens.py --dir zh2en|en2zh --out <dir> [--device cuda:0]
        [--limit N]   (smoke: first N items of each set)
Writes <out>/prompts/<dir>/<id>_voice.wav, <out>/tokens_<arm>_<dir>.jsonl (arm v1 v2 v3) and <out>/tokens_<dir>.meta.jsonl.
Refuses existing token files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea6_e2e_train import (CKPT_C, FIT, FIT_SHA1, HOP, L, RC_PATH, SR,  # noqa: E402
                                                               post_numpy, prior_terms)

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
BENCH = REPO / "data/open_voicecond_v1/bench_dev"
WS = int(L * SR)


def window_starts(n):
    if n < WS:
        return np.zeros(0, int)
    K = int((n - WS) // (HOP * SR)) + 1
    return (np.arange(K) * HOP * SR).astype(int)


class Feats:
    """window encodes of one audio segment: ends (s, segment time), z (frozen, 192), uC (C, 192), vt (5)"""

    def __init__(self, P0, Pc, demucs, dev):
        self.P0, self.Pc, self.demucs, self.dev = P0, Pc, demucs, dev

    @torch.no_grad()
    def __call__(self, y):
        from experiments.mist_w3po_v1.voicecond.idea2_demucs import separate
        from experiments.mist_w3po_v1.voicecond.idea2_feats2 import window_vt
        st = window_starts(len(y))
        if not len(st):
            return {"end": np.zeros(0), "z": np.zeros((0, 192)), "uC": np.zeros((0, 192)), "vt": np.zeros((0, 5))}
        ends = (st + WS) / SR
        z = self.P0.phi_wave([y[s:s + WS] for s in st]).reshape(len(st), 192)
        with torch.no_grad():
            ref = torch.from_numpy(np.stack([self.Pc.ref_clip(y[s:s + WS]) for s in st])).float().to(self.Pc.device)
            uC = self.Pc._u_from_ref(ref).reshape(len(st), 192).cpu().numpy()
        vh, bh = separate(self.demucs, y.astype(np.float32), self.dev)
        vt = window_vt(vh, bh, ends, L)
        return {"end": ends, "z": z.astype(np.float64), "uC": uC.astype(np.float64), "vt": vt.astype(np.float64)}


def shrink_post(pf, lang, z, vt):
    """Idea 2 shrink posterior (idea2_game_tokens.posterior, mode full) with pi = 1; prompt space"""
    lam = pf.lam(vt)
    y = z - (1 - lam)[:, None] * pf.cvec(lang)[None]
    Rfi = np.linalg.inv(pf.R * (pf.L / HOP))
    S0i = np.linalg.inv(pf.Sig0)
    m = np.linalg.solve(S0i + (lam * lam).sum() * Rfi, S0i @ pf.mu0 + Rfi @ (lam[:, None] * y).sum(0))
    return m + pf.bM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, choices=["zh2en", "en2zh"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    files = {k: out / f"tokens_{k}_{a.dir}.jsonl" for k in ("v1", "v2", "v3")}
    metaf = out / f"tokens_{a.dir}.meta.jsonl"
    for f in [*files.values(), metaf]:
        if f.exists():
            sys.exit(f"拒绝覆盖已有输出: {f}")
    pdir = out / "prompts" / a.dir
    pdir.mkdir(parents=True, exist_ok=True)
    lang = "zh" if a.dir == "zh2en" else "en"

    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    from experiments.mist_w3po_v1.voicecond.idea1_zne import read_mono64
    from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams
    from experiments.mist_w3po_v1.voicecond.idea2_stream_tokens import v_tokens
    from torchaudio.models import hdemucs_high
    assert hashlib.sha1(FIT.read_bytes()).hexdigest() == FIT_SHA1
    pf = ShrinkParams.load(FIT)
    RC = np.load(RC_PATH)["R_C"]
    S0i, e0C, G, m0 = prior_terms(pf, RC)
    dev = torch.device(a.device)
    P0 = BiCodecPhi(a.device)
    Pc = load_encoder(BiCodecPhi(a.device), str(CKPT_C))
    Pc.model.eval()
    dm = hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    dm.load_state_dict(torch.load("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/checkpoints/hdemucs_high_trained.pt",
                                  map_location="cpu"))
    dm = dm.eval().to(dev)
    feats = Feats(P0, Pc, dm, dev)

    items = [json.loads(l) for l in open(S1 / "data" / f"items_{a.dir}.jsonl", encoding="utf-8")]
    meta = {r["id"]: r for r in map(json.loads, open(S1 / "data" / f"meta_{a.dir}.jsonl", encoding="utf-8"))}
    blocks = {r["id"]: r["blocks"] for r in map(json.loads, open(S1 / f"thinker_{a.dir}/blocks_{a.dir}_w1.jsonl",
                                                                  encoding="utf-8"))}
    # set (b): session start time of each turn (refs_windows of bench_dev/<cond>/<dir>)
    t0 = {}
    for cond, label in (("id2", "noisy"), ("clean", "clean")):
        for w in map(json.loads, open(BENCH / cond / a.dir / "refs_windows.jsonl", encoding="utf-8")):
            t0[f"b_{label}__{w['id']}"] = w["start_s"]
    streams = defaultdict(list)
    for it in items:
        m = meta[it["id"]]
        if m["set"] == "b":
            streams[(m["cond"], m["stream"])].append(it["id"])
    byid = {it["id"]: it for it in items}
    if a.limit:
        keep, n = [], defaultdict(int)
        for it in items:
            m = meta[it["id"]]
            k = (m["set"], m["cond"])
            if n[k] < a.limit:
                keep.append(it)
                n[k] += 1
        items = keep

    turn_cache = {}

    def turn_feats(i):
        if i not in turn_cache:
            x, sr = sf.read(byid[i]["audio"], dtype="float32", always_2d=True)
            assert sr == SR
            turn_cache[i] = feats(x.mean(axis=1)[: int(byid[i]["dur_ms"]) * 16].astype(np.float64))
            if len(turn_cache) > 64:
                turn_cache.pop(next(iter(turn_cache)))
        return turn_cache[i]

    rows = {k: [] for k in files}
    metas = []
    for n_done, it in enumerate(items, 1):
        if not blocks.get(it["id"]):
            continue
        m = meta[it["id"]]
        # the prompt, exactly as si_infer writes it
        x, sr = sf.read(it["audio"], dtype="float32", always_2d=True)
        assert sr == SR
        wav = x.mean(axis=1)[0: int(it["dur_ms"]) * 16]
        first = blocks[it["id"]][0]["write_ms"]
        h_ms = max(first, 1000)
        p = pdir / f"{it['id'].replace('/', '__')}_voice.wav"
        sf.write(p, wav[: h_ms * 16], 16000)
        sha = hashlib.sha1(p.read_bytes()).hexdigest()
        key = p.stem
        y = read_mono64(p)
        # V1: C on the prompt
        ids1 = [int(v) for v in Pc.tokens(Pc.phi_wave(y))]
        # windows heard by the first write: earlier turns of the stream (b), then the prompt
        segs = []
        hist = []
        if m["set"] == "b":
            T = t0[it["id"]] + h_ms / 1000.0
            for j in streams[(m["cond"], m["stream"])]:
                if j == it["id"] or t0[j] >= t0[it["id"]]:
                    continue
                f = turn_feats(j)
                ok = t0[j] + f["end"] <= T + 1e-9
                if ok.any():
                    segs.append({k: v[ok] for k, v in f.items()})
                    hist.append(j)
        segs.append(feats(y))
        z = np.concatenate([s["z"] for s in segs])
        uC = np.concatenate([s["uC"] for s in segs])
        vt = np.concatenate([s["vt"] for s in segs])
        nwin = len(z)
        if nwin == 0:
            vfb = [int(v) for v in v_tokens(P0, p, out / "vocals_prompt" / a.dir, dm, dev)]
            ids2 = ids3 = vfb
        else:
            ids2 = [int(v) for v in P0.tokens(shrink_post(pf, lang, z, vt).reshape(32, 6).astype(np.float32))]
            u3 = post_numpy(uC, np.zeros_like(uC), np.ones(nwin), S0i, e0C, G, m0)
            ids3 = [int(v) for v in P0.tokens(u3.reshape(32, 6).astype(np.float32))]
        for k, ids in (("v1", ids1), ("v2", ids2), ("v3", ids3)):
            rows[k].append({"key": key, "id": it["id"], "sha1": sha, "ids": ids})
        metas.append({"key": key, "id": it["id"], "set": m["set"], "cond": m["cond"], "stream": m["stream"],
                      "first_ms": first, "prompt_ms": h_ms, "n_win": nwin, "n_win_prompt": int(len(segs[-1]["z"])),
                      "hist_turns": len(hist), "fallback": nwin == 0})
        if n_done % 100 == 0:
            print(f"[v-tokens] {a.dir} {n_done}/{len(items)}", flush=True)
    for k, f in files.items():
        with open(f, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows[k])
    with open(metaf, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in metas)
    nw = np.array([r["n_win"] for r in metas])
    ham = lambda A, B: float(np.mean([sum(x != y for x, y in zip(r["ids"], s["ids"])) for r, s in zip(A, B)]))  # noqa: E731
    print(json.dumps({"dir": a.dir, "n": len(metas), "fallback": int((nw == 0).sum()), "n_win_median": float(np.median(nw)),
                      "ham_v1_v2": ham(rows["v1"], rows["v2"]), "ham_v1_v3": ham(rows["v1"], rows["v3"]),
                      "fit_sha1": FIT_SHA1}), flush=True)


if __name__ == "__main__":
    main()
