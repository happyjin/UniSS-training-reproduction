"""Idea 2 e2e test (prereg addendum 2026-11-19): voice-token tables of an e2e checkpoint, in the format of
idea2_stream_tokens.py (jsonl rows {key, id, ..., sha1, ids} + <tag>.sidecar.json {key: ids}), for the identical items.

build     the M2 computation (idea2_stream_tokens.py) with the trained modules: the features cache of M2
          (streams_cache/<dir>_<key>/<m>.npz: window ends, v~, slot, pi) is REQUIRED and reused; only the window codes
          z are recomputed with the trained encoder (BiCodecPhi.phi_wave with the checkpoint's speaker_encoder /
          perceiver_sampler / project_in); lambda / log r from the trained head, c_lang from the checkpoint, R / Sigma0 /
          mu0 / b_M of the full fit (sha1 25aaf6cd):
            u = (S0^-1 + sum pi lam^2 e^-r Rf^-1)^-1 (S0^-1 mu0 + Rf^-1 sum pi lam e^-r (z - (1 - lam) c)) + b_M
          Items without a usable window: the V fallback ids of the M2 table (--m2-tag), as in M2.
selftest  cvt DEV only: (a) frozen encoder + initial head reproduce the stored z_mix and the numpy shrink posterior;
          (b) with the checkpoint, code error and token recovery on the cvt dev items vs the training log's dev record
          at the checkpoint step.
    python -I experiments/mist_w3po_v1/voicecond/idea2_e2e_tokens.py build --ckpt <ckpt> --m2-tag <M2 tag> --tag <tag>
        <idea2_stream_tokens.py arguments: --data --dir --run --windows [--lang] [--meetings] [--session] [--diar]>
    python -I experiments/mist_w3po_v1/voicecond/idea2_e2e_tokens.py selftest --ckpt <ckpt> --out <json>
Refuses to overwrite outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import DIAR_DELAY, I2  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams  # noqa: E402
import experiments.mist_w3po_v1.voicecond.idea2_stream_tokens as ST  # noqa: E402

FIT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all"
FIT_SHA1 = "25aaf6cd7f6a69b1e7972b947459d1217744e037"
SR, HOP, L = 16000, 0.5, 2.0
WS = int(L * SR)


class Trained:
    """frozen-structure posterior with the trained modules of an e2e checkpoint (main setting); ckpt=None -> initial"""

    def __init__(self, ckpt, device):
        from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
        from experiments.mist_w3po_v1.voicecond.idea2_e2e_train import LamHead
        assert hashlib.sha1((FIT / "full.npz").read_bytes()).hexdigest() == FIT_SHA1
        self.pf = pf = ShrinkParams.load(FIT / "full.npz")
        self.P = BiCodecPhi(device)
        self.head = LamHead(pf.alpha, pf.beta)
        self.c = {k: np.asarray(v, np.float64) for k, v in pf.c.items()}
        self.meta = {"ckpt": None}
        if ckpt:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            assert ck["ablation"] == "main" and ck["fit_sha1"] == FIT_SHA1, (ck["ablation"], ck["fit_sha1"])
            self.P.se.speaker_encoder.load_state_dict(ck["encoder"]["speaker_encoder"])
            self.P.se.perceiver_sampler.load_state_dict(ck["encoder"]["perceiver_sampler"])
            self.P.rq.project_in.load_state_dict(ck["encoder"]["project_in"])
            self.head.load_state_dict(ck["head"])
            self.c = {k: v.double().numpy() for k, v in ck["c_lang"].items()}
            h = hashlib.sha1()
            with open(ckpt, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            self.meta = {"ckpt": str(ckpt), "ckpt_step": int(ck["step"]), "ckpt_sha1": h.hexdigest()}
        self.P.model.eval()
        self.head.eval()
        self.Rfi = torch.as_tensor(np.linalg.inv(pf.R * (pf.L / HOP)), dtype=torch.float64)
        self.S0i = torch.as_tensor(np.linalg.inv(pf.Sig0), dtype=torch.float64)
        self.e0 = self.S0i @ torch.as_tensor(pf.mu0, dtype=torch.float64)

    def z_windows(self, y, starts, bs=256):
        out = [self.P.phi_wave([y[s:s + WS] for s in starts[i:i + bs]]) for i in range(0, len(starts), bs)]
        return np.concatenate(out).reshape(len(starts), -1).astype(np.float64) if out else np.zeros((0, 192))

    @torch.no_grad()
    def posterior(self, z, vt_raw, pi, lang):
        """z [n,192], raw v~ [n,5], pi [n] -> u (prompt space, 192) as float64 numpy"""
        pf = self.pf
        vt = torch.as_tensor((np.asarray(vt_raw, np.float64) - pf.vt_mean) / pf.vt_std, dtype=torch.float32)
        lg, lr = self.head(vt)
        lam, er = torch.sigmoid(lg).double(), torch.exp(-lr.double())
        c = torch.as_tensor(self.c[lang], dtype=torch.float64)
        y = torch.as_tensor(z, dtype=torch.float64) - (1 - lam)[:, None] * c[None]
        w = torch.as_tensor(np.asarray(pi, np.float64))
        A = (w * lam * lam * er).sum()
        Y = ((w * lam * er)[:, None] * y).sum(0)
        s = torch.linalg.solve(self.S0i + A * self.Rfi, self.e0 + self.Rfi @ Y)
        return s.numpy() + pf.bM

    def tokens(self, u):
        return [int(v) for v in self.P.tokens(np.asarray(u).reshape(32, 6).astype(np.float32))]


def cmd_build(a):
    TOK = ST.TOK
    out, side = TOK / f"{a.tag}.jsonl", TOK / f"{a.tag}.sidecar.json"
    if out.exists() or side.exists():
        sys.exit(f"refuse to overwrite {out}")
    ST.SESS["sess"], ST.SESS["diar"] = a.session, a.diar
    data = Path(a.data) if Path(a.data).is_absolute() else REPO / a.data
    lang = a.lang or ("zh" if a.dir == "zh2en" else "en")
    M = Trained(a.ckpt, a.device)
    m2 = {r["key"]: r for r in map(json.loads, open(TOK / f"{a.m2_tag}.jsonl", encoding="utf-8"))}
    S = REPO / "eval_outputs/mist_w3po_v1/speech" / a.run
    items = [json.loads(l) for l in open(S / "talker_in/items.jsonl", encoding="utf-8")]
    blocks = {r["id"]: r["blocks"] for r in map(json.loads, open(S / "talker_in/blocks.jsonl", encoding="utf-8"))}
    wpath = Path(a.windows) if Path(a.windows).is_absolute() else REPO / a.windows
    W = {w["id"]: w for w in map(json.loads, open(wpath, encoding="utf-8"))}
    key = hashlib.sha1(f"{data.resolve()}|{a.session}|{a.diar}".encode()).hexdigest()[:12]
    cache = I2 / "streams_cache" / f"{a.dir}_{key}"
    feats, rows, seen, t0 = {}, [], set(), time.time()
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
            cp = cache / f"{m}.npz"
            assert cp.exists(), f"M2 feature cache missing: {cp}"
            f = dict(np.load(cp))
            y, sr = sf.read(str(ST.session_audio(data, a.dir, m)), dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            assert sr == SR
            K = len(f["end"])
            st = (np.arange(K) * HOP * SR).astype(int)
            assert np.allclose((st + WS) / SR, f["end"]), m
            f["z"] = M.z_windows(y, st)
            feats[m] = f
            print(f"[e2e tokens] {a.tag} {m}: {K} windows re-encoded ({time.time() - t0:.0f}s)", flush=True)
        f = feats[m]
        t = w["start_s"] + min(b["write_ms"] for b in blocks[it["id"]]) / 1000.0
        s = int(k) - 1
        sel = np.nonzero((f["slot"] == s) & (f["end"] + DIAR_DELAY <= t + 1e-9))[0]
        r = {"key": p.stem, "id": it["id"], "mode": "e2e", **M.meta, "fit_sha1": FIT_SHA1,
             "sha1": hashlib.sha1(p.read_bytes()).hexdigest(), "t_commit": round(t, 3), "n_win": int(len(sel)),
             "pi_sum": float(f["pi"][sel].sum()), "fallback": len(sel) == 0}
        r2 = m2[p.stem]
        assert r2["sha1"] == r["sha1"] and r2["n_win"] == r["n_win"] and r2["fallback"] == r["fallback"], (p.stem, r2, r)
        if len(sel) == 0:
            r["ids"] = [int(v) for v in r2["ids"]]          # M2's V fallback tokens
        else:
            r["ids"] = M.tokens(M.posterior(f["z"][sel], f["vt"][sel], f["pi"][sel], lang))
        rows.append(r)
    assert len(rows) == len(m2), (len(rows), len(m2))
    TOK.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(side, "x", encoding="utf-8") as fh:
        json.dump({r["key"]: r["ids"] for r in rows}, fh)
    same = np.mean([np.mean(np.array(r["ids"]) == np.array(m2[r["key"]]["ids"])) for r in rows if not r["fallback"]])
    print(json.dumps({"out": str(out), "n": len(rows), "fallback": int(sum(r["fallback"] for r in rows)),
                      "token_agreement_with_M2_nonfallback": float(same), **M.meta}))


def cmd_selftest(a):
    from experiments.mist_w3po_v1.voicecond.idea2_e2e_train import CV, DEV_SETS, build_items, latest_refs
    outp = Path(a.out)
    if outp.exists():
        sys.exit(f"refuse to overwrite {outp}")
    rep = {}
    for mode, ck in (("init", None), ("ckpt", a.ckpt)):
        M = Trained(ck, a.device)
        items, _ = build_items(DEV_SETS, M.pf, latest_refs(CV), print)
        n_lim = a.limit if mode == "ckpt" else (a.limit or 40)
        if n_lim:
            items = items[:: max(1, len(items) // n_lim)][:n_lim]
        zs = {}
        for sid in sorted({it["sid"] for it in items}):
            y = sf.read(str(CV / "streams" / sid / "mix.wav"), dtype="float32")[0]
            need = sorted({int(k) for it in items if it["sid"] == sid for k in it["sel"]})
            zz = M.z_windows(y, (np.array(need) * HOP * SR).astype(int))
            zs[sid] = {k: zz[i] for i, k in enumerate(need)}
        acc = {}
        dz, du = [], []
        for it in items:
            z = np.stack([zs[it["sid"]][int(k)] for k in it["sel"]])
            vt_raw = it["vt"].astype(np.float64) * M.pf.vt_std + M.pf.vt_mean
            u = M.posterior(z, vt_raw, it["pi"], it["lang"])
            g = M.tokens(u)
            d = "zh2en" if it["set"].startswith("zh2en") else "en2zh"
            x = acc.setdefault(d, {"n": 0, "code_sq": 0.0, "tok": 0.0})
            x["n"] += 1
            x["code_sq"] += float(((u - it["ustar"]) ** 2).sum())
            x["tok"] += float(np.mean(np.array(g) == it["gstar"]))
            if mode == "init":
                dz.append(float(np.abs(z - it["z0"]).max()))
                lam = M.pf.lam(vt_raw)
                yy = it["z0"].astype(np.float64) - (1 - lam)[:, None] * M.pf.cvec(it["lang"])[None]
                w = it["pi"].astype(np.float64)
                Rfi, S0i = np.linalg.inv(M.pf.R * 4), np.linalg.inv(M.pf.Sig0)
                m = np.linalg.solve(S0i + (w * lam * lam).sum() * Rfi, S0i @ M.pf.mu0 + Rfi @ ((w * lam)[:, None] * yy).sum(0))
                du.append(float(np.abs(u - (m + M.pf.bM)).max()))
        res = {d: {"n": x["n"], "code_sq": x["code_sq"] / x["n"], "tok": x["tok"] / x["n"]} for d, x in acc.items()}
        if mode == "init":
            res["z_maxabs_vs_stored"] = max(dz)
            res["u_maxabs_vs_numpy_shrink"] = max(du)
        else:
            log = Path(a.ckpt).parent / "train_log.jsonl"
            devrec = [r for r in map(json.loads, open(log)) if "dev" in r and r["step"] == M.meta["ckpt_step"]]
            res["train_log_dev"] = devrec[0]["dev"] if devrec else None
            if devrec and not a.limit:
                res["max_abs_diff_vs_train_log"] = max(abs(res[d][k] - devrec[0]["dev"][d][k])
                                                       for d in ("zh2en", "en2zh") for k in ("code_sq", "tok"))
            res.update(M.meta)
        rep[mode] = res
        print(json.dumps({mode: res}, indent=1), flush=True)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "x") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "selftest"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--m2-tag", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--dir", choices=["zh2en", "en2zh"])
    ap.add_argument("--run", default="")
    ap.add_argument("--windows", default="")
    ap.add_argument("--lang", default="")
    ap.add_argument("--meetings", nargs="*", default=[])
    ap.add_argument("--session", default="")
    ap.add_argument("--diar", default="nemotron_lowlat")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="selftest: number of dev items (0 = all for the ckpt check)")
    a = ap.parse_args()
    cmd_build(a) if a.cmd == "build" else cmd_selftest(a)
