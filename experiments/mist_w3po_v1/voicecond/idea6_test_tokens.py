"""Idea 6 full-system test (prereg "完整系统测试预登记"): voice-token tables of the two arms C_post and E2E_post for the 9
test sets, in the format of idea2_stream_tokens.py (jsonl rows {key, id, ..., sha1, ids} + <tag>.sidecar.json {key: ids}),
for exactly M2's items (Idea 2 frozen shrink+Vfb tables, voicecond_idea2/tokens/<m2 tag>.jsonl).

Windows, slots, pi and t_commit come from M2's feature cache (data/open_voicecond_v1/idea2/streams_cache/<dir>_<key>/
<m>.npz); usable windows: own slot, end + 1.04 s <= t_commit (M2's rule). Items without a usable window keep M2's V
fallback ids. Otherwise u_hat = closed-form posterior (prior N(mu0 + bM, Sig0) of the full fit sha1 25aaf6cd,
G = inv(R_C L / hop), Lambda_k = pi_k D_k^1/2 G D_k^1/2) with
  C_post   (arm c)  u_k = C encoder (Idea 1 m2/ctrl/encoder.pt), alpha_k = 0
  E2E_post (arm e)  u_k, alpha_k from idea6_e2e_main_v1 ckpt_step03000.pt (encoder + precision head)
and the 32 ids = FSQ(u_hat).

  encode    window encodes (u_C, u_E, alpha) of every meeting the 9 sets need, sharded:
            --shard i --num-shards n --device cuda:0 -> eval_outputs/mist_w3po_v1/idea6_test/wincache/<dir>_<key>/<m>.npz
  build     --set <name> --arm c|e -> eval_outputs/mist_w3po_v1/idea6_test/tokens/i6<arm>_<set>.{jsonl,sidecar.json}
  selftest  Idea 2 synthetic DEV streams only, through the same encode / posterior functions: (a) step-0 checkpoint:
            E2E tokens == C tokens; (b) step-3000 checkpoint: E2E_post errors == the dev gate's units (<= 1e-4), C arm ==
            precheck A's C_post.  --out <json>
Refuses to overwrite outputs (encode skips finished meetings).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import DIAR_DELAY, I2  # noqa: E402
import experiments.mist_w3po_v1.voicecond.idea2_stream_tokens as ST  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea6_e2e_train import (CKPT_C, FIT, FIT_SHA1, HOP, L, RC_PATH, SR,  # noqa: E402
                                                               enc_forward, load_trained, post_numpy, prior_terms)

OUT = REPO / "eval_outputs/mist_w3po_v1/idea6_test"
WC, TOKD = OUT / "wincache", OUT / "tokens"
M2TOK = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/tokens"
E2E_RUN = REPO / "eval_outputs/mist_w3po_v1/idea6_e2e/idea6_e2e_main_v1"
CK_E = E2E_RUN / "ckpt_step03000.pt"
WS = int(L * SR)
DD = "eval_outputs/mist_w3po_v1/data"
RT = "joint_v8h6k_20261113T073658Z_s06000__t1c"


def set_specs():
    S = {}
    for c in ("id2", "ood1", "clean"):
        for d in ("zh2en", "en2zh"):
            S[f"{c}_{d}"] = {"data": f"data/open_voicecond_v1/bench/{c}/{d}", "dir": d, "session": "", "diar": "nemotron_lowlat",
                             "run": f"obsi_b_{c}_{d}__t1c", "meetings": [], "m2": f"obsi_{c}_{d}",
                             "windows": f"{DD}/windows_obsi_{c}_{d}_c2g06_v1/windows_test_online_{d}.jsonl"}
    S["ami"] = {"data": "data/ami_multispk_si_v1", "dir": "zh2en", "session": "zh_mix_v8", "diar": "nemotron_lowlat_zh8",
                "run": f"jtest_bami_{RT}", "windows": f"{DD}/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl",
                "meetings": ["EN2002d", "IS1009c"], "m2": "ood3_ami"}
    S["en"] = {"data": "data/ami_multispk_si_v1", "dir": "en2zh", "session": "ihm_mix", "diar": "nemotron_lowlat",
               "run": f"jtest_ben_{RT}", "windows": f"{DD}/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl",
               "meetings": ["EN2002d", "IS1009c"], "m2": "ood3_en"}
    S["ali"] = {"data": "data/alimeeting_si_v1", "dir": "zh2en", "session": "near_mix", "diar": "nemotron_lowlat",
                "run": f"jtest_bali_{RT}", "windows": f"{DD}/windows_test_c2g06_v1/windows_test_online_ali.jsonl",
                "meetings": ["R8009_M8021", "R8009_M8024"], "m2": "ood3_ali"}
    return S


def cache_key(sp):
    data = REPO / sp["data"]
    return f"{sp['dir']}_" + hashlib.sha1(f"{data.resolve()}|{sp['session']}|{sp['diar']}".encode()).hexdigest()[:12]


def set_items(sp):
    """M2's item loop (idea2_stream_tokens.main): unique prompts in item order with (item, window, k, t_commit)"""
    S = REPO / "eval_outputs/mist_w3po_v1/speech" / sp["run"]
    items = [json.loads(l) for l in open(S / "talker_in/items.jsonl", encoding="utf-8")]
    blocks = {r["id"]: r["blocks"] for r in map(json.loads, open(S / "talker_in/blocks.jsonl", encoding="utf-8"))}
    W = {w["id"]: w for w in map(json.loads, open(REPO / sp["windows"], encoding="utf-8"))}
    out, seen = [], set()
    for it in items:
        p = Path(it["voice_audio"])
        p = p if p.is_absolute() else REPO / p
        if p.stem in seen:
            continue
        wid, k = it["id"].rsplit("__S", 1)
        w = W[wid]
        if sp["meetings"] and w["meeting"] not in sp["meetings"]:
            continue
        seen.add(p.stem)
        t = w["start_s"] + min(b["write_ms"] for b in blocks[it["id"]]) / 1000.0
        out.append({"key": p.stem, "id": it["id"], "path": p, "meeting": w["meeting"], "slot": int(k) - 1, "t": t})
    return out


class Encoders:
    def __init__(self, device, ck_e=CK_E):
        from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
        self.Pc = BiCodecPhi(device)
        load_trained(self.Pc, torch.load(CKPT_C, map_location="cpu"))
        self.Pe = BiCodecPhi(device)
        self.ck_e = str(ck_e)
        self.head = load_trained(self.Pe, torch.load(ck_e, map_location="cpu", weights_only=False))
        self.Pc.model.eval(); self.Pe.model.eval()

    @torch.no_grad()
    def encode(self, y, starts, bs=256):
        uC, uE, al = [], [], []
        for i in range(0, len(starts), bs):
            ref = torch.from_numpy(np.stack([self.Pc.ref_clip(y[s:s + WS]) for s in starts[i:i + bs]])).float().to(self.Pc.device)
            uC.append(self.Pc._u_from_ref(ref).reshape(len(ref), 192).cpu().numpy())
            u, pooled = enc_forward(self.Pe, ref)
            uE.append(u.cpu().numpy())
            al.append(self.head(pooled, u).cpu().numpy())
        cat = lambda x: np.concatenate(x).astype(np.float32) if x else np.zeros((0, 192), np.float32)  # noqa: E731
        return cat(uC), cat(uE), cat(al)


class Post:
    def __init__(self):
        from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams
        assert hashlib.sha1(FIT.read_bytes()).hexdigest() == FIT_SHA1
        self.pf = ShrinkParams.load(FIT)
        self.RC = np.load(RC_PATH)["R_C"]
        self.S0i, self.e0C, self.G, self.m0 = prior_terms(self.pf, self.RC)

    def __call__(self, arm, f, sel):
        """the posterior of both arms on windows sel of a cache dict f {uC, uE, alpha, pi}"""
        w = f["pi"][sel].astype(np.float64)
        if arm == "c":
            return post_numpy(f["uC"][sel].astype(np.float64), np.zeros((len(sel), 192)), w, self.S0i, self.e0C, self.G, self.m0)
        return post_numpy(f["uE"][sel].astype(np.float64), f["alpha"][sel].astype(np.float64), w, self.S0i, self.e0C, self.G,
                          self.m0)


def tokens(P, u):
    return [int(v) for v in P.tokens(np.asarray(u).reshape(32, 6).astype(np.float32))]


# ------------------------------------------------------------------------------------------------ encode
def jobs():
    out = {}
    for name, sp in set_specs().items():
        ck = cache_key(sp)
        for m in sorted({it["meeting"] for it in set_items(sp)}):
            out.setdefault((ck, m), (name, sp))
    return sorted(out.items())


def cmd_encode(a):
    torch.set_num_threads(4)
    J = jobs()[a.shard:: a.num_shards]
    E = Encoders(a.device)
    print(f"[enc] shard {a.shard}/{a.num_shards}: {len(J)} meetings on {a.device}", flush=True)
    t0 = time.time()
    for i, ((ck, m), (name, sp)) in enumerate(J, 1):
        dst = WC / ck / f"{m}.npz"
        if dst.exists():
            continue
        f = np.load(I2 / "streams_cache" / ck / f"{m}.npz")
        ST.SESS["sess"], ST.SESS["diar"] = sp["session"], sp["diar"]
        y, sr = sf.read(str(ST.session_audio(REPO / sp["data"], sp["dir"], m)), dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        assert sr == SR
        K = len(f["end"])
        st = (np.arange(K) * HOP * SR).astype(int)
        assert np.allclose((st + WS) / SR, f["end"]), m
        uC, uE, al = E.encode(y, st)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f".{m}.partial.npz"
        np.savez(tmp, end=f["end"], uC=uC, uE=uE, alpha=al, ckpt_e=E.ck_e, ckpt_c=str(CKPT_C))
        os.rename(tmp, dst)
        print(f"[enc] {i}/{len(J)} {name} {ck}/{m} K={K} {time.time() - t0:.0f}s", flush=True)
    print("[enc] DONE", flush=True)


# ------------------------------------------------------------------------------------------------ build
def cmd_build(a):
    sp = set_specs()[a.set]
    tag = f"i6{a.arm}_{a.set}"
    out, side = TOKD / f"{tag}.jsonl", TOKD / f"{tag}.sidecar.json"
    if out.exists() or side.exists():
        sys.exit(f"refuse to overwrite {out}")
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi("cpu")
    post = Post()
    ck = cache_key(sp)
    m2 = {r["key"]: r for r in map(json.loads, open(M2TOK / f"{sp['m2']}.jsonl", encoding="utf-8"))}
    feats, rows = {}, []
    for it in set_items(sp):
        m = it["meeting"]
        if m not in feats:
            f = dict(np.load(I2 / "streams_cache" / ck / f"{m}.npz"))
            g = np.load(WC / ck / f"{m}.npz")
            assert np.array_equal(g["end"], f["end"]), m
            f.update(uC=g["uC"], uE=g["uE"], alpha=g["alpha"])
            feats[m] = f
        f = feats[m]
        sel = np.nonzero((f["slot"] == it["slot"]) & (f["end"] + DIAR_DELAY <= it["t"] + 1e-9))[0]
        r2 = m2[it["key"]]
        sha = hashlib.sha1(it["path"].read_bytes()).hexdigest()
        assert r2["sha1"] == sha and r2["n_win"] == len(sel) and r2["fallback"] == (len(sel) == 0) \
            and abs(r2["t_commit"] - round(it["t"], 3)) < 1e-9, (it["key"], r2["n_win"], len(sel))
        r = {"key": it["key"], "id": it["id"], "mode": {"c": "C_post", "e": "E2E_post"}[a.arm], "fit_sha1": FIT_SHA1,
             "R_C": str(RC_PATH), "ckpt": str(CKPT_C) if a.arm == "c" else str(CK_E), "sha1": sha,
             "t_commit": round(it["t"], 3), "n_win": int(len(sel)), "pi_sum": float(f["pi"][sel].sum()),
             "fallback": len(sel) == 0}
        r["ids"] = [int(v) for v in r2["ids"]] if len(sel) == 0 else tokens(P, post(a.arm, f, sel))
        rows.append(r)
    assert len(rows) == len(m2), (len(rows), len(m2))
    TOKD.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(side, "x", encoding="utf-8") as fh:
        json.dump({r["key"]: r["ids"] for r in rows}, fh)
    nf = [r for r in rows if not r["fallback"]]
    agree = float(np.mean([np.mean(np.array(r["ids"]) == np.array(m2[r["key"]]["ids"])) for r in nf])) if nf else None
    print(json.dumps({"out": str(out), "n": len(rows), "fallback": len(rows) - len(nf), "token_agreement_with_M2_nonfallback": agree}))


# ------------------------------------------------------------------------------------------------ selftest (dev only)
def cmd_selftest(a):
    from experiments.mist_w3po_v1.voicecond.idea2_common import T_EVAL, load_refs, load_stream, slot_for_speaker, speech_clock
    outp = Path(a.out)
    if outp.exists():
        sys.exit(f"refuse to overwrite {outp}")
    ss = a.streams.split(",")
    assert all(s.startswith("dev_") for s in ss)
    post, refs = Post(), load_refs()
    gate = {(u["sid"], u["spk"]): u for u in np.load(E2E_RUN / "dev_gate_step03000/units.npz", allow_pickle=True)["units"]}
    rep = {"streams": ss}
    for tagck, ck in (("step0", E2E_RUN / "ckpt_step00000.pt"), ("step3000", CK_E)):
        E = Encoders(a.device, ck)
        n_same, n_tok, de, dc = 0, 0, 0.0, 0.0
        for sid in ss:
            d = load_stream(sid, L)
            K = len(d["end"])
            y = sf.read(str(I2 / "streams" / sid / "mix.wav"), dtype="float32")[0]
            st = (np.arange(K) * HOP * SR).astype(int)
            uC, uE, al = E.encode(y, st)
            f = {"uC": uC, "uE": uE, "alpha": al, "pi": d["pi"], "slot": d["slot"], "end": d["end"]}
            for j, spk in enumerate(d["meta"]["speakers"]):
                s = slot_for_speaker(d, j)
                tau, _ = speech_clock(d["meta"], j)
                g = gate[(sid, spk)]
                for ti, t in enumerate(T_EVAL):
                    tt = tau(t)
                    if tt is None:
                        continue
                    sel = np.nonzero((f["slot"] == s) & (f["end"] + DIAR_DELAY <= tt + 1e-9))[0]
                    ue, uc = post("e", f, sel), post("c", f, sel)
                    if tagck == "step0":
                        n_same += int(tokens(E.Pc, ue) == tokens(E.Pc, uc))
                        n_tok += 1
                    else:
                        de = max(de, abs(float(np.linalg.norm(ue - refs[spk][0])) - g["err"]["E2E_post"][ti]))
                        dc = max(dc, abs(float(np.linalg.norm(uc - refs[spk][0])) - g["err"]["C_post"][ti]))
        rep[tagck] = ({"tokens_identical": n_same, "n": n_tok, "pass": n_same == n_tok} if tagck == "step0" else
                      {"E2E_err_maxabs_vs_dev_gate": de, "C_err_maxabs_vs_precheckA": dc, "pass": bool(de <= 1e-4 and dc <= 1e-4)})
        print(json.dumps({tagck: rep[tagck]}), flush=True)
        del E
        torch.cuda.empty_cache()
    rep["pass"] = bool(rep["step0"]["pass"] and rep["step3000"]["pass"])
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "x") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "build", "selftest", "jobs"])
    ap.add_argument("--set", default="")
    ap.add_argument("--arm", choices=["c", "e"], default="c")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--streams", default="dev_en_0000,dev_zh_0005,dev_en_0017,dev_zh_0030")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "jobs":
        J = jobs()
        print(len(J), {k: sum(1 for (c, _), _ in J if c == k) for k in sorted({c for (c, _), _ in J})})
    else:
        {"encode": cmd_encode, "build": cmd_build, "selftest": cmd_selftest}[a.cmd](a)
