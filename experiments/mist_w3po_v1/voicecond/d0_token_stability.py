"""Robust D0 (CPU only): intrinsic stability of the frozen BiCodec global-token encoder phi0 vs the information in short noisy
prompts vs the denoisers (reports/mist_w3po_v1/robust_d0_token_stability.zh-CN.md).

Prompt span = what the Talker gets (oracle_voice_tokens.py): item audio [0, max(first write, 1 s)), first write from the item's
own S1 Thinker blocks (wave2_S1/thinker_<dir>/blocks_<dir>_w1.jsonl); every view is round-tripped through a PCM_16 wav exactly
like si_infer's sf.write prompt file.

Views per direction (all deterministic, seeded by crc32 of the view id):
  base      every clean item, its own span          (Q1 reference; Q1d same-speaker + Q2 EER on set b without crosstalk `ovl`,
            speaker = meta `stream` = meeting x speaker)
  shift10   sample of 300 clean items: span start +10 ms, same length
  long200   same sample: span + 0.2 s (only if the item is long enough)
  wn45      same sample: + white noise at -45 dB relative to active-speech power (20 ms frames within 30 dB of the loudest)
  full      same sample: the whole utterance (phi0 uses its first 6 s)                               (Q3)
  noisy     every noisy item over its own span; twin = its clean twin (same set+pair) over the NOISY item's span   (Q2)
Stages (each refuses to overwrite its outputs):
  vllm_omni_030 python -I -m ... --stage phi    -> <out>/phi_<dir>.npz   (u, bounded, ids, d-vector, soft d-vector, ECAPA x-vector)
  nemo_diar     python -I -m ... --stage sv     -> <out>/sv_<dir>.npz    (TitaNet-large, WavLM-base-plus-sv)
  any python with numpy  ... --stage report     -> <out>/d0_summary.json (+ markdown tables on stdout)
CUDA_VISIBLE_DEVICES="" always (CPU only).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zlib
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
ORACLE = REPO / "eval_outputs/mist_w3po_v1/oracle_voice/tokens"
DIRS = ("zh2en", "en2zh")
NSAMPLE = 300
SR = 16000
NEMO = ("/opt/dlami/nvme/jasonleeeli/cache/hf_v6/models--nvidia--speakerverification_en_titanet_large/snapshots/"
        "0dc382f40121a5fbd34db10a2bb04d826c2be6a8/speakerverification_en_titanet_large.nemo")
WAVLM = "/opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def item_wav(it):
    x, sr = sf.read(it["audio"] if Path(it["audio"]).is_absolute() else str(REPO / it["audio"]), dtype="float32",
                    always_2d=True)
    assert sr == SR, it["audio"]
    return x.mean(axis=1)[0: int(it["dur_ms"]) * 16]


def pcm16(w):
    b = io.BytesIO()
    sf.write(b, np.asarray(w, dtype=np.float32), SR, format="WAV")    # default subtype PCM_16, as si_infer's prompt file
    b.seek(0)
    return sf.read(b, dtype="float32")[0]


def speech_power(w):
    f = w[: len(w) // 320 * 320].reshape(-1, 320).astype(np.float64)
    e = (f ** 2).mean(1)
    act = e[e >= e.max() * 1e-3]
    return float(act.mean())


def seed(s):
    return zlib.crc32(s.encode())


def build(d):
    """-> meta list (dicts) and a generator of waveforms in the same order"""
    meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
    items = {it["id"]: it for it in jl(S1 / "data" / f"items_{d}.jsonl")}
    first = {r["id"]: r["blocks"][0]["write_ms"] for r in jl(S1 / f"thinker_{d}" / f"blocks_{d}_w1.jsonl") if r.get("blocks")}
    span = {i: max(first[i], 1000) * 16 for i in items}
    twin = {(m["set"], m["pair"]): i for i, m in meta.items() if m["cond"] == "clean"}
    clean = sorted(i for i, m in meta.items() if m["cond"] == "clean")
    noisy = sorted(i for i, m in meta.items() if m["cond"] == "noisy")
    rng = np.random.default_rng(20261121)
    sample = sorted(rng.choice(clean, size=NSAMPLE, replace=False).tolist())
    spk = lambda i: meta[i]["stream"] if meta[i]["set"] == "b" else None     # speaker = meeting x speaker (wave2_s1_prep)
    V = []
    for i in clean:
        V.append(dict(vid=f"base|{i}", kind="base", item=i, set=meta[i]["set"], spk=spk(i), a=0, n=span[i],
                      ovl=bool(meta[i]["ovl"])))
    for i in sample:
        L, D = span[i], int(items[i]["dur_ms"]) * 16
        V.append(dict(vid=f"shift10|{i}", kind="shift10", item=i, set=meta[i]["set"], spk=spk(i), a=160, n=L))
        if D >= L + 3200:
            V.append(dict(vid=f"long200|{i}", kind="long200", item=i, set=meta[i]["set"], spk=spk(i), a=0, n=L + 3200))
        V.append(dict(vid=f"wn45|{i}", kind="wn45", item=i, set=meta[i]["set"], spk=spk(i), a=0, n=L))
        V.append(dict(vid=f"full|{i}", kind="full", item=i, set=meta[i]["set"], spk=spk(i), a=0, n=D))
    for i in noisy:
        t = twin[(meta[i]["set"], meta[i]["pair"])]
        V.append(dict(vid=f"noisy|{i}", kind="noisy", item=i, set=meta[i]["set"], spk=spk(i), a=0, n=span[i], twin=t,
                      ovl=bool(meta[i]["ovl"])))
        V.append(dict(vid=f"twin|{i}", kind="twin", item=t, set=meta[i]["set"], spk=spk(i), a=0, n=span[i], noisy=i))

    def waves():
        cache = {}
        for v in V:
            if v["item"] not in cache:
                cache = {v["item"]: item_wav(items[v["item"]])}
            w = cache[v["item"]][v["a"]: v["a"] + v["n"]]
            if v["kind"] == "wn45":
                p = speech_power(w)
                w = w + np.random.default_rng(seed(v["vid"])).standard_normal(len(w)).astype(np.float32) * np.sqrt(p * 10 ** -4.5)
            yield pcm16(w)
    return V, waves


def stage_phi(out, threads, limit):
    import torch
    torch.set_num_threads(threads)
    sys.path.insert(0, str(REPO))
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi("cpu")
    se, rq, fsq = P.se, P.rq, P.fsq

    @torch.no_grad()
    def run(ws):
        ref = torch.from_numpy(np.stack([P.ref_clip(w, True) for w in ws])).float()
        mel = P.model.mel_transformer(ref).squeeze(1)
        xv, feats = se.speaker_encoder(mel.transpose(1, 2), True)
        x = se.perceiver_sampler(feats.transpose(1, 2)).transpose(1, 2).transpose(1, 2)
        u = rq.project_in(x)
        b = fsq.bound(u)
        _, idx = fsq(u)
        ids = idx.reshape(idx.shape[0], -1).long()
        dv = se.detokenize(ids[:, None, :])                      # (B,1,32) as BiCodecTokenizer.detokenize

        def soft(c):
            zq = rq.project_out(c).transpose(1, 2)
            return se.project(zq.reshape(zq.shape[0], -1))
        dsoft = soft(b / 2)
        dchk = soft(torch.round(b) / 2)
        return dict(u=u.numpy(), bounded=b.numpy(), ids=ids.numpy(), dvec=dv.numpy(), dsoft=dsoft.numpy(),
                    xvec=xv.numpy(), dchk_err=(dchk - dv).abs().max(1).values.numpy())

    for d in DIRS:
        f = out / f"phi_{d}.npz"
        if f.exists():
            sys.exit(f"拒绝覆盖已有输出: {f}")
        V, waves = build(d)
        V = V[:limit] if limit else V
        acc, buf = {}, []
        for k, w in enumerate(waves(), 1):
            if limit and k > limit:
                break
            buf.append(w)
            if len(buf) == 64 or k == len(V):
                r = run(buf)
                for key, val in r.items():
                    acc.setdefault(key, []).append(val)
                buf = []
                print(f"[d0-phi] {d} {k}/{len(V)}", flush=True)
        arr = {k: np.concatenate(v) for k, v in acc.items()}
        print(f"[d0-phi] {d} soft-vs-hard d-vector check max|err| = {arr['dchk_err'].max():.2e}", flush=True)
        with open(f, "xb") as fh:
            np.savez(fh, vid=np.array([v["vid"] for v in V]), **arr)


def stage_sv(out, threads, limit):
    import torch
    torch.set_num_threads(threads)
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    from transformers import AutoFeatureExtractor, WavLMForXVector
    tn = EncDecSpeakerLabelModel.restore_from(NEMO, map_location="cpu").eval()
    fe = AutoFeatureExtractor.from_pretrained(WAVLM, local_files_only=True)
    wl = WavLMForXVector.from_pretrained(WAVLM, local_files_only=True).eval()

    @torch.no_grad()
    def emb(w):
        x = torch.from_numpy(w)[None]
        _, e1 = tn.forward(input_signal=x, input_signal_length=torch.tensor([x.shape[1]]))
        inp = fe(w, sampling_rate=SR, return_tensors="pt")
        e2 = wl(**inp).embeddings
        return e1[0].float().numpy(), e2[0].float().numpy()

    for d in DIRS:
        f = out / f"sv_{d}.npz"
        if f.exists():
            sys.exit(f"拒绝覆盖已有输出: {f}")
        V, waves = build(d)
        V = V[:limit] if limit else V
        T, W = [], []
        for k, w in enumerate(waves(), 1):
            if limit and k > limit:
                break
            a, b = emb(w)
            T.append(a)
            W.append(b)
            if k % 200 == 0:
                print(f"[d0-sv] {d} {k}/{len(V)}", flush=True)
        with open(f, "xb") as fh:
            np.savez(fh, vid=np.array([v["vid"] for v in V]), titanet=np.stack(T), wavlm=np.stack(W))


# ----------------------------------------------------------------------------------------------------------------- report
def nrm(x):
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def eer(same, diff):
    s = np.concatenate([same, diff])
    y = np.concatenate([np.ones(len(same)), np.zeros(len(diff))])
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y) / len(same)
    fp = np.cumsum(1 - y) / len(diff)
    fnr = 1 - tp
    k = np.argmin(np.abs(fnr - fp))
    return float((fnr[k] + fp[k]) / 2)


def near_boundary(b, tol=0.1):
    bd = np.array([-1.5, -0.5, 0.5])
    return np.abs(b.reshape(*b.shape, 1) - bd).min(-1) < tol


def stage_report(out):
    S = {}
    md = []
    for d in DIRS:
        V, _ = build(d)
        P = np.load(out / f"phi_{d}.npz")
        Q = np.load(out / f"sv_{d}.npz")
        assert list(P["vid"]) == [v["vid"] for v in V] == list(Q["vid"]), d
        ix = {v["vid"]: k for k, v in enumerate(V)}
        ids, B = P["ids"], P["bounded"].reshape(len(V), -1)
        lev = np.round(B).astype(int)
        E = {"phi0_dvec": nrm(P["dvec"]), "phi0_dsoft": nrm(P["dsoft"]), "phi0_u": nrm(P["u"]),
             "phi0_ecapa_x": nrm(P["xvec"]), "titanet": nrm(Q["titanet"]), "wavlm_sv": nrm(Q["wavlm"])}
        R = {"n_views": len(V), "soft_hard_check_maxerr": float(P["dchk_err"].max())}

        def pair_stats(pairs):
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            nb_a = near_boundary(B[a])
            flip = lev[a] != lev[b]
            r = {"n": len(pairs), "tok_match": float((ids[a] == ids[b]).mean()),
                 "tok_match_median": float(np.median((ids[a] == ids[b]).mean(1))),
                 "all32_same": float((ids[a] == ids[b]).all(1).mean()),
                 "dim_match": float((~flip).mean()),
                 "near_bnd_ref": float(nb_a.mean()), "near_bnd_pert": float(near_boundary(B[b]).mean()),
                 "abs_dbounded_median": float(np.median(np.abs(B[a] - B[b]))),
                 "flips_near_bnd": float(nb_a[flip].mean()) if flip.any() else float("nan")}
            for k, e in E.items():
                r[f"cos_{k}"] = float((e[a] * e[b]).sum(1).mean())
            return r

        # sanity: CPU base tokens vs the GPU oracle clean rows
        orc = {r["id"]: r["ids"] for r in jl(ORACLE / f"tokens_oracle_{d}.jsonl")}
        bases = [v for v in V if v["kind"] == "base" and v["item"] in orc]
        m = np.array([np.mean(np.array(orc[v["item"]]) == ids[ix[v["vid"]]]) for v in bases])
        R["sanity_cpu_vs_gpu_oracle"] = {"n": len(bases), "tok_match": float(m.mean()), "all32_same": float((m == 1).mean())}

        # Q1
        q1 = {}
        for kind in ("shift10", "long200", "wn45"):
            q1[kind] = pair_stats([(ix[f"base|{v['item']}"], ix[v["vid"]]) for v in V if v["kind"] == kind])
        bb = [v for v in V if v["kind"] == "base" and v["set"] == "b" and not v["ovl"]]   # speaker analyses: no crosstalk
        byspk = {}
        for v in bb:
            byspk.setdefault(v["spk"], []).append(ix[v["vid"]])
        same = [(L[j], L[(j + 1) % len(L)]) for L in byspk.values() if len(L) > 1 for j in range(len(L))]
        rng = np.random.default_rng(7)
        spks = sorted(byspk)
        diff = []
        for s_, L in byspk.items():
            for j in L:
                o = spks[(spks.index(s_) + 1 + rng.integers(len(spks) - 1)) % len(spks)]
                diff.append((j, int(rng.choice(byspk[o]))))
        q1["same_spk_other_utt"] = pair_stats(same)
        q1["diff_spk_ref"] = pair_stats(diff)
        allb = np.array([ix[f"base|{v['item']}"] for v in V if v["kind"] == "base"])
        q1["base_near_bnd_all"] = float(near_boundary(B[allb]).mean())
        # chance per-dim level match given the marginal level distribution per dim
        pl = np.stack([(lev[allb] == L_).mean(0) for L_ in (-2, -1, 0, 1)])
        q1["chance_dim_match"] = float((pl ** 2).sum(0).mean())
        R["Q1"] = q1

        # Q2 noise invariance + separation
        q2 = {}
        tw = [(ix[f"twin|{v['item']}"], ix[v["vid"]], v["set"]) for v in V if v["kind"] == "noisy"]
        bi = np.array([ix[v["vid"]] for v in bb])
        bspk = np.array([v["spk"] for v in bb])
        bpair = np.array([v["item"] for v in bb])
        triu = np.triu_indices(len(bi), 1)
        same_m = (bspk[:, None] == bspk[None, :])[triu]
        nz = [v for v in V if v["kind"] == "noisy" and v["set"] == "b" and not v["ovl"]]
        ni = np.array([ix[v["vid"]] for v in nz])
        nspk = np.array([v["spk"] for v in nz])
        ntwin = np.array([v["twin"] for v in nz])
        for k, e in E.items():
            c = np.array([(e[a] * e[b]).sum() for a, b, _ in tw])
            sets = np.array([s for _, _, s in tw])
            G = e[bi] @ e[bi].T
            gs, gd = G[triu][same_m], G[triu][~same_m]
            X = e[ni] @ e[bi].T                                         # noisy probe vs clean enrol (other utterances)
            msk_same = (nspk[:, None] == bspk[None, :]) & (ntwin[:, None] != bpair[None, :])
            msk_diff = nspk[:, None] != bspk[None, :]
            md_ = float(gd.mean())
            q2[k] = {"twin_cos_mean": float(c.mean()), "twin_cos_median": float(np.median(c)),
                     "twin_cos_set_a": float(c[sets == "a"].mean()), "twin_cos_set_b": float(c[sets == "b"].mean()),
                     "same_spk_cos": float(gs.mean()), "diff_spk_cos": md_,
                     "dprime": float((gs.mean() - gd.mean()) / np.sqrt((gs.var() + gd.var()) / 2)),
                     "NI": float((c.mean() - md_) / (1 - md_)),
                     "twin_z": float((c.mean() - md_) / gd.std()),
                     "eer_clean": eer(gs, gd), "eer_noisy_vs_clean": eer(X[msk_same], X[msk_diff])}
        tk = np.array([(ids[a] == ids[b]).mean() for a, b, _ in tw])
        q2["phi0_tok_match_noisy_vs_twin"] = float(tk.mean())
        q2["phi0_dim_match_noisy_vs_twin"] = float(np.mean([(lev[a] == lev[b]).mean() for a, b, _ in tw]))
        q2["n_pairs"] = len(tw)
        q2["n_b_clean_noovl"] = len(bi)
        q2["n_speakers"] = len(set(bspk.tolist()))
        q2["n_same_pairs"] = int(same_m.sum())
        R["Q2"] = q2

        # Q3 span vs full
        fv = [v for v in V if v["kind"] == "full"]
        pr = [(ix[f"base|{v['item']}"], ix[v["vid"]]) for v in fv]
        q3 = {"all": pair_stats(pr)}
        spanlen = np.array([V[a]["n"] / SR for a, _ in pr])
        fulllen = np.array([V[b]["n"] / SR for _, b in pr])
        q3["span_s_median"] = float(np.median(spanlen))
        q3["full_s_median"] = float(np.median(fulllen))
        for lo, hi in ((1, 2), (2, 3), (3, 4), (4, 99)):
            sel = [p for p, s in zip(pr, spanlen) if lo <= s < hi]
            if sel:
                q3[f"span_{lo}-{hi}s"] = pair_stats(sel)
        R["Q3"] = q3
        S[d] = R
    f = out / "d0_summary.json"
    if f.exists():
        sys.exit(f"拒绝覆盖已有输出: {f}")
    with open(f, "x", encoding="utf-8") as fh:
        json.dump(S, fh, indent=1)
    print(json.dumps(S, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("phi", "sv", "report"), required=True)
    ap.add_argument("--out", default=str(REPO / "eval_outputs/mist_w3po_v1/d0"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N views per direction")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    {"phi": lambda: stage_phi(out, a.threads, a.limit), "sv": lambda: stage_sv(out, a.threads, a.limit),
     "report": lambda: stage_report(out)}[a.stage]()


if __name__ == "__main__":
    main()
