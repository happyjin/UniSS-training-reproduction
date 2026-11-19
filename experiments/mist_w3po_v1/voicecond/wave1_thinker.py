"""Robust wave 1 Thinker prechecks K2 (P2a, P2b + logit linearity), K3, K7, K8 (+ 2 reference arms) on the shared
CoVoST 2 validation set of wave1_data.py (prereg reports/mist_w3po_v1/robust_wave1_prereg.zh-CN.md; K5 / K6 dropped by
the user 2026-11-19). One setting per method, no sweep.

Thinker = B: joint_v8h6k merged_s06000/thinker_merged + the identity crf ckpt (crf/decode_crf_ckpt.apply_ckpt), loaded
with core/engine.Engine.from_merged; never changed. Each sentence is ONE window decoded exactly as B's test decoder
(online_decode --policy wait --context 0, reads every 1 s, greedy, max_new 300, loop guard k 6 / maxp 12, budget 25
tok/s + 50, --final-read-post close_detect with close_detect = duration + 1.04 s: the final read sees every in-window
posterior frame). Posteriors: the sentence's synthetic post (wave1_data.synth_post), identical in every arm and branch.

Arms (unit = arm, cond, dir, sentence shard; outputs runs/<arm>_<cond>_<dir>/s<k>.jsonl):
  base   B on the clean / noisy sentence (P2a = text BLEU clean - noisy)
  p2b    background-contrastive decoding: every step of every read, the main window and the same window with the
         sentence's equal-length background-only clip (same post, same committed prefix) in one batch;
         l = l_main + w (l_main - l_bg), w = min(3, (1 - lam) / lam), lam = Idea 2 lambda head on the noisy sentence
         (data/lambda.jsonl); only tokens with p_main >= 0.1 max p_main are allowed (adaptive plausibility). Starts
         only after the P2a gate file says the en2zh noise loss >= 1.0 BLEU (P2A_GATE.json, written by the first worker
         that sees every base unit done).
  k3     feature offset: delta = mean(log-mel of the 3 s gap) - mean(training-silence log-mel, data/k3_silence.npz);
         gap frames from the log-mel of [gap; noisy sentence] (frames 0 .. 295); every read's input features (valid
         frames) - delta
  k3dist per noisy sentence: mean frame L2 distance to the clean twin, of the log-mel input and of the audio-encoder
         output, before / after the K3 correction (no decode)
  k7     online soft subspace Wiener at the audio encoder's upper half (layers 16-31 of 32, output of each layer, fixed,
         no selection rule; idea doc 6.3): per layer second moments (uncentred, frame level) Sigma_s (300 clean CoVoST
         train sentences) and Sigma_g (300 training-noise clips) from k7stats; semantic directions = eigenvectors of
         Sigma_s covering 95 % of its trace; per sentence Sigma_hat = (S_gap + (nu - d - 1) Sigma_g) / (n_gap + nu - d - 1),
         nu = d + 1 + 50 (inverse-Wishart posterior mean; S_gap, n_gap = second-moment sum and frame count of the 3 s gap;
         clean condition: a 3 s digital-silence gap, as in the clean twin's gaps); Q = eigenvectors of Sigma_hat covering
         95 % of its trace whose max |cos| with every semantic direction < 0.1; gamma_i = s2n_i / (s2n_i + s2s_i),
         s2n_i = q_i' Sigma_hat q_i, s2s_i = q_i' Sigma_s q_i; a <- a - sum_i gamma_i (q_i' a) q_i at every read
  k8     x = a* HDemucs(x) + (1 - a*) x, a* = s2n / (s2n + s2a), s2n = (1 - lam) / lam (lambda of that condition's
         sentence), s2a = data/k8cal.json sigma2_a; k8h pure HDemucs vocals; k8f fixed a = 0.5
  lin    Thinker logit linearity in sigma (P1a's recipe): first 100 sentences per direction; x(sigma) = sqrt(1 - sigma) s
         + sqrt(sigma) b', b' = the background under the speech rescaled to the speech energy, sigma in {0,.25,.5,.75,1};
         final-read prompt teacher-forced on the clean B decode (+ <eos>); per step and vocabulary entry a linear and a
         quadratic fit in sigma (centred on the mean over sigma): summed SS_tot, SS_res(lin), SS_res(quad) per sentence
         (raw logits; log-softmax as a supplement)
  k7stats once (any worker): the K7 calibration statistics -> k7_stats.pt

Workers pull units from one queue (claim files, O_EXCL); arms are interleaved so every method runs at the same time;
units whose inputs are missing (k7 before k7_stats.pt, lin before its base-clean unit, p2b before the gate) are deferred.
Outputs are written to .partial and renamed; an existing output is never rewritten.

    CUDA_VISIBLE_DEVICES=<card> python -I experiments/mist_w3po_v1/voicecond/wave1_thinker.py --worker k [--max-gib 62]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
OUT = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5"
DATA = OUT / "data"
RUNS, CLAIMS = OUT / "runs", OUT / "claims"
JO = REPO / "eval_outputs/mist_w3po_v1/joint/joint_v8h6k_20261113T073658Z/merged_s06000"
THINKER, IDN = JO / "thinker_merged", JO / "identity"
DIRS = ("en2zh", "zh2en")
N_SHARD = 8
UPPER = list(range(16, 32))
ENERGY, DELTA_COS, NU_EXTRA = 0.95, 0.1, 50
W_CAP, PLAUS = 3.0, 0.1
SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_LIN = 100
GAP_FRAMES = 295
LOOKAHEAD_MS = 1040
SR = 16000


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------- data
def manifest() -> dict[str, list[dict]]:
    M = {d: [] for d in DIRS}
    for x in map(json.loads, open(DATA / "manifest.jsonl", encoding="utf-8")):
        M[x["dir"]].append(x)
    return M


def lambdas() -> dict:
    return {(r["id"], r["cond"]): r for r in map(json.loads, open(DATA / "lambda.jsonl", encoding="utf-8"))}


def wav(rel: str) -> np.ndarray:
    x, sr = sf.read(str(REPO / rel), dtype="float32", always_2d=True)
    assert sr == SR
    return x.mean(1)


def units() -> list[tuple]:
    """(arm, cond, dir, shard), arms interleaved shard by shard"""
    arms = [("base", "clean"), ("base", "noisy"), ("k3", "noisy"), ("k7", "noisy"), ("k8", "noisy"), ("k8h", "noisy"),
            ("k8f", "noisy"), ("k7", "clean"), ("k8", "clean"), ("k8h", "clean"), ("k8f", "clean"), ("k3dist", "noisy"),
            ("lin", "noisy"), ("p2b", "noisy")]
    out = [("k7stats", "-", "-", 0)]
    for k in range(N_SHARD):
        for d in DIRS:
            for a, c in arms:
                out.append((a, c, d, k))
    return out


def uname(u) -> str:
    return "k7stats" if u[0] == "k7stats" else f"{u[0]}_{u[1]}_{u[2]}/s{u[3]}"


def out_path(u) -> Path:
    return OUT / "k7_stats.pt" if u[0] == "k7stats" else RUNS / f"{u[0]}_{u[1]}_{u[2]}" / f"s{u[3]}.jsonl"


# ---------------------------------------------------------------------------------------------------------- engine
def build_engine(max_gib: int):
    from experiments.mist_w3po_v1.core import engine as E
    from experiments.mist_w3po_v1.crf import decode_crf_ckpt as dc

    class WEngine(E.Engine):
        """Engine with three optional input-side interventions in audio_embeds (k3 feature offset, k7 layer projector)
        and a capture hook on the upper encoder layers; read() is the engine's own."""
        mode = "base"

        def setup(self):
            self.delta, self.proj, self._ctx, self._cap = {}, {}, None, None
            enc = self.thinker.audio_tower
            for li in UPPER:
                enc.layers[li].register_forward_hook(self._hook(li))

        def _hook(self, li):
            def fn(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                if self._cap is not None:
                    self._cap[li].append(h.detach().float())
                if self._ctx is None:
                    return None
                lens, Ps = self._ctx
                assert sum(lens) == h.shape[0], (lens, h.shape)
                hn, pos = h.clone(), 0
                for n, P in zip(lens, Ps):
                    if P is not None:
                        x = h[pos: pos + n].float()
                        hn[pos: pos + n] = (x - x @ P[li]).to(h.dtype)
                    pos += n
                return (hn,) + tuple(output[1:]) if isinstance(output, tuple) else hn
            return fn

        @torch.no_grad()
        def encode(self, wavs: list[np.ndarray], feat_fn=None, Ps=None, capture=False):
            """the engine's feature + encoder pass; feat_fn(j, feats[j], n_frames) may edit features in place"""
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths
            f = self.pr.proc.feature_extractor(wavs, sampling_rate=16000, padding=True, truncation=False,
                                               return_attention_mask=True, return_tensors="pt")
            feats, am = f["input_features"], f["attention_mask"]
            if feat_fn is not None:
                for j in range(len(wavs)):
                    feat_fn(j, feats[j], int(am[j].sum()))
            n_audio = _get_feat_extract_output_lengths(am.sum(1)).tolist()
            self._ctx = (n_audio, Ps) if Ps is not None and any(P is not None for P in Ps) else None
            if capture:
                self._cap = {li: [] for li in UPPER}
            try:
                a = self.thinker.get_audio_features(feats.to(self.dev, torch.bfloat16),
                                                    am.to(self.dev)).last_hidden_state
            finally:
                self._ctx = None
            cap, self._cap = self._cap, None
            return a, n_audio, feats, am, cap

        @torch.no_grad()
        def audio_embeds(self, items, emb_dtype, lookahead_ms=E.LOOKAHEAD_MS, cache=None):
            from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import interleave, post_to_tokens
            assert cache is None
            keys, uniq = [], {}
            for j, it in enumerate(items):
                k = it.akey or f"__{j}"
                keys.append(k)
                uniq.setdefault(k, j)
            order = list(uniq)
            wavs = []
            for k in order:
                it = items[uniq[k]]
                x = it.wav[: it.t_ms * 16]
                wavs.append(x if len(x) >= 1600 else np.pad(x, (0, 1600 - len(x))))
            sids = [k.split("|")[0] for k in order]
            feat_fn = None
            if self.mode == "k3":
                def feat_fn(j, F, n):
                    F[:, :n] -= torch.as_tensor(self.delta[sids[j]], dtype=F.dtype)[:, None]
            Ps = [self.proj.get(s) for s in sids] if self.mode == "k7" else None
            a, n_audio, _, _, _ = self.encode(wavs, feat_fn, Ps)
            a = a.to(emb_dtype)
            out, pos = {}, 0
            for k, n, x in zip(order, n_audio, wavs):
                it = items[uniq[k]]
                ai = a[pos: pos + n]
                pos += n
                post = it.post.copy()
                if not it.full:
                    post[max(0, it.t_ms - lookahead_ms) // 80:] = 0.0
                pt = torch.as_tensor(post_to_tokens(post, n, len(x) / 16000)).to(self.dev, torch.bfloat16)
                sp = self.sd_proj(pt[None])[0]
                out[k] = (n, interleave(ai, sp.to(emb_dtype)))
            return [out[k] for k in keys]

        @torch.no_grad()
        def read_cd(self, items, bg_items, ws, max_new=300):
            """E.Engine.read (greedy) with background-contrastive logits: rows B.. are the bg branch of rows ..B"""
            from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import AUDIO_PAD, EOS_ID
            B = len(items)
            allit = list(items) + list(bg_items)
            emb_dtype = self.emb_layer.weight.dtype
            ae = self.audio_embeds(allit, emb_dtype)
            prompts = [self.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(allit, ae)]
            L = max(len(p) for p in prompts)
            ids = torch.full((2 * B, L), self.tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((2 * B, L), dtype=torch.long)
            for i, p in enumerate(prompts):
                ids[i, L - len(p):] = torch.as_tensor(p)
                mask[i, L - len(p):] = 1
            ids, mask = ids.to(self.dev), mask.to(self.dev)
            emb = self.emb_layer(ids)
            emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
            o = self.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=True, return_dict=True)
            pkv, h = o.past_key_values, o.last_hidden_state[:, -1]
            outs = [E.ReadOut(prompt_len=len(p)) for p in prompts[:B]]
            bar = torch.tensor([it.final or it.bar_wait for it in items], device=self.dev)
            done = torch.zeros(B, dtype=torch.bool, device=self.dev)
            w = torch.tensor(ws, device=self.dev, dtype=torch.float32)[:, None]
            stats = {"steps": 0, "changed": 0}
            for step in range(max_new):
                lg = self.logits(h).to(self.dev)
                lm, lb = lg[:B].clone(), lg[B:]
                lm[bar, E.WAIT_ID] = -1e9
                pm = torch.softmax(lm, -1)
                ok = pm >= PLAUS * pm.max(-1, keepdim=True).values
                comb = torch.where(ok, lm + w * (lm - lb), torch.full_like(lm, -float("inf")))
                comb[bar, E.WAIT_ID] = -1e9
                nxt = comb.argmax(-1)
                act = ~done
                stats["steps"] += int(act.sum())
                stats["changed"] += int(((nxt != lm.argmax(-1)) & act).sum())
                nl = nxt.tolist()
                for i in range(B):
                    if not done[i]:
                        outs[i].ids.append(nl[i])
                done |= (nxt == EOS_ID) | (nxt == E.WAIT_ID)
                if bool(done.all()):
                    break
                mask = torch.cat([mask, torch.ones((2 * B, 1), dtype=mask.dtype, device=self.dev)], 1)
                n2 = torch.cat([nxt, nxt])[:, None]
                o = self.thinker.model(inputs_embeds=self.emb_layer(n2), attention_mask=mask, past_key_values=pkv,
                                       use_cache=True, return_dict=True)
                pkv, h = o.past_key_values, o.last_hidden_state[:, -1]
            del pkv, o
            return outs, stats

    eng = WEngine.from_merged(str(THINKER), "en2zh", max_gib)
    off = [nm for nm, p in eng.thinker.named_parameters() if p.device.type != "cuda"]
    assert not off, f"{len(off)} thinker parameters not on the GPU, e.g. {off[:3]}"
    info = dc.apply_ckpt(eng, IDN)
    log("B loaded: identity ckpt applied " + json.dumps(info))
    eng.setup()
    eng.prompters = {"en2zh": eng.pr}
    from experiments.mist_w3po_v1.core.engine import Prompter
    eng.prompters["zh2en"] = Prompter("zh2en")
    return eng


# ------------------------------------------------------------------------------------------------------ decoding
def decode(eng, d: str, sents: list[dict], wavs: dict, mode: str, batch: int, extra: dict | None = None) -> list[dict]:
    """online_decode --policy wait --context 0 on one-window sentences (B's test flags); mode base|k3|k7|p2b"""
    from experiments.mist_w3po_v1.core.engine import EOS_ID, WAIT_ID, Parser, ReadItem
    from experiments.mist_w3po_v1.evaluation.loop_guard import guard
    eng.pr = eng.prompters[d]
    eng.mode = mode if mode in ("k3", "k7") else "base"
    tok = eng.tok
    queue = list(sents)
    active, res = [], []
    while queue or active:
        while queue and len(active) < batch:
            x = queue.pop(0)
            post = np.load(REPO / x["post"]).astype(np.float32)
            wv = wavs[x["id"]]
            st = {"x": x, "wav": wv, "post": post, "committed": [], "parser": Parser(tok), "blocks": [],
                  "dur": int(round(len(wv) / 16)), "r": 0, "log": [], "cd": {"steps": 0, "changed": 0}}
            if mode == "p2b":
                st["bg"] = extra["bg"][x["id"]]
                st["w"] = extra["w"][x["id"]]
            active.append(st)
        items, bgs = [], []
        for s in active:
            s["r"] += 1
            s["t"] = min(s["r"] * 1000, s["dur"])
            fin = s["t"] >= s["dur"]
            post, full = s["post"], False
            if fin:   # online_decode_finalpost close_detect with close_detect = dur + 1.04 s
                cut = max(0, s["t"] - LOOKAHEAD_MS, min(s["t"], s["dur"] + LOOKAHEAD_MS - LOOKAHEAD_MS))
                post = s["post"].copy()
                post[cut // 80:] = 0.0
                full = True
            items.append(ReadItem(wav=s["wav"], post=post, t_ms=s["t"], final=fin, committed=s["committed"],
                                  akey=f"{s['x']['id']}|{s['t']}", full=full))
            if mode == "p2b":
                bgs.append(ReadItem(wav=s["bg"], post=post, t_ms=s["t"], final=fin, committed=s["committed"],
                                    akey=f"{s['x']['id']}|bg|{s['t']}", full=full))
        if mode == "p2b":
            outs, cst = eng.read_cd(items, bgs, [s["w"] for s in active])
        else:
            outs = eng.read(items, max_new=300)
        fin_now = []
        for s, it, o in zip(active, items, outs):
            new = [i for i in o.ids if i not in (EOS_ID, WAIT_ID)]
            new, why = guard(new, s["committed"], 6, 12, int(25 * s["dur"] / 1000) + 50)
            s["committed"] = s["committed"] + new
            s["log"].append({"t": s["t"], "new": tok.decode(new, skip_special_tokens=False), **({"guard": why} if why else {})})
            s["blocks"] += s["parser"].feed(new, s["t"])
            if it.final:
                s["parser"].close(s["dur"] / 1000, s["t"])
                fin_now.append(s)
        if mode == "p2b":
            for s in active:   # per-read stats are batch totals: attribute them evenly (diagnostic only)
                s["cd"]["steps"] += cst["steps"] / len(active)
                s["cd"]["changed"] += cst["changed"] / len(active)
        for s in fin_now:
            active.remove(s)
            x = s["x"]
            r = {"id": x["id"], "dir": d, "ids": s["committed"],
                 "translation": ("" if d == "en2zh" else " ").join(z["text"].strip() for z in s["parser"].segments),
                 "segments": s["parser"].segments, "reads": s["r"], "read_log": s["log"]}
            if mode == "p2b":
                r["w"] = s["w"]
                r["cd_batch_share"] = {k: round(v, 2) for k, v in s["cd"].items()}
            res.append(r)
    return res


# ---------------------------------------------------------------------------------------------------- statistics
def second_moments(eng, wav_list: list[np.ndarray], bs: int = 8) -> tuple[dict, int]:
    M = {li: torch.zeros(1280, 1280, dtype=torch.float64, device=eng.dev) for li in UPPER}
    n = 0
    for i in range(0, len(wav_list), bs):
        _, n_audio, _, _, cap = eng.encode(wav_list[i: i + bs], capture=True)
        for li in UPPER:
            h = torch.cat(cap[li], 0).double()
            M[li] += h.T @ h
        n += sum(n_audio)
    return M, n


def eig_desc(S: torch.Tensor):
    ev, V = torch.linalg.eigh(S)
    return ev.flip(0).clamp_min(0), V.flip(1)


def top_energy(ev: torch.Tensor, frac: float) -> int:
    c = torch.cumsum(ev, 0) / ev.sum()
    return int(torch.searchsorted(c, torch.tensor(frac, dtype=c.dtype, device=c.device)).item()) + 1


def run_k7stats(eng, limit: int = 0) -> dict:
    rows = [json.loads(l) for l in open(DATA / "k7src.jsonl", encoding="utf-8")]
    clean = [wav(r["wav"]) for r in rows if r["kind"] == "clean"]
    noise = [wav(r["wav"]) for r in rows if r["kind"] == "noise"]
    if limit:
        clean, noise = clean[:limit], noise[:limit]
    eng.pr = eng.prompters["en2zh"]
    Ms, ns = second_moments(eng, clean)
    Mg, ng = second_moments(eng, noise)
    st = {"layers": UPPER, "n_clean_clips": len(clean), "n_noise_clips": len(noise), "n_clean_frames": ns,
          "n_noise_frames": ng, "Sigma_s": {}, "Sigma_g": {}, "Vs": {}, "k_sem": {}}
    for li in UPPER:
        Ss, Sg = (Ms[li] / ns), (Mg[li] / ng)
        ev, V = eig_desc(Ss)
        k = top_energy(ev, ENERGY)
        st["Sigma_s"][li], st["Sigma_g"][li] = Ss.float().cpu(), Sg.float().cpu()
        st["Vs"][li], st["k_sem"][li] = V[:, :k].float().cpu(), k
    log(f"k7stats: clean frames {ns}, noise frames {ng}, semantic ranks {st['k_sem']}")
    return st


def k7_projectors(eng, st: dict, gap: np.ndarray) -> tuple[dict, dict]:
    """per upper layer the [d, d] Wiener projector sum_i gamma_i q_i q_i' from the 3 s gap"""
    _, n_audio, _, _, cap = eng.encode([gap], capture=True)
    n = n_audio[0]
    d = 1280
    P, info = {}, {"n_gap": n, "r": {}, "r95": {}, "gamma_mean": {}}
    for li in UPPER:
        h = torch.cat(cap[li], 0).double()
        S = h.T @ h
        Sg = st["Sigma_g"][li].to(eng.dev).double()
        Sh = (S + NU_EXTRA * Sg) / (n + NU_EXTRA)          # nu - d - 1 = NU_EXTRA
        ev, V = eig_desc(Sh)
        k = top_energy(ev, ENERGY)
        Q = V[:, :k]
        Vs = st["Vs"][li].to(eng.dev).double()
        cosmax = (Q.T @ Vs).abs().max(1).values
        keep = cosmax < DELTA_COS
        Q, s2n = Q[:, keep], ev[:k][keep]
        s2s = ((Q.T @ st["Sigma_s"][li].to(eng.dev).double()) * Q.T).sum(1)
        g = s2n / (s2n + s2s + 1e-12)
        P[li] = ((Q * g) @ Q.T).float()
        info["r95"][li], info["r"][li] = k, int(keep.sum())
        info["gamma_mean"][li] = round(float(g.mean()), 4) if len(g) else None
        _ = d
    return P, info


def k3_delta(eng, gap: np.ndarray, noisy: np.ndarray, mu_sil: np.ndarray) -> np.ndarray:
    f = eng.pr.proc.feature_extractor([np.concatenate([gap, noisy])], sampling_rate=16000, padding=True,
                                      truncation=False, return_attention_mask=True, return_tensors="np")
    F = f["input_features"][0]
    return (F[:, :GAP_FRAMES].mean(1) - mu_sil).astype(np.float32)


# --------------------------------------------------------------------------------------------------- unit runners
def bleu_dir(d: str, hyps: list[str], refs: list[str]) -> float:
    from sacrebleu.metrics import BLEU
    return BLEU(tokenize="zh" if d == "en2zh" else "13a").corpus_score(hyps, [refs]).score


def p2a_gate(M: dict) -> dict | None:
    gate = OUT / "P2A_GATE.json"
    if gate.exists():
        return json.load(open(gate))
    for c in ("clean", "noisy"):
        for d in DIRS:
            for k in range(N_SHARD):
                if not out_path(("base", c, d, k)).exists():
                    return None
    res = {}
    for d in DIRS:
        ref = {x["id"]: x["ref"] for x in M[d]}
        b = {}
        for c in ("clean", "noisy"):
            H = {}
            for k in range(N_SHARD):
                for r in map(json.loads, open(out_path(("base", c, d, k)), encoding="utf-8")):
                    H[r["id"]] = r["translation"]
            ids = sorted(ref)
            b[c] = bleu_dir(d, [H[i] for i in ids], [ref[i] for i in ids])
        res[d] = {"clean": round(b["clean"], 2), "noisy": round(b["noisy"], 2), "loss": round(b["clean"] - b["noisy"], 2)}
    res["pass"] = res["en2zh"]["loss"] >= 1.0
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    tmp = OUT / f".P2A_GATE.{os.getpid()}.partial"
    with open(tmp, "x") as fh:
        json.dump(res, fh, indent=1)
    try:
        os.link(tmp, gate)       # atomic, never replaces an existing gate
    except FileExistsError:
        pass
    os.rename(tmp, OUT / f".P2A_GATE.{os.getpid()}.written")
    log(f"P2A gate: {json.dumps(res)}")
    return json.load(open(gate))


def ready(u, M) -> bool | None:
    """True: run; False: not yet; None: never (gate failed)"""
    a, c, d, k = u
    if a in ("k7",):
        return (OUT / "k7_stats.pt").exists()
    if a == "lin":
        return out_path(("base", "clean", d, k)).exists()
    if a == "p2b":
        g = p2a_gate(M)
        if g is None:
            return False
        return True if g["pass"] else None
    return True


def run_unit(eng, u, M, LAM, st_cache: dict, batch: int, sents: list | None = None, smoke: Path | None = None):
    a, c, d, k = u
    if sents is None:
        sents = [x for i, x in enumerate(M[d]) if i % N_SHARD == k]
    eng.pr = eng.prompters[d]
    if a in ("base", "k3", "k7", "k8", "k8h", "k8f"):
        base_w = {x["id"]: wav(x[c]) for x in sents}
        if a in ("k8", "k8h", "k8f"):
            s2a = json.load(open(DATA / "k8cal.json"))["sigma2_a"]
            W = {}
            for x in sents:
                lr = LAM[(x["id"], c)]
                vh = wav(lr["vhat"])[: len(base_w[x["id"]])]
                if a == "k8h":
                    al = 1.0
                elif a == "k8f":
                    al = 0.5
                else:
                    s2n = (1 - lr["lam"]) / lr["lam"]
                    al = s2n / (s2n + s2a)
                W[x["id"]] = (al * vh + (1 - al) * base_w[x["id"]]).astype(np.float32)
                x["_alpha"] = al
            res = decode(eng, d, sents, W, "base", batch)
            al = {x["id"]: x.pop("_alpha") for x in sents}
            for r in res:
                r["alpha"] = round(al[r["id"]], 5)
            return res
        if a == "k3":
            mu = np.load(DATA / "k3_silence.npz")["mean"]
            eng.delta = {x["id"]: k3_delta(eng, wav(x["gap"]), base_w[x["id"]], mu) for x in sents}
            res = decode(eng, d, sents, base_w, "k3", batch)
            for r in res:
                r["delta_l2"] = round(float(np.linalg.norm(eng.delta[r["id"]])), 4)
            eng.delta = {}
            return res
        if a == "k7":
            if "k7" not in st_cache:
                st_cache["k7"] = torch.load((smoke or OUT) / "k7_stats.pt", map_location="cpu")
            st = st_cache["k7"]
            infos = {}
            eng.proj = {}
            res = []
            for i in range(0, len(sents), batch):   # projectors of one batch at a time (memory)
                part = sents[i: i + batch]
                for x in part:
                    gap = wav(x["gap"]) if c == "noisy" else np.zeros(int(3.0 * SR), np.float32)
                    eng.proj[x["id"]], infos[x["id"]] = k7_projectors(eng, st, gap)
                res += decode(eng, d, part, base_w, "k7", batch)
                eng.proj = {}
            for r in res:
                inf = infos[r["id"]]
                r["k7"] = {"n_gap": inf["n_gap"], "r_mean": round(float(np.mean(list(inf["r"].values()))), 2),
                           "r95_mean": round(float(np.mean(list(inf["r95"].values()))), 2),
                           "gamma_mean": round(float(np.mean([v for v in inf["gamma_mean"].values() if v is not None]
                                                             or [0.0])), 4)}
            return res
        return decode(eng, d, sents, base_w, "base", batch)
    if a == "p2b":
        W = {x["id"]: wav(x["noisy"]) for x in sents}
        lam = {x["id"]: LAM[(x["id"], "noisy")]["lam"] for x in sents}
        extra = {"bg": {x["id"]: wav(x["bgonly"]) for x in sents},
                 "w": {i: min(W_CAP, (1 - l) / max(l, 1e-6)) for i, l in lam.items()}}
        res = decode(eng, d, sents, W, "p2b", batch, extra)
        for r in res:
            r["lam"] = lam[r["id"]]
        return res
    if a == "k3dist":
        mu = np.load(DATA / "k3_silence.npz")["mean"]
        res = []
        for x in sents:
            cl, nz = wav(x["clean"]), wav(x["noisy"])
            dl = k3_delta(eng, wav(x["gap"]), nz, mu)
            fn = lambda j, F, n: F[:, :n].sub_(torch.as_tensor(dl)[:, None]) if j == 2 else None  # noqa: E731
            a3, n3, feats, am, _ = eng.encode([cl, nz, nz], fn)
            n = n3[0]
            E0, E1, E2 = a3[:n].float(), a3[n: 2 * n].float(), a3[2 * n: 3 * n].float()
            m = int(am[0].sum())
            F0, F1, F2 = feats[0][:, :m], feats[1][:, :m], feats[2][:, :m]
            res.append({"id": x["id"], "dir": d, "delta_l2": round(float(np.linalg.norm(dl)), 4),
                        "enc_before": float((E1 - E0).norm(dim=1).mean()), "enc_after": float((E2 - E0).norm(dim=1).mean()),
                        "mel_before": float((F1 - F0).norm(dim=0).mean()), "mel_after": float((F2 - F0).norm(dim=0).mean())})
        return res
    if a == "lin":
        bp = smoke / f"base_clean_{d}.jsonl" if smoke else out_path(("base", "clean", d, k))
        B0 = {r["id"]: r for r in map(json.loads, open(bp, encoding="utf-8"))}
        keep = {x["id"] for x in M[d][:N_LIN]}
        res = []
        for x in sents:
            if x["id"] not in keep:
                continue
            res.append(lin_one(eng, x, B0[x["id"]]["ids"]))
        return res
    raise ValueError(a)


@torch.no_grad()
def lin_one(eng, x: dict, C: list[int]) -> dict:
    from experiments.mist_w3po_v1.core.engine import ReadItem
    from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import AUDIO_PAD, EOS_ID
    s, bu = wav(x["clean"]), wav(x["bgunder"])
    bp = bu * math.sqrt(float((s.astype(np.float64) ** 2).sum()) / max(float((bu.astype(np.float64) ** 2).sum()), 1e-12))
    W = [(math.sqrt(1 - g) * s + math.sqrt(g) * bp).astype(np.float32) for g in SIGMAS]
    post = np.load(REPO / x["post"]).astype(np.float32)
    dur = int(round(len(s) / 16))
    post[dur // 80:] = 0.0
    items = [ReadItem(wav=w, post=post, t_ms=dur, final=True, committed=[], akey=f"{x['id']}|s{j}", full=True)
             for j, w in enumerate(W)]
    eng.mode = "base"
    emb_dtype = eng.emb_layer.weight.dtype
    ae = eng.audio_embeds(items, emb_dtype)
    n0 = ae[0][0]
    assert all(n == n0 for n, _ in ae)
    P0 = len(eng.pr.ids(n0, "", []))
    prompt = eng.pr.ids(n0, "", C) + [EOS_ID]
    ids = torch.as_tensor([prompt] * len(W), device=eng.dev)
    emb = eng.emb_layer(ids)
    emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
    o = eng.thinker.model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
    h = o.last_hidden_state[:, P0 - 1: P0 + len(C)]
    out = {"id": x["id"], "dir": x["dir"], "T": int(h.shape[1])}
    xs = torch.tensor(SIGMAS, device=eng.dev, dtype=torch.float64)
    xc = xs - xs.mean()
    p2 = xc ** 2 - (xc ** 2).mean()
    p2 = p2 - (p2 @ xc) / (xc @ xc) * xc
    for name, fn in (("logit", lambda z: z), ("logsoftmax", lambda z: torch.log_softmax(z, -1))):
        tot = lin = quad = 0.0
        for t0 in range(0, h.shape[1], 16):
            Y = fn(eng.logits(h[:, t0: t0 + 16]).double())          # [5, t, V]
            Yc = Y - Y.mean(0, keepdim=True)
            sst = (Yc ** 2).sum(0)
            b1 = torch.einsum("s,stv->tv", xc, Yc) ** 2 / (xc @ xc)
            b2 = torch.einsum("s,stv->tv", p2, Yc) ** 2 / (p2 @ p2)
            tot += float(sst.sum())
            lin += float((sst - b1).sum())
            quad += float((sst - b1 - b2).sum())
        out[name] = {"ss_tot": tot, "ss_res_lin": lin, "ss_res_quad": quad}
    return out


# --------------------------------------------------------------------------------------------------------- worker
def claim(u, wid: int) -> bool:
    p = CLAIMS / (uname(u).replace("/", "__") + ".claim")
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"worker {wid} pid {os.getpid()} host {socket.gethostname()} {time.strftime('%F %T')}\n".encode())
    os.close(fd)
    return True


def claimed(u) -> bool:
    return (CLAIMS / (uname(u).replace("/", "__") + ".claim")).exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--max-gib", type=int, default=66)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--only", default="", help="comma list of arms (debug)")
    ap.add_argument("--limit-units", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="every arm on the first N en2zh sentences -> OUT/smoke_<pid>/")
    a = ap.parse_args()
    for p in (RUNS, CLAIMS):
        p.mkdir(parents=True, exist_ok=True)
    M, LAM = manifest(), lambdas()
    eng = build_engine(a.max_gib)
    log(f"worker {a.worker} ready on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"{torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB")
    if a.smoke:
        sm = OUT / f"smoke_{os.getpid()}"
        sm.mkdir(parents=True)
        torch.save(run_k7stats(eng, 24), sm / "k7_stats.pt")
        for arm, c in (("base", "clean"), ("base", "noisy"), ("k3", "noisy"), ("k3dist", "noisy"), ("k7", "noisy"),
                       ("k7", "clean"), ("k8", "noisy"), ("k8h", "clean"), ("k8f", "noisy"), ("p2b", "noisy"),
                       ("lin", "noisy")):
            t0 = time.time()
            res = run_unit(eng, (arm, c, "en2zh", 0), M, LAM, {}, a.batch, [dict(x) for x in M["en2zh"][: a.smoke]], sm)
            with open(sm / f"{arm}_{c}_en2zh.jsonl", "x", encoding="utf-8") as fh:
                for r in res:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"smoke {arm} {c}: {len(res)} rows in {time.time() - t0:.0f}s; peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
        return
    only = set(a.only.split(",")) if a.only else None
    U = [u for u in units() if only is None or u[0] in only]
    st_cache, n_done = {}, 0
    while True:
        todo = [u for u in U if not claimed(u)]
        if not todo:
            break
        ran = False
        for u in todo:
            r = ready(u, M)
            if r is None:
                if claim(u, a.worker):
                    log(f"skip {uname(u)}: P2a gate failed (en2zh noise loss < 1.0 BLEU)")
                continue
            if not r or not claim(u, a.worker):
                continue
            op = out_path(u)
            if op.exists():
                raise SystemExit(f"拒绝覆盖已有输出: {op}")
            t0 = time.time()
            log(f"start {uname(u)}")
            op.parent.mkdir(parents=True, exist_ok=True)
            tmp = op.with_name(op.name + ".partial")
            if tmp.exists():
                raise SystemExit(f"拒绝覆盖已有输出: {tmp}")
            try:
                if u[0] == "k7stats":
                    torch.save(run_k7stats(eng), tmp)
                else:
                    res = run_unit(eng, u, M, LAM, st_cache, a.batch)
                    with open(tmp, "x", encoding="utf-8") as fh:
                        for r in res:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception:   # noqa: BLE001  the unit stays claimed (no output); FAILED marker for the record
                import traceback
                tb = traceback.format_exc()
                log(f"FAILED {uname(u)}:\n{tb}")
                with open(CLAIMS / (uname(u).replace("/", "__") + f".FAILED.{os.getpid()}"), "x") as fh:
                    fh.write(tb)
                torch.cuda.empty_cache()
                continue
            os.rename(tmp, op)
            n_done += 1
            log(f"done {uname(u)} in {time.time() - t0:.0f}s")
            ran = True
            if a.limit_units and n_done >= a.limit_units:
                return
            break
        if not ran:
            time.sleep(15)
    log(f"worker {a.worker}: queue empty after {n_done} units")


if __name__ == "__main__":
    main()
