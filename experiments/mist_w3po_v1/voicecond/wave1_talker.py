"""Robust wave 1, K1 / K4 / K7v: the T1c Talker under HF transformers (bf16), teacher-forced or generating.

lin   P1a: per item, the 5 sigma-mix voice-token sets (prep tokens s000..s100) teacher-forced on the item's cached T1c(g*)
      sequence (cvt teacher_out train_ids; voice tokens at positions 4..35 replaced). Per code position (positions
      lf-1 .. lf+nc-2, the ones that predict a semantic code) and per semantic code (8192): the fp32 logit row
      l = h W_sem^T. Primary: the log-softmax over the 8192 semantic codes (the distribution the Talker samples from;
      per-step constants removed). Per (position, code) a line in sigma with intercept: R^2 = 1 - SS_res / SS_tot pooled
      over all positions, codes and items; quadratic share = SS of the orthogonal quadratic term / SS_tot.
      Supplements: the same on the raw semantic logits, and weighted by the mean probability over sigma.
feats P4a: the mid-layer (14 of 28: output of decoder layer 13) hidden state averaged over the code positions of every
      cvt train / dev teacher sequence (g*), with the sequence's rate = target text tokens / (n_codes * 0.02 s).
gen   P1b / P4b / K7v: sampling exactly as si_infer's T1c talk job for a single final block (--speed-policy fixed
      --history-ban 8): SIGuard (make_si_guard: expected codes, final, ngram 8, min_frac 0.3, penalty 1.3, empty
      history), temperature 0.7, top-p 0.8 (vLLM rule), stop at <|end_semantic|> / <|wait_read|>, max 1500 tokens.
      Arms: B (raw noisy tokens), C, K7v (positive tokens only); ZNE / CFG1: two branches in one batch, the positive
      (C tokens) and the negative (neg = C tokens of the stream background), both fed the sampled tokens; raw logits
      l = l+ + w (l+ - l-) with w = item w_zne (ZNE) or 1 (CFG1), then the guard and the sampler. The BiCodec decoder
      always gets the positive (C) tokens. Common random numbers: the sampler's uniforms come from a generator seeded by
      (item id, seed) only, so all arms of an item / seed share them.
      --probe / --rhat (P4b): at the mid layer, on every position that predicts a code (last prompt position and each
      generated one), h <- h - (r'h - rhat) r in both branches; r = the P4a probe direction, rhat per item.
      Each job also records the mean (pre-clamp) mid hidden state over its code positions (hmid.npy).

    <venvs/vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.wave1_talker lin --items <prep>/items.jsonl --out <dir>
    ... feats --out <dir> [--sets zh2en_train,en2zh_train,zh2en_dev,en2zh_dev] [--shard k --num-shards n]
    ... gen --items <prep>/items.jsonl --arm B|C|ZNE|CFG1|K7v --seeds 20261021 20261109 --out <dir>
            [--k7v <tokens.jsonl>] [--probe <probe.npz> --rhat <rhat.json>]
Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
T1C = REPO / "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
CV = REPO / "data/open_voicecond_v1/idea2/cvt"
GLOBAL_0, SEMANTIC_0, SEMANTIC_LAST = 151665, 155761, 163952
START_GLOBAL, END_GLOBAL, START_SEMANTIC, END_SEMANTIC, WAIT_READ = 180387, 180391, 180388, 180392, 180395
START_CONTENT, END_CONTENT = 180386, 180390
SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SKEYS = ("s000", "s025", "s050", "s075", "s100")
MID = 14
TEMP, TOP_P, MAX_TOKENS = 0.7, 0.8, 1500


def log(s):
    print(f"{time.strftime('%F %T')} {s}", flush=True)


def load_talker(dev):
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.enable_cudnn_sdp(False)
    m = AutoModelForCausalLM.from_pretrained(str(T1C), dtype=torch.bfloat16).to(dev).eval()
    assert m.config.num_hidden_layers == 2 * MID
    return m


def with_voice(ids, voice):
    assert ids[3] == START_GLOBAL and ids[36] == END_GLOBAL and len(voice) == 32
    return ids[:4] + [GLOBAL_0 + int(v) for v in voice] + ids[36:]


class MidHook:
    """forward hook on decoder layer MID-1: records the last position's hidden state (pre-clamp) and optionally clamps
    its projection on r to rhat (per row)"""

    def __init__(self, layer):
        self.clamp = None
        self.rec = None
        layer.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        last = hs[:, -1, :]
        self.rec = last.float().clone()
        if self.clamp is not None:
            r, rhat = self.clamp
            x = last.float()
            hs[:, -1, :] = (x - (x @ r - rhat)[:, None] * r[None]).to(hs.dtype)
        return out


# ------------------------------------------------------------------------------------------------ lin (P1a)
def cmd_lin(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    dev = torch.device("cuda:0")
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    model = load_talker(dev)
    body = model.model
    Wsem = model.lm_head.weight[SEMANTIC_0:SEMANTIC_0 + 8192].float()
    x = torch.tensor(SIGMAS, dtype=torch.float64, device=dev) - 0.5
    sxx = float((x ** 2).sum())
    p2 = x ** 2 - float((x ** 2).mean())
    spp = float((p2 ** 2).sum())
    keys = ("tot", "lin", "quad", "wtot", "wlin", "wquad", "rtot", "rlin", "rquad")
    per = []
    t0 = time.time()
    for n, it in enumerate(items):
        seqs = [with_voice(it["ids"], it["tok"][k]) for k in SKEYS]
        ids = torch.tensor(seqs, device=dev)
        lf, nc = it["lf"], it["nc"]
        pos = torch.arange(lf - 1, lf - 1 + nc, device=dev)
        with torch.no_grad():
            hs = body(input_ids=ids).last_hidden_state[:, pos].float()     # (5, nc, 1536)
            raw = (hs @ Wsem.T).double()                                    # (5, nc, 8192)
        acc = {}
        for tag, y in (("", torch.log_softmax(raw, -1)), ("r", raw)):
            yc = y - y.mean(0, keepdim=True)
            tot = (yc ** 2).sum(0)
            lin = (torch.einsum("s,snk->nk", x, y) ** 2) / sxx
            quad = (torch.einsum("s,snk->nk", p2, y) ** 2) / spp
            acc[tag + "tot"], acc[tag + "lin"], acc[tag + "quad"] = float(tot.sum()), float(lin.sum()), float(quad.sum())
            if tag == "":
                wgt = torch.exp(y).mean(0)
                acc["wtot"], acc["wlin"], acc["wquad"] = (float((wgt * tot).sum()), float((wgt * lin).sum()),
                                                         float((wgt * quad).sum()))
        # how different the five conditions are: mean KL(sigma=0 || sigma) per position
        lp = torch.log_softmax(raw, -1)
        kl = [float((lp[0].exp() * (lp[0] - lp[k])).sum(-1).mean()) for k in range(5)]
        per.append({"id": it["id"], "dir": it["dir"], "nc": nc, **acc, "kl0": kl,
                    "ham0": [sum(u != v for u, v in zip(it["tok"]["s000"], it["tok"][k])) for k in SKEYS]})
        if n % 20 == 0:
            log(f"[lin] {n + 1}/{len(items)} R2 so far {1 - (sum(p['tot'] - p['lin'] for p in per) / sum(p['tot'] for p in per)):.4f}")

    def pooled(rows, pre=""):
        T = sum(r[pre + "tot"] for r in rows)
        L = sum(r[pre + "lin"] for r in rows)
        Q = sum(r[pre + "quad"] for r in rows)
        return {"R2_lin": L / T, "quad_share": Q / T, "R2_lin_plus_quad": (L + Q) / T}

    def boot(rows, pre="", n=2000):
        rng = np.random.default_rng(0)
        T = np.array([r[pre + "tot"] for r in rows])
        L = np.array([r[pre + "lin"] for r in rows])
        v = []
        for _ in range(n):
            i = rng.integers(0, len(rows), len(rows))
            v.append(L[i].sum() / T[i].sum())
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    res = {"n_items": len(per), "n_positions": int(sum(p["nc"] for p in per)),
           "primary_logsoftmax_sem": {**pooled(per), "R2_ci95_items": boot(per)},
           "weighted_by_mean_prob": pooled(per, "w"), "raw_sem_logits": pooled(per, "r"),
           "per_dir": {d: {**pooled([p for p in per if p["dir"] == d]),
                           "weighted": pooled([p for p in per if p["dir"] == d], "w")} for d in ("zh2en", "en2zh")},
           "per_item_R2_median": float(np.median([p["lin"] / p["tot"] for p in per])),
           "mean_KL_sigma0_vs": [float(np.mean([p["kl0"][k] for p in per])) for k in range(5)],
           "mean_ham_sigma0_vs": [float(np.mean([p["ham0"][k] for p in per])) for k in range(5)],
           "gate_R2_ge_0.8": None, "sec": round(time.time() - t0, 1)}
    res["gate_R2_ge_0.8"] = bool(res["primary_logsoftmax_sem"]["R2_lin"] >= 0.8)
    out.mkdir(parents=True)
    with open(out / "per_item.jsonl", "x") as fh:
        for p in per:
            fh.write(json.dumps(p) + "\n")
    (out / "lin.json").write_text(json.dumps(res, indent=1))
    log(json.dumps(res))


# ------------------------------------------------------------------------------------------------ feats (P4a)
def cmd_feats(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    dev = torch.device("cuda:0")
    recs = []
    for s in a.sets.split(","):
        for p in sorted(glob.glob(str(CV / "teacher_out" / s / "part_*.jsonl"))):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                assert r["error"] is None and len(r["blocks"]) == 1
                b = r["blocks"][0]
                ids = [int(v) for v in b["train_ids"]]
                lf, nc = int(b["label_from"]), len(b["code_ids"])
                n_text = ids.index(END_CONTENT) - ids.index(START_CONTENT) - 1
                recs.append({"id": r["id"], "set": s, "ids": ids, "lf": lf, "nc": nc, "n_text": n_text})
    recs = recs[a.shard::a.num_shards]
    model = load_talker(dev)
    body = model.model
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]["ids"]))
    F = np.zeros((len(recs), model.config.hidden_size), np.float32)
    t0 = time.time()
    for b0 in range(0, len(order), a.bs):
        bi = order[b0:b0 + a.bs]
        Lm = max(len(recs[i]["ids"]) for i in bi)
        ids = torch.zeros(len(bi), Lm, dtype=torch.long)
        mask = torch.zeros(len(bi), Lm, dtype=torch.long)
        for n, i in enumerate(bi):
            ids[n, :len(recs[i]["ids"])] = torch.tensor(recs[i]["ids"])
            mask[n, :len(recs[i]["ids"])] = 1
        with torch.no_grad():
            o = body(input_ids=ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True)
            H = o.hidden_states[MID]
            for n, i in enumerate(bi):
                lf, nc = recs[i]["lf"], recs[i]["nc"]
                F[i] = H[n, lf - 1:lf - 1 + nc].float().mean(0).cpu().numpy()
        if (b0 // a.bs) % 20 == 0:
            log(f"[feats] {b0 + len(bi)}/{len(recs)} {time.time() - t0:.0f}s")
    out.mkdir(parents=True)
    np.savez(out / "feats.npz", id=np.array([r["id"] for r in recs]), set=np.array([r["set"] for r in recs]),
             feat=F, nc=np.array([r["nc"] for r in recs]), n_text=np.array([r["n_text"] for r in recs]))
    log(f"[feats] done {len(recs)} in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ gen
def top_p_sample(logits: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """temperature + vLLM top-p rule + inverse-CDF draw with given uniforms; logits (J, V) fp32"""
    probs = torch.softmax(logits / TEMP, -1)
    ps, idx = probs.sort(-1, descending=False)
    cs = ps.cumsum(-1)
    drop = cs <= (1 - TOP_P)
    drop[:, -1] = False
    ps = ps.masked_fill(drop, 0.0)
    ps = ps / ps.sum(-1, keepdim=True)
    ps, idx = ps.flip(-1), idx.flip(-1)                  # descending: deterministic order for the draw
    c = ps.cumsum(-1)
    k = torch.searchsorted(c, u[:, None].to(c.dtype) * c[:, -1:]).clamp(max=ps.shape[-1] - 1)
    return idx.gather(-1, k)[:, 0]


def cmd_gen(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import soundfile as sf
    from experiments.uniss_qwen3_omni_v1.evaluation.si_guard import make_si_guard
    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import expected_codes
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    dev = torch.device("cuda:0")
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    items = [it for it in items if it["p1b"]]
    if a.limit:
        items = items[:a.limit]
    k7v = {}
    if a.arm == "K7v":
        k7v = {r["id"]: r["ids"] for r in map(json.loads, open(a.k7v, encoding="utf-8"))}
    probe = None
    if a.probe:
        z = np.load(a.probe)
        rh = json.load(open(a.rhat))
        probe = (torch.tensor(z["r"], dtype=torch.float32, device=dev), rh)
    jobs = []
    for seed in a.seeds:
        for it in items:
            pos = {"B": it["tok"]["B"], "C": it["tok"]["C"], "ZNE": it["tok"]["C"], "CFG1": it["tok"]["C"],
                   "K7v": k7v.get(it["id"])}[a.arm]
            neg = it["tok"]["neg"] if a.arm in ("ZNE", "CFG1") else None
            w = it["w_zne"] if a.arm == "ZNE" else 1.0
            jobs.append({"key": f"{a.arm}__s{seed}__{it['id']}", "id": it["id"], "seed": seed, "pos": pos, "neg": neg,
                         "w": w, "prompt": it["ids"][:it["lf"]], "expected": expected_codes(it["text"], it["tgt"]),
                         "rhat": (probe[1][it["id"]] if probe else None)})
    model = load_talker(dev)
    hook = MidHook(model.model.layers[MID - 1])
    P = BiCodecPhi(str(dev))
    out.mkdir(parents=True)
    (out / "wav").mkdir()
    (out / "cfg.json").write_text(json.dumps({**vars(a), "n_jobs": len(jobs)}, indent=1))
    fh = open(out / "rows.jsonl", "x", encoding="utf-8")
    hmid = {}
    jobs.sort(key=lambda j: len(j["prompt"]))
    t0 = time.time()
    nrow = 2 if a.arm in ("ZNE", "CFG1") else 1
    per = max(1, a.rows // nrow)
    for b0 in range(0, len(jobs), per):
        J = jobs[b0:b0 + per]
        nj = len(J)
        seqs = [with_voice(j["prompt"], j["pos"]) for j in J] + ([with_voice(j["prompt"], j["neg"]) for j in J] if nrow == 2 else [])
        R = len(seqs)
        Lm = max(len(s) for s in seqs)
        ids = torch.zeros(R, Lm, dtype=torch.long)
        mask = torch.zeros(R, Lm, dtype=torch.long)
        for n, s in enumerate(seqs):                       # left padding
            ids[n, Lm - len(s):] = torch.tensor(s)
            mask[n, Lm - len(s):] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        posid = (mask.cumsum(1) - 1).clamp(min=0)
        guards = [make_si_guard(expected=j["expected"], final=True, history=[], hist_n=8, ngram=8, min_frac=0.3) for j in J]
        gens = [torch.Generator().manual_seed(int(hashlib.sha1(f"{j['id']}|{j['seed']}".encode()).hexdigest(), 16) % (2 ** 63))
                for j in J]
        wvec = torch.tensor([j["w"] for j in J], dtype=torch.float32, device=dev)[:, None]
        if probe is not None:
            rh = torch.tensor([j["rhat"] for j in J] * nrow, dtype=torch.float32, device=dev)
            hook.clamp = (probe[0], rh)
        else:
            hook.clamp = None
        outs = [[] for _ in J]
        done = [False] * nj
        hsum = torch.zeros(nj, model.config.hidden_size, dtype=torch.float64, device=dev)
        hcnt = torch.zeros(nj, dtype=torch.float64, device=dev)
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=mask, position_ids=posid, use_cache=True, logits_to_keep=1)
            for step in range(MAX_TOKENS):
                lg = o.logits[:, -1].float()
                hm = hook.rec[:nj].double()
                l = lg[:nj] + wvec * (lg[:nj] - lg[nj:]) if nrow == 2 else lg
                l = l.clone()
                for k in range(nj):
                    if not done[k]:
                        l[k] = guards[k](outs[k], l[k])
                u = torch.tensor([float(torch.rand(1, generator=g)) for g in gens], device=dev)
                tok = top_p_sample(l, u).tolist()
                for k in range(nj):
                    if done[k]:
                        continue
                    t = int(tok[k])
                    if SEMANTIC_0 <= t <= SEMANTIC_LAST:
                        hsum[k] += hm[k]
                        hcnt[k] += 1
                    outs[k].append(t)
                    if t in (END_SEMANTIC, WAIT_READ) or len(outs[k]) >= MAX_TOKENS:
                        done[k] = True
                if all(done):
                    break
                nxt = torch.tensor([outs[k][-1] for k in range(nj)] * nrow, device=dev)[:, None]
                mask = torch.cat([mask, torch.ones(R, 1, dtype=mask.dtype, device=dev)], 1)
                posid = posid[:, -1:] + 1
                o = model(input_ids=nxt, attention_mask=mask, past_key_values=o.past_key_values, position_ids=posid,
                          use_cache=True)
        for k, j in enumerate(J):
            out_ids = outs[k]
            codes = [t - SEMANTIC_0 for t in out_ids if SEMANTIC_0 <= t <= SEMANTIC_LAST]
            exp = j["expected"]
            row = {"key": j["key"], "id": j["id"], "arm": a.arm, "seed": j["seed"], "w": j["w"], "n_codes": len(codes),
                   "closing": out_ids[-1] if out_ids else None, "forced": len(codes) >= max(int(2.2 * exp), exp + 100),
                   "expected": exp, "rhat": j["rhat"], "clamp": probe is not None}
            if codes:
                wav = np.asarray(P.tok.decode_tokens_to_audio(torch.tensor([*j["pos"], *codes], dtype=torch.long)),
                                 dtype=np.float32).reshape(-1)
                wp = out / "wav" / f"{j['key']}.wav"
                sf.write(str(wp), wav, 16000)
                row.update({"wav": str(wp), "dur_s": round(len(wav) / 16000, 3)})
            else:
                row["error"] = "no codes"
            if hcnt[k] > 0:
                hmid[j["key"]] = (hsum[k] / hcnt[k]).float().cpu().numpy()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        log(f"[gen {a.arm}] {min(b0 + per, len(jobs))}/{len(jobs)} jobs, {time.time() - t0:.0f}s")
    fh.close()
    ks = sorted(hmid)
    np.savez(out / "hmid.npz", key=np.array(ks), h=np.stack([hmid[k] for k in ks]) if ks else np.zeros((0, 1536)))
    (out / "DONE").write_text(f"{len(jobs)} jobs {time.time() - t0:.0f}s\n")
    log(f"[gen {a.arm}] done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lin")
    s.add_argument("--items", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("feats")
    s.add_argument("--out", required=True)
    s.add_argument("--sets", default="zh2en_train,en2zh_train,zh2en_dev,en2zh_dev")
    s.add_argument("--bs", type=int, default=32)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    s = sub.add_parser("gen")
    s.add_argument("--items", required=True)
    s.add_argument("--arm", required=True, choices=["B", "C", "ZNE", "CFG1", "K7v"])
    s.add_argument("--seeds", type=int, nargs="+", default=[20261021, 20261109])
    s.add_argument("--out", required=True)
    s.add_argument("--k7v", default="")
    s.add_argument("--probe", default="")
    s.add_argument("--rhat", default="")
    s.add_argument("--rows", type=int, default=96, help="batch rows (a two-branch job takes two)")
    s.add_argument("--limit", type=int, default=0, help="smoke only: first n items")
    a = ap.parse_args()
    {"lin": cmd_lin, "feats": cmd_feats, "gen": cmd_gen}[a.cmd](a)


if __name__ == "__main__":
    main()
