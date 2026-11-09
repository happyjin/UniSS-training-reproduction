"""Diagnose why TurnCRF (CPU stage) lost <=1 s turn recall. Read-only on existing outputs; writes one JSON to argv[1]."""
import json
import multiprocessing as mp
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, ".")
from experiments.mist_w3po_v1.crf import eval_turn_crf as E  # noqa: E402
from experiments.mist_w3po_v1.crf.turn_crf import build_model  # noqa: E402

SHORT = E.SHORT_FRAMES
G = {}


def init():
    torch.set_num_threads(1)
    for kind in ("semi", "chain"):
        ck = torch.load(f"eval_outputs/mist_w3po_v1/crf/turn_v1/{kind}/params.pt", weights_only=False)
        m = build_model(ck["kind"], **ck["model_kw"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        G[kind] = m


def soft(m, P, tau, T):
    _, mu = m.marginals(torch.from_numpy(P[:tau].astype(np.float64)), censored=tau < T)
    return m.slot_marginals(mu).numpy()


def match(rr, hh, thr):
    pairs = []
    for i, (a, b) in enumerate(rr):
        for j, (c, d) in enumerate(hh):
            inter = min(b, d) - max(a, c)
            if inter > 0:
                iou = inter / (max(b, d) - min(a, c))
                if iou >= thr:
                    pairs.append((-iou, i, j))
    ur, uh = set(), set()
    for _, i, j in sorted(pairs):
        if i not in ur and j not in uh:
            ur.add(i)
            uh.add(j)
    return ur, uh


def one(args):
    P, R, dur = args
    T = len(P)
    taus = E.read_prefixes(dur, T)
    Pf = P.astype(np.float64)
    B = Pf >= 0.5
    hyp, sft = {}, {}
    hyp["raw|offline"] = hyp["raw|streaming"] = B
    sft["raw|offline"] = sft["raw|streaming"] = Pf
    hyp["med5|offline"] = E.med5(B)
    hyp["med5|streaming"] = E.stream(lambda t: E.med5(B[:t]), T, taus)
    for k in ("chain", "semi"):
        m = G[k]
        off = soft(m, P, T, T)
        st = np.zeros((T, 8))
        prev = 0
        for t in taus:
            st[prev:t] = soft(m, P, t, T)[prev:t]
            prev = t
        sft[f"{k}|offline"], sft[f"{k}|streaming"] = off, st
        hyp[f"{k}|offline"], hyp[f"{k}|streaming"] = off >= 0.5, st >= 0.5
    c = defaultdict(float)
    others = R.sum(1)
    for key, H in hyp.items():
        for s in range(8):
            rr, hh = E.runs(R[:, s]), E.runs(H[:, s])
            m50 = match(rr, hh, 0.5)
            m20 = match(rr, hh, 0.2)
            m0 = match(rr, hh, 1e-9)
            for j, (a, b) in enumerate(hh):
                if b - a <= SHORT:
                    c[key + "|hyp_short"] += 1
                    c[key + "|hyp_short_ok50"] += j in m50[1]
                    c[key + "|hyp_short_noref"] += j not in m0[1]
            for i, (a, b) in enumerate(rr):
                if b - a > SHORT:
                    continue
                ovl = (others[a:b] >= 2).mean() >= 0.5
                g = "ov" if ovl else "iso"
                c[key + "|ref_short"] += 1
                c[key + f"|ref_short_{g}"] += 1
                c[key + "|hit50"] += i in m50[0]
                c[key + f"|hit50_{g}"] += i in m50[0]
                c[key + "|hit20"] += i in m20[0]
                c[key + "|hit_any"] += i in m0[0]
                if key in sft:
                    mx = sft[key][a:b, s].max()
                    c[key + "|soft_ge05"] += mx >= 0.5
                    c[key + "|soft_ge03"] += mx >= 0.3
                    c[key + "|soft_ge01"] += mx >= 0.1
                    c[key + "|soft_mean_sum"] += sft[key][a:b, s].mean()
    return dict(c)


def main():
    d = np.load("eval_outputs/mist_w3po_v1/crf/turn_refs_v1/dev.npz", allow_pickle=True)
    dur = {}
    for line in open("eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"):
        w = json.loads(line)
        dur[w["id"]] = w["dur_ms"]
    o = d["offs"]
    jobs = [(d["P"][o[i]:o[i + 1]], d["R"][o[i]:o[i + 1]], dur[d["ids"][i]]) for i in range(len(d["ids"]))]
    tot = defaultdict(float)
    with mp.Pool(96, initializer=init) as pool:
        for c in pool.imap_unordered(one, jobs, chunksize=8):
            for k, v in c.items():
                tot[k] += v
    out = {}
    for sysmode in sorted({k.rsplit("|", 1)[0] for k in tot}):
        g = lambda n: tot.get(f"{sysmode}|{n}", 0.0)
        rs = max(1, g("ref_short"))
        row = {"ref_short": g("ref_short"), "share_ov": g("ref_short_ov") / rs,
               "recall50": g("hit50") / rs, "recall20": g("hit20") / rs, "recall_any": g("hit_any") / rs,
               "recall50_iso": g("hit50_iso") / max(1, g("ref_short_iso")),
               "recall50_ov": g("hit50_ov") / max(1, g("ref_short_ov")),
               "hyp_short": g("hyp_short"), "short_prec50": g("hyp_short_ok50") / max(1, g("hyp_short")),
               "hyp_short_noref_share": g("hyp_short_noref") / max(1, g("hyp_short"))}
        if g("soft_mean_sum"):
            row.update({"soft_max_ge05": g("soft_ge05") / rs, "soft_max_ge03": g("soft_ge03") / rs,
                        "soft_max_ge01": g("soft_ge01") / rs, "soft_mean": g("soft_mean_sum") / rs})
        out[sysmode] = {k: round(v, 4) for k, v in row.items()}
    tr = np.load("eval_outputs/mist_w3po_v1/crf/turn_refs_v1/train.npz", allow_pickle=True)
    R, to = tr["R"], tr["offs"]
    nt = ns = fs = ft = 0
    for i in range(len(to) - 1):
        Rw = R[to[i]:to[i + 1]]
        for s in range(8):
            for a, b in E.runs(Rw[:, s]):
                nt += 1
                ft += b - a
                if b - a <= SHORT:
                    ns += 1
                    fs += b - a
    out["train_ref"] = {"turns": nt, "short_turn_share": round(ns / nt, 4), "short_frame_share": round(fs / ft, 4)}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
