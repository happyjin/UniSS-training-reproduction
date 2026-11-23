"""sst_hsmm_v1 P5 (prereg change 8) playout proxy on DECODED blocks: replays hsmm.lattice_risk_block's state dynamics on
a system's actual block write times -> per item latency L (s, LAAL form: mean_i [min(t_i + s'_i, X) - i X / n]) and pause
count P (Talker per-block overhead OB bins: the first token of a block starts its speech at max(s, OB); it pauses iff
started, previous token not punctuation and s <= -0.3 + OB). Tokens = tokenizer of each block's text (no specials).
Used for (i) calibrating proxy vs measured D2 / NP_in on dev systems (report), (ii) the per-item budgets from the P1a
rollouts of the training items (train_risk2).
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.proxy --report
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOKBIN = {"en2zh": 3, "zh2en": 3}            # round(345.6 / 100), round(332.5 / 100)
OB = 2                                        # Talker per-block overhead 215.5 / 235.3 ms -> 2 bins (0.2 s)
SMIN, SMAX, PAUSE = -30, 80, -3               # bins of 0.1 s: s in [-3, 8]; pause iff s <= -0.3 (+ OB for a block start)
PUNCT = re.compile(r"^[\s，。,.!?！？；;：:、…\"“”'‘’()（）《》\-—]+$")


def tok_fn():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(REPO / "eval_outputs/sst_hsmm_v1/p4/merged"))
    return tk


def item_proxy(blocks, X_ms, d, tk, ob=OB):
    X = X_ms / 1000.0
    C = max(1, math.ceil(X_ms / 1000 - 1e-9))
    s, f, started, c_prev = 0, 0, 0, 1
    lat, P, i = [], 0, 0
    toks = []
    for b in blocks:
        c = C if b.get("final") else min(int(round(b["write_ms"] / 1000)), C)
        t = X if c == C else float(c)
        s = max(s - 10 * (c - c_prev), SMIN)
        c_prev = c
        for k, tid in enumerate(tk(b["text"], add_special_tokens=False)["input_ids"]):
            p = bool(PUNCT.match(tk.decode([tid]) or "x"))
            first = k == 0
            if started and f == 0 and s <= PAUSE + (ob if first else 0):
                P += 1
            s = min(max(s, ob if first else 0) + (0 if p else TOKBIN[d]), SMAX)
            lat.append(min(t + s / 10.0, X))
            f, started = int(p), 1
            toks.append(tid)
    n = len(lat)
    L = sum(l - j * X / n for j, l in enumerate(lat)) / n if n else 0.0
    return L, P, n


def system(path, d, tk, X_of):
    out = {}
    for l in open(path, encoding="utf-8"):
        x = json.loads(l)
        out[x["id"]] = item_proxy(x["blocks"], X_of(x), d, tk)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="more roots under eval_outputs/sst_hsmm_v1")
    ap.add_argument("--out")
    a = ap.parse_args()
    tk = tok_fn()
    data = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
    dur = {}
    for d in ("en2zh", "zh2en"):
        for l in open(data / f"items_{d}.jsonl", encoding="utf-8"):
            x = json.loads(l)
            dur[x["id"]] = x["dur_ms"]
    roots = {"ctrl": REPO / "eval_outputs/mist_w3po_v1/wave17/eval_enc_wise05"}
    for r in ["p2", "p2b", "p2c", *a.extra]:
        roots[r] = REPO / "eval_outputs/sst_hsmm_v1" / r
    res = {}
    for name, root in roots.items():
        for d in ("en2zh", "zh2en"):
            f = root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"
            if not f.exists():
                continue
            v = system(f, d, tk, lambda x: dur[x["id"]])
            ids = sorted(v)
            m = {"L": sum(v[k][0] for k in ids) / len(ids), "P": sum(v[k][1] for k in ids) / len(ids), "n": len(ids)}
            if name != "ctrl":
                s = json.load(open(root / "summary.json"))["pooled"][d] if (root / "summary.json").exists() else None
                if s:
                    m.update(d2_diff=round(s["d2"]["diff"], 4), np_in_rel=round(s["NP_in"]["rel"], 4),
                             np_in=round(s["NP_in"]["arm"], 4))
            else:
                s = json.load(open(REPO / "eval_outputs/sst_hsmm_v1/p2/summary.json"))["pooled"][d]
                m.update(np_in=round(s["NP_in"]["ref"], 4))
            res[f"{name}/{d}"] = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in m.items()}
            print(name, d, res[f"{name}/{d}"], flush=True)
    if a.out:
        with open(a.out, "x", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
