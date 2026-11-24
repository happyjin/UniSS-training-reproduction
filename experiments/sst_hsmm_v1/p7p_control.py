"""sst_hsmm_v1 P7' (prereg change 15): p6b_control with lam_hi 10 (lam0 = lamP unchanged = sqrt(0.003 * 1)) and 8 dev points (+ p2h, p2i). Was: sst_hsmm_v1 P6b (prereg change 13): as p6_control.py but with the SPEECH lag (playback queue) and p2g as a sixth dev point:
block (write w, n non-punct tokens) plays from max(w + OB_MS, previous end) for n * TOKMS ms; Wp(t) = tokens played by t (linear
within a block); lag = t - Wp(t) / rho at reads t = 1, 2, ... s with t > L* only... (fit uses all reads t >= 1 s). Controller
updates only when t / 1000 > L* (before that a lag below the setpoint is unavoidable). Was: sst_hsmm_v1 P6 (prereg change 11): constants of the inference-time online controller -> <out> (control.json).
rho[dir] = non-punctuation reference tokens per source second over the P1 training rows (data/train_<dir>.jsonl);
lag of a system = mean causal lag t - W(t) / rho over all non-final reads of its dev blocks (t = 1, 2, ... s < duration;
W(t) = non-punctuation tokens of the blocks written at <= t; Qwen tokenizer, e2e PUNCT). L*[dir] = the lag at which the
OLS line D2diff = a + b * lag over the HSMM dev systems (p2, p2b, p2c, p2e, p2f) gives D2diff = -0.10 s (the change-8
target); stops if b <= 0. N's own lag (wave17/eval_enc_wise05) is reported.
eta 0.5 per read per second of lag error, lam0 = lamP = geometric middle of [LAM_LO, LAM_HI]. Also prints the same lag
for earlier HSMM dev systems (report only). Refuses to overwrite.
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.p6_control --out eval_outputs/sst_hsmm_v1/p6/control.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from experiments.sst_hsmm_v1 import hsmm as H

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
DEV = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
CTRL = REPO / "eval_outputs/mist_w3po_v1/wave17/eval_enc_wise05"
PUNCT = re.compile(r"^[\s，。,.!?！？；;：:、…\"“”'‘’()（）《》\-—]+$")
ETA = 0.5
OB_MS = 200.0
TOKMS = {"en2zh": 345.6, "zh2en": 332.5}


def mean_lag(blocks_file, dur, rho, tk, punct, tokms):
    lags = []
    for l in open(blocks_file, encoding="utf-8"):
        x = json.loads(l)
        D = dur[x["id"]]
        wt = [(b["write_ms"], sum(1 for i in tk(b["text"], add_special_tokens=False)["input_ids"] if i not in punct))
              for b in x["blocks"]]
        seg, end = [], 0.0
        for w, n in sorted(wt):
            st_ = max(w + OB_MS, end); end = st_ + n * tokms
            seg.append((st_, end, n))
        t = 1000
        while t < D:
            W = sum(n * min(max((t - a0) / max(b0 - a0, 1e-9), 0.0), 1.0) for a0, b0, n in seg if n > 0)
            lags.append(t / 1000.0 - W / rho)
            t += 1000
    return sum(lags) / len(lags), len(lags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(REPO / "eval_outputs/sst_hsmm_v1/p4/merged"))
    punct = {i for i in range(len(tk)) if PUNCT.match(tk.decode([i]) or "x")}
    lam_mid = math.sqrt(H.LAM_LO * H.LAM_HI)
    res = {"lag": "speech", "ob_ms": OB_MS, "tokms": TOKMS, "eta": ETA, "lam0": lam_mid, "lamP": lam_mid, "lam_lo": H.LAM_LO, "lam_hi": 10.0}
    for d in ("en2zh", "zh2en"):
        ntok = sec = 0.0
        for l in open(DATA / f"train_{d}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            ntok += len(r["tok"]) - sum(r["punct"])
            sec += r["X"] / 1000.0
        rho = ntok / sec
        dur = {json.loads(l)["id"]: json.loads(l)["dur_ms"] for l in open(DEV / f"items_{d}.jsonl", encoding="utf-8")}
        LN, n = mean_lag(CTRL / f"thinker_{d}" / f"blocks_{d}_w1.jsonl", dur, rho, tk, punct, TOKMS[d])
        pts = []
        for s in ("p2", "p2b", "p2c", "p2e", "p2f", "p2g", "p2h", "p2i"):
            f = REPO / "eval_outputs/sst_hsmm_v1" / s / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"
            d2 = json.load(open(REPO / "eval_outputs/sst_hsmm_v1" / s / "summary.json"))["pooled"][d]["d2"]["diff"]
            pts.append((mean_lag(f, dur, rho, tk, punct, TOKMS[d])[0], d2, s))
        mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
        b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
        a0 = my - b * mx
        if b <= 0:
            raise SystemExit(f"slope <= 0 ({d})")
        r2 = 1 - sum((p[1] - a0 - b * p[0]) ** 2 for p in pts) / sum((p[1] - my) ** 2 for p in pts)
        L = (-0.10 - a0) / b
        res[d] = {"rho": rho, "Lstar": L, "lag_N": LN, "fit": [a0, b, r2], "points": pts, "n_reads": n}
        print(d, "rho", round(rho, 3), "L*", round(L, 3), "lag N", round(LN, 3), "fit", [round(a0, 3), round(b, 3), round(r2, 4)],
              [(s, round(x, 3), y) for x, y, s in pts], flush=True)
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
