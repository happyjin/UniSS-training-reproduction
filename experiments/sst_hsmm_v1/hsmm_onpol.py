"""sst_hsmm_v1 prereg change 36 (HSC6-on-policy): the HSC6 model / objective of hsmm_anchor.py (not modified), refitted on the union of
the N-policy rollouts (h1/roll_*) and rollouts of the deployed policy itself (hsc9_si_thinker n2 theta 0.75 on h14, h20/roll_*), i.e. one
DAgger-style aggregation step. The per-read commit count is cv_new in both record kinds (the thinker stores len(new) as cv_new and N's cut as
n_new; the first version used n_new for on-policy records, which broke the lp0 alignment), so the walk is hsmm_anchor.walk unchanged.
Precheck on the held-out (sha1 % 5 == 0) on-policy instances only: GO if ECE_new <= ECE_h14 and AUC_new >= AUC_h14 - 0.005.
    <qwen3omni python> -m experiments.sst_hsmm_v1.hsmm_onpol fit --roots 'eval_outputs/sst_hsmm_v1/h1/roll_*,eval_outputs/sst_hsmm_v1/h20/roll_*' \
        --onpol-root 'eval_outputs/sst_hsmm_v1/h20/roll_*' --pre-out eval_outputs/sst_hsmm_v1/h20/precheck.json --out eval_outputs/sst_hsmm_v1/h20/params.json"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import hsmm_anchor as HA  # noqa: E402


walk = HA.walk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fit",))
    ap.add_argument("--roots", required=True)
    ap.add_argument("--onpol-root", required=True)
    ap.add_argument("--pre-out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=str(REPO / "eval_outputs/sst_hsmm_v1/h14/params.json"))
    a = ap.parse_args()
    for f in (a.pre_out, a.out):
        if Path(f).exists():
            sys.exit(f"拒绝覆盖: {f}")
    held = lambda i: int(hashlib.sha1(i.encode()).hexdigest(), 16) % 5 == 0
    E1 = REPO / "eval_outputs/sst_hsmm_v1/e1"
    onp = set()
    for d in ("en2zh", "zh2en"):
        onp |= set(glob.glob(f"{a.onpol_root}/blocks_{d}_w1.jsonl"))
    seqs, inst = [], []
    for d in ("en2zh", "zh2en"):
        relv = HA.relv_from_views(E1 / "data" / f"items_{d}.jsonl", E1 / "views" / d)
        n0, m0 = len(seqs), len(inst)
        for root in a.roots.split(","):
            for f in sorted(glob.glob(f"{root}/blocks_{d}_w1.jsonl")):
                s, i = walk(f, d.endswith("zh"), relv, held)
                seqs += s
                if f in onp:
                    inst += i
        print(d, "sequences", len(seqs) - n0, "held-out on-policy instances", len(inst) - m0, flush=True)
    assert inst, "no on-policy instances"
    train = [(o, u, y) for i, o, u, y in seqs if not held(i)]
    print("train sequences", len(train), flush=True)
    p80, h80, chk = HA.fit(train)
    print("torch filtering vs numpy posterior max abs diff", chk, flush=True)
    Pn = HA.HJ.NParams(p80)
    Pr = HA.HJ.NParams(json.load(open(a.ref))["params"])
    y = [x[2] for x in inst]; Tn = [x[3] for x in inst]
    rr = HA.report("h14", [HA.HJ.posterior_stable(o, u, Pr) for o, u, _, _ in inst], y, Tn)
    rn = HA.report("onpol_80", [HA.HJ.posterior_stable(o, u, Pn) for o, u, _, _ in inst], y, Tn)
    go = rn["ece"] <= rr["ece"] and rn["auc"] >= rr["auc"] - 0.005
    res = {"go": bool(go), "rule": "on-policy held-out: ECE_new <= ECE_h14 and AUC_new >= AUC_h14 - 0.005", "h14": rr, "onpol_80": rn,
           "hist_80": h80, "filter_check": chk, "n_train": len(train), "beta": HA.BETA}
    Path(a.pre_out).write_text(json.dumps(res, indent=1))
    print("precheck GO" if go else "precheck NO-GO", flush=True)
    if not go:
        sys.exit(3)
    pall, hall, chk2 = HA.fit([(o, u, y) for _, o, u, y in seqs])
    out = {"lmax": HA.LMAX, "eps": HA.EPS, "joint": True, "anchored": True, "onpol": True, "beta": HA.BETA, "params": pall, "hist": hall,
           "filter_check": chk2, "n_seq": len(seqs)}
    out["dirs"] = {d: {"params": pall} for d in ("en2zh", "zh2en")}      # the same single model for both directions
    Path(a.out).write_text(json.dumps(out))
    print("final fit written", a.out, flush=True)


if __name__ == "__main__":
    main()
