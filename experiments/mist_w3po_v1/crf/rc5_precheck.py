"""RC5's CPU precheck (reports/mist_w3po_v1/thinker_rc5_prereg_v1.zh-CN.md §2); every line must pass:

  R1  build: sched_t5_rc_w10_v1's builder / trainer checks (n_infeasible 0, n_bad 0) and the same 22,671 windows as T5
  R2  label path unchanged: over all windows, the label read of every word (utterance, word order) equals the one in
      sched_t5_pieces4_w10_v1 for >= 99.9% of the words; the mean label-path latency heuristic (summary
      L_ref_mean_lat_heur_windows) within 1% (relative) of T5's
  R3  granularity: words per piece in [3.5, 5.0] (T5 2.89, VE1's merge3 5.98)
  R4  trainer: run_crf_v2p.sh's arguments + EXTRA parse to latent_write / t5; the DART manifest loads against the plan
      (draws and plan identical to the trainer's) and was made in mode t5; the unit / regression tests pass
      (test_t5_rc, test_dart_aug, test_latent_write_crf, test_train_crf_sft)
Reported too: pieces, final-read label word share, same-read cuts removed, per-step lambda* on the first steps.

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.rc5_precheck --out <new json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

if not torch.cuda.is_available():
    torch.accelerator.current_accelerator = lambda *a, **k: None

from experiments.mist_w3po_v1.crf import dart_aug as DA  # noqa: E402
from experiments.mist_w3po_v1.crf import train_crf_sft as T  # noqa: E402

C = "eval_outputs/mist_w3po_v1/crf"
SCHED = f"{C}/sched_t5_rc_w10_v1"
REF = f"{C}/sched_t5_pieces4_w10_v1"
DART = f"{C}/t5rc_dart_w10_v1"
EXTRA = f"--sched {SCHED} --objective latent_write --lw-mode t5 --dart {DART}"
BASE = ("--objective serial_v2p --tau 0.5 --final-read-full-post 1 --out /nonexistent --steps 3000 --save-every 500 "
        "--feat-per-row 1 --fast-moe-bwd 1 --prefetch 1 --empty-cache 0 --tok-budget 32000 --mem-cap-gib 120")
TESTS = ["test_t5_rc", "test_dart_aug", "test_latent_write_crf", "test_train_crf_sft"]


def word_reads(rec: dict) -> dict:
    out = defaultdict(list)
    for c in sorted(rec["chunks"], key=lambda c: (c["utt"], c["k"])):
        out[c["utt"]] += [c["e_read"]] * len(c["text"].split())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-tests", action="store_true", help="debug only: R4 then cannot pass")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {}
    sm, sr = json.loads(Path(SCHED, "summary.json").read_text()), json.loads(Path(REF, "summary.json").read_text())
    wins, ref = T.load_windows(SCHED), {w.rec["id"]: w.rec for w in T.load_windows(REF)}
    res["R1"] = {"windows": len(wins), "windows_t5": len(ref), "n_infeasible": sm["n_infeasible"], "n_bad": sm["n_bad"]}
    res["R1"]["pass"] = bool(sm["n_infeasible"] == 0 and sm["n_bad"] == 0 and len(wins) == len(ref) == 22671
                             and {w.rec["id"] for w in wins} == set(ref))
    # R2
    eq = tot = 0
    bad_w = []
    for w in wins:
        a_, b_ = word_reads(w.rec), word_reads(ref[w.rec["id"]])
        ok = True
        for u in set(a_) | set(b_):
            x, y = a_.get(u, []), b_.get(u, [])
            if len(x) != len(y):
                ok = False
                tot += max(len(x), len(y))
                continue
            eq += sum(p == q for p, q in zip(x, y))
            tot += len(x)
            ok &= x == y
        if not ok:
            bad_w.append(w.rec["id"])
    lh, lh5 = sm["L_ref_mean_lat_heur_windows"], sr["L_ref_mean_lat_heur_windows"]
    res["R2"] = {"words": tot, "same_label_read": eq, "share": round(eq / tot, 6), "windows_differing": len(bad_w),
                 "examples": bad_w[:10], "lat_heur_rc": lh, "lat_heur_t5": lh5, "lat_heur_rel_diff": round(lh / lh5 - 1, 5)}
    res["R2"]["pass"] = bool(eq / tot >= 0.999 and abs(lh / lh5 - 1) <= 0.01)
    # R3 (+ reported statistics)
    nw = [c["n_words"] for w in wins for c in w.rec["chunks"]]
    nw5 = [c["n_words"] for r in ref.values() for c in r["chunks"]]
    fin = sum(c["n_words"] for w in wins for c in w.rec["chunks"] if c["e_read"] == w.R - 1)
    res["R3"] = {"pieces": len(nw), "pieces_t5": len(nw5), "words_per_piece": round(float(np.mean(nw)), 3),
                 "words_per_piece_t5": round(float(np.mean(nw5)), 3), "share_1w": round(float(np.mean(np.array(nw) == 1)), 4),
                 "share_1w_t5": round(float(np.mean(np.array(nw5) == 1)), 4), "p90": float(np.percentile(nw, 90)),
                 "final_read_label_word_share": round(fin / sum(nw), 4), "final_word_share_t5": sr["final_word_share"]}
    res["R3"]["pass"] = bool(3.5 <= res["R3"]["words_per_piece"] <= 5.0)
    # R4
    args = T.build_args((BASE + " " + EXTRA).split())
    plan = T.plan_steps([w.R for w in wins], 20261109, 3000, 64)
    d = DA.load_manifest(args.dart, wins, plan, args.sched, args.seed, args.steps, args.rows_per_step)
    tests = {}
    if not a.skip_tests:
        # CPU only (CUDA_VISIBLE_DEVICES=): torch.accelerator.current_accelerator() raises without a device, which breaks
        # transformers' lazy imports in tests that do not patch it themselves (test_train_crf_sft); patch it the way
        # test_dart_aug / this module do, then run the test module as __main__
        boot = ("import torch, runpy\nif not torch.cuda.is_available():\n"
                "    torch.accelerator.current_accelerator = lambda *a, **k: None\n"
                "runpy.run_module('experiments.mist_w3po_v1.crf.{}', run_name='__main__')")
        for t in TESTS:
            p = subprocess.run([sys.executable, "-c", boot.format(t)], capture_output=True, text=True, timeout=3600)
            tests[t] = {"rc": p.returncode, "tail": (p.stdout + p.stderr).strip().splitlines()[-1:]}
    res["R4"] = {"objective": args.objective, "lw_mode": args.lw_mode, "dart_entries": sum(len(v) for v in d.values()),
                 "manifest_mode": DA.manifest_mode(args.dart), "plan_windows": len({i for s in plan for i in s}),
                 "tests": tests}
    res["R4"]["pass"] = bool(args.objective == "latent_write" and args.lw_mode == "t5" and res["R4"]["manifest_mode"] == "t5"
                             and res["R4"]["dart_entries"] > 0 and tests and all(v["rc"] == 0 for v in tests.values()))
    res["pass"] = all(res[k]["pass"] for k in ("R1", "R2", "R3", "R4"))
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
