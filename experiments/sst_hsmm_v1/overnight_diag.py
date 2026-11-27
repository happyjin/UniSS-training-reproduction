"""overnight 2026-11-27 diagnosis (CPU, existing outputs only): why the drafter-side schemes did not move the frontier.
Per system x direction x domain (edev / RealSI / OBSI), from the Thinker block files (non-final reads):
  ant      mean len(draft_t) / len(final) at mid-utterance reads (0.3 <= t/dur <= 0.7): how far drafts run ahead of the audio
  instab   share of the previous read's uncommitted tail that the next read rewrites
  degen    share of non-final drafts with a loop (a unit or unit bigram repeated >= 4 times in a row) or len > 2.5 x final
  settle   mean t_settle / dur of final units (first read from which the unit stays in every draft up to its commit)
  commit   mean t_commit / dur of final units; wait = commit - settle (detection delay, in reads of 1 s)
  calib    on uncommitted positions with a posterior: mean P vs survival rate, bins [0, .3) [.3, .6) [.6, 1]
  p2f_*    (S8 only, draft_p2f present) degen / agreement-with-d0 of the evidence drafts
    python3 experiments/sst_hsmm_v1/overnight_diag.py <out.json>
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units  # noqa: E402

D0 = REPO / "eval_outputs/sst_hsmm_v1"
META = {d: {json.loads(l)["id"]: json.loads(l) for l in open(REPO / f"eval_outputs/mist_w3po_v1/test_robust/data/meta_{d}.jsonl")}
        for d in ("en2zh", "zh2en")}
SYS = {  # name: {dir: (test glob, dev glob)}
    "p3hk": {"en2zh": ("p3hk/thinker_en2zh_*/blocks_en2zh_*.jsonl", "p45/run_en2zh_*/blocks_en2zh_heal_0.75.jsonl"),
             "zh2en": ("p3hk/thinker_zh2en_*/blocks_zh2en_*.jsonl", "d3/dev_w17/run_zh2en_*/blocks_zh2en_base_0.75.jsonl")},
    "S1_d3x0.5": {"en2zh": ("p3d3s05/thinker_en2zh_*/blocks_en2zh_*.jsonl", "d3s05/dev/run_en2zh_*/blocks_en2zh_heal_0.60.jsonl"),
                  "zh2en": ("p3d3s05/thinker_zh2en_*/blocks_zh2en_*.jsonl", "d3s05/dev/run_zh2en_*/blocks_zh2en_base_0.60.jsonl")},
    "S3_p2f_ff": {"en2zh": ("p3p2fxff/thinker_en2zh_*/blocks_en2zh_*.jsonl", "p2fxff/dev/run_en2zh_*/blocks_en2zh_heal_0.60.jsonl"),
                  "zh2en": ("p3p2fxff/thinker_zh2en_*/blocks_zh2en_*.jsonl", "p2fxff/dev/run_zh2en_*/blocks_zh2en_base_0.60.jsonl")},
    "S8_aac": {"en2zh": ("p3aac/thinker_en2zh_*/blocks_en2zh_*.jsonl", "aac/dev/run_en2zh_*/blocks_en2zh_heal_0.75.jsonl"),
               "zh2en": ("p3aac/thinker_zh2en_*/blocks_zh2en_*.jsonl", "aac/dev/run_zh2en_*/blocks_zh2en_base_0.70.jsonl")},
}


def lcp(a, b):
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def degenerate(u, nfin):
    if len(u) > 2.5 * max(nfin, 4):
        return True
    for w in (1, 2):
        run = 1
        for i in range(w, len(u) - w + 1, 1):
            if u[i:i + w] == u[i - w:i]:
                run += 1
                if run >= 4:
                    return True
            else:
                run = 1
    return False


def acc():
    return defaultdict(lambda: [0.0, 0])


def add(a, k, v):
    a[k][0] += v; a[k][1] += 1


def item_stats(r, zh, A, dur):
    fin = units(r["translation"], zh)
    if not fin:
        return
    D = r["drafts"]
    ds = [units(x["draft"], zh) for x in D]
    n, ns = 0, []
    for x in D:
        ns.append(n); n += int(x.get("cv_new", 0) or 0)
    nf = len(D) - 1                                     # index of the final read
    for k in range(nf):
        x, d = D[k], ds[k]
        f = x["t"] / dur
        if 0.3 <= f <= 0.7:
            add(A, "ant", len(d) / len(fin))
        add(A, "degen", int(degenerate(d, len(fin))))
        if k + 1 < nf:
            n1 = ns[k + 1]
            if len(d) > n1:
                add(A, "instab", 1 - (max(lcp(d, ds[k + 1]), n1) - n1) / (len(d) - n1))
        if "P" in x:
            y_ok = lcp(d, fin)
            c = int(x.get("cv_new", 0) or 0)
            for j, p in enumerate(x["P"]):
                if j < c:
                    continue
                i = ns[k] + j
                b = "lo" if p < 0.3 else ("mid" if p < 0.6 else "hi")
                add(A, f"P_{b}", p); add(A, f"y_{b}", int(i < y_ok))
        if "draft_p2f" in x:
            dp = units(x["draft_p2f"], zh)
            add(A, "p2f_degen", int(degenerate(dp, len(fin))))
            tail = len(d) - ns[k]
            if tail > 0:
                add(A, "p2f_agree_tail", (max(lcp(d, dp), ns[k]) - ns[k]) / tail)
    # settle / commit per final unit
    cum = [ns[k] + int(D[k].get("cv_new", 0) or 0) for k in range(len(D))]
    for i in range(len(fin)):
        kc = next((k for k in range(len(D)) if cum[k] > i), nf)
        ks = kc
        while ks - 1 >= 0 and lcp(ds[ks - 1], fin) > i:
            ks -= 1
        add(A, "settle", D[ks]["t"] / dur); add(A, "commit", D[kc]["t"] / dur); add(A, "wait", kc - ks)
        add(A, "excess_s", (D[ks]["t"] - dur * (i + 1) / len(fin)) / 1000)      # settle time beyond a proportional (monotonic) schedule
        add(A, "wait_s", (D[kc]["t"] - D[ks]["t"]) / 1000); add(A, "dur_s", dur / 1000)


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"refuse to overwrite {out}")
    res = {}
    for s, dd in SYS.items():
        for d, (tg, dg) in dd.items():
            zh = d.endswith("zh")
            for split, g in (("test", tg), ("edev", dg)):
                files = sorted(glob.glob(str(D0 / g)))
                if not files:
                    print("missing", s, d, split, g); continue
                accs = defaultdict(acc)
                for f in files:
                    for line in open(f, encoding="utf-8"):
                        r = json.loads(line)
                        if split == "test":
                            m = META[d].get(r["id"])
                            if m is None:
                                continue
                            dom, dur = m["set"], m["dur_ms"]
                        else:
                            dom, dur = "edev", r["drafts"][-1]["t"]
                        item_stats(r, zh, accs[dom], dur)
                for dom, A in accs.items():
                    res[f"{s}|{d}|{dom}"] = {k: round(v[0] / v[1], 4) for k, v in A.items() if v[1]}
                    res[f"{s}|{d}|{dom}"]["n_reads"] = A["degen"][1]
    out.write_text(json.dumps(res, indent=1))
    keys = ["dur_s", "excess_s", "wait_s", "ant", "instab", "degen", "P_hi", "y_hi", "p2f_agree_tail"]
    print("system|dir|domain".ljust(28) + " ".join(k.rjust(8) for k in keys))
    for k_, v in sorted(res.items(), key=lambda kv: (kv[0].split("|")[1], kv[0].split("|")[2], kv[0].split("|")[0])):
        print(k_.ljust(28) + " ".join((f"{v[k]:8.3f}" if k in v else "       -") for k in keys))


if __name__ == "__main__":
    main()
