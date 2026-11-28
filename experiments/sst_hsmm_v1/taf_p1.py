"""change 67 precheck analysis (taf_precheck.sh): P0 real time + P1 discrimination of the TAF-S veto.
P0: per-round wall times (combo json round_s, first round dropped) of the TAF and base arms on 8 edev items per direction;
    GO iff p95(TAF) <= 1.0 s in both directions.
P1: --taf-shadow runs (edev + cvdev shard 0). For every non-final read record with rec["taf"] = [cut, cap, n_valid, n], the units
    j in [n, cut) are the HSMM commits; survival(j) = j < LCP(draft units, units of the final read's unprefixed whole-audio draft
    rec["shadow"]); vetoed(j) = j >= cap. GO iff in both directions surv(not vetoed) - surv(vetoed) >= 0.15 and
    vetoed share <= 0.30 (pooled over the two dev sets).
    python3 taf_p1.py --root <pre dir> --k K --out <json>   (refuses to overwrite)"""
import argparse, glob, json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--k", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if Path(a.out).exists():
    raise SystemExit(f"refuse to overwrite {a.out}")
units = lambda t, zh: [c for c in t if not c.isspace()] if zh else t.split()


def lcp(x, y):
    j = 0
    while j < len(x) and j < len(y) and x[j] == y[j]: j += 1
    return j


def p95(v):
    v = sorted(v); return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))] if v else None


res = {"k": a.k, "p0": {}, "p1": {}}
for d in ("en2zh", "zh2en"):
    zh = d == "en2zh"
    p0 = {}
    for arm in ("taf", "base"):
        f = glob.glob(f"{a.root}/p0/{d}_{arm}/job_{d}.combo.json")
        rs = json.load(open(f[0]))["round_s"][1:] if f else []
        p0[arm] = {"rounds": len(rs), "p50": sorted(rs)[len(rs) // 2] if rs else None, "p95": p95(rs)}
    p0["go"] = p0["taf"]["p95"] is not None and p0["taf"]["p95"] <= 1.0
    res["p0"][d] = p0
    c = {"kept": 0, "kept_surv": 0, "vet": 0, "vet_surv": 0, "reads": 0, "reads_capped": 0, "valid": 0}
    per_ds = {}
    for ds in ("edev", "cvdev"):
        cc = dict.fromkeys(c, 0)
        for f in glob.glob(f"{a.root}/p1/{ds}_{d}/blocks_{d}_*.jsonl"):
            for row in map(json.loads, open(f, encoding="utf-8")):
                sh = [x.get("shadow") for x in row["drafts"] if x.get("shadow") is not None]
                if not sh:
                    continue
                S = units(sh[-1], zh)
                for x in row["drafts"]:
                    if "taf" not in x:
                        continue
                    cut, cap, nv, n = x["taf"]
                    D = units(x["draft"], zh); L = lcp(D, S)
                    cc["reads"] += 1; cc["reads_capped"] += int(cut > cap); cc["valid"] += nv
                    for j in range(n, cut):
                        s = int(j < L)
                        if j >= cap: cc["vet"] += 1; cc["vet_surv"] += s
                        else: cc["kept"] += 1; cc["kept_surv"] += s
        per_ds[ds] = cc
        for k_ in c: c[k_] += cc[k_]
    sk = c["kept_surv"] / max(1, c["kept"]); sv = c["vet_surv"] / max(1, c["vet"]); share = c["vet"] / max(1, c["vet"] + c["kept"])
    res["p1"][d] = {"surv_kept": round(sk, 4), "surv_vetoed": round(sv, 4), "gap": round(sk - sv, 4), "vetoed_share": round(share, 4),
                    "units_committed": c["vet"] + c["kept"], "reads": c["reads"], "reads_capped": c["reads_capped"],
                    "mean_valid_futures": round(c["valid"] / max(1, c["reads"]), 2), "per_devset": per_ds,
                    "go": (sk - sv >= 0.15) and share <= 0.30 and c["vet"] > 0}
    print(d, "P0", json.dumps(p0), flush=True)
    print(d, "P1", json.dumps({k_: v for k_, v in res["p1"][d].items() if k_ != "per_devset"}), flush=True)
res["go_p0"] = all(res["p0"][d]["go"] for d in res["p0"])
res["go_p1"] = all(res["p1"][d]["go"] for d in res["p1"])
res["go"] = res["go_p0"] and res["go_p1"]
json.dump(res, open(a.out, "x"), indent=1)
print("GO" if res["go"] else "NOGO", "p0", res["go_p0"], "p1", res["go_p1"])
