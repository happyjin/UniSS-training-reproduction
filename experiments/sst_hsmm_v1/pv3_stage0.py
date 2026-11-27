"""plan v3 (change 56) stage 0 (CPU, dev only): split B0's content-stabilisation lag on edev into
  reorder  = E(q) - P(q): when the reference content at relative position q has been heard (OmniAlign support_omni_ms of the
             reference's target words, running max) minus the proportional (monotone) schedule dur * q
  rest     = S - max(E, P): from evidence heard to the unit settling in the drafts = recognition + hesitation + read quantisation
             (reads every 1 s: a perfect drafter still settles on average ~0.5 s after the evidence)
per final system unit i (q = (i + 1) / len(final)); S = first read from which the unit stays in every draft (overnight_diag).
H5 rule (prereg change 56): run H5 only if rest - 0.5 s >= 0.3 s.
    python3 experiments/sst_hsmm_v1/pv3_stage0.py <out.json>
"""
import glob, json, sys
from pathlib import Path
import pyarrow.parquet as pq
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units  # noqa: E402
D0 = REPO / "eval_outputs/sst_hsmm_v1"
RUNS = {"en2zh": "pfb0/dev/edev/run_en2zh_*/blocks_en2zh_heal_0.75.jsonl", "zh2en": "pfb0/dev/edev/run_zh2en_*/blocks_zh2en_base_0.75.jsonl"}


def lcp(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def main():
    out = Path(sys.argv[1])
    if out.exists():
        sys.exit(f"exists: {out}")
    items = {d: [json.loads(l) for f in sorted(glob.glob(str(D0 / RUNS[d]))) for l in open(f)] for d in RUNS}
    ids = {r["id"].split("__", 1)[1] for d in items for r in items[d]}
    cm = {}
    for f in sorted((REPO / "data/unist_simul_v1/rows").glob("part-*.parquet")):
        for row in pq.read_table(f, columns=["id", "support_omni_ms"]).to_pylist():
            if row["id"] in ids and row["id"] not in cm:
                m, c = 0, []
                for v in json.loads(row["support_omni_ms"]):
                    if v is not None:
                        m = max(m, int(v))
                    c.append(m)
                cm[row["id"]] = c
    res = {}
    for d, rs in items.items():
        zh = d == "en2zh"
        A = {"excess": [], "reorder": [], "rest": [], "settle_minus_P": []}
        miss = 0
        for r in rs:
            c = cm.get(r["id"].split("__", 1)[1])
            fin = units(r["translation"], zh)
            D = r["drafts"]
            dur = D[-1]["t"]
            if not c or not fin:
                miss += 1; continue
            ds = [units(x["draft"], zh) for x in D]
            nf = len(D) - 1
            n, cum = 0, []
            for x in D:
                n += int(x.get("cv_new", 0) or 0); cum.append(n)
            for i in range(len(fin)):
                kc = next((k for k in range(len(D)) if cum[k] > i), nf)
                ks = kc
                while ks - 1 >= 0 and lcp(ds[ks - 1], fin) > i:
                    ks -= 1
                q = (i + 1) / len(fin)
                P = dur * q
                E = min(c[min(len(c) - 1, max(0, int(round(q * len(c))) - 1))], dur)
                S = D[ks]["t"]
                A["excess"].append((S - P) / 1000); A["reorder"].append(max(0.0, E - P) / 1000)
                A["rest"].append((S - max(E, P)) / 1000)
        res[d] = {k: round(sum(v) / max(1, len(v)), 3) for k, v in A.items() if v}
        res[d].update(n_units=len(A["excess"]), n_items=len(rs), missing=miss)
        res[d]["rest_minus_quant"] = round(res[d]["rest"] - 0.5, 3)
        res[d]["h5_run"] = res[d]["rest_minus_quant"] >= 0.3
    json.dump(res, open(out, "x"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
