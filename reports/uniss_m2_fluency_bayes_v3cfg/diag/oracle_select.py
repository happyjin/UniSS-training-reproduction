"""Upper bounds for test-time adaptation: pick, per talk or per sentence, one run out of a set (same system at
different biases) so that mean LAAL stays at the base run's, and see how much corpus ASR-BLEU that could buy.
A noise floor comes from the same selection among runs that differ only in sampling seed."""
import itertools, json, math, sys
import numpy as np
sys.path.insert(0, "/opt/dlami/nvme/jasonleeeli/projects/UniSS")
from evaluation.text_metrics import corpus_bleu
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units
E = "/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500"

def load(sysname, d):
    run = f"{E}/system_{sysname}/realsi_{d}"
    recs = [json.loads(l) for l in open(run + "/results.jsonl")]
    heard = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(run + "/asr_speech.jsonl"))}
    lang = recs[0]["tgt_lang"]; out = {}
    for r in recs:
        s = corpus_bleu([heard.get(r["id"], "")], [r["translation_ref"]], language=lang)
        ds = [b["start_ms"] for b in r["blocks"] for _ in units(b["text"], lang)]
        lat = laal(ds, r["source_duration_ms"], len(units(r["translation_ref"], lang))) / 1000 if ds else None
        out[r["id"]] = (np.array(s["counts"] + s["totals"] + [s["sys_len"], s["ref_len"]], float), lat)
    return out

def bleu(st):
    c, t, h, r = st[:4], st[4:8], st[8], st[9]
    if min(c) == 0: return 0.0
    bp = 1.0 if h > r else math.exp(1 - r / max(h, 1))
    return 100 * bp * math.exp(sum(math.log(c[i] / t[i]) for i in range(4)) / 4)

def evaluate(runs, choice, ids):
    st = sum(runs[choice[i]][i][0] for i in ids)
    lat = np.mean([runs[choice[i]][i][1] for i in ids if runs[choice[i]][i][1] is not None])
    return bleu(st), lat

def sentence_oracle(runs, ids, base, lat_cap, iters=6):
    best = None
    for lam in np.linspace(0, 30, 61):
        ch = {i: base for i in ids}
        st = sum(runs[base][i][0] for i in ids); lats = {i: runs[base][i][1] or 0 for i in ids}
        for _ in range(iters):
            for i in ids:
                cur = ch[i]; bestv, bestk = None, cur
                for k in runs:
                    st2 = st - runs[cur][i][0] + runs[k][i][0]
                    v = bleu(st2) - lam * (runs[k][i][1] or 0) / len(ids)
                    if bestv is None or v > bestv: bestv, bestk = v, k
                if bestk != cur:
                    st = st - runs[cur][i][0] + runs[bestk][i][0]; ch[i] = bestk
        b, l = evaluate(runs, ch, ids)
        if l <= lat_cap + 1e-9 and (best is None or b > best[0]): best = (b, l, lam)
    return best

def talk_oracle(runs, ids, lat_cap):
    talks = sorted({i.rsplit("_", 1)[0] for i in ids}); keys = list(runs); best = None
    agg = {(t, k): (sum(runs[k][i][0] for i in ids if i.rsplit("_", 1)[0] == t),
                    sum(runs[k][i][1] for i in ids if i.rsplit("_", 1)[0] == t and runs[k][i][1] is not None),
                    sum(1 for i in ids if i.rsplit("_", 1)[0] == t and runs[k][i][1] is not None)) for t in talks for k in keys}
    for combo in itertools.product(range(len(keys)), repeat=len(talks)):
        st = 0; ls = 0; n = 0
        for t, c in zip(talks, combo):
            a = agg[(t, keys[c])]; st = st + a[0]; ls += a[1]; n += a[2]
        l = ls / n
        if l <= lat_cap + 1e-9:
            b = bleu(st)
            if best is None or b > best[0]: best = (b, l, [keys[c] for c in combo])
    return best

# zh2en is also run with base "b-0.5" (oracle_zh2en_mid.json): the b-1 base is the lowest-latency run, so a
# selection capped at its latency has nothing to trade.
SETS = {
 "en2zh": {"base": "b1", "runs": {"b0.5": "flu7-sq_m0.5", "b1": "flu7-sq-b1_m0.5", "b1.5": "bay-v3-b1.5_m0.5", "b2": "bay-v3-b2_m0.5"},
           "noise": {"s0": "la2_m1", "s1": "la2-seed1_m1", "s2": "la2-seed2_m1"}},
 "zh2en": {"base": "b-1", "runs": {"b-1": "flu7-sq-zh_m1", "b-0.5": "flu7-sq-bm0p5-zh_m1", "b0": "bay-v3-zh-b0_m1", "b0.5": "bay-v3-zh-b0.5_m1"},
           "noise": {"s0": "la2_m1", "s1": "la2-seed1_m1", "s2": "la2-seed2_m1"}},
}
res = {}
for d, cfg in SETS.items():
    runs = {k: load(v, d) for k, v in cfg["runs"].items()}
    ids = sorted(runs[cfg["base"]])
    b0, l0 = evaluate(runs, {i: cfg["base"] for i in ids}, ids)
    per_run = {k: evaluate(runs, {i: k for i in ids}, ids) for k in runs}
    to = talk_oracle(runs, ids, l0); so = sentence_oracle(runs, ids, cfg["base"], l0)
    nruns = {k: load(v, d) for k, v in cfg["noise"].items()}
    nb, nl = evaluate(nruns, {i: "s0" for i in ids}, ids)
    nso = sentence_oracle(nruns, ids, "s0", 99)
    nto = talk_oracle(nruns, ids, 99)
    res[d] = {"base": [round(b0, 2), round(l0, 3)], "per_run": {k: [round(a, 2), round(b, 3)] for k, (a, b) in per_run.items()},
              "talk_oracle": [round(to[0], 2), round(to[1], 3), list(to[2])], "sentence_oracle": [round(so[0], 2), round(so[1], 3)],
              "noise_floor_seeds": {"base": round(nb, 2), "talk_oracle": round(nto[0], 2), "sentence_oracle": round(nso[0], 2)}}
    print(d, json.dumps(res[d]), flush=True)
json.dump(res, open(sys.argv[1], "x"), indent=1)
