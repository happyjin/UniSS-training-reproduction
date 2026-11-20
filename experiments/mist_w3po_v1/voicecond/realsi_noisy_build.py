"""RealSI-noisy v1: the noisy twin of the RealSI test set (reports/mist_w3po_v1/robust_test_prereg_20261120.zh-CN.md).
TEST ONLY: nothing is trained, fitted, selected or tuned on it. CPU only; refuses an existing output directory; built once.

Units = the RealSI evaluation units of the paper's LA2 baseline, i.e. si_infer.realsi_items(<dir>) (en2zh 346, zh2en 431
sentences): sentence k of talk <vid> = samples [start, end) ms of data/raw/realsi/RealSI_data/data/<dir>/wav/<vid>.wav,
channels averaged, reference = trg_text. The twin keeps every id, the cut (same samples, same length, same reference);
only the source audio gains a background:

  background  one continuous timeline per talk (bench_build.bg_schedule, seeded "OpenBroadcast-SI-v1|realsi:bg:<dir>:<vid>")
              rendered by bench_build.render_bg over the OpenBroadcast-SI id2 pools (bench/bgsrc/pools.json: held-out
              MUSAN music = bed / music, held-out FSD50K crowd = amb / crowd, held-out FSD50K sfx + MUSAN noise = sfx), a
              2.5 s lead-in dropped (bed fade-in), each layer rendered separately (per-sentence layer energy shares).
  level       per sentence, BVR target = quantile of the OpenBroadcast-SI BVR statistic (bench/meta/stats.json bvr_db, the
              same values bench_build.plan uses) at the stratified q_i = 0.05 + 0.90 (perm_i + 0.5) / n, perm = shuffle
              of range(n) by random.Random("OpenBroadcast-SI-v1|realsi:bvr:<dir>"), items in realsi_items order;
              alpha_i = sqrt(10^(BVR/10) * E(clean cut) / E(background cut)) (bench_build.mix's alpha over the cut), no
              ducking (a cut holds one sentence: ducking would only add a 50 ms attack ramp at the cut start), the bench
              limiter (whole mixture scaled under 0.99 if its peak exceeds 0.99, speech and background alike).
Why per-sentence cuts and a per-sentence level (not one noisy talk wav): the baseline evaluates every sentence alone (the
Thinker hears only the cut; the Talker's voice prompt and the V1 token are the cut's first max(first write, 1 s)), so
cutting the clean talk exactly as realsi_items does keeps the time axis and every read window identical, and a
per-sentence BVR draw follows the target distribution with n = 346 / 431 draws instead of 10 talks.

Writes (FLOAT wav: the clean cut is bit-identical to what realsi_items feeds the models):
  <out>/audio/<dir>/{clean,noisy}/<id>.wav
  <out>/items_{clean,noisy}_<dir>.jsonl   {id, audio, dur_ms, translation, src_text} (si_thinker_stream / si_infer --items-file)
  <out>/manifest_<dir>.jsonl              per sentence: cut, BVR target / realised / speech-active, alpha, limiter,
                                          layer energy shares, dominant layer, background events in the cut (+ licences)
  <out>/sessions_<dir>.json               per talk: schedule, seed, rendered events
  <out>/build_info.json                   git revision, script sha1, seeds, pools, BVR source
  <out>/verify.json                       (step verify)

    python experiments/mist_w3po_v1/voicecond/realsi_noisy_build.py build  [--procs 20]
    python experiments/mist_w3po_v1/voicecond/realsi_noisy_build.py verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import bench_build as BB  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items  # noqa: E402

OUT = REPO / "data/open_voicecond_v1/realsi_noisy_v1"
SR = 16000
SALT = BB.SALT
LEAD_S = 2.5
COND = "id2"
LAYERS = ("bed", "amb", "sfx")
LAYER_NAME = {"bed": "music", "amb": "crowd", "sfx": "sfx"}
BG_INDEX = REPO / "data/open_voicecond_v1/idea2/index/backgrounds.jsonl"
N_EXPECT = {"en2zh": 346, "zh2en": 431}


def pools() -> dict:
    return json.load(open(BB.BENCH / "bgsrc/pools.json"))["pools"]


def bvr_values() -> np.ndarray:
    return np.array(json.load(open(BB.BENCH / "meta/stats.json"))["bvr_db"]["values"])


def heldout_check(P: dict) -> dict:
    """every id2 pool file is held out (listed + the sha1 rule) and in no training / dev background pool"""
    H = set().union(*BB.heldout_set().values())
    used = {r["src"] for k in BB.LAYER_POOLS[COND].values() for r in P[k]}
    not_listed = sorted(s for s in used if s not in H)
    not_rule = sorted(s for s in used if not BB.held(s))
    idx = {}
    for r in map(json.loads, open(BG_INDEX, encoding="utf-8")):
        idx.setdefault(r["split"], set()).add(r["rel"].split("/", 1)[1])
    inter = {sp: sorted(used & v) for sp, v in idx.items()}
    res = {"id2_pool_files": len(used), "not_in_heldout_lists": not_listed, "fails_heldout_rule": not_rule,
           "intersection_with_idea2_background_index": {k: len(v) for k, v in inter.items()},
           "idea2_index_sizes": {k: len(v) for k, v in idx.items()},
           "note": ("idea2 index = the only background source of every robust-line training run (wave1_data.bg_pools "
                    "'train' for CoVoST-noisy ERM / wave5 / wave14 UniST / wave17; 'dev' for dev (a) noisy); idea1 "
                    "(V1 voice encoder) pools drop held-out files by construction (idea1_build_diag.bg_pools(held))")}
    assert not not_listed and not not_rule and not any(inter.values()), res
    return res


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _talk_job(a):
    d, vid, wav, items, P = a
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    assert sr == SR, wav
    x = x.mean(axis=1)
    T = len(x) / SR
    rng = random.Random(f"{SALT}realsi:bg:{d}:{vid}")
    sched = BB.bg_schedule(rng, T + LEAD_S)
    n = len(x) + int(LEAD_S * SR)
    L, used = {}, []
    for layer in LAYERS:
        sub = {k: (sched[k] if k == layer else []) for k in LAYERS}
        b, u = BB.render_bg(sub, P, COND, n)
        L[layer] = b[int(LEAD_S * SR):]
        used += u
    bg = L["bed"] + L["amb"] + L["sfx"]
    lic = {r["src"]: r.get("licence", "") for k in BB.LAYER_POOLS[COND].values() for r in P[k]}
    rows = []
    for it in items:
        s0, s1 = it["start"] * 16, it["end"] * 16
        c = x[s0:s1]
        b = bg[s0:s1]
        es, eb = float((c.astype(np.float64) ** 2).sum()), float((b.astype(np.float64) ** 2).sum())
        alpha = math.sqrt(10 ** (it["bvr_db"] / 10) * es / max(eb, 1e-12))
        y = c + alpha * b
        peak = float(np.abs(y).max())
        lim = 0.99 / peak if peak > 0.99 else 1.0
        y = (y * lim).astype(np.float32)
        cp = OUT / "audio" / d / "clean" / f"{it['id']}.wav"
        npth = OUT / "audio" / d / "noisy" / f"{it['id']}.wav"
        for p in (cp, npth):
            if p.exists():
                raise SystemExit(f"拒绝覆盖已有输出: {p}")
        sf.write(str(cp), c, SR, subtype="FLOAT")
        sf.write(str(npth), y, SR, subtype="FLOAT")
        bd = (y - c * lim).astype(np.float64)
        bvr_real = 10 * math.log10(float((bd ** 2).sum()) / max(float(((c * lim).astype(np.float64) ** 2).sum()), 1e-12))
        db = BB.frame_db(c)
        act = np.repeat(db > db.max() - 30.0, 160)[: len(c)]
        act = np.pad(act, (0, len(c) - len(act)))
        bvr_act = 10 * math.log10(float((bd[act] ** 2).sum()) / max(float(((c * lim)[act].astype(np.float64) ** 2).sum()), 1e-12))
        sh = {LAYER_NAME[k]: float((L[k][s0:s1].astype(np.float64) ** 2).sum()) for k in LAYERS}
        tot = sum(sh.values()) or 1.0
        sh = {k: round(v / tot, 4) for k, v in sh.items()}
        a0, a1 = LEAD_S + it["start"] / 1000, LEAD_S + it["end"] / 1000
        ev = [{"layer": LAYER_NAME[e["layer"]], "src": e["src"], "t_in_cut_s": round(e["t"] - a0, 3), "len_s": e["len_s"],
               "licence": lic.get(e["src"], "")} for e in used if e["t"] < a1 and e["t"] + e["len_s"] > a0]
        rows.append({"id": it["id"], "vid": vid, "start_ms": it["start"], "end_ms": it["end"], "dur_ms": len(c) // 16,
                     "bvr_q": it["bvr_q"], "bvr_target_db": it["bvr_db"], "bvr_real_db": round(bvr_real, 3),
                     "bvr_speech_active_db": round(bvr_act, 3), "alpha": round(alpha, 6), "limiter": round(lim, 5),
                     "peak": round(float(np.abs(y).max()), 4), "layer_share": sh, "dominant": max(sh, key=sh.get),
                     "events": ev, "clean": str(cp.relative_to(REPO)), "noisy": str(npth.relative_to(REPO)),
                     "translation": it["trg_text"], "src_text": it["src_text"]})
    return d, vid, rows, {"vid": vid, "dur_s": round(T, 3), "seed": f"{SALT}realsi:bg:{d}:{vid}", "schedule": sched,
                          "events": used}


def build(a) -> None:
    import multiprocessing as mp
    if OUT.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {OUT}")
    P = pools()
    hc = heldout_check(P)
    bv = bvr_values()
    jobs, plan = [], {}
    for d in ("en2zh", "zh2en"):
        items = realsi_items(d)
        assert len(items) == N_EXPECT[d], (d, len(items))
        n = len(items)
        perm = list(range(n))
        random.Random(f"{SALT}realsi:bvr:{d}").shuffle(perm)
        for i, it in enumerate(items):
            q = 0.05 + 0.90 * (perm[i] + 0.5) / n
            it["bvr_q"] = round(q, 5)
            it["bvr_db"] = round(float(np.quantile(bv, q)), 3)
        plan[d] = [it["id"] for it in items]
        by = {}
        for it in items:
            by.setdefault(it["wav"], []).append(it)
        for wav, its in by.items():
            jobs.append((d, Path(wav).stem, wav, its, P))
        for c in ("clean", "noisy"):
            (OUT / "audio" / d / c).mkdir(parents=True)
    with mp.get_context("fork").Pool(min(a.procs, len(jobs))) as pool:
        R = pool.map(_talk_job, jobs, chunksize=1)
    for d in ("en2zh", "zh2en"):
        rows = {r["id"]: r for dd, _, rr, _ in R if dd == d for r in rr}
        sess = {vid: s for dd, vid, _, s in R if dd == d}
        with BB.x_open(OUT / f"manifest_{d}.jsonl") as fm, BB.x_open(OUT / f"items_clean_{d}.jsonl") as fc, \
                BB.x_open(OUT / f"items_noisy_{d}.jsonl") as fn:
            for sid in plan[d]:
                r = rows[sid]
                fm.write(json.dumps(r, ensure_ascii=False) + "\n")
                for f, c in ((fc, "clean"), (fn, "noisy")):
                    f.write(json.dumps({"id": sid, "audio": str(REPO / r[c]), "dur_ms": r["dur_ms"],
                                        "translation": r["translation"], "src_text": r["src_text"]}, ensure_ascii=False) + "\n")
        with BB.x_open(OUT / f"sessions_{d}.json") as fh:
            json.dump(sess, fh, ensure_ascii=False)
    used = sorted({e["src"] for _, _, _, s in R for e in s["events"]})
    lic = {r["src"]: r.get("licence", "") for k in BB.LAYER_POOLS[COND].values() for r in P[k]}
    with BB.x_open(OUT / "build_info.json") as fh:
        json.dump({"git_rev": git_rev(), "script": str(Path(__file__).relative_to(REPO)),
                   "script_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest(),
                   "units": "si_infer.realsi_items (paper LA2 baseline units)", "n": {d: len(v) for d, v in plan.items()},
                   "bvr_source": "data/open_voicecond_v1/bench/meta/stats.json bvr_db.values (OpenBroadcast-SI statistic)",
                   "bvr_seed": f"{SALT}realsi:bvr:<dir>", "bg_seed": f"{SALT}realsi:bg:<dir>:<vid>", "lead_in_s": LEAD_S,
                   "pools": {k: f"bench/bgsrc/pools.json[{v}]" for k, v in BB.LAYER_POOLS[COND].items()},
                   "ducking": "none", "limiter": "whole mixture <= 0.99", "heldout_check": hc,
                   "background_files_used": len(used), "licences_used": sorted({lic.get(s, "") for s in used}),
                   "background_files": [{"src": s, "licence": lic.get(s, "")} for s in used]}, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"n": {d: len(v) for d, v in plan.items()}, "bg_files_used": len(used)}))


def verify(a) -> None:
    out = OUT / "verify.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    bv = bvr_values()
    res = {"heldout_check": heldout_check(pools())}
    rng = random.Random("realsi_noisy_v1|verify")
    for d in ("en2zh", "zh2en"):
        items = realsi_items(d)
        M = [json.loads(l) for l in open(OUT / f"manifest_{d}.jsonl", encoding="utf-8")]
        ic = [json.loads(l) for l in open(OUT / f"items_clean_{d}.jsonl", encoding="utf-8")]
        inn = [json.loads(l) for l in open(OUT / f"items_noisy_{d}.jsonl", encoding="utf-8")]
        r = {"n": len(M), "n_expected": N_EXPECT[d], "ids_match_realsi_items": [m["id"] for m in M] == [i["id"] for i in items],
             "ids_match_items_files": [x["id"] for x in ic] == [x["id"] for x in inn] == [m["id"] for m in M],
             "refs_match": all(m["translation"] == i["trg_text"] for m, i in zip(M, items))}
        cache, exact, lens, bfile, peaks, maxdev = {}, 0, 0, [], [], 0.0
        for m, it in zip(M, items):
            if it["wav"] not in cache:
                x, _ = sf.read(it["wav"], dtype="float32", always_2d=True)
                cache = {it["wav"]: x.mean(axis=1)}
            ref = cache[it["wav"]][it["start"] * 16: it["end"] * 16]
            c, _ = sf.read(REPO / m["clean"], dtype="float32")
            y, _ = sf.read(REPO / m["noisy"], dtype="float32")
            exact += int(np.array_equal(c, ref))
            lens += int(len(c) == len(y) == len(ref))
            bd = y.astype(np.float64) - c.astype(np.float64) * m["limiter"]
            bf = 10 * math.log10((bd ** 2).sum() / ((c.astype(np.float64) * m["limiter"]) ** 2).sum())
            bfile.append(bf)
            maxdev = max(maxdev, abs(bf - m["bvr_target_db"]))
            peaks.append(float(np.abs(y).max()))
        r.update({"clean_cut_bitexact_vs_realsi_items": f"{exact}/{len(M)}", "lengths_equal": f"{lens}/{len(M)}",
                  "bvr_from_files_max_abs_dev_from_target_db": round(maxdev, 5),
                  "limiter_applied": sum(m["limiter"] < 1.0 for m in M), "peak_max": round(max(peaks), 4),
                  "peak_over_0.99": sum(p > 0.99 + 1e-4 for p in peaks)})
        Q = (5, 10, 25, 50, 75, 90, 95)
        lo, hi = np.quantile(bv, 0.05), np.quantile(bv, 0.95)
        src = bv[(bv >= lo) & (bv <= hi)]
        tgt = np.array([m["bvr_target_db"] for m in M])
        act = np.array([m["bvr_speech_active_db"] for m in M])
        from scipy.stats import ks_2samp
        r["bvr_quantiles_db"] = {"target": {q: round(float(np.percentile(tgt, q)), 2) for q in Q},
                                 "realised_from_files": {q: round(float(np.percentile(bfile, q)), 2) for q in Q},
                                 "speech_active_frames": {q: round(float(np.percentile(act, q)), 2) for q in Q},
                                 "obsi_statistic_q05_q95": {q: round(float(np.percentile(src, q)), 2) for q in Q}}
        r["ks_target_vs_obsi_q05_q95"] = {"stat": round(float(ks_2samp(tgt, src).statistic), 4),
                                          "p": round(float(ks_2samp(tgt, src).pvalue), 4)}
        dom = {}
        for m in M:
            dom[m["dominant"]] = dom.get(m["dominant"], 0) + 1
        r["dominant_layer_counts"] = dom
        r["layer_share_mean"] = {k: round(float(np.mean([m["layer_share"][k] for m in M])), 4) for k in ("music", "crowd", "sfx")}
        smp = rng.sample(range(len(M)), 8)
        S = []
        for k in smp:
            m = M[k]
            c, _ = sf.read(REPO / m["clean"], dtype="float32")
            y, _ = sf.read(REPO / m["noisy"], dtype="float32")
            S.append({"id": m["id"], "dur_s": round(len(y) / SR, 2), "clean_rms_dbfs": round(BB.active_rms_db(c), 2),
                      "noisy_rms_dbfs": round(BB.active_rms_db(y), 2), "noisy_peak": round(float(np.abs(y).max()), 4),
                      "frac_samples_over_0.98": round(float((np.abs(y) > 0.98).mean()), 6), "bvr_target_db": m["bvr_target_db"]})
        r["spot_check"] = S
        res[d] = r
        assert r["n"] == N_EXPECT[d] and r["ids_match_realsi_items"] and r["ids_match_items_files"] and r["refs_match"], r
        assert exact == len(M) and r["peak_over_0.99"] == 0 and maxdev < 0.01, r
    res["PASS"] = True
    with BB.x_open(out) as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({d: {k: res[d][k] for k in ("n", "clean_cut_bitexact_vs_realsi_items", "limiter_applied",
                                                  "bvr_quantiles_db", "ks_target_vs_obsi_q05_q95", "dominant_layer_counts")}
                      for d in ("en2zh", "zh2en")}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("build", "verify"))
    ap.add_argument("--procs", type=int, default=20)
    a = ap.parse_args()
    build(a) if a.step == "build" else verify(a)


if __name__ == "__main__":
    main()
