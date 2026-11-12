"""Three LAAL definitions on one code path, for the multi-speaker system (mist_w3po_v1 dev) and the single-speaker QABC
system (RealSI, the paper's stored outputs). CPU only, read-only on existing outputs; descriptive (no training, tuning
or selection on anything). Report: reports/mist_w3po_v1/laal_definitions_compare_v1.zh-CN.md.

All three use the LAAL of Papi et al. (2022): mean over i = 1..tau of (d_i - (i - 1) * |X| / max(|Y|, |Y*|)), tau = the
first i with d_i >= |X| (all of them if none). They differ in what one "i" is and when it is timed:

  D0  SimulEval default (simuleval/evaluator/scorers/latency_scorer.py LAALScorer, non-computation-aware, speech output;
      instance.py SpeechOutputInstance.receive_prediction): ONE delay per emitted non-empty output speech segment,
      d_i = source time read when the segment was emitted (step_to_delay; NOT its playback start -- the playback queue
      only enters EndOffset / RTF / discontinuity), |Y| = number of segments, |Y*| = reference length (latency_unit
      word; for a Chinese target the same unit count as D1, see D0_char for SimulEval's raw "char" count), and
      LAAL = d_1 if d_1 > |X|. laal_simuleval() is LAALScorer.compute verbatim.
  D1  segment LAAL (the single-speaker paper, appendix "Latency metric"): one delay per output unit (English word /
      Chinese character), d_i = playback START of the speech block carrying unit i (queue included), |Y| = output units.
  D0w sensitivity, SimulEval's arithmetic with one delay per output UNIT at its block's emission time (= D0 if every
      unit were its own speech segment; = SimulEval's text-output LAAL on the committed text; multi-speaker: the
      reports' text_laal_online on the speech unit set). D0 depends on how coarse the speech segments are; D0w does not.
  D2  speech_laal_online (mist_w3po_v1 reports, speech_latency_lit): one delay per output unit, d_i = when the unit has
      been SPOKEN, a block's units spread evenly over its playback ((k + 1) / n of the block), |Y| = output units.

Multi-speaker (speech_latency_lit's own block / segment / unit construction, metric_headroom.assign): the source is
one scored reference unit (a speaker's utterances < 1 s apart, as tcpBLEU), |X| = its duration, |Y*| = its reference
words; delays are relative to the unit start and NOT capped at |X| (the meeting goes on after the unit ends). A D0
segment is a talker block (= a text segment of the decode) with a piece assigned to the unit; its emission time is the
block's ONLINE commit (write time clamped by the window detector: >= open_detect, a final-read block >= close_detect).
D0_raw: the raw Thinker write time (no detector). D1_rec / D2_rec: the talker's recorded timeline (queue on raw write
times, no detector). Unit set for every speech definition: speech_latency_lit's speech_laal_online set (units with an
assigned block and a spoken word), so D2 here reproduces lat_*.json unit.speech_laal_online exactly.
Text only (--text runs, no talker): D0_text = one delay per assigned written piece's segment at its online commit
(D0_text_raw: raw write), text_laal = speech_latency_lit's text_laal_online.

Single-speaker (paper/bayes_simul_s2st_v11/scripts/collect_results.py realsi_points, same runs and filters): per
RealSI sentence, |X| = source_duration_ms, |Y*| = si_metrics.units(reference); D0 segments = blocks with speech
(dur_ms > 0) at min(write_ms, |X|) (SimulEval cannot read past the source); D1 = the paper's code; D2 as above.
There is no detector, so D0_raw = D0.

    /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python \
        eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112/laal_defs.py --out-dir <new dir> [--jobs 16]

Writes <out-dir>/<job>.json per job and <out-dir>/summary.json; refuses an existing out-dir.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.evaluation import metric_headroom as M  # noqa: E402  (also puts meeting_score on the path)
from experiments.mist_w3po_v1.evaluation import speech_latency_lit as SL  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal as si_laal, units as si_units  # noqa: E402

EO = "eval_outputs/mist_w3po_v1"
WIN_FULL = f"{EO}/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
WIN_DEV5 = f"{EO}/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl"

def multi_jobs() -> list:
    """(job name, system label, seed, decode dir, talker dir or None, windows, stored json to reproduce, its key)"""
    jobs = []
    for sys_, label, lat in (
            ("dev_crf_v2p_s3000_finalpost_s8_c2g06", "A' (c1b1s2)", f"{EO}/speech_laal_v1/Ap{{s}}.json"),
            ("dev_crf_v2p_s3000_finalpost_s8_c2g06__plem_c1b1s2", "A' + PL-EM",
             f"{EO}/talker_crf/eval/plem/lat_dev_crf_v2p_s3000_finalpost_s8_c2g06__plem_c1b1s2{{s}}.json"),
            ("dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem", "A'rl + PL-EM",
             f"{EO}/crf/t_eval/dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem/lat_dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem{{s}}.json"),
            ("dev_ve1_s3000__rlplem", "VE1rl + PL-EM", f"{EO}/crf/t_eval/dev_ve1_s3000__rlplem/lat_dev_ve1_s3000__rlplem{{s}}.json"),
            ("dev_ve1s2_s3000__rlplem", "VE1-s2rl + PL-EM",
             f"{EO}/crf/t_eval/dev_ve1s2_s3000__rlplem/lat_dev_ve1s2_s3000__rlplem{{s}}.json"),
            ("dev_ve1_s3000__segplem", "VE1 (un-merged) + PL-EM",
             f"{EO}/crf/t_eval/dev_ve1_s3000__segplem/lat_dev_ve1_s3000__segplem{{s}}.json")):
        for s in ("", "_tts2"):
            lj = lat.format(s=s)
            st_ = json.load(open(REPO / lj, encoding="utf-8"))   # the run / talker the stored number was computed on
            jobs.append({"job": f"multi__{sys_}{s}", "system": label, "seed": s or "_tts1", "run": st_["run"],
                         "talker": st_["talker"], "win": WIN_FULL, "stored": lj, "stored_key": "speech_laal_online"})
    # text only, dev5 (T5 / T4 have no talker); VE1 and A' on the same dev5 windows as references
    for job, label, run, stored in (
            ("text__dev5_t5_s3000", "T5 (text, dev5)", f"{EO}/decode/dev5_t5_s3000",
             f"{EO}/crf/t_eval/dev5_t5_s3000/textlat_dev5_t5_s3000.json"),
            ("text__dev5_t4_s3000", "T4 (text, dev5)", f"{EO}/decode/dev5_t4_s3000",
             f"{EO}/crf/t_eval/dev5_t4_s3000/textlat_dev5_t4_s3000.json"),
            ("text__dev5_ve1_s3000", "VE1 (text, dev5)", f"{EO}/decode/dev5_ve1_s3000",
             f"{EO}/crf/t_eval/dev5_ve1_s3000/textlat_dev5_ve1_s3000.json"),
            ("text__dev5cut_aprime", "A' (text, dev5 windows)", f"{EO}/decode/dev_crf_v2p_s3000_finalpost_s8_c2g06", None)):
        jobs.append({"job": job, "system": label, "seed": "", "run": run, "talker": None, "win": WIN_DEV5,
                     "stored": stored, "stored_key": "text_laal_online"})
    return jobs


SI_E = "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500"
SI_RESULTS = "paper/bayes_simul_s2st_v11/generated/results.json"   # the paper's collected points (segment LAAL = D1)
SI_RX = re.compile(r"system_(.+?)(-seed\d+(?:r\d*)?|-r\d*)?_m([\d.]+)")   # collect_results.RX
SI_FAMILY = {"swb-q-b1": "AmortRisk", "swb-q-b1-zh": "AmortRisk", "swb-q-b0": "AmortRisk-b0", "swb-q-b0-zh": "AmortRisk-b0",
             "swb-q-bm1": "AmortRisk-bm1", "swb-q-bm1-zh": "AmortRisk-bm1"}   # QABC (= ABC+SDR) with b0 = 1 / 0 / -1
SI_B0 = {"AmortRisk": 1, "AmortRisk-b0": 0, "AmortRisk-bm1": -1}
# SimulS2ST-Omni's sentence-level tiers (LAAL s, ASR-BLEU), as make_tables.OMNI (reported, not recomputed)
OMNI = {"en2zh": [(1.18, 19.80), (1.98, 24.21), (2.46, 25.01), (2.98, 25.54)],
        "zh2en": [(1.18, 14.89), (1.92, 19.55), (2.40, 21.07), (2.98, 22.47)]}


# ---------------------------------------------------------------- the formula
def laal_simuleval(delays: list, source_length: float, target_length: int) -> float:
    """D0: SimulEval LAALScorer.compute, verbatim (delays in emission order, any time unit)."""
    if delays[0] > source_length:
        return delays[0]
    LAAL = 0
    gamma = max(len(delays), target_length) / source_length
    tau = 0
    for t_minus_1, d in enumerate(delays):
        LAAL += d - t_minus_1 / gamma
        tau = t_minus_1 + 1
        if d >= source_length:
            break
    LAAL /= tau
    return LAAL


def laal_unitwise(d: list, X: float, ref_len: int) -> float:
    """D1 / D2: speech_latency_lit.laal (= si_metrics.laal on sorted delays) with |X| and |Y*| given directly."""
    return SL.laal(d, {"a": 0.0, "b": X, "ref": " ".join(["w"] * ref_len)})


# ---------------------------------------------------------------- multi-speaker
def online(w, wm):
    """speech_latency_lit.main's online commit (window detector clamp), verbatim."""
    c = max(wm, w["open_detect_ms"])
    if wm >= w["dur_ms"]:
        c = max(c, w["close_detect_ms"])
    return c


def unit_defs(pw: list, u: dict, speech: bool = True) -> dict:
    """all definitions for one scored unit u ({a, b, ref}) from its assigned pieces pw = [(segment, piece text)],
    sorted by segment start (speech_latency_lit's pw). A segment carries commit / raw_commit (s, absolute) and blk
    (the talker block, or None) with write / avail / on_start / on_end / rec_start / rec_end (s, absolute)."""
    a = u["a"]
    X = max(u["b"] - u["a"], 1e-3)
    T = len(u["ref"].split())
    out = {"X": X, "T": T}
    seen, segs = set(), []
    for x, t in pw:
        if id(x) not in seen and t.split():
            seen.add(id(x))
            segs.append(x)
    td = [x["commit"] - a for x, t in pw for _ in t.split()]
    if td:
        out["text_laal"] = SL.laal(td, u)
        out["text_laal_raw"] = SL.laal([x["raw_commit"] - a for x, t in pw for _ in t.split()], u)
        out["D0_text"] = laal_simuleval(sorted(x["commit"] - a for x in segs), X, T)
        out["D0_text_raw"] = laal_simuleval(sorted(x["raw_commit"] - a for x in segs), X, T)
        out["n_text_seg"] = len(segs)
    if not speech:
        return out
    sd = SL.speech_word_delays(pw, a)
    seen, bs = set(), []
    for x, t in pw:
        b = x.get("blk")
        if b is not None and id(b) not in seen:
            seen.add(id(b))
            bs.append(b)
    if not bs or not sd:
        return out
    # D1: every spoken word at its block's playback start (same word set as speech_word_delays)
    # D0w (sensitivity): every spoken word at its block's emission (online commit; _raw: raw write)
    d1, d1r, dw, dwr = [], [], [], []
    for x, t in pw:
        b = x.get("blk")
        if b is None or not b["text"].split() or not t.split():
            continue
        n = len(t.split())
        d1 += [b["on_start"] - a] * n
        d1r += [b["rec_start"] - a] * n
        dw += [b["avail"] - a] * n
        dwr += [b["write"] - a] * n
    out.update({"D0": laal_simuleval(sorted(b["avail"] - a for b in bs), X, T),
                "D0_raw": laal_simuleval(sorted(b["write"] - a for b in bs), X, T),
                "D0w": SL.laal(dw, u), "D0w_raw": SL.laal(dwr, u), "D1": SL.laal(d1, u), "D2": SL.laal(sd, u),
                "D1_rec": SL.laal(d1r, u), "D2_rec": SL.laal(SL.speech_word_delays(pw, a, "rec_start", "rec_end"), u),
                "n_seg": len(bs), "n_words": len(sd), "neg_first_delay": float(min(b["avail"] for b in bs) < a)})
    return out


def multi_job(J: dict) -> dict:
    """speech_latency_lit.main's blocks / segments / assignment, then unit_defs per unit."""
    from meeting_score import universe
    W = [json.loads(l) for l in open(REPO / J["win"], encoding="utf-8")]
    wins = {w["id"]: w for w in W}
    H = {h["id"]: h for h in map(json.loads, open(REPO / J["run"] / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(REPO / M.REF_WIN, encoding="utf-8")]
    U = universe(M.DR, {w["meeting"] for w in W})
    sc = M.scored_utts(W, RW)
    speech = J["talker"] is not None
    rows = {}
    for l in (open(REPO / J["talker"] / "results.jsonl", encoding="utf-8") if speech else []):
        r = json.loads(l)
        if not r.get("error"):
            rows[r["id"]] = r
    blk = {}
    for k, r in rows.items():
        w = wins.get(k.split("__S")[0])
        if w is None:
            continue
        t0, prev = w["start_s"], None
        for i, b in enumerate(r["blocks"]):
            av = online(w, b["write_ms"])
            stt = av if prev is None else max(av, prev)
            prev = stt + b["dur_ms"]
            blk[(k, i)] = {"write": t0 + b["write_ms"] / 1000, "avail": t0 + av / 1000,
                           "rec_start": t0 + b["start_ms"] / 1000, "rec_end": t0 + (b["start_ms"] + b["dur_ms"]) / 1000,
                           "on_start": t0 + stt / 1000, "on_end": t0 + prev / 1000, "text": b["text"]}
    segs = defaultdict(list)
    miss = 0
    for w in W:
        h = H.get(w["id"])
        if h is None:
            continue
        idx = defaultdict(int)
        for s in h["segments"]:
            if not s["text"].strip():
                continue
            k = f"{w['id']}__S{s['spk']}"
            i = idx[s["spk"]]
            idx[s["spk"]] += 1
            b = blk.get((k, i))
            if b is None or b["text"].strip() != s["text"].strip():
                miss += speech
                b = None
            segs[w["meeting"]].append({"spk": s["spk"], "a": w["start_s"] + s["start_s"],
                                       "b": w["start_s"] + max(s["end_s"], s["start_s"] + 0.08), "text": s["text"].strip(),
                                       "commit": w["start_s"] + online(w, s["write_ms"]) / 1000,
                                       "raw_commit": w["start_s"] + s["write_ms"] / 1000, "blk": b})
    recs = []
    for m in sorted(segs):
        A = M.assign(m, segs[m], sc, U[m])
        for uid, u in A["units"].items():
            pw = sorted(A["got"].get(uid, []), key=lambda xt: xt[0]["a"])
            if not pw:
                continue
            r = unit_defs(pw, u, speech)
            if len(r) > 2:
                recs.append({"uid": uid, "meeting": m, "reg": u["reg"], **r})
    return {"job": J, "segment_block_mismatch": miss, "units": recs}


# ---------------------------------------------------------------- single-speaker
def sentence_defs(r: dict) -> dict | None:
    """all definitions (ms) for one RealSI sentence of si_infer's results.jsonl; None where the paper skips it (no
    output unit)."""
    lang = r["tgt_lang"]
    X = r["source_duration_ms"]
    T = len(si_units(r["translation_ref"], lang))
    bl = r["blocks"]
    ds = [b["start_ms"] for b in bl for _ in si_units(b["text"], lang)]   # collect_results.realsi_points, verbatim
    if not ds:
        return None
    d2 = []
    for b in bl:
        n = len(si_units(b["text"], lang))
        d2 += [b["start_ms"] + (k + 1) / n * b["dur_ms"] for k in range(n)]
    seg = [min(b["write_ms"], X) for b in bl if b["dur_ms"] > 0]
    if not seg:   # no speech at all (does not occur in the stored runs)
        return None
    seg_u = [min(b["write_ms"], X) for b in bl if b["dur_ms"] > 0 and si_units(b["text"], lang)]
    dw = [min(b["write_ms"], X) for b in bl for _ in si_units(b["text"], lang)]
    out = {"X": X, "T": T, "D1": si_laal(ds, X, T), "D1_sl": laal_unitwise(ds, X, T), "D2": laal_unitwise(d2, X, T),
           "D0": laal_simuleval(seg, X, T), "D0w": si_laal(dw, X, T), "D0_unitblocks": laal_simuleval(seg_u, X, T), "n_seg": len(seg),
           "n_units": len(ds)}
    if lang == "cmn":   # SimulEval latency_unit char: len(reference.strip()), punctuation included
        out["D0_char"] = laal_simuleval(seg, X, len(r["translation_ref"].strip()))
    return out


def sentence_d0_chunks(r: dict, chunk_ms: float) -> float | None:
    """D0 if every block's speech were emitted as ceil(dur_ms / chunk_ms) segments at the block's write time (how a
    talker that streams fixed-length audio chunks would look to SimulEval); same sentence set as sentence_defs."""
    lang = r["tgt_lang"]
    X = r["source_duration_ms"]
    bl = r["blocks"]
    if not any(si_units(b["text"], lang) for b in bl):
        return None
    seg = [min(b["write_ms"], X) for b in bl if b["dur_ms"] > 0 for _ in range(-(-b["dur_ms"] // chunk_ms))]
    return laal_simuleval(seg, X, len(si_units(r["translation_ref"], lang))) if seg else None


def single_chunks_job(run: str, chunks=(80, 160, 320, 640, 1000)) -> dict:
    recs = [json.loads(l) for l in open(REPO / run / "results.jsonl", encoding="utf-8")]
    out = {}
    for c in chunks:
        v = [sentence_d0_chunks(r, c) for r in recs]
        out[f"D0_chunk{c}"] = float(np.mean([x for x in v if x is not None])) / 1000
    return out


def single_job(run: str) -> dict:
    run = Path(run)
    recs = [json.loads(l) for l in open(REPO / run / "results.jsonl", encoding="utf-8")]
    heard = {json.loads(l)["id"] for l in open(REPO / run / "asr_speech.jsonl", encoding="utf-8")}
    per = [sentence_defs(r) for r in recs]
    per = [p for p in per if p is not None]
    keys = [k for k in per[0] if k.startswith("D")]
    res = {k: float(np.mean([p[k] for p in per if k in p])) / 1000 for k in keys if any(k in p for p in per)}
    return {"run": str(run), "n_sent": len(recs), "n_scored": len(per), "asr_cover_ok": len(heard) >= len(recs) * 0.99,
            "mean_s": res, "seg_per_sent": float(np.mean([p["n_seg"] for p in per])),
            "units_per_sent": float(np.mean([p["n_units"] for p in per]))}


def single_runs() -> list:
    out = []
    for run in sorted(glob.glob(str(REPO / SI_E / "system_*_m*/realsi_*"))):
        run = Path(run)
        m = SI_RX.fullmatch(run.parent.name)
        if not m or m.group(1) not in SI_FAMILY or not (run / "asr_speech.jsonl").exists():
            continue
        out.append(str(run.relative_to(REPO)))
    return out


# ---------------------------------------------------------------- summaries
def q(a):
    a = np.asarray([x for x in a if x is not None], dtype=float)
    if not len(a):
        return None
    return {"mean": float(a.mean()), "p50": round(float(np.median(a)), 3), "p90": round(float(np.percentile(a, 90)), 3),
            "n": int(len(a))}


def summarize_multi(res: dict) -> dict:
    U = res["units"]
    keys = ["D0", "D0_raw", "D0w", "D0w_raw", "D1", "D2", "D1_rec", "D2_rec", "D0_text", "D0_text_raw", "text_laal",
            "text_laal_raw", "n_seg", "n_words",
            "n_text_seg", "X", "T", "neg_first_delay"]
    out = {}
    for k in keys:
        sel = [u for u in U if k in u]
        if sel:
            out[k] = {g: q([u[k] for u in sel if g == "all" or u["reg"] == g]) for g in ("all", "ov", "1spk")}
    return out


def tier_points() -> dict:
    """the paper's tier table (tab_realsi_main) QABC cell per (direction, tier): make_tables.best over the
    ABC+SDR families, by the paper's own D1 values."""
    R = json.load(open(REPO / SI_RESULTS, encoding="utf-8"))
    out = {}
    for d, tiers in OMNI.items():
        pts = [dict(p, fam=f) for f in SI_B0 for p in R["realsi"].get(f"{d}|{f}", [])]
        for k, (x, ob) in enumerate(tiers):
            ok = [p for p in pts if p["laal"] <= x + 1e-9]
            p = max(ok, key=lambda p: p["bleu"]) if ok else None
            out[f"{d}|m{k + 1}"] = {"budget": x, "omni_bleu": ob, "point": p}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--chunks-only", action="store_true",
                    help="only the single-speaker D0 granularity sensitivity (sentence_d0_chunks) -> <out-dir>/"
                         "single_d0_chunks.json (out-dir may exist; the file may not)")
    a = ap.parse_args()
    if a.chunks_only:
        out = Path(a.out_dir) / "single_d0_chunks.json"
        runs = single_runs()
        with ProcessPoolExecutor(a.jobs) as ex:
            res = dict(zip(runs, ex.map(single_chunks_job, runs)))
        groups = defaultdict(list)
        for run, v in res.items():
            p = Path(run)
            m = SI_RX.fullmatch(p.parent.name)
            groups[f"{p.name.split('_')[1]}|{SI_FAMILY[m.group(1)]}|{float(m.group(3)):g}"].append(v)
        means = {k: {c: round(st.mean(x[c] for x in v), 4) for c in v[0]} | {"n_seeds": len(v)} for k, v in sorted(groups.items())}
        with open(out, "x", encoding="utf-8") as fh:
            json.dump({"per_run": res, "seed_means": means}, fh, ensure_ascii=False, indent=1)
        print(json.dumps(means, indent=1))
        return
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=False)   # refuses an existing dir: never overwrites
    summary = {"multi": {}, "single": {}, "checks": {}}
    with ProcessPoolExecutor(a.jobs) as ex:
        fm = {J["job"]: ex.submit(multi_job, J) for J in multi_jobs()}
        runs = single_runs()
        fs = {r: ex.submit(single_job, r) for r in runs}
        for name, f in fm.items():
            res = f.result()
            with open(od / f"{name}.json", "x", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False)
            J = res["job"]
            s = summarize_multi(res)
            s["segment_block_mismatch"] = res["segment_block_mismatch"]
            summary["multi"][name] = {"system": J["system"], "seed": J["seed"], "run": J["run"], "talker": J["talker"],
                                      "win": J["win"], "defs": s}
            if J["stored"]:
                stored = json.load(open(REPO / J["stored"], encoding="utf-8"))["unit"][J["stored_key"]]["all"]
                mine = s["D2" if J["talker"] else "text_laal"]["all"]
                summary["checks"][name] = {"stored": J["stored"], "key": J["stored_key"], "stored_mean": stored["mean"],
                                           "stored_n": stored["n"], "mine_mean": round(mine["mean"], 3), "mine_n": mine["n"],
                                           "ok": abs(round(mine["mean"], 3) - stored["mean"]) < 1e-9 and mine["n"] == stored["n"]}
            print(name, json.dumps(summary["checks"].get(name)), flush=True)
        single = {r: f.result() for r, f in fs.items()}
    with open(od / "single_runs.json", "x", encoding="utf-8") as fh:
        json.dump(single, fh, ensure_ascii=False, indent=1)
    # single: seed means per (direction, family, read window), as collect_results.realsi_points
    groups = defaultdict(list)
    for run, v in single.items():
        if not v["asr_cover_ok"]:
            continue
        p = Path(run)
        m = SI_RX.fullmatch(p.parent.name)
        groups[(p.name.split("_")[1], SI_FAMILY[m.group(1)], float(m.group(3)))].append(v)
    R = json.load(open(REPO / SI_RESULTS, encoding="utf-8"))
    for (d, fam, w), v in sorted(groups.items()):
        keys = sorted({k for x in v for k in x["mean_s"]})
        e = {"direction": d, "family": fam, "b0": SI_B0[fam], "window_s": w, "n_seeds": len(v),
             **{k: round(st.mean(x["mean_s"][k] for x in v), 4) for k in keys},
             "seg_per_sent": round(st.mean(x["seg_per_sent"] for x in v), 2),
             "units_per_sent": round(st.mean(x["units_per_sent"] for x in v), 2)}
        paper = [p for p in R["realsi"].get(f"{d}|{fam}", []) if p["m"] == w]
        e["paper_laal_D1"] = paper[0]["laal"] if paper else None
        e["paper_bleu"] = paper[0]["bleu"] if paper else None
        e["D1_reproduces_paper"] = bool(paper) and round(e["D1"], 2) == paper[0]["laal"]
        summary["single"][f"{d}|{fam}|{w:g}"] = e
    summary["checks"]["single_D1_all_reproduce"] = all(e["D1_reproduces_paper"] for e in summary["single"].values()
                                                       if e["paper_laal_D1"] is not None)
    summary["checks"]["single_D1_eq_D1_sl"] = all(abs(e["D1"] - e["D1_sl"]) < 1e-9 for e in summary["single"].values())
    summary["tiers"] = tier_points()
    for k, t in summary["tiers"].items():
        p = t["point"]
        t["ours"] = None if p is None else summary["single"].get(f"{k.split('|')[0]}|{p['fam']}|{p['m']:g}")
    with open(od / "summary.json", "x", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps(summary["checks"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
