"""Paper v10 tables, macros and statistics (docs/mist_v10_plan.zh-CN.md §4.0). Reads evaluation outputs only (never writes
outside paper/gstar_si_v10/generated):

  eval_outputs/mist_w3po_v1/decode/<run>/score_text.json, <run>/score_asr.json        overlap scorer (per-meeting stats)
  eval_outputs/mist_w3po_v1/crf/t_eval/<run>/textlat_chunk_<run>.json                 chunk-protocol text LAAL
  eval_outputs/mist_w3po_v1/crf/t_eval/<speech run>/lat_chunk_<speech run>{,_tts2}.json chunk-protocol speech latency
  eval_outputs/mist_w3po_v1/joint/{test_summary_v1,wb_test_summary_v1}.json            joint test summaries (cross-check)
  eval_outputs/mist_w3po_v1/joint/joint_v2p*/selection.json                           stage-2 / plan-B checkpoint rules
  logs/mist_w3po_v1/joint/wb/summary.tsv                                              write-bias calibration (dev5)
  logs/mist_w3po_v1/joint_v2p{,l}_*.log                                               per-step posterior statistics
  eval_outputs/mist_w3po_v1/crf/t_eval/final_test_v1/{ami/table_v1,ali/table_v1_g1}.json  talker ablation (separate zh->en)
  generated/v10_inputs/*_online.json                                                  block-start ("single-speaker") LAAL
  eval_outputs/mist_w3po_v1/decode/jtest_{ami,ali,en}_seamless_default/...           E7 (optional, other agent)

Every number of the v10 text is a \\num{v10:...} macro written here. A missing input never fails the script: its macros
print as \\textbf{??} and the key + source file are listed in generated/v10_missing.json. Numbers that exist only in a
report's prose are constants below, each with the report file and section it comes from.

Statistics (E3): per test set, paired comparisons over meetings on the BLEU sufficient statistics: exact two-sided
sign-flip permutation p over all 2^M flips (M = 16 / 20 meetings), and a 95% interval from 10,000 meeting-bootstrap
resamples. Speech items pool the two TTS seeds per meeting (mean of the per-meeting statistics, as gate_2seed). Holm over
the family of test comparisons (12 tests: 3 sets x {text all M-B, speech M-B, speech E2-B, speech M-E2}).

    python scripts/make_tables_v10.py          (from the paper folder; needs numpy)
"""
from __future__ import annotations

import glob
import json
import math
import re
import shutil
from decimal import ROUND_HALF_UP, Decimal
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
GEN = PAPER / "generated"
INP = GEN / "v10_inputs"
ROOT = PAPER.parents[1]
EO = ROOT / "eval_outputs/mist_w3po_v1"
DEC = EO / "decode"
TE = EO / "crf/t_eval"
JO = EO / "joint"
LOGS = ROOT / "logs/mist_w3po_v1"
M2SRC = Path("/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/88583ca0-585a-4fad-ada7-4b193b743b50/"
             "scratchpad/m2laal/out")

PH = r"\textbf{??}"
MAC: dict[str, str] = {}
MISSING: dict[str, str] = {}
RNG = np.random.default_rng(20261114)
NBOOT = 10000

RT_B = "joint_v8h6k_20261113T073658Z_s06000"          # stage 1: joint deadline SFT, iter 6000 (strict baseline)
RT_M = "joint_v2p_20261113T120415Z_s03500"            # stage 2: joint SerialCRF v2', step 3500
RT_W = RT_M + "_wbzm1em1"                             # + write bias delta_w = -1 (both directions)
DELTA_W = -1                                          # fixed decoding constant (LOG 2026-11-14 12:45 Prague)
SETS = ("ami", "ali", "en")
DIRS = ("z", "e")
SYSS = ("b", "e2", "m0", "m")


# ---------------------------------------------------------------------------------------------------------- helpers
def j(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def hfmt(fmt, v):
    """format with round-half-up on the decimal representation (the reports round this way: 19.235 -> 19.24)."""
    m = re.fullmatch(r"\{:(\+?)\.(\d)f\}", fmt)
    if m is None or not isinstance(v, float):
        return fmt.format(v)
    q = Decimal(f"{float(v):.9f}").quantize(Decimal(1).scaleb(-int(m.group(2))), rounding=ROUND_HALF_UP)
    s = f"{q:.{m.group(2)}f}"
    if s.startswith("-") and float(q) == 0:
        s = s[1:]
    return ("+" + s if m.group(1) and not s.startswith("-") else s)


def put(key, v, fmt="{:.2f}", src=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        MAC[key] = PH
        MISSING[key] = str(src)
    elif isinstance(v, str):
        MAC[key] = v
    else:
        s = hfmt(fmt, v)
        if s.startswith("-"):
            s = "$-$" + s[1:]
        MAC[key] = s


def putd(key, v, fmt="{:+.2f}", src=""):
    """signed difference: +0.12 / $-$0.12"""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        MAC[key] = PH
        MISSING[key] = str(src)
        return
    s = hfmt(fmt, float(v))
    MAC[key] = s.replace("-", "$-$") if s.startswith("-") else s


def mean2(a, b):
    return None if a is None or b is None else (a + b) / 2.0


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def bleu_vec(S):
    """corpus BLEU (sacrebleu defaults: exp smoothing, 4-grams) of each row of S (N x 10 sacrebleu statistics);
    same function as scripts/bootstrap_v7.bleu_vec."""
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    hyp, ref, cor, tot = S[:, 0], S[:, 1], S[:, 2:6], S[:, 6:10]
    k = np.cumsum(cor == 0, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(cor > 0, 100.0 * cor / np.maximum(tot, 1), 100.0 / (np.power(2.0, k) * np.maximum(tot, 1)))
        p = np.where(tot > 0, p, 0.0)
        logp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), -np.inf)
        bp = np.where(hyp < ref, np.exp(1 - ref / np.maximum(hyp, 1e-9)), 1.0)
        out = np.where(hyp > 0, bp * np.exp(logp.mean(1)), 0.0)
    return np.nan_to_num(out, nan=0.0, neginf=0.0)


def fp(p):
    return PH if p is None else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


# ---------------------------------------------------------------------------------------------------------- systems
def sysdef():
    """name map: (split, set/dir, system) -> text run, speech run"""
    D = {}
    for d in DIRS:
        D[("dev", d, "b")] = (f"jdev_b{d}_{RT_B}", f"jdev_b{d}_{RT_B}__t1c")
        D[("dev", d, "e2")] = (f"jdev_b{d}_{RT_B}", f"jdev_b{d}_{RT_B}__rlplem")
        D[("dev", d, "m0")] = (f"jdev_m{d}_{RT_M}", f"jdev_m{d}_{RT_M}__rlplem")
        D[("dev", d, "m")] = (f"jdev_m{d}_{RT_W}", f"jdev_m{d}_{RT_W}__rlplem")
    for s in SETS:
        D[("test", s, "b")] = (f"jtest_b{s}_{RT_B}", f"jtest_b{s}_{RT_B}__t1c")
        D[("test", s, "e2")] = (f"jtest_b{s}_{RT_B}", f"jtest_b{s}_{RT_B}__rlplem")
        D[("test", s, "m0")] = (f"jtest_m{s}_{RT_M}", f"jtest_m{s}_{RT_M}__rlplem")
        D[("test", s, "m")] = (f"jtest_m{s}_{RT_W}", f"jtest_m{s}_{RT_W}__rlplem")
    return D


def load_system(text, speech):
    """all numbers of one system; None where an input is missing."""
    t = j(DEC / text / "score_text.json")
    tl = j(TE / text / f"textlat_chunk_{text}.json")
    seeds = [speech, speech + "_tts2"]
    asr = [j(DEC / s / "score_asr.json") for s in seeds]
    lat = [j(TE / speech / f"lat_chunk_{s}.json") for s in seeds]
    r = {"src_text": DEC / text / "score_text.json", "src_tlat": TE / text / f"textlat_chunk_{text}.json",
         "src_asr": DEC / speech / "score_asr.json", "src_lat": TE / speech / f"lat_chunk_{speech}.json",
         "score_text": t, "asr_runs": asr}
    r["all"], r["ov"], r["1spk"] = g(t, "tcpBLEU_all"), g(t, "tcpBLEU_ov"), g(t, "tcpBLEU_1spk")
    r["pre"] = g(t, "pre_end_share")
    r["dup"] = g(t, "duplicate_rate")
    r["cr"] = g(t, "CR")
    r["meetings"] = g(t, "meetings")
    r["tlaal"] = g(tl, "unit", "text_laal_online", "all", "mean")
    both = all(a is not None for a in asr)
    r["asr"] = mean2(*[g(a, "tcpBLEU_all") for a in asr]) if both else None
    r["asr_ov"] = mean2(*[g(a, "tcpBLEU_ov") for a in asr]) if both else None
    bothl = all(x is not None for x in lat)

    def lm(*ks):
        if not bothl:
            return None
        v = [g(x, *ks) for x in lat]
        return None if any(y is None for y in v) else sum(v) / len(v)
    r["slaal"] = lm("unit", "speech_laal_online", "all", "mean")
    r["endoff"] = lm("unit", "speech_end_online", "all", "mean")
    r["sstart"] = lm("unit", "speech_start_online", "all", "mean")
    r["tlaal_sp"] = lm("unit", "text_laal_online", "all", "mean")
    for k in ("model_lag", "detector_wait", "queue_wait", "last_block_dur"):
        r["p_" + k] = lm("unit_end_parts_online", k, "mean")
    r["load"] = lm("load", "speech_over_src_union")
    r["blockdur"] = lm("load", "block_dur", "mean")
    # BLEU = BP x precision part, from the summed sufficient statistics
    if t is not None:
        S = np.array([v["all_stats"] for v in t["per_meeting"].values() if v.get("all_stats")], float).sum(0)
        hyp, ref = S[0], S[1]
        bp = 1.0 if hyp >= ref else math.exp(1 - ref / max(hyp, 1e-9))
        r["lenratio"], r["bp"] = hyp / ref, bp
        r["nobp"] = float(bleu_vec(S)[0]) / bp
    else:
        r["lenratio"] = r["bp"] = r["nobp"] = None
    return r


# ---------------------------------------------------------------------------------------------------------- stats
def per_meeting(score, reg):
    return {m: np.asarray(v.get(f"{reg}_stats") or [0.0] * 10, float) for m, v in score["per_meeting"].items()}


def pooled(scores, reg):
    ms = set.intersection(*[set(s["per_meeting"]) for s in scores])
    return {m: np.mean([np.asarray(s["per_meeting"][m].get(f"{reg}_stats") or [0.0] * 10, float) for s in scores], 0)
            for m in ms}


def paired(a: dict, b: dict):
    """B - A on meeting-level BLEU statistics: exact two-sided sign-flip p, 95% bootstrap interval."""
    ms = sorted(set(a) & set(b))
    if len(ms) < 3:
        return None
    A = np.stack([a[m] for m in ms])
    B = np.stack([b[m] for m in ms])
    obs = float(bleu_vec(B.sum(0))[0] - bleu_vec(A.sum(0))[0])
    M_ = len(ms)
    idx = np.arange(2 ** M_, dtype=np.int64)
    F = ((idx[:, None] >> np.arange(M_)[None, :]) & 1).astype(np.float64)
    D = B - A
    Bs = B.sum(0)[None, :] - F @ D            # flip: meeting swaps its two systems
    As = A.sum(0)[None, :] + F @ D
    d = bleu_vec(Bs) - bleu_vec(As)
    p = float((np.abs(d) >= abs(obs) - 1e-9).mean())
    J = RNG.integers(0, M_, (NBOOT, M_))
    ds = bleu_vec(B[J].sum(1)) - bleu_vec(A[J].sum(1))
    return {"diff": round(obs, 2), "p": p, "ci": [round(float(np.percentile(ds, 2.5)), 2),
                                                  round(float(np.percentile(ds, 97.5)), 2)], "meetings": M_}


def holm(tests):
    ks = [k for k, v in tests.items() if v is not None]
    order = sorted(ks, key=lambda k: tests[k]["p"])
    m, run = len(order), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * tests[k]["p"]))
        tests[k]["p_holm"] = run
    return tests


def speech_scores(r):
    return r["asr_runs"] if all(x is not None for x in r["asr_runs"]) else None


# ---------------------------------------------------------------------------------------------------------- inputs
def copy_m2():
    """copy the block-start LAAL jsons from the scratchpad into generated/v10_inputs (never overwrite)."""
    INP.mkdir(parents=True, exist_ok=True)
    if M2SRC.exists():
        for f in sorted(M2SRC.glob("*_online.json")):
            dst = INP / f.name
            if not dst.exists():
                shutil.copy2(f, dst)


def parse_trainlog(p, hi=3500):
    rows = {}
    if not Path(p).exists():
        return None
    for line in open(p, errors="ignore"):
        i = line.find("{")
        if i < 0 or '"q_label"' not in line:
            continue
        try:
            r = json.loads(line[i:])
        except Exception:
            continue
        if "step" in r:
            rows[r["step"]] = r
    s = [rows[k] for k in sorted(rows) if k <= hi]
    if not s:
        return None
    W = sum(r["windows"] for r in s)
    out = {"steps": len(s), "q": float(np.mean([r["q_label"] for r in s])),
           "defer_win": sum(r["exp_defer"] for r in s) / W, "defer_step": float(np.mean([r["exp_defer"] for r in s])),
           "win_step": W / len(s)}
    first = [r for r in s if r["step"] <= 100]
    last = [r for r in s if r["step"] > hi - 500]
    out["q_first"] = float(np.mean([r["q_label"] for r in first]))
    out["q_last"] = float(np.mean([r["q_label"] for r in last]))
    out["defer_win_first"] = sum(r["exp_defer"] for r in first) / sum(r["windows"] for r in first)
    out["defer_win_last"] = sum(r["exp_defer"] for r in last) / sum(r["windows"] for r in last)
    if "lat_E0" in s[0]:
        out["E0"] = float(np.mean([r["lat_E0"] for r in s])) / out["win_step"]
        out["E"] = float(np.mean([r["lat_E"] for r in s])) / out["win_step"]
        out["E0_first"] = float(np.mean([r["lat_E0"] / r["windows"] for r in first]))
        out["E0_last"] = float(np.mean([r["lat_E0"] / r["windows"] for r in last]))
        out["lam"] = float(np.mean([r["lat_lam_mean"] for r in s]))
    return out


# ---------------------------------------------------------------------------------------------------------- constants
# Numbers that exist only in report prose (not in a machine-readable output). Each cites its source.
C = {
    # reports/mist_w3po_v1/aprime_plem_crf_formal_and_latency.zh-CN.md §2.1 (separate zh->en A' system, dev, 5421 units)
    "dec:units": "5{,}421", "dec:unitdur": 3.48, "dec:unitp50": 2.11, "dec:short": 48,
    "dec:raw": 2.77, "dec:online": 3.47, "dec:speech": 4.15, "dec:det": 0.70, "dec:play": 0.68, "dec:final": 56,
    # same report §2.3 (upper bound: same text, commit at unit end + lag)
    "ub:l03": 0.26, "ub:h03": 0.36, "ub:h0": 0.73,
    # reports/mist_w3po_v1/joint_baseline_vs_method_20261114.zh-CN.md "方法为什么开口更晚" table (reads with output, %)
    "rw:devz:b": 45.9, "rw:devz:m0": 36.2, "rw:deve:b": 57.2, "rw:deve:m0": 48.6,
    "rw:ami:b": 44.1, "rw:ami:m0": 35.8, "rw:ali:b": 58.1, "rw:ali:m0": 38.0, "rw:en:b": 55.6, "rw:en:m0": 47.0,
    # same report, same section: median first-write lag (s), baseline -> method (delta_w = 0)
    "fw:devz:b": 1.96, "fw:devz:m0": 2.56, "fw:ali:b": 2.26, "fw:ali:m0": 3.00,
    # same report §C (AliMeeting / AMI test, delta_w = 0): duplicate words and de-duplicated BLEU (diagnostic script)
    "dup:ali:b:words": "27{,}697", "dup:ali:m0:words": "12{,}689", "dup:ali:b:att": "85{,}053",
    "dup:ali:b:share": 33, "dup:ali:b:nondup": "57{,}356", "dup:ali:m0:nondup": "69{,}881", "dup:ali:gain": 22,
    "dup:ami:b:words": "5{,}103", "dup:ami:m0:words": "3{,}667",
    "dd:ali:b": 13.16, "dd:ali:m0": 15.49, "dd:ami:b": 22.53, "dd:ami:m0": 22.95, "dd:devz:b": 25.62, "dd:devz:m0": 26.15,
    # same report §D: Ali text drop also in the separate v2' systems (A' s3000 18.14, VE1 18.74 vs 21.43)
    "ali:sepA": 18.14,
    # PMC (earlier-write lattice [d-1, d+1] + e-path budget), separate zh->en system, full dev:
    # reports/mist_w3po_v1/LOG.zh-CN.md 2026-11-12 13:57 Prague (vs A' / A'rl+PL-EM), chunk LAAL from
    # reports/mist_w3po_v1/latency_chunk_protocol_table_v1.zh-CN.md §3
    "pmc:text": -1.92, "pmc:ov": -2.53, "pmc:asr": -1.70, "pmc:slaal": 2.86, "pmc:ref": 3.86,
    # PL-EM training: reports/mist_w3po_v1/talker_crf_results_v1.zh-CN.md §2 (zh->en) and LOG 2026-11-14 07:05 (En->Zh)
    "plem:lam:z": 0.497, "plem:lam:e": 0.703, "plem:blocks:z": "19{,}077", "plem:rows:z": "111{,}487",
    "plem:rows:e": "190{,}995", "plem:steps:z": "1{,}200", "plem:it:e": 400, "plem:meet:z": 15, "plem:budgetoff": 0.3,
    # joint training: reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md §2.1 and amendment 4
    "j:rows": "960{,}000", "j:s1steps": "6{,}000", "j:s2step": "3{,}500", "j:s2rule": "4{,}000",
    # pre-registered selection threshold (same prereg §5): feasible if dev5 text LAAL <= ref + 0.1 s
    "sel:tol": 0.1,
    # SerialCRF v2' scale (reports/mist_w3po_v1/serial_v2p_train.zh-CN.md §1 "规模", separate zh->en run, CPU check of
    # all 27,953 training windows): label-path rows identical to deadline rows; windows with a real order choice; rows
    "crf:win": "27{,}953", "crf:realwin": "6{,}801", "crf:pass1": 61.5, "crf:pass2": 100.4,
    # PL-EM objective (reports/mist_w3po_v1/aprime_plem_crf_formal_and_latency.zh-CN.md §1.2): 35 rate tokens, weights
    # 1.0 (rate) / 0.5 (KL to T1c); candidates with q >= 0.01 kept (talker_crf_results_v1 §2); brute-force check 1e-9
    "plem:rates": 35, "plem:wrate": 1.0, "plem:wkl": 0.5, "plem:minq": 0.01,
    # chunk protocol (reports/mist_w3po_v1/latency_chunk_protocol_table_v1.zh-CN.md §1-§2): 12 systems re-scored,
    # identical rankings; speech LAAL shifts 0.10-0.30 s; open 316 ms / close 401 ms earlier on average (dev)
    "chunk:nsys": 12, "chunk:shiftlo": 0.10, "chunk:shifthi": 0.30, "chunk:open": 316, "chunk:close": 401,
    # reports/mist_w3po_v1/joint_baseline_vs_method_20261114.zh-CN.md §D: stage-2 dev5 BLEU without BP and length ratio
    "s2:nobp3000": 29.85, "s2:nobp3500": 30.05, "s2:nobp4000": 30.17, "s2:len3500": "0.934", "s2:len4000": "0.922",
    # stage-2 rule override (LOG 2026-11-13 21:31 Prague): user choice on dev numbers only, before any s4000 method test job
    "ovr:time": "21:31",
    # talker ablation inference config: c1+b1+s2 = queue-adaptive rate, 2 samples + CTC choice, +2 rate steps
    # (joint_bidir_plan_prereg_v1 amendment 3 §2)
}


# ---------------------------------------------------------------------------------------------------------- main
def main():
    GEN.mkdir(exist_ok=True)
    copy_m2()
    SD = sysdef()
    R = {k: load_system(*v) for k, v in SD.items()}

    # --- per-system macros
    fields = (("all", "{:.2f}", "src_text"), ("ov", "{:.2f}", "src_text"), ("1spk", "{:.2f}", "src_text"),
              ("asr", "{:.2f}", "src_asr"), ("asr_ov", "{:.2f}", "src_asr"), ("tlaal", "{:.2f}", "src_tlat"),
              ("slaal", "{:.2f}", "src_lat"), ("endoff", "{:.2f}", "src_lat"), ("sstart", "{:.2f}", "src_lat"),
              ("p_model_lag", "{:.2f}", "src_lat"), ("p_detector_wait", "{:.2f}", "src_lat"),
              ("p_queue_wait", "{:.2f}", "src_lat"), ("p_last_block_dur", "{:.2f}", "src_lat"),
              ("load", "{:.2f}", "src_lat"), ("lenratio", "{:.3f}", "src_text"), ("bp", "{:.3f}", "src_text"),
              ("nobp", "{:.2f}", "src_text"), ("dup", "{:.1f}", "src_text"), ("pre", "{:.1f}", "src_text"),
              ("cr", "{:.1f}", "src_text"), ("meetings", "{:d}", "src_text"))
    for (sp, s, sy), r in R.items():
        for f, fmt, src in fields:
            v = r[f]
            if f in ("dup", "pre", "cr") and v is not None:
                v = 100 * v
            put(f"v10:{sp}:{s}:{sy}:{f}", v, fmt, r[src])
    # differences
    for (sp, s, sy), r in R.items():
        for ref in SYSS:
            if ref == sy:
                continue
            q = R[(sp, s, ref)]
            for f in ("all", "ov", "1spk", "asr", "tlaal", "slaal", "endoff", "sstart", "p_queue_wait"):
                a, b = r[f], q[f]
                putd(f"v10:{sp}:{s}:{sy}-{ref}:{f}", None if a is None or b is None else a - b,
                     src=r["src_lat"] if a is None else q["src_lat"])
    # ranges used in the abstract / intro (test, MIST v10 vs strict baseline)
    def rng_of(f, sets, a="m", b="b", fmt="{:.2f}"):
        v = [None if R[("test", s, a)][f] is None or R[("test", s, b)][f] is None
             else R[("test", s, a)][f] - R[("test", s, b)][f] for s in sets]
        return v
    for f in ("asr", "slaal", "endoff", "all"):
        v = rng_of(f, ("ami", "en"))
        src = "E1 test files (AMI Zh->En / En->Zh)"
        put(f"v10:test:amiboth:m-b:{f}:min", None if None in v else min(v), "{:.2f}", src)
        put(f"v10:test:amiboth:m-b:{f}:max", None if None in v else max(v), "{:.2f}", src)
        va = [None if x is None else abs(x) for x in v]
        put(f"v10:test:amiboth:m-b:{f}:absmin", None if None in va else min(va), "{:.2f}", src)
        put(f"v10:test:amiboth:m-b:{f}:absmax", None if None in va else max(va), "{:.2f}", src)

    # --- statistics (E3 on test; descriptive on dev)
    tests, info = {}, {}
    for sp, keys in (("test", SETS), ("dev", DIRS)):
        for s in keys:
            B, E2, M0, M = (R[(sp, s, x)] for x in SYSS)
            comps = {}
            if B["score_text"] is not None and M["score_text"] is not None:
                comps["text:m-b"] = paired(per_meeting(B["score_text"], "all"), per_meeting(M["score_text"], "all"))
                comps["textov:m-b"] = paired(per_meeting(B["score_text"], "ov"), per_meeting(M["score_text"], "ov"))
            if B["score_text"] is not None and M0["score_text"] is not None:
                comps["text:m0-b"] = paired(per_meeting(B["score_text"], "all"), per_meeting(M0["score_text"], "all"))
            sb, se, sm0, sm = (speech_scores(x) for x in (B, E2, M0, M))
            for name, a, b in (("speech:m-b", sb, sm), ("speech:e2-b", sb, se), ("speech:m-e2", se, sm),
                               ("speech:m0-b", sb, sm0)):
                if a is not None and b is not None:
                    comps[name] = paired(pooled(a, "all"), pooled(b, "all"))
            for name, v in comps.items():
                info[f"{sp}:{s}:{name}"] = v
                if sp == "test" and name in ("text:m-b", "speech:m-b", "speech:e2-b", "speech:m-e2"):
                    tests[f"{s}:{name}"] = v
    holm(tests)
    FAMILY = [f"{s}:{n}" for s in SETS for n in ("text:m-b", "speech:m-b", "speech:e2-b", "speech:m-e2")]
    for key in [f"{sp}:{s}:{n}" for sp, ks in (("test", SETS), ("dev", DIRS)) for s in ks
                for n in ("text:m-b", "textov:m-b", "text:m0-b", "speech:m-b", "speech:e2-b", "speech:m-e2",
                          "speech:m0-b")]:
        v = info.get(key)
        mk = "v10:sig:" + key
        src = "score files of " + key
        putd(mk + ":diff", None if v is None else v["diff"], src=src)
        if v is None:
            put(mk + ":ci", None, src=src)
            put(mk + ":p", None, src=src)
        else:
            lo, hi = v["ci"]
            MAC[mk + ":ci"] = "[" + f"{lo:.2f}".replace("-", "$-$") + ", " + f"{hi:.2f}".replace("-", "$-$") + "]"
            MAC[mk + ":p"] = fp(v["p"])
        tk = key.split(":", 1)[1]
        if key.startswith("test:") and tk in tests:
            t = tests[tk]
            MAC[mk + ":holm"] = PH if t is None or "p_holm" not in t else fp(t["p_holm"])
    put("v10:sig:ntests", len([k for k in FAMILY if tests.get(k) is not None]), "{:d}")
    json.dump({"tests": tests, "info": info}, open(GEN / "v10_tests.json", "w"), indent=1, default=str)

    # --- calibration (dev5 text), delta = 0 method from the stage-2 proxy, plan B
    cal = {}
    tsv = LOGS / "joint/wb/summary.tsv"
    if tsv.exists():
        for line in open(tsv).read().splitlines()[1:]:
            n, a, o, one, l = line.split("\t")
            cal[n] = (float(a), float(l))
    for d, dd in (("z", "z"), ("e", "e")):
        b = cal.get(f"wb{d}_dev5_base_s6000")
        put(f"v10:cal:{d}:b:bleu", None if b is None else b[0], src=tsv)
        put(f"v10:cal:{d}:b:laal", None if b is None else b[1], src=tsv)
        for k in (1, 2, 3, 5):
            v = cal.get(f"wb{d}_dev5_m3500_bm{k}")
            put(f"v10:cal:{d}:m{k}:bleu", None if v is None else v[0], src=tsv)
            put(f"v10:cal:{d}:m{k}:laal", None if v is None else v[1], src=tsv)
        n0 = f"s2{dd}_dev5_{RT_M}"
        t0 = j(DEC / n0 / "score_text.json")
        l0 = j(TE / n0 / f"textlat_chunk_{n0}.json")
        put(f"v10:cal:{d}:m0:bleu", g(t0, "tcpBLEU_all"), src=DEC / n0)
        put(f"v10:cal:{d}:m0:laal", g(l0, "unit", "text_laal_online", "all", "mean"), src=TE / n0)
        # plan B (rho = 0.5) at every evaluated step
        for st in range(3000, 6500, 500):
            n = f"s2{dd}_dev5_joint_v2pl_20261114T130043Z_s{st:05d}"
            t = j(DEC / n / "score_text.json")
            lt = j(TE / n / f"textlat_chunk_{n}.json")
            put(f"v10:pb:{d}:{st}:bleu", g(t, "tcpBLEU_all"), src=DEC / n)
            put(f"v10:pb:{d}:{st}:laal", g(lt, "unit", "text_laal_online", "all", "mean"), src=TE / n)
        # stage-2 v2' steps (chunk text LAAL; rule protocol numbers below)
        for st in (3000, 3500, 4000):
            n = f"s2{dd}_dev5_joint_v2p_20261113T120415Z_s{st:05d}"
            t = j(DEC / n / "score_text.json")
            lt = j(TE / n / f"textlat_chunk_{n}.json")
            put(f"v10:s2:{d}:{st}:bleu", g(t, "tcpBLEU_all"), src=DEC / n)
            put(f"v10:s2:{d}:{st}:laal", g(lt, "unit", "text_laal_online", "all", "mean"), src=TE / n)
    for d in DIRS:
        bl = cal.get(f"wb{d}_dev5_base_s6000")
        for k in ("m1",):
            v = cal.get(f"wb{d}_dev5_m3500_b{k}")
            putd(f"v10:cal:{d}:{k}-b:laal", None if v is None or bl is None else v[1] - bl[1], src=tsv)
            putd(f"v10:cal:{d}:{k}-b:bleu", None if v is None or bl is None else v[0] - bl[0], src=tsv)
    put("v10:delta", DELTA_W, "{:d}")

    # selection files
    for tag, run in (("s2", "joint_v2p_20261113T120415Z"), ("pb", "joint_v2pl_20261114T130043Z")):
        sel = j(JO / run / "selection.json")
        put(f"v10:{tag}:chosen", g(sel, "chosen"), "{:,}", src=JO / run / "selection.json")
        put(f"v10:{tag}:decision", g(sel, "decision_step"), "{:,}", src=JO / run / "selection.json")
        for dk, d in (("zh2en", "z"), ("en2zh", "e")):
            put(f"v10:{tag}:refL:{d}", g(sel, "ref_L", dk), "{:.3f}", src=JO / run / "selection.json")
        for p in (sel or {}).get("points", []):
            if p.get("status") != "ok":
                continue
            st = p["step"]
            for dk, d in (("zh2en", "z"), ("en2zh", "e")):
                put(f"v10:{tag}:{d}:{st}:Q", p[dk]["Q"], "{:.2f}")
                put(f"v10:{tag}:{d}:{st}:L", p[dk]["L"], "{:.3f}")
                put(f"v10:{tag}:{d}:{st}:dL", p[dk]["L"] - sel["ref_L"][dk], "{:+.3f}")
            put(f"v10:{tag}:{st}:J", p["J"], "{:.2f}")
            MAC[f"v10:{tag}:{st}:feas"] = "yes" if p["feasible"] else "no"
    # plan-B effect at its selected step vs v2' at 3500 (dev5 chunk text)
    for d in DIRS:
        try:
            pb = float(MAC[f"v10:pb:{d}:3500:laal"])
            m0 = float(MAC[f"v10:cal:{d}:m0:laal"])
            putd(f"v10:pb:{d}:3500-m0:laal", pb - m0)
            pbq = float(MAC[f"v10:pb:{d}:3500:bleu"])
            m0q = float(MAC[f"v10:cal:{d}:m0:bleu"])
            putd(f"v10:pb:{d}:3500-m0:bleu", pbq - m0q)
            m1 = float(MAC[f"v10:cal:{d}:m1:laal"])
            putd(f"v10:cal:{d}:m1-m0:laal", m1 - m0)
            putd(f"v10:cal:{d}:m1-m0:bleu", float(MAC[f"v10:cal:{d}:m1:bleu"]) - m0q)
        except (ValueError, KeyError):
            for k in ("pb:%s:3500-m0:laal", "pb:%s:3500-m0:bleu", "cal:%s:m1-m0:laal", "cal:%s:m1-m0:bleu"):
                putd("v10:" + k % d, None, src="calibration inputs")

    # --- drift statistics from the training logs (steps 1..3500 = the selected checkpoint's history)
    dv = parse_trainlog(LOGS / "joint_v2p_20261113T120415Z.log")
    dp = parse_trainlog(LOGS / "joint_v2pl_20261114T130043Z.log")
    for tag, dd, src in (("v2p", dv, LOGS / "joint_v2p_20261113T120415Z.log"),
                         ("pb", dp, LOGS / "joint_v2pl_20261114T130043Z.log")):
        for k, fmt in (("q", "{:.2f}"), ("q_first", "{:.2f}"), ("q_last", "{:.2f}"), ("defer_win", "{:.2f}"),
                       ("defer_win_first", "{:.2f}"), ("defer_win_last", "{:.2f}"), ("defer_step", "{:.1f}"),
                       ("win_step", "{:.1f}"), ("E0", "{:.2f}"), ("E", "{:.2f}"), ("E0_first", "{:.2f}"),
                       ("E0_last", "{:.2f}"), ("lam", "{:.2f}")):
            if tag == "v2p" and k in ("E0", "E", "E0_first", "E0_last", "lam"):
                continue
            put(f"v10:drift:{tag}:{k}", None if dd is None else dd.get(k), fmt, src=src)

    # --- talker ablation (separate zh->en system, test)
    for s, f in (("ami", TE / "final_test_v1/ami/table_v1.json"), ("ali", TE / "final_test_v1/ali/table_v1_g1.json")):
        d = j(f)
        sp = (d or {}).get("speech", {})
        for lab in ("a", "b", "c"):
            key = next((k for k in sp if k.startswith(f"({lab})")), None)
            e = sp.get(key) if key else None
            put(f"v10:tk:{s}:{lab}:asr", None if e is None else float(np.mean(e["bleu"]["all"])), src=f)
            put(f"v10:tk:{s}:{lab}:slaal", g(e, "lat", "chunk", "speech_laal_online", "all"), src=f)
            put(f"v10:tk:{s}:{lab}:endoff", g(e, "lat", "chunk", "speech_end_online", "all"), src=f)
            put(f"v10:tk:{s}:{lab}:wer", None if e is None else 100 * float(np.mean(e["listen"]["talker_wer"])),
                "{:.1f}", src=f)
        put(f"v10:tk:{s}:text", g(d, "text", "A′", "bleu", "all"), src=f)

    # --- block-start ("single-speaker") speech LAAL (copied inputs)
    for tag, sp, s, sy in (("devz", "dev", "z", "b"), ("devz", "dev", "z", "m0"), ("devz", "dev", "z", "m"),
                           ("deve", "dev", "e", "b"), ("deve", "dev", "e", "m0"), ("deve", "dev", "e", "m"),
                           ("tami", "test", "ami", "b"), ("tami", "test", "ami", "m0"),
                           ("tali", "test", "ali", "b"), ("tali", "test", "ali", "m0"),
                           ("ten", "test", "en", "b"), ("ten", "test", "en", "m0")):
        c = {"b": "b", "m0": "m", "m": "w"}[sy]
        v = [g(j(INP / f"{tag}_{c}_s{k}_online.json"), "unit", "speech_laal_online", "all", "mean") for k in (1, 2)]
        put(f"v10:m2:{sp}:{s}:{sy}", None if None in v else sum(v) / 2, "{:.2f}", src=INP / f"{tag}_{c}_s1_online.json")

    # --- E7 (SeamlessStreaming, optional; other agent's outputs, read only)
    e7 = {}
    for s in SETS:
        n = f"jtest_{s}_seamless_default"
        t = j(DEC / n / "score_text.json")
        a = j(DEC / n / "score_asr.json")
        lat_t = lat_s = None
        for fjs in sorted(glob.glob(str(TE / f"{n}*" / "*.json")) + sorted(glob.glob(str(EO / "e7_seamless" / "**" / f"*{s}*.json"), recursive=True))):
            dj = j(fjs)
            if not isinstance(dj, dict) or "unit" not in dj or "_1spk" in fjs:
                continue
            if "speech_laal_online" in dj["unit"] and lat_s is None:
                lat_s = dj
            if "text_laal_online" in dj["unit"] and "speech_laal_online" not in dj["unit"] and lat_t is None:
                lat_t = dj
        e7[s] = t is not None
        src = DEC / n
        put(f"v10:test:{s}:e7:all", g(t, "tcpBLEU_all"), src=src)
        put(f"v10:test:{s}:e7:ov", g(t, "tcpBLEU_ov"), src=src)
        put(f"v10:test:{s}:e7:1spk", g(t, "tcpBLEU_1spk"), src=src)
        put(f"v10:test:{s}:e7:asr", g(a, "tcpBLEU_all"), src=src)
        put(f"v10:test:{s}:e7:tlaal", g(lat_t, "unit", "text_laal_online", "all", "mean")
            or g(lat_s, "unit", "text_laal_online", "all", "mean"), src=TE / n)
        put(f"v10:test:{s}:e7:slaal", g(lat_s, "unit", "speech_laal_online", "all", "mean"), src=TE / n)
        put(f"v10:test:{s}:e7:endoff", g(lat_s, "unit", "speech_end_online", "all", "mean"), src=TE / n)
    e7_any = any(e7.values())

    # --- constants
    for k, v in C.items():
        if isinstance(v, str):
            MAC["v10:c:" + k] = v
        elif isinstance(v, int):
            MAC["v10:c:" + k] = str(v)
        else:
            put("v10:c:" + k, v, "{:.1f}" if k.startswith("rw:") else "{:.2f}")
    # derived from constants
    putd("v10:c:dd:ali:diff", C["dd:ali:m0"] - C["dd:ali:b"])
    putd("v10:c:dd:ami:diff", C["dd:ami:m0"] - C["dd:ami:b"])
    putd("v10:c:dd:devz:diff", C["dd:devz:m0"] - C["dd:devz:b"])

    # ------------------------------------------------------------------------------------------------ write
    write_macros(e7_any, R)
    write_tables(R, e7, e7_any)
    compact_v7_main()
    json.dump(MISSING, open(GEN / "v10_missing.json", "w"), indent=1, sort_keys=True)
    print(f"macros {len(MAC)}, placeholders {len(MISSING)}")
    groups = {}
    for k, v in MISSING.items():
        groups.setdefault(v, []).append(k)
    for v, ks in sorted(groups.items()):
        print(f"  ?? {len(ks):3d} from {v}")


def write_macros(e7_any, R):
    with open(GEN / "macros_v10.tex", "w") as fh:
        fh.write("% generated by scripts/make_tables_v10.py -- do not edit\n")
        fh.write(r"\providecommand{\num}[1]{\csname n@#1\endcsname}" + "\n")
        fh.write(r"\newif\ifEseven" + ("\n\\Eseventrue\n" if e7_any else "\n\\Esevenfalse\n"))
        # E1 / E2 completeness flags (text may branch on them)
        e1 = all(R[("test", s, "m")][f] is not None for s in SETS for f in ("all", "asr", "slaal"))
        e2 = all(R[("test", s, "e2")][f] is not None for s in SETS for f in ("asr", "slaal"))
        fh.write(r"\newif\ifEonedone" + ("\n\\Eonedonetrue\n" if e1 else "\n\\Eonedonefalse\n"))
        fh.write(r"\newif\ifEtwodone" + ("\n\\Etwodonetrue\n" if e2 else "\n\\Etwodonefalse\n"))
        # claim switch (docs/mist_v10_plan §3.6 item 3; user decision 2026-11-14): AliMeeting is claimed only when the
        # final system's text tcpBLEU is not below the strict baseline's; while E1 is pending the narrow (AMI) wording
        # is used.
        aw = R[("test", "ali", "m")]["all"], R[("test", "ali", "b")]["all"]
        ali_win = None not in aw and aw[0] >= aw[1]
        fh.write(r"\newif\ifAliwin" + ("\n\\Aliwintrue\n" if ali_win else "\n\\Aliwinfalse\n"))
        for k in sorted(MAC):
            fh.write(f"\\expandafter\\def\\csname n@{k}\\endcsname{{{MAC[k]}}}\n")


def N(k):
    return MAC.get(k, PH)


def compact_v7_main():
    """Main-text version of v7's Table (generated/tab_v7_main.tex, make_tables_v7.py): offline and oracle blocks only,
    without the (empty) LAAL columns; the full table is kept in the appendix under the label tab:main7full."""
    src = GEN / "tab_v7_main.tex"
    if not src.exists():
        return
    lines = src.read_text().splitlines()
    full = [l.replace(r"\label{tab:main7}", r"\label{tab:main7full}") for l in lines]
    (GEN / "tab_v7_main_full.tex").write_text("\n".join(full) + "\n")
    out, skip = [], False
    for l in lines:
        if l.startswith(r"\multicolumn{11}{l}{\emph{Simultaneous"):
            skip = True
            continue
        if skip:
            if l.startswith(r"\addlinespace"):
                skip = False
            continue
        if "turns + 2" in l:
            continue
        if l.count("&") == 10:
            cells = l.rstrip().rstrip("\\").split("&")
            cells = [c for k, c in enumerate(cells) if k not in (1, 6)]
            l = "&".join(cells) + r"\\"
        l = l.replace("{lccccc@{\\hskip 10pt}ccccc}", "{lcccc@{\\hskip 10pt}cccc}")
        l = l.replace(r"\multicolumn{11}", r"\multicolumn{9}").replace(r"\multicolumn{5}", r"\multicolumn{4}")
        l = l.replace(r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
        if l.startswith(r"\caption{"):
            l = (r"\caption{Speaker-attributed translation (text) on the reference windows of the AMI test meetings "
                 r"(2{,}375 En$\rightarrow$Zh and 3{,}317 Zh$\rightarrow$En windows, 16 meetings), offline. cpBLEU$_\text{sess}$ "
                 r"(primary) matches output to reference speakers once per meeting; BLEU$_\text{blind}$ ignores speakers; Spk: "
                 r"windows with the right number of speakers (\%); DER (\%). $^\dagger$: better than the strongest "
                 r"non-oracle baseline (Table~\ref{tab:sig7}). Simultaneous v9 rows: Table~\ref{tab:main7full}.}")
        l = l.replace(r"\begin{table*}", r"\begin{table}").replace(r"\end{table*}", r"\end{table}")
        l = l.replace(r"\resizebox{\textwidth}", r"\resizebox{\columnwidth}")
        out.append(l)
    (GEN / "tab_v7_main_compact.tex").write_text("\n".join(out) + "\n")


def write_tables(R, e7, e7_any):
    ARR = {"ami": r"AMI Zh$\rightarrow$En", "ali": r"AliMeeting Zh$\rightarrow$En", "en": r"AMI En$\rightarrow$Zh"}
    NAME = {"b": r"Deadline SFT (strict baseline)", "e2": r"\quad + rl + PL-EM Talker",
            "m0": r"\sys{} v10, $\bias=0$", "m": r"\sys{} v10 ($\bias=-1$)", "e7": r"SeamlessStreaming$^\dagger$"}

    # main test table
    L = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{text tcpBLEU $\uparrow$} & speech & \multicolumn{3}{c}{latency (s) $\downarrow$} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){7-9}",
         r"Test set & System & all & ov & 1spk & ASR-BLEU $\uparrow$ & text LAAL & speech LAAL & EndOffset \\",
         r"\midrule"]
    for i, s in enumerate(SETS):
        rows = ["b", "e2", "m"] + (["e7"] if e7.get(s) else [])
        for k, sy in enumerate(rows):
            p = f"v10:test:{s}:{sy}"
            lab = (r"\multirow{%d}{*}{%s}" % (len(rows), ARR[s])) if k == 0 else ""
            nm = NAME[sy]
            row = [lab, nm] + [N(f"{p}:{f}") for f in ("all", "ov", "1spk", "asr", "tlaal", "slaal", "endoff")]
            if sy == "m":
                row = row[:2] + [r"\textbf{" + x + "}" if x != PH else x for x in row[2:]]
            L.append(" & ".join(row) + r" \\")
        if i < len(SETS) - 1:
            L.append(r"\midrule")
    if not e7_any:
        L.append("% E7 (SeamlessStreaming) rows appear here once eval_outputs/mist_w3po_v1/decode/"
                 "jtest_<set>_seamless_default/score_text.json exists (reports/mist_w3po_v1/e7_seamless_multispk_*).")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Test sets, decoded once. The strict baseline is the joint deadline-trained Thinker with the frozen "
          r"T1c Talker; the second row keeps its text and speaks it with read-level merging and the PL-EM Talker; "
          r"\sys{} v10 is the latent-write-aligned Thinker (SerialCRF v2$'$) with the fixed write bias $\bias=-1$ and the "
          r"same speech layer. Text tcpBLEU on whole meetings (all / overlapped / single-speaker units); speech: "
          r"ASR-BLEU (tcpBLEU of the ASR transcript, mean of two TTS seeds); latency under the chunk protocol, per "
          r"speaker stream: text LAAL, speech LAAL (primary) and EndOffset (end of source unit to end of its last heard "
          r"word). Bold: \sys{} v10. Significance: Table~\ref{tab:sig10}; $\bias=0$: Table~\ref{tab:delta0}."
          + (r" $^\dagger$External S2ST system on our online windows with causal diarizer relabelling, one TTS "
             r"draw (Appendix~\ref{app:e7})." if e7_any else "") + "}",
          r"\label{tab:main10}", r"\end{table*}"]
    (GEN / "tab_v10_main.tex").write_text("\n".join(L) + "\n")

    # dev ablation table (single column)
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrrr}", r"\toprule",
         r" & tcpBLEU & ASR-BLEU & text & speech & End- \\",
         r"Dev system & all & all & LAAL & LAAL & Offset \\", r"\midrule"]
    ROWN = {"b": r"(1) deadline SFT + T1c", "e2": r"(2) \quad + rl + PL-EM", "m0": r"(3) \quad + SerialCRF v2$'$",
            "m": r"(4) \quad + $\bias=-1$ (\sys{} v10)"}
    for d, lab in (("z", r"Zh$\rightarrow$En"), ("e", r"En$\rightarrow$Zh")):
        L.append(r"\multicolumn{6}{l}{\emph{%s}} \\" % lab)
        for sy in SYSS:
            p = f"v10:dev:{d}:{sy}"
            L.append(" & ".join([ROWN[sy]] + [N(f"{p}:{f}") for f in ("all", "asr", "tlaal", "slaal", "endoff")]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Full development set (" + N("v10:dev:z:b:meetings") + " / " + N("v10:dev:e:b:meetings") + r" meetings), one change per row. Rows (1)--(2) share their text. "
          r"Latency in seconds, chunk protocol. A training-time latency budget and an earlier-write lattice are "
          r"compared on text in \S\ref{sec:res-v10} and Appendix~\ref{app:calib}.}",
          r"\label{tab:abl10}", r"\end{table}"]
    (GEN / "tab_v10_ablation.tex").write_text("\n".join(L) + "\n")

    # delta = 0 test table (appendix)
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{2.5pt}",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{llrrrrr}", r"\toprule",
         r"Set & System & tcpBLEU & ov & ASR-BLEU & sp.\ LAAL & EndOff. \\", r"\midrule"]
    for i, s in enumerate(SETS):
        for k, sy in enumerate(("b", "m0", "m")):
            p = f"v10:test:{s}:{sy}"
            lab = {"ami": "AMI Zh$\\rightarrow$En", "ali": "AliMeeting", "en": "AMI En$\\rightarrow$Zh"}[s] if k == 0 else ""
            nm = {"b": "strict baseline", "m0": r"$\bias=0$ (first run)", "m": r"$\bias=-1$"}[sy]
            L.append(" & ".join([lab, nm] + [N(f"{p}:{f}") for f in ("all", "ov", "asr", "slaal", "endoff")]) + r" \\")
        if i < 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Test results of the latent-write-aligned system without the write bias ($\bias=0$), the run that "
          r"preceded the introduction of $\bias$, next to the strict baseline and the final system. Same speech layer "
          r"(rl + PL-EM) for both $\bias$ values; chunk protocol.}", r"\label{tab:delta0}", r"\end{table}"]
    (GEN / "tab_v10_delta0.tex").write_text("\n".join(L) + "\n")

    # calibration (dev5) incl. plan B
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Zh$\rightarrow$En} & \multicolumn{2}{c}{En$\rightarrow$Zh} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"dev5, text only & tcpBLEU & LAAL & tcpBLEU & LAAL \\", r"\midrule"]
    for key, nm in (("b", "deadline SFT (strict baseline)"), ("m0", r"SerialCRF v2$'$, $\bias=0$"),
                    ("m1", r"\quad $\bias=-1$ (chosen)"), ("m2", r"\quad $\bias=-2$"), ("m3", r"\quad $\bias=-3$"),
                    ("m5", r"\quad $\bias=-5$")):
        L.append(" & ".join([nm] + [N(f"v10:cal:{d}:{key}:{f}") for d in DIRS for f in ("bleu", "laal")]) + r" \\")
    L.append(r"\midrule")
    for st, nm in ((3500, r"E-step budget $\rho=0.5$, step 3{,}500 (rule)"), (5000, r"\quad step 5{,}000")):
        L.append(" & ".join([nm] + [N(f"v10:pb:{d}:{st}:{f}") for d in DIRS for f in ("bleu", "laal")]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Write-bias calibration on the five development meetings (dev5; text decoding, chunk-protocol text "
          r"LAAL in seconds). The rule, fixed before decoding: per direction, the $\bias$ whose text LAAL is closest to "
          r"the strict baseline's; both directions gave $\bias=-1$, which was then confirmed once on the full development "
          r"set. Bottom: SerialCRF v2$'$ retrained with an E-step latency budget (plan B, $\rho=0.5$, a single "
          r"pre-registered value), checkpoint chosen by the same rule as stage 2, $\bias=0$.}",
          r"\label{tab:calib10}", r"\end{table}"]
    (GEN / "tab_v10_calib.tex").write_text("\n".join(L) + "\n")

    # stage-2 selection (3000/3500/4000)
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{rrrrrrc}", r"\toprule",
         r" & \multicolumn{2}{c}{Zh$\rightarrow$En} & \multicolumn{2}{c}{En$\rightarrow$Zh} & & \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"step & $Q$ & $L-L^{\mathrm{ref}}$ & $Q$ & $L-L^{\mathrm{ref}}$ & $J$ & feasible \\",
         r"\midrule"]
    for st in (3000, 3500, 4000):
        L.append(" & ".join([f"{st:,}".replace(",", "{,}")] + [N(f"v10:s2:{d}:{st}:{f}") for d in DIRS for f in ("Q", "dL")]
                            + [N(f"v10:s2:{st}:J"), N(f"v10:s2:{st}:feas")]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Stage-2 checkpoints on dev5 under the pre-registered rule ($Q$: text tcpBLEU-all; $L$: text LAAL "
          r"of the rule's protocol, s; $J$: mean $Q$ of the two directions; feasible: $L\le L^{\mathrm{ref}}+0.1$\,s in "
          r"both directions). The rule selects step 4{,}000; we use step 3{,}500 (Appendix~\ref{app:v10train}).}",
          r"\label{tab:s2sel10}", r"\end{table}"]
    (GEN / "tab_v10_s2sel.tex").write_text("\n".join(L) + "\n")

    # drift table
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{lrrr}", r"\toprule",
         r"Training posterior (steps 1--3{,}500) & v2$'$ & $\rho=0.5$ \\", r"\midrule",
         r"label-path mass $q(\text{label})$, mean & " + N("v10:drift:v2p:q") + " & " + N("v10:drift:pb:q") + r" \\",
         r"\quad first 100 / last 500 steps & " + N("v10:drift:v2p:q_first") + " / " + N("v10:drift:v2p:q_last")
         + " & " + N("v10:drift:pb:q_first") + " / " + N("v10:drift:pb:q_last") + r" \\",
         r"$\mathbb{E}_q$[deferred chunks] per window & " + N("v10:drift:v2p:defer_win") + " & " + N("v10:drift:pb:defer_win") + r" \\",
         r"\quad first 100 / last 500 steps & " + N("v10:drift:v2p:defer_win_first") + " / " + N("v10:drift:v2p:defer_win_last")
         + " & " + N("v10:drift:pb:defer_win_first") + " / " + N("v10:drift:pb:defer_win_last") + r" \\",
         r"$\mathbb{E}_{q_0}$[deferral cost] per window (unit$\cdot$s) & -- & " + N("v10:drift:pb:E0") + r" \\",
         r"\quad first 100 / last 500 steps & -- & " + N("v10:drift:pb:E0_first") + " / " + N("v10:drift:pb:E0_last") + r" \\",
         r"\bottomrule", r"\end{tabular}}"]
    L += [r"\vspace{4pt}", r"\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrr}", r"\toprule",
          r"Decoding (full dev) & B & v2$'$ $\bias{=}0$ & v2$'$ $\bias{=}{-}1$ \\", r"\midrule"]
    for d, lab in (("z", r"Zh$\rightarrow$En"), ("e", r"En$\rightarrow$Zh")):
        L.append(lab + r": words before unit end (\%) & " + " & ".join(N(f"v10:dev:{d}:{sy}:pre") for sy in ("b", "m0", "m")) + r" \\")
        L.append(r"\quad speech start offset (s) & " + " & ".join(N(f"v10:dev:{d}:{sy}:sstart") for sy in ("b", "m0", "m")) + r" \\")
        L.append(r"\quad text LAAL (s) & " + " & ".join(N(f"v10:dev:{d}:{sy}:tlaal") for sy in ("b", "m0", "m")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Late-write drift. Top: posterior statistics logged at every stage-2 training step (about "
          + N("v10:drift:v2p:win_step") + r" windows per step); $\rho=0.5$ is the run with an E-step latency budget "
          r"(its $\mathbb{E}_{q_0}$ is the cost under the unconstrained posterior of the current model). Bottom: full "
          r"development set, strict baseline (B) and the latent-write-aligned Thinker without and with the write bias.}",
          r"\label{tab:drift10}", r"\end{table}"]
    (GEN / "tab_v10_drift.tex").write_text("\n".join(L) + "\n")

    # block-start LAAL
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{block start} & \multicolumn{2}{c}{word heard (ours)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Set & B & \sys{} & B & \sys{} \\", r"\midrule"]
    for sp, s, lab in (("dev", "z", r"dev Zh$\rightarrow$En"), ("dev", "e", r"dev En$\rightarrow$Zh"),
                       ("test", "ami", r"test AMI Zh$\rightarrow$En"), ("test", "ali", "test AliMeeting"),
                       ("test", "en", r"test AMI En$\rightarrow$Zh")):
        msys = "m" if sp == "dev" else "m0"
        L.append(" & ".join([lab, N(f"v10:m2:{sp}:{s}:b"), N(f"v10:m2:{sp}:{s}:{msys}"),
                             N(f"v10:{sp}:{s}:b:slaal"), N(f"v10:{sp}:{s}:{msys}:slaal")]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Speech LAAL with the single-speaker convention (every word of a block is delayed until the block "
          r"\emph{starts} playing) next to ours (each word at the moment it is heard). B: strict baseline; \sys{}: "
          r"$\bias=-1$ on dev, $\bias=0$ on test (the $\bias=-1$ test runs were not rescored with this convention). "
          r"Chunk protocol, mean of two TTS seeds, seconds.}",
          r"\label{tab:m2laal10}", r"\end{table}"]
    (GEN / "tab_v10_m2laal.tex").write_text("\n".join(L) + "\n")

    # BP / duplicates diagnostic
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{llrrrrr}", r"\toprule",
         r"Set & System & BLEU & len.\ ratio & BP & BLEU/BP & dup.\ units (\%) \\", r"\midrule"]
    for i, (sp, s, lab) in enumerate((("test", "ali", "AliMeeting"), ("test", "ami", r"AMI Zh$\rightarrow$En"),
                                      ("test", "en", r"AMI En$\rightarrow$Zh"), ("dev", "z", r"dev Zh$\rightarrow$En"))):
        for k, sy in enumerate(("b", "m0", "m")):
            p = f"v10:{sp}:{s}:{sy}"
            nm = {"b": "B", "m0": r"$\bias=0$", "m": r"$\bias=-1$"}[sy]
            L.append(" & ".join([lab if k == 0 else "", nm] + [N(f"{p}:{f}") for f in ("all", "lenratio", "bp", "nobp", "dup")]) + r" \\")
        if i < 3:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Text tcpBLEU split into its brevity penalty (BP) and the precision part (BLEU/BP), with the share of "
          r"reference units that received two near-identical translations (dup.). Computed from the official score "
          r"files; diagnostic only.}", r"\label{tab:dup10}", r"\end{table}"]
    (GEN / "tab_v10_dup.tex").write_text("\n".join(L) + "\n")

    # talker ablation
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{3}{c}{AMI Zh$\rightarrow$En} & \multicolumn{3}{c}{AliMeeting} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Talker & ASR-BLEU & LAAL & EndOff. & ASR-BLEU & LAAL & EndOff. \\", r"\midrule"]
    for lab, nm in (("a", "rule (queue rate, 2-best CTC)"), ("b", "PL-EM"), ("c", "rl + PL-EM")):
        L.append(" & ".join([nm] + [N(f"v10:tk:{s}:{lab}:{f}") for s in ("ami", "ali") for f in ("asr", "slaal", "endoff")]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Speech layer alone, on the separately trained Zh$\rightarrow$En v2$'$ Thinker (same text in all rows: "
          r"tcpBLEU " + N("v10:tk:ami:text") + " / " + N("v10:tk:ali:text") + r"); test sets, two TTS seeds, speech "
          r"LAAL and EndOffset in seconds (chunk protocol). rl: read-level merging.}",
          r"\label{tab:talker10}", r"\end{table}"]
    (GEN / "tab_v10_talker.tex").write_text("\n".join(L) + "\n")

    # EndOffset decomposition
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{2.5pt}",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Set & Sys. & start & model & detector & queue & last block & EndOff. & load \\", r"\midrule"]
    for i, (sp, s, lab) in enumerate((("dev", "z", r"dev Zh$\rightarrow$En"), ("dev", "e", r"dev En$\rightarrow$Zh"),
                                      ("test", "ami", r"AMI Zh$\rightarrow$En"), ("test", "ali", "AliMeeting"),
                                      ("test", "en", r"AMI En$\rightarrow$Zh"))):
        for k, sy in enumerate(("b", "e2", "m")):
            p = f"v10:{sp}:{s}:{sy}"
            nm = {"b": "B", "e2": "B+PL", "m": r"\sys{}"}[sy]
            L.append(" & ".join([lab if k == 0 else "", nm] + [N(f"{p}:{f}") for f in
                                ("sstart", "p_model_lag", "p_detector_wait", "p_queue_wait", "p_last_block_dur", "endoff", "load")]) + r" \\")
        if i < 4:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Where the speech delay of a source unit comes from (chunk protocol, mean over units and two TTS "
          r"seeds, seconds). EndOffset = model lag (last commit after the unit's end) + detector wait (online window "
          r"release) + queue wait (earlier speech of the same stream) + duration of the last block. start: speech "
          r"start offset from the unit's start. load: synthesised speech per second of source speech. B+PL: strict "
          r"baseline text with rl + PL-EM.}", r"\label{tab:decomp10}", r"\end{table}"]
    (GEN / "tab_v10_decomp.tex").write_text("\n".join(L) + "\n")

    # significance table
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{2.5pt}",
         r"\resizebox{\columnwidth}{!}{\begin{tabular}{llrrrr}", r"\toprule",
         r"Set & Comparison & $\Delta$ & 95\% interval & $p$ & $p_\text{Holm}$ \\", r"\midrule"]
    CN = {"text:m-b": r"text: \sys{} $-$ B", "speech:m-b": r"speech: \sys{} $-$ B", "speech:e2-b": r"speech: B+PL $-$ B",
          "speech:m-e2": r"speech: \sys{} $-$ B+PL"}
    for i, s in enumerate(SETS):
        for k, c in enumerate(CN):
            mk = f"v10:sig:test:{s}:{c}"
            lab = {"ami": "AMI Zh$\\rightarrow$En", "ali": "AliMeeting", "en": "AMI En$\\rightarrow$Zh"}[s] if k == 0 else ""
            L.append(" & ".join([lab, CN[c], N(mk + ":diff"), N(mk + ":ci"), N(mk + ":p"), N(mk + ":holm")]) + r" \\")
        if i < 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Paired tests over meetings on the test sets (16 AMI, 20 AliMeeting meetings): exact two-sided "
          r"sign-flip permutation $p$ over all $2^M$ flips, 95\% meeting-bootstrap interval (10{,}000 resamples), Holm "
          r"over the " + N("v10:sig:ntests") + r" tests. Speech pools the two TTS seeds per meeting. B: strict baseline; "
          r"B+PL: its text with rl + PL-EM; \sys{}: v10 with $\bias=-1$. Text of B+PL equals B by construction.}",
          r"\label{tab:sig10}", r"\end{table}"]
    (GEN / "tab_v10_sig.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
