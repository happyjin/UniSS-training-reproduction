"""SpecClose CPU precheck P1-P4 (reports/mist_w3po_v1/thinker_detector_specclose_prereg_v1.zh-CN.md §2). CPU only;
dev only. Writes eval_outputs/mist_w3po_v1/detector_v1/specclose_precheck_v1.json (refuses to overwrite); prints it.

  P1  code: data/test_specclose and crf/test_decode_crf_ckpt_specclose + test_decode_crf_ckpt_finalpost pass; the
      judge-off windows (windows_chunk_v1) equal the C2 windows row by row (id, start, duration, audio) with chunk times
      never later; speech_latency_lit on them reproduces the e_chunk bound (A'rl + PL-EM 3.859 / 3.854, +- 0.001);
      specclose_eval_orch --dry decodes with decode_crf_ckpt_specclose only
  P2  judge on dev (windows_specclose_v1/summary.json): false-close rate <= 2.0 %, split C2 windows <= 54, recall >= 50 %
  P3  speech_latency_lit of the existing A'rl + PL-EM outputs (2 seeds) on windows_dev_retime.jsonl: mean per-stream
      speech LAAL <= 3.757 s (chunk baseline 3.857 - 0.10)
  P4  utterance cut rate on the SpecClose windows (c2_gap_stats rule: no window covers >= 90 %) <= 2.91 %
    python experiments/mist_w3po_v1/scripts/specclose_precheck.py
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
ENV = {"PYTHONPATH": ".:third_party/Megatron-LM", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8",
       "PATH": "/usr/bin:/bin"}
D = Path("eval_outputs/mist_w3po_v1")
DV = D / "detector_v1"
C2 = D / "data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
WC = DV / "windows_chunk_v1/windows_dev_online.jsonl"
WS = DV / "windows_specclose_v1/windows_dev_online.jsonl"
WR = DV / "windows_specclose_v1/windows_dev_retime.jsonl"
A = "dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem"
CK = "eval_outputs/mist_w3po_v1/crf/train/crf_v2p_20261110T075131Z/ckpt/step_03000"
OUT = DV / "specclose_precheck_v1.json"
LINES = {"P2_fcr_max": 0.02, "P2_split_max": 54, "P2_recall_min": 0.50, "P3_laal_max": 3.757, "P4_cut_max": 0.0291,
         "P1_chunk_ref": [3.859, 3.854]}


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ)
    env.update({k: v for k, v in ENV.items() if k != "PATH"})
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def cut_rate(win: Path) -> dict:
    W = jl(win)
    spans = defaultdict(list)
    for w in W:
        spans[w["meeting"]].append((w["start_s"], w["start_s"] + w["dur_ms"] / 1000))
    cut = tot = 0
    for m, sp in spans.items():
        for u in map(json.loads, open(f"data/ami_multispk_si_v1/zh_mix_v8/utts/{m}.jsonl", encoding="utf-8")):
            if u["hes"] or not u["placed"]:
                continue
            tot += 1
            b, e = u["zh_begin"], u["zh_end"]
            if not any(min(we, e) - max(ws, b) >= 0.9 * (e - b) for ws, we in sp):
                cut += 1
    return {"utts": tot, "cut": cut, "cut_rate": round(cut / max(tot, 1), 4), "windows": len(W)}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {OUT}")
    res = {"lines": LINES}
    # ---- P1
    p1 = {}
    for name, mod in (("test_specclose", "experiments.mist_w3po_v1.data.test_specclose"),
                      ("test_decode_specclose", "experiments.mist_w3po_v1.crf.test_decode_crf_ckpt_specclose"),
                      ("test_decode_finalpost", "experiments.mist_w3po_v1.crf.test_decode_crf_ckpt_finalpost")):
        r = run([PY, "-m", mod])
        p1[name] = r.returncode == 0 and ("ALL " in r.stdout)
    a, c = jl(C2), jl(WC)
    p1["chunk_windows_equal_c2"] = len(a) == len(c) and all(
        x["id"] == y["id"] and x["start_s"] == y["start_s"] and x["dur_ms"] == y["dur_ms"] and x["audio"] == y["audio"]
        and y["open_detect_ms"] <= x["open_detect_ms"] and y["close_detect_ms"] <= x["close_detect_ms"]
        for x, y in zip(a, c))
    ref = [json.load(open(DV / f"ref_chunk/lat_{n}.json"))["unit"]["speech_laal_online"]["all"]["mean"]
           for n in (A, A + "_tts2")]
    p1["chunk_ref_laal"] = ref
    p1["chunk_ref_reproduces_bound"] = all(abs(x - y) <= 0.001 for x, y in zip(ref, LINES["P1_chunk_ref"]))
    r = run([PY, "experiments/mist_w3po_v1/scripts/specclose_eval_orch.py", "--run", "/nonexistent/specclose_dry",
             "--name", "dev_specclose_aprime_s3000", "--ckpt", CK, "--talker", "none", "--window", "1",
             "--scorer-port0", "19601", "--win", str(WS), "--dry"])
    dec = [l for l in r.stdout.splitlines() if " dec " in l]
    p1["orch_dry_ok"] = (r.returncode == 0 and len(dec) == 8
                         and all("decode_crf_ckpt_specclose --crf-ckpt" in l and "final-read-post" not in l
                                 and str(WS) in l for l in dec))
    p1["pass"] = all(v for k, v in p1.items() if isinstance(v, bool))
    res["P1"] = p1
    # ---- P2
    s = json.load(open(DV / "windows_specclose_v1/summary.json"))
    by_u = defaultdict(Counter)
    for j in jl(DV / "windows_specclose_v1/judge.jsonl"):
        k = ("FC" if j["close"] else "TH") if j["resumes"] else ("TC" if j["close"] else "MC")
        by_u[j["u"]][k] += 1
    res["P2"] = {"calls": s["calls"], "hold_truth": s["calls_hold_truth"], "false_close": s["false_close"],
                 "true_close": s["true_close"], "missed_close": s["missed_close"], "fcr": s["fcr"],
                 "recall": s["recall"], "split_c2_windows": s["base_split"], "new_pieces": s["new_piece"],
                 "kinds": {k[5:]: v for k, v in s.items() if k.startswith("kind_")},
                 "by_u": {u: dict(v) for u, v in sorted(by_u.items())}}
    res["P2"]["pass"] = bool(s["fcr"] <= LINES["P2_fcr_max"] and s["base_split"] <= LINES["P2_split_max"]
                             and s["recall"] >= LINES["P2_recall_min"])
    # ---- P3
    lat = []
    for n in (A, A + "_tts2"):
        o = DV / f"precheck_retime_lat_{n}.json"
        if not o.exists():
            r = run([PY, "-m", "experiments.mist_w3po_v1.evaluation.speech_latency_lit", "--run", str(D / "decode" / n),
                     "--talker", str(D / "speech" / n / "talker"), "--win", str(WR), "--out", str(o)])
            if r.returncode != 0:
                raise SystemExit(r.stderr[-2000:])
        u = json.load(open(o))["unit"]
        lat.append({"speech_laal": u["speech_laal_online"]["all"]["mean"], "ov": u["speech_laal_online"]["ov"]["mean"],
                    "1spk": u["speech_laal_online"]["1spk"]["mean"], "text_laal": u["text_laal_online"]["all"]["mean"],
                    "end_offset": u["speech_end_online"]["all"]["mean"]})
    m = round(sum(x["speech_laal"] for x in lat) / 2, 4)
    res["P3"] = {"per_seed": lat, "mean_speech_laal": m, "chunk_baseline": round(sum(ref) / 2, 4),
                 "old_protocol": 4.050, "pass": bool(m <= LINES["P3_laal_max"])}
    # ---- P4
    cs, cc = cut_rate(WS), cut_rate(C2)
    res["P4"] = {"specclose": cs, "c2": cc, "pass": bool(cs["cut_rate"] <= LINES["P4_cut_max"])}
    res["GO"] = bool(res["P1"]["pass"] and res["P2"]["pass"] and res["P3"]["pass"] and res["P4"]["pass"])
    with open(OUT, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
