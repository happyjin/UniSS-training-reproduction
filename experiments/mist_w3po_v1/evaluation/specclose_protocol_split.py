"""SpecClose latency split (descriptive; reports/mist_w3po_v1/thinker_detector_specclose_prereg_v1.zh-CN.md 9.3: report
what the chunk protocol gives and what SpecClose adds beyond it). CPU only, dev only, read-only on existing outputs.

The SpecClose evaluation stamps every SpecClose window with its own detection times (windows_specclose_v1: chunk
open T(st); close = the speculative close time tau_f for "spec" windows, else the chunk close T(j - 1)). The same
decode + talker outputs are re-stamped here, only the two detection stamps (open_detect_ms, close_detect_ms) changed,
every other window field kept:
  spec   the SpecClose windows as they are (what the evaluation reports)
  chunk  every window id that is also a C2 window: the chunk-protocol stamps of windows_chunk_v1 (so a spec close is
         replaced by its chunk close T(j - 1)); the fragments of the windows SpecClose cut keep their own stamps
  old    the same ids: the C2 stamps (the current protocol, f * 80 ms + 1.04 s); fragments keep their own stamps
spec - chunk = the timing SpecClose adds beyond the chunk protocol on its own decode; chunk - old = the accounting
correction. speech_latency_lit on both TTS seeds of the speech run, and --text-only on the text decode.

    python -m experiments.mist_w3po_v1.evaluation.specclose_protocol_split --speech dev_specclose_aprime_s3000__rlplem \
        --text dev_specclose_aprime_s3000 --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E = Path("eval_outputs/mist_w3po_v1")
WS = E / "detector_v1/windows_specclose_v1/windows_dev_online.jsonl"
WOLD = E / "data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
WCHUNK = E / "detector_v1/windows_chunk_v1/windows_dev_online.jsonl"
STAMPS = ("open_detect_ms", "close_detect_ms")
KEYS = ("speech_laal_online", "speech_end_online", "speech_laal_online_1ch", "text_laal_online")
REGS = ("all", "ov", "1spk")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def restamp(spec: list[dict], other: dict[str, dict]) -> tuple[list[dict], Counter]:
    """only an uncut window (its orig_id has one SpecClose row, id == orig_id, same start_s and dur_ms as the other
    file's row) takes the other stamps; the fragments of a cut window keep their own."""
    n_orig = Counter(w.get("orig_id", w["id"]) for w in spec)
    out, c = [], Counter()
    for w in spec:
        z = dict(w)
        o = other.get(w["id"])
        if o is not None and (n_orig[w.get("orig_id", w["id"])] > 1 or w.get("orig_id", w["id"]) != w["id"]
                              or o["start_s"] != w["start_s"] or o["dur_ms"] != w["dur_ms"]):
            o = None
            c["fragment_or_changed"] += 1
        if o is not None:
            for k in STAMPS:
                z[k] = o[k]
            c["restamped_" + w.get("close_kind", "?")] += 1
        else:
            c["kept_" + w.get("close_kind", "?")] += 1
        out.append(z)
    return out, c


def metrics(p: Path) -> dict:
    u = json.load(open(p))["unit"]
    return {k: {r: (u[k][r]["mean"] if u[k].get(r) else None) for r in REGS} for k in KEYS if k in u}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speech", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    spec = jl(WS)
    old = {w["id"]: w for w in jl(WOLD)}
    chunk = {w["id"]: w for w in jl(WCHUNK)}
    assert old.keys() == chunk.keys()
    win = {"spec": WS}
    out.mkdir(parents=True)
    counts = {}
    for name, other in (("chunk", chunk), ("old", old)):
        rows, c = restamp(spec, other)
        counts[name] = dict(c)
        win[name] = out / f"windows_{name}.jsonl"
        with open(win[name], "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ".:third_party/Megatron-LM", "OMP_NUM_THREADS": "4"})
    jobs = []
    for proto in ("spec", "chunk", "old"):
        (out / proto).mkdir()
        for s in (a.speech, a.speech + "_tts2"):
            jobs.append((out / proto / f"lat_{s}.json",
                         ["--run", str(E / "decode" / s), "--talker", str(E / "speech" / s / "talker")], proto))
        jobs.append((out / proto / f"textlat_{a.text}.json", ["--run", str(E / "decode" / a.text), "--text-only"], proto))

    def run(j):
        o, args, proto = j
        with open(str(o) + ".log", "x", encoding="utf-8") as fh:
            return subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.speech_latency_lit", *args,
                                   "--win", str(win[proto]), "--out", str(o)], stdout=fh, stderr=subprocess.STDOUT,
                                  env=env).returncode
    with ThreadPoolExecutor(9) as ex:
        rcs = list(ex.map(run, jobs))
    if any(rcs):
        raise SystemExit(f"speech_latency_lit failed: {[(str(j[0]), r) for j, r in zip(jobs, rcs) if r]}")
    res = {"windows": len(spec), "close_kind": dict(Counter(w.get("close_kind", "?") for w in spec)),
           "restamp_counts": counts, "protocols": {}}
    for proto in ("spec", "chunk", "old"):
        m1 = metrics(out / proto / f"lat_{a.speech}.json")
        m2 = metrics(out / proto / f"lat_{a.speech}_tts2.json")
        mt = metrics(out / proto / f"textlat_{a.text}.json")
        res["protocols"][proto] = {
            "speech_2seed_mean": {k: {r: (round((m1[k][r] + m2[k][r]) / 2, 4) if m1[k][r] is not None
                                          and m2[k][r] is not None else None) for r in REGS} for k in m1},
            "seed1": m1, "seed2": m2, "text_only": mt}
    sp = {p: res["protocols"][p]["speech_2seed_mean"]["speech_laal_online"]["all"] for p in ("spec", "chunk", "old")}
    res["split_speech_laal_all"] = {"spec": sp["spec"], "chunk": sp["chunk"], "old": sp["old"],
                                    "chunk_minus_old": round(sp["chunk"] - sp["old"], 4),
                                    "spec_minus_chunk": round(sp["spec"] - sp["chunk"], 4)}
    with open(out / "split.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res["split_speech_laal_all"]))


if __name__ == "__main__":
    main()
