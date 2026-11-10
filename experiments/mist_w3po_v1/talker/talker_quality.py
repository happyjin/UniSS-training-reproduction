"""Naturalness and voice of a MIST talker run (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md §4): UTMOS
(utmos22_strong, the local SpeechMOS hub copy reports/uniss_m2_fluency_results_v2 used) of each stream's speech-only
wav (>= 0.5 s), and WavLM-SV speaker similarity (experiments/uniss_omni_s2st_v1/evaluation/sim_o.py,
wavlm-base-plus-sv) between that wav and the stream's voice reference (talker_in items.jsonl voice_audio: the
causal voice prompt the talker was given). GPU; writes <out> (refuses to overwrite; one directory per run: sim_o adds per_sample_sim_o.jsonl
next to it) and <out>.items.jsonl.

    python -m experiments.mist_w3po_v1.talker.talker_quality --talker <dir with results.jsonl> --items <items.jsonl> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
HUB = "/opt/dlami/nvme/jasonleeeli/cache/torch/hub/tarepan_SpeechMOS_v1.2.0"
WAVLM = "/opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--talker", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() or (out.parent / "per_sample_sim_o.jsonl").exists():   # sim_o writes that file next to its output
        raise SystemExit(f"拒绝覆盖已有输出: {out} (one directory per run)")
    out.parent.mkdir(parents=True, exist_ok=True)
    voice = {r["id"]: r.get("voice_audio") for r in map(json.loads, open(a.items, encoding="utf-8"))}
    rows = [r for r in map(json.loads, open(Path(a.talker) / "results.jsonl", encoding="utf-8"))
            if not r.get("error") and r.get("speech_path")]
    m = torch.hub.load(HUB, "utmos22_strong", source="local", trust_repo=True).eval().to("cuda:0")
    per = {}
    with torch.no_grad():
        for r in rows:
            x, sr = sf.read(r["speech_path"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if len(x) < 0.5 * sr:
                continue
            per[r["id"]] = {"utmos": float(m(torch.from_numpy(x).unsqueeze(0).to("cuda:0"), sr).item())}
    sim_in = out.with_suffix(".sim_in.jsonl")
    with open(sim_in, "x", encoding="utf-8") as fh:
        for r in rows:
            v = voice.get(r["id"])
            if r["id"] in per and v:
                fh.write(json.dumps({"id": r["id"], "audio_path": r["speech_path"], "src_lang": "cmn", "tgt_lang": "eng",
                                     "reference_audio_path": str(REPO / v) if not os.path.isabs(v) else v}) + "\n")
    sim_out = out.with_suffix(".sim.json")
    subprocess.run([sys.executable, "-m", "experiments.uniss_omni_s2st_v1.evaluation.sim_o", "--input", str(sim_in),
                    "--output", str(sim_out), "--model", WAVLM, "--device", "cuda:0"], check=True, cwd=REPO)
    sim = json.load(open(sim_out))
    sim_mean = float(next(iter(sim["groups"].values()))["mean"])
    u = [v["utmos"] for v in per.values()]
    res = {"talker": a.talker, "n_utmos": len(u), "utmos_mean": round(float(np.mean(u)), 4) if u else None,
           "sim_mean": round(sim_mean, 4), "sim_groups": sim["groups"]}
    with open(str(out) + ".items.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps({"id": k, **v}) + "\n" for k, v in per.items())
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "sim_groups"}))


if __name__ == "__main__":
    main()
