"""OpenBroadcast-SI DEV twin (D0, reports/mist_w3po_v1/robust_wave1_prereg.zh-CN.md): bench_eval.py UNCHANGED (every
metric, both TTS seeds, metrics.json / items_<n>.jsonl under eval_outputs/mist_w3po_v1/voicecond_bench/<name>/), with
only its set roots swapped in this process:
  data     data/open_voicecond_v1/bench -> data/open_voicecond_v1/bench_dev (meetings, refs, utts, src/<dir>/pool.jsonl)
  windows  windows_obsi_* -> windows_obsidev_* (C2 windows and the chunk-protocol windows of the speech LAAL)
  --cond   id2 | clean only; --name must start with "obsidev_"
chunk_lat is bench_eval.chunk_lat with the reference windows taken from the dev root (the only hard-coded path).

    python -m experiments.mist_w3po_v1.voicecond.bench_dev_eval --cond id2|clean --dir zh2en|en2zh --name obsidev_<run>
        [--seeds 1|2] [--procs 32]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from experiments.mist_w3po_v1.voicecond import bench_eval as E

E.BD = E.REPO / "data/open_voicecond_v1/bench_dev"


def wins(cond: str, d: str) -> tuple[str, str]:
    return (f"{E.WD}/windows_obsidev_{cond}_{d}_c2g06_v1/windows_test_online_{d}.jsonl",
            f"{E.WD}/windows_obsidev_{cond}_{d}_c2g06_chunk_v1/windows_dev_online.jsonl")


def chunk_lat(cond: str, d: str, n: str, out):
    f = out / f"lat_chunk_{n}.json"
    if not f.exists():
        ref = str((E.BD / cond / d / "refs_windows.jsonl").relative_to(E.REPO))
        cw = wins(cond, d)[1]
        if d == "zh2en":
            cmd = [sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.on_refset", "--ref-win", ref, "--dr", "ali",
                   "experiments.mist_w3po_v1.evaluation.speech_latency_lit", "--run", str(E.DEC / n), "--talker",
                   str(E.SPD / n / "talker"), "--win", cw, "--out", str(f)]
        else:
            cmd = [sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.en2zh_lat", "--ref-win", ref, "--work",
                   str(f) + ".work", "--run", str(E.DEC / n), "--talker", str(E.SPD / n / "talker"), "--win", cw, "--out", str(f)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{E.REPO}:{E.REPO}/third_party/Megatron-LM")
        with open(str(f) + ".log", "w") as lg:
            r = subprocess.run(cmd, cwd=E.REPO, env=env, stdout=lg, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not f.exists():
            print(f"chunk latency failed for {n} (see {f}.log)", flush=True)
            return None
    L = json.load(open(f))
    u = L["unit"]
    g = lambda k, s="all": (u.get(k) or {}).get(s) or {}  # noqa: E731
    return {"speech_laal": g("speech_laal_online").get("mean"), "speech_laal_ov": g("speech_laal_online", "ov").get("mean"),
            "speech_laal_1spk": g("speech_laal_online", "1spk").get("mean"), "end_offset": g("speech_end_online").get("mean"),
            "text_laal": g("text_laal_online").get("mean"), "n_units": g("speech_laal_online").get("n"),
            "file": str(f.relative_to(E.REPO))}


E.wins = wins
E.chunk_lat = chunk_lat


def main() -> None:
    argv = sys.argv[1:]
    if "--cond" not in argv or argv[argv.index("--cond") + 1] not in ("id2", "clean"):
        raise SystemExit("--cond id2|clean required (dev twin)")
    if "--name" not in argv or not argv[argv.index("--name") + 1].startswith("obsidev_"):
        raise SystemExit("--name obsidev_<...> required (dev twin run names)")
    E.main()


if __name__ == "__main__":
    main()
