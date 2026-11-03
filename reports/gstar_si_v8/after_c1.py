"""Paper v8: re-plan the slow decodes once C1 training has finished (both merged speaker-blind models exist).

  1. stop, by PID, the 8-card spread LA2 resume decoders (hyp_r1) and the fixed-lag 2 s decoders (hyp_s0): spread
     decoding runs ~10x slower than 2-card shards; their outputs are kept and never re-decoded;
  2. LA2 (prereg E-pol): the windows not yet decoded, as 2-card shards (--max-gib 34, no CPU offload) in the v8 queue;
  3. fixed lag 2 s (exploratory): narrowed to the 8 AMI / 10 AliMeeting meetings of the v6/v7 subset (as v7's fixed lag),
     only the windows not yet decoded, as 2-card shards after the LA2 shards.
Writes reports/gstar_si_v8/jobs/after_c1.json and adds it to the queue. Refuses to run twice.

    python3 reports/gstar_si_v8/after_c1.py
"""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = "eval_outputs/uniss_qwen3_omni_v1"
R = REPO / EV / "gstar_v8_20261103/runs"
ON = REPO / "data/online_v6_20261031"
JOBS = REPO / "reports/gstar_si_v8/jobs/after_c1.json"
T = {"en2zh": f"{EV}/thinker_gstar_s2dl_20261029b/thinker_merged",
     "zh2en": f"{EV}/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged",
     "ali": f"{EV}/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged"}
PY = ("CUDA_VISIBLE_DEVICES=$CVD PYTHONPATH=.:third_party/Megatron-LM /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/"
      "python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online_v6s")


def pids(marks):
    out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    res = []
    for ln in out:
        pid, _, args = ln.strip().partition(" ")
        if "gstar_si_stream_online_v6s" in args and args.startswith("/") and any(m in args for m in marks):
            res.append(int(pid))
    return res


def done_ids(d):
    got = set()
    for f in d.glob("hyp_*.jsonl"):
        for ln in open(f, encoding="utf-8"):
            try:
                got.add(json.loads(ln)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return got


def main():
    if JOBS.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {JOBS}")
    marks = [f"runs/on_{s}_S2dlLA2/hyp_r1" for s in ("en2zh", "zh2en")] + \
            [f"runs/on_{s}_S2dllag2/hyp_s0" for s in ("en2zh", "zh2en", "ali")]
    ps = pids(marks)
    for p in ps:
        os.kill(p, signal.SIGTERM)
    for _ in range(60):
        if not pids(marks):
            break
        time.sleep(5)
    print("stopped", ps, "left", pids(marks), flush=True)
    sub = json.load(open(ON / "subset_meetings.json"))
    jobs = []
    plan = [("S2dlLA2", s, "--policy la2", None) for s in ("en2zh", "zh2en")] + \
           [("S2dllag2", s, "--policy lag --lag-ms 2000", set(sub["ali" if s == "ali" else "ami"]))
            for s in ("en2zh", "zh2en", "ali")]
    for name, s, pol, meetings in plan:
        d = R / f"on_{s}_{name}"
        got = done_ids(d)
        W = [json.loads(ln) for ln in open(ON / f"windows_test_online_{s}.jsonl")]
        rest = [w for w in W if w["id"] not in got and (meetings is None or w["meeting"] in meetings)]
        f = d / "resume2.windows.jsonl"
        with open(f, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(w) + "\n" for w in rest)
        nsh = max(1, min(4, round(len(rest) / 500)))
        print(f"{d.name}: {len(got)} done, {len(rest)} to decode in {nsh} shards", flush=True)
        dr = "en2zh" if s == "en2zh" else "zh2en"
        for k in range(nsh):
            jobs.append({"id": f"mist8r_{d.name}_r2s{k}", "gpus": 2, "mem": 46000, "kind": "decode", "deps": [],
                         "cmd": f"{PY} --thinker {T[s]} --windows {f.relative_to(REPO)} --out {(d / f'hyp_r2s{k}.jsonl').relative_to(REPO)} "
                                f"--shard {k} --num-shards {nsh} {pol} --window 1 --post-key lowlat --dir {dr} --batch 32 "
                                f"--max-gib 34 > {(d / f'infer_r2s{k}.log').relative_to(REPO)} 2>&1"})
    with open(JOBS, "x") as fh:
        json.dump(jobs, fh, indent=1)
    q = REPO / "logs/multispk/v8/queue/add/010_after_c1.json"
    with open(q, "x") as fh:
        json.dump(jobs, fh, indent=1)
    print(len(jobs), "jobs queued")


if __name__ == "__main__":
    main()
