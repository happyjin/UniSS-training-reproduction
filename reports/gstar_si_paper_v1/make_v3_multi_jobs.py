"""The rest of the v3 evaluation (jobs/v3_jobs.json, minus what logs/v3_eval5_run.log shows done) and the AliMeeting fix
chain (jobs/v3_ali_fix_jobs.json) as one queue in which the HF decoding runs go K to a card through gstar_multi (one
copy of the Thinker, one worker per run, each run's argv, output and log as in its own job).

  * runs grouped by (Thinker, dependencies), K per group; a group is one exclusive 1-card job;
  * a run still going under the old queue (status "start") becomes a no-card job that waits for its python PID to exit
    and then checks its log for the done line, so the score jobs after it keep their order;
  * everything else (scores, segmentation, vLLM jobs, ali_mt, ...) is kept as it was; dependencies are mapped to the
    groups that hold them.

    python3 reports/gstar_si_paper_v1/make_v3_multi_jobs.py K out_jobs.json runs_dir
"""
import json, os, re, shlex, subprocess, sys

K, OUT, RUNS = int(sys.argv[1]), sys.argv[2], sys.argv[3]
J = "reports/gstar_si_paper_v1/jobs"
PYM = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
DEC = re.compile(r"^mkdir -p (\S+) && CUDA_VISIBLE_DEVICES=\$GPU0 PYTHONPATH=\.:third_party/Megatron-LM " + re.escape(PYM)
                 + r" -m experiments\.uniss_qwen3_omni_v1\.evaluation\.(gstar_si_stream(?:_online)?) (.*) > (\S+) 2>&1$")
MARK = {"gstar_si_stream": "GSTAR_STREAM_DONE", "gstar_si_stream_online": "GSTAR_ONLINE_DONE"}

st = {}
for line in open("logs/v3_eval5_run.log"):
    m = re.match(r"\S+ (start|done|fail|skip) (\S+)", line)
    if m:
        st[m.group(2)] = m.group(1)
main = [j for j in json.load(open(f"{J}/v3_jobs.json")) if st.get(j["id"], "wait") in ("wait", "start")]
ali = json.load(open(f"{J}/v3_ali_fix_jobs.json"))
assert not set(j["id"] for j in main) & set(j["id"] for j in ali)
done = {k for k, v in st.items() if v == "done"}
ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout.splitlines()

jobs, group_of, groups = [], {}, {}
for j in main + ali:
    j = dict(j, deps=[d for d in j["deps"] if d not in done])
    m = DEC.match(j["cmd"])
    if st.get(j["id"]) == "start":
        assert m, j["id"]
        argv = shlex.split(m.group(3))
        out = argv[argv.index("--out") + 1]
        pids = [l.split()[0] for l in ps if f"--out {out} " in l + " " and l.split()[1] == PYM]
        assert len(pids) <= 1, (j["id"], pids)
        wait = f"while kill -0 {pids[0]} 2>/dev/null; do sleep 20; done; " if pids else ""
        jobs.append({"id": j["id"], "gpus": 1, "vllm": False, "share": True, "need_mib": 0, "max_used_mib": 200000,
                     "deps": [], "cmd": wait + f"grep -q {MARK[m.group(2)]} {m.group(4)}"})
        continue
    if not m:
        jobs.append(j)
        continue
    argv = shlex.split(m.group(3))
    key = (argv[argv.index("--thinker") + 1], tuple(sorted(j["deps"])))
    groups.setdefault(key, []).append({"id": j["id"], "module": m.group(2), "argv": argv, "log": m.group(4),
                                       "mkdir": m.group(1)})

os.makedirs(RUNS, exist_ok=False)
for (thinker, deps), runs in groups.items():
    for c in range(0, len(runs), K):
        chunk = runs[c: c + K]
        gid = f"multi_{chunk[0]['id']}" + (f"_{len(chunk)}" if len(chunk) > 1 else "")
        f = f"{RUNS}/{gid}.json"
        with open(f, "x") as fh:
            json.dump([{k: r[k] for k in ("module", "argv", "log")} for r in chunk], fh, indent=1)
        dirs = " ".join(sorted({r["mkdir"] for r in chunk}))
        jobs.append({"id": gid, "gpus": 1, "vllm": False, "deps": list(deps), "members": [r["id"] for r in chunk],
                     "cmd": f"mkdir -p {dirs} && CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=.:third_party/Megatron-LM {PYM} "
                            f"-m experiments.uniss_qwen3_omni_v1.evaluation.gstar_multi --thinker {thinker} {f}"})
        for r in chunk:
            group_of[r["id"]] = gid
ids = {j["id"] for j in jobs}
for j in jobs:
    j["deps"] = sorted({group_of.get(d, d) for d in j["deps"]})
    assert all(d in ids for d in j["deps"]), (j["id"], j["deps"])
with open(OUT, "x") as fh:
    json.dump(jobs, fh, indent=1)
n_multi = sum(1 for j in jobs if j["id"].startswith("multi_"))
print(f"{len(jobs)} jobs: {n_multi} groups of <= {K} runs ({len(group_of)} runs), "
      f"{sum(1 for j in jobs if j.get('share') and j.get('max_used_mib') == 200000)} waits on running runs, "
      f"{len(jobs) - n_multi} other")
