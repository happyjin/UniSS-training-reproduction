"""Paper v9: idempotent re-spawn after an external kill of all v9 processes (11-04 04:02, 11-05 02:10 and 03:20 UTC).

If the launcher (memq9b.py) is running, prints the status and does nothing else. Otherwise:
  1. for every unfinished experiment (Lag1 / Lag3 / Blk2 subset, SDM S / Rel) the windows not yet in any hyp_*.jsonl of
     its run dir are written to a new resume_r<N>.windows.jsonl and queued as new jobs m9r<N>_* with new output files
     (never overwriting); the waiting jobs of earlier rounds are cancelled;
  2. the launcher and the P1-4 trigger are started again (setsid nohup).
Status line: remaining windows per experiment, running decoders, launcher alive. Exit code 0 = launcher was alive,
3 = it was dead and has been re-spawned.

    python3 reports/gstar_si_v9/kick.py
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
os.chdir(REPO)
EV = "eval_outputs/uniss_qwen3_omni_v1"
V9 = f"{EV}/gstar_v9_20261104/runs"
Q = "logs/multispk/v9/q"
LOG = "logs/multispk/v9/memq9.log"
T = {"en2zh": f"{EV}/thinker_gstar_s2dl_20261029b/thinker_merged",
     "zh2en": f"{EV}/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged",
     "ali": f"{EV}/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged"}
PY = ("CUDA_VISIBLE_DEVICES=$CVD PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH "
      "PYTHONPATH=.:third_party/Megatron-LM /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python "
      "-m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online_v6s")
EXP = [(f"{L}_{st}", f"{V9}/on_{st}_S2dl{L}", f"{V9}/on_{st}_S2dl{L}/sub.windows.jsonl", T[st], pol, st, prio)
       for L, pol, prio in (("lag1", "--policy lag --lag-ms 1000 --window 1", 3), ("lag3", "--policy lag --lag-ms 3000 --window 1", 3),
                            ("blk2", "--policy lag --lag-ms 0 --window 2", 7)) for st in ("en2zh", "zh2en", "ali")]
EXP += [("S_sdm", f"{V9}/sdm_S_sdm", f"{EV}/gstar_v9_20261104/sdm/windows_test_online_sdm.jsonl", T["en2zh"], "--policy wait --window 1", "en2zh", 5),
        ("Rel_sdm", f"{V9}/sdm_Rel_sdm", f"{EV}/gstar_v9_20261104/sdm/windows_test_online_sdm.jsonl",
         f"{EV}/thinker_gstar_s2dlnosd_en2zh_v8/thinker_merged", "--policy wait --zero-sd --window 1", "en2zh", 5)]


def got(d):
    g = set()
    for f in glob.glob(f"{d}/hyp_*.jsonl"):
        for ln in open(f, encoding="utf-8"):
            try:
                g.add(json.loads(ln)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return g


def alive(pattern):
    return any(a.startswith(pattern) for a in subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines())


def main():
    rem = {}
    for name, d, wf, *_ in EXP:
        rem[name] = sum(1 for ln in open(wf) if json.loads(ln)["id"] not in got(d))
    dec = sum(1 for a in subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
              if a.startswith("/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python") and "gstar_si_stream_online" in a)
    up = alive("python3 reports/gstar_si_v9/memq9b.py")
    print(json.dumps({"remaining": rem, "decoders": dec, "launcher": up}))
    if up:
        return 0
    rounds = [int(Path(f).stem.split("_r")[-1]) for f in glob.glob(f"{Q}/*_resume_r*.json")]
    N = max(rounds + [2]) + 1
    cancel = [j["id"] for f in glob.glob(f"{Q}/*_resume_r*.json") for j in json.load(open(f))]
    jobs = []
    for name, d, wf, th, pol, st, prio in EXP:
        W = [ln for ln in open(wf) if json.loads(ln)["id"] not in got(d)]
        if not W:
            continue
        rw = f"{d}/resume_r{N}.windows.jsonl"
        with open(rw, "x") as fh:
            fh.writelines(W)
        n = max(1, round(len(W) / 120))
        dr = "en2zh" if st == "en2zh" else "zh2en"
        for k in range(n):
            jobs.append({"id": f"m9r{N}_{name}_s{k}", "gpus": 2, "need": 46000, "prio": prio,
                         "cmd": f"{PY} --thinker {th} --windows {rw} --out {d}/hyp_r{N}s{k}.jsonl --shard {k} --num-shards {n} "
                                f"{pol} --post-key lowlat --dir {dr} --batch 16 --max-gib 34 > {d}/infer_r{N}s{k}.log 2>&1"})
    with open(f"{Q}/cancel", "a") as fh:
        fh.write("\n".join(cancel) + "\n")
    with open(f"{Q}/0{30 + N}_resume_r{N}.json", "x") as fh:
        json.dump(jobs, fh, indent=1)
    with open(LOG, "a") as fh:
        fh.write(subprocess.check_output(["date", "-u", "+%m-%d %H:%M:%S"], text=True).strip() +
                 f" note launcher found dead; kick.py round r{N}: {len(jobs)} resume jobs, {dec} decoders were running\n")
    subprocess.Popen(f"setsid nohup python3 reports/gstar_si_v9/memq9b.py {Q} {LOG} > logs/multispk/v9/memq9_r{N}.stdout 2>&1 < /dev/null &",
                     shell=True)
    if not alive("bash reports/gstar_si_v9/trigger_p14.sh") and not os.path.exists(f"{Q}/040_p14_seeds.json"):
        subprocess.Popen("setsid nohup bash reports/gstar_si_v9/trigger_p14.sh >> logs/multispk/v9/trigger_p14.log 2>&1 < /dev/null &",
                         shell=True)
    print(f"RESPAWNED round r{N}: {len(jobs)} jobs")
    return 3


if __name__ == "__main__":
    sys.exit(main())
