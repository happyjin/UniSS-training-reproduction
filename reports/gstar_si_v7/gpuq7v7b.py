"""gpuq7v7b (replaces gpuq7v7.py from 2026-11-01 14:35 UTC; same rules plus two job fields):
  "cards": [..]  the job may only start on these cards (a training lane and a decode lane);
  "needs": [..]  paths that must exist before the job may start (e.g. a merged model trained by an earlier queue).

gpuq7v7: the single GPU queue of the paper-v7 (MIST) branch. As scripts/gpu_queue/gpuq7.py (stacks jobs beside
other people's work, per-card memory CAP, OUR_MAX, 2-minute steadiness, 90 s between our starts on a card), plus:

  * jobs can be added while it runs: every *.json in <logdir>/add/ (a list of jobs) is read once, in name order;
    a job id seen before is ignored (logged), so a file can never re-run a job;
  * <logdir>/cancel (one job id per line) marks waiting jobs as cancelled (prereg amendments); running jobs go on;
  * it keeps running while <logdir>/FINISH does not exist, so one queue serves the whole v7 run;
  * every job runs with MIST_V7_JOB=<id> and MIST_V7_KIND=<kind> (job field "kind": train / decode / speech / cpu,
    default decode) in its environment, which the memory watchdog (mem_watchdog_v7.py) uses to tell our processes;
  * a job with "gpus": 0 runs at once on the CPU (scoring, assembling), outside the memory rules.

Job: {"id", "gpus", "mem" (MiB per card), "deps", "cmd", "kind"}; cmd is run by bash with GPUS="a b ...", GPU0..GPU7,
CVD (comma-separated cards). A failed or cancelled dependency skips the job.

    GPUQ_CARDS=0,1,2,3,4,5,6,7 CAP=120000 OUR_MAX=70000 python gpuq7v7.py <logdir>
"""
import glob
import json
import os
import subprocess
import sys
import time

logdir = sys.argv[1]
os.makedirs(os.path.join(logdir, "add"), exist_ok=True)
CARDS = {int(c) for c in os.environ["GPUQ_CARDS"].split(",")}
OUR_MAX = int(os.environ.get("OUR_MAX", 70000))
CAP = int(os.environ.get("CAP", 120000))
REPO = "/opt/dlami/nvme/jasonleeeli/projects/UniSS"
jobs, state, order, seen_files = {}, {}, [], set()
running, last_start, hist = {}, {}, {g: [] for g in CARDS}
T0 = time.time()


def stamp():
    return time.strftime("%m-%d %H:%M:%S", time.gmtime())


def say(msg):
    print(f"{stamp()} {msg}", flush=True)


def used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines()) if int(a) in CARDS}


def load_new():
    for f in sorted(glob.glob(os.path.join(logdir, "add", "*.json"))):
        if f in seen_files:
            continue
        seen_files.add(f)
        try:
            new = json.load(open(f))
        except (OSError, ValueError) as e:
            say(f"cannot read {f}: {e}")
            continue
        n = 0
        for j in new:
            if j["id"] in jobs:
                say(f"ignored duplicate job id {j['id']} in {f}")
                continue
            j.setdefault("deps", [])
            j.setdefault("kind", "decode")
            jobs[j["id"]] = j
            state[j["id"]] = "wait"
            order.append(j["id"])
            n += 1
        say(f"loaded {n} jobs from {os.path.basename(f)}")


def cancels():
    p = os.path.join(logdir, "cancel")
    if not os.path.exists(p):
        return
    for k in (l.strip() for l in open(p)):
        if k in state and state[k] == "wait":
            state[k] = "cancel"
            say(f"cancel {k}")


def start(k, cards):
    j = jobs[k]
    env = dict(os.environ, MIST_V7_JOB=k, MIST_V7_KIND=j["kind"], GPUS=" ".join(map(str, cards)),
               CVD=",".join(map(str, cards)))
    for i, c in enumerate(cards):
        env[f"GPU{i}"] = str(c)
    log = open(os.path.join(logdir, f"{k}.log"), "a")
    running[k] = subprocess.Popen(["bash", "-c", j["cmd"]], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    j["_cards"], j["_t"] = cards, time.time()
    state[k] = "run"
    say(f"start {k} on {cards}")


while not os.path.exists(os.path.join(logdir, "FINISH")) or any(s in ("wait", "run") for s in state.values()):
    load_new()
    cancels()
    now = time.time()
    for k, p in list(running.items()):
        if p.poll() is not None:
            state[k] = "done" if p.returncode == 0 else "fail"
            del running[k]
            say(f"{state[k]} {k} (rc {p.returncode})")
    for k in order:
        if state[k] == "wait" and any(state.get(d) in ("fail", "cancel") for d in jobs[k]["deps"]):
            state[k] = "fail"
            say(f"skip {k} (dependency failed or cancelled)")
    mem = used_mib()
    for g, v in mem.items():
        hist[g] = [(t, m) for t, m in hist[g] if now - t <= 120] + [(now, v)]
    ours = {g: sum(jobs[k]["mem"] for k in running if g in jobs[k].get("_cards", [])) for g in CARDS}
    pending = {g: sum(jobs[k]["mem"] for k in running if g in jobs[k].get("_cards", []) and now - jobs[k]["_t"] < 300)
               for g in CARDS}
    for k in order:
        if state[k] != "wait" or not all(state.get(d) == "done" for d in jobs[k]["deps"]):
            continue
        if not all(os.path.exists(os.path.join(REPO, p)) for p in jobs[k].get("needs", [])):
            continue
        j = jobs[k]
        if j["gpus"] == 0:
            start(k, [])
            continue
        span = lambda g: max(m for _, m in hist[g]) - min(m for _, m in hist[g])   # noqa: E731
        ok = [g for g in sorted(CARDS, key=lambda g: mem[g] + pending[g])
              if ours[g] + j["mem"] <= OUR_MAX and mem[g] + pending[g] + j["mem"] <= CAP
              and now - T0 >= 120 and span(g) < 3000 and now - last_start.get(g, 0.0) >= 90]
        if j.get("cards"):
            ok = [g for g in ok if g in j["cards"]]
        if len(ok) < j["gpus"]:
            continue
        cards = sorted(ok[: j["gpus"]])
        start(k, cards)
        for g in cards:
            last_start[g] = now
            ours[g] += j["mem"]
            pending[g] += j["mem"]
    with open(os.path.join(logdir, "state.json.tmp"), "w") as fh:
        json.dump({k: state[k] for k in order}, fh, indent=0)
    os.replace(os.path.join(logdir, "state.json.tmp"), os.path.join(logdir, "state.json"))
    time.sleep(15)
say(f"QUEUE_DONE done={sum(s == 'done' for s in state.values())} fail={sum(s == 'fail' for s in state.values())} "
    f"cancel={sum(s == 'cancel' for s in state.values())}")
