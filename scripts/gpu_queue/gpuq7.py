"""gpuq7: a GPU work queue that stacks jobs beside other people's work instead of waiting for empty cards.

Each job declares the memory it takes per card ("mem", MiB). A job starts on cards where
  * our own jobs on the card plus this one stay within OUR_MAX (default 62,000 MiB), so a training run that restarts
    on the card (about 76 GB) still fits;
  * the card's used memory plus what our jobs started in the last 5 minutes may not have allocated yet plus "mem"
    stays within CAP (default 138,000 MiB);
  * the card's used memory has been steady for 2 minutes (max - min < 3,000 MiB), so nobody's engine is profiling;
  * our last start on the card was >= 90 s ago (vLLM engines on one card must not profile together);
  * the card already holds >= PIGGY_MIN MiB of someone else's work (default 20,000), so empty cards stay free for the
    queue that needs whole cards; touch <logdir>/allow_empty to lift this.
Otherwise as gpuq6: jobs json [{"id", "gpus", "vllm", "deps", "cmd", "mem"}], cmd run by bash with GPUS="a b", GPU0, GPU1;
dependencies; a failed dependency skips the job. Never touches GPU holders.
Several gpuq7 queues may run at once: each writes its per-card reservations to $GPUQ7_LEDGER (default
logs/gpuq7_ledger/<pid>.json) and counts those of the other live queues as its own for OUR_MAX.

    GPUQ_CARDS=0,1,2,3,4,5,6,7 python gpuq7.py jobs.json logdir
"""
import json, os, subprocess, sys, time

jobs = {j["id"]: j for j in json.load(open(sys.argv[1]))}
logdir = sys.argv[2]
os.makedirs(logdir, exist_ok=True)
CARDS = {int(c) for c in os.environ["GPUQ_CARDS"].split(",")}
OUR_MAX = int(os.environ.get("OUR_MAX", 62000))
CAP = int(os.environ.get("CAP", 138000))
PIGGY_MIN = int(os.environ.get("PIGGY_MIN", 20000))
state = {k: "wait" for k in jobs}
running, last_start, hist = {}, {}, {g: [] for g in CARDS}
T0 = time.time()   # memory watched since
LEDGER = os.environ.get("GPUQ7_LEDGER", "/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/gpuq7_ledger")
os.makedirs(LEDGER, exist_ok=True)
ME = os.path.join(LEDGER, f"{os.getpid()}.json")


def others():
    tot = {g: 0 for g in CARDS}
    for f in os.listdir(LEDGER):
        pid = f.split(".")[0]
        if f == os.path.basename(ME) or not pid.isdigit() or not os.path.exists(f"/proc/{pid}"):
            continue
        try:
            for g, m in json.load(open(os.path.join(LEDGER, f))).items():
                if int(g) in tot:
                    tot[int(g)] += m
        except (OSError, ValueError):
            pass
    return tot


def stamp():
    return subprocess.run(["bash", "-c", "TZ=Europe/Prague date +%H:%M:%S"], capture_output=True, text=True).stdout.strip()


def used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines()) if int(a) in CARDS}


while any(s in ("wait", "run") for s in state.values()):
    now = time.time()
    for k, p in list(running.items()):
        if p.poll() is not None:
            state[k] = "done" if p.returncode == 0 else "fail"
            del running[k]
            print(f"{stamp()} {state[k]} {k}", flush=True)
    for k in jobs:
        if state[k] == "wait" and any(state[d] == "fail" for d in jobs[k]["deps"]):
            state[k] = "fail"
            print(f"{stamp()} skip {k} (dependency failed)", flush=True)
    mem = used_mib()
    for g, v in mem.items():
        hist[g] = [(t, m) for t, m in hist[g] if now - t <= 120] + [(now, v)]
    mine = {g: sum(jobs[k]["mem"] for k in running if g in jobs[k]["_cards"]) for g in CARDS}
    oth = others()
    ours = {g: mine[g] + oth[g] for g in CARDS}
    pending = {g: sum(jobs[k]["mem"] for k in running if g in jobs[k]["_cards"] and now - jobs[k]["_t"] < 300)
               for g in CARDS}
    allow_empty = os.path.exists(os.path.join(logdir, "allow_empty"))
    ready = [k for k in jobs if state[k] == "wait" and all(state[d] == "done" for d in jobs[k]["deps"])]
    for k in ready:
        j = jobs[k]
        span = lambda g: max(m for _, m in hist[g]) - min(m for _, m in hist[g])   # noqa: E731
        ok = [g for g in sorted(CARDS, key=lambda g: mem[g] + pending[g])
              if ours[g] + j["mem"] <= OUR_MAX and mem[g] + pending[g] + j["mem"] <= CAP
              and now - T0 >= 120 and span(g) < 3000 and now - last_start.get(g, 0.0) >= 90
              and (allow_empty or mem[g] - ours[g] >= PIGGY_MIN)]
        if len(ok) < j["gpus"]:
            continue
        cards = ok[: j["gpus"]]
        env = dict(os.environ, GPUS=" ".join(map(str, cards)), GPU0=str(cards[0]), GPU1=str(cards[-1]))
        log = open(os.path.join(logdir, f"{k}.log"), "w")
        running[k] = subprocess.Popen(["bash", "-c", j["cmd"]], env=env, stdout=log, stderr=subprocess.STDOUT,
                                      cwd="/opt/dlami/nvme/jasonleeeli/projects/UniSS")
        j["_cards"], j["_t"] = cards, now
        state[k] = "run"
        for g in cards:
            last_start[g] = now
            ours[g] += j["mem"]
            pending[g] += j["mem"]
        print(f"{stamp()} start {k} on {cards}", flush=True)
        mine = {g: mine[g] + (j["mem"] if g in cards else 0) for g in CARDS}
    json.dump({str(g): v for g, v in mine.items()}, open(ME + ".tmp", "w"))
    os.replace(ME + ".tmp", ME)
    time.sleep(15)
print(f"{stamp()} QUEUE_DONE done={sum(s == 'done' for s in state.values())} fail={sum(s == 'fail' for s in state.values())}",
      flush=True)
