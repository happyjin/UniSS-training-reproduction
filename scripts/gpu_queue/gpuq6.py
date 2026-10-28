"""gpuq6 (this session): gpuq5 restricted to the cards in $GPUQ_CARDS and without any GPU holder (another
experiment shares the machine and its cards drop below 2 GB between its own jobs).
gpuq5: as gpuq3, and an idle card that no waiting job can use runs its own GPU holder until a job can.
A small GPU work queue: start each job as soon as its dependencies are done and enough cards are free.
A job with "share": true (light work, e.g. ASR) takes the card with the most free memory without holding it;
while it runs, no exclusive job starts on that card (gpuq3: the light job may not have allocated memory yet).

jobs json: [{"id": str, "gpus": 1 | 2, "vllm": bool, "deps": [ids], "cmd": str}], cmd run by bash with GPUS="a b" and
GPU0/GPU1 set. A card is free when no job of ours holds it and it uses < 2 GB (someone else's work is never touched).
vLLM starts on the same card are kept >= 60 s apart (an engine profiling memory fails when another on its card
moves); cards apart do not disturb each other. The GPU holder is stopped while jobs wait
and restarted at the end.

    python gpuq.py jobs.json logdir
"""
import json, os, subprocess, sys, time

jobs = {j["id"]: j for j in json.load(open(sys.argv[1]))}
logdir = sys.argv[2]
os.makedirs(logdir, exist_ok=True)
L = os.path.dirname(os.path.abspath(__file__))   # holder_on.sh / holder_off.sh live beside this file
state = {k: "wait" for k in jobs}
running, held, last_start = {}, set(), {}


def stamp():
    return subprocess.run(["bash", "-c", "TZ=Europe/Prague date +%H:%M:%S"], capture_output=True, text=True).stdout.strip()


CARDS = {int(c) for c in os.environ["GPUQ_CARDS"].split(",")}
def used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines()) if int(a) in CARDS}


holders = set()   # gpuq4: cards running a per-card holder while no job needs them


def holders_off():
    if holders:
        subprocess.run(["bash", f"{L}/holder_off.sh", ",".join(map(str, sorted(holders)))])
        holders.clear()
        time.sleep(10)


while any(s in ("wait", "run") for s in state.values()):
    for k, p in list(running.items()):
        if p.poll() is not None:
            state[k] = "done" if p.returncode == 0 else "fail"
            held -= set(jobs[k]["_cards"])
            del running[k]
            print(f"{stamp()} {state[k]} {k}", flush=True)
    for k in jobs:   # a job whose dependency failed cannot run
        if state[k] == "wait" and any(state[d] == "fail" for d in jobs[k]["deps"]):
            state[k] = "fail"
            print(f"{stamp()} skip {k} (dependency failed)", flush=True)
    ready = [k for k in jobs if state[k] == "wait" and all(state[d] == "done" for d in jobs[k]["deps"])]
    if False:   # gpuq6: never touches holders
        subprocess.run(["bash", f"{L}/holder_off.sh", "all"])
        time.sleep(10)
    mem = used_mib()
    busy_share = {g for k2, p2 in running.items() if jobs[k2].get("share") for g in jobs[k2].get("_share_cards", [])}
    free = [g for g in sorted(mem) if g not in held and mem[g] < 2000 and g not in busy_share]   # a light job may not have allocated yet
    calm = [g for g in free if time.time() - last_start.get(g, 0.0) >= 60]
    unplaced = []
    for k in ready:
        j = jobs[k]
        if j.get("share"):   # light: the emptiest card with room, not held
            room = [g for g in sorted(mem, key=mem.get) if mem[g] < j.get("max_used_mib", 90000)
                    and time.time() - last_start.get(g, 0.0) >= 90]   # never beside an engine still profiling memory
            if not room:
                unplaced.append(k)
                continue
            cards = room[:1]
            mem[cards[0]] += j.get("need_mib", 8000)
        else:
            pool = calm if j["vllm"] else free
            if len(pool) < j["gpus"]:
                unplaced.append(k)
                continue
            cards = pool[: j["gpus"]]
            free, calm = [g for g in free if g not in cards], [g for g in calm if g not in cards]
        env = dict(os.environ, GPUS=" ".join(map(str, cards)), GPU0=str(cards[0]), GPU1=str(cards[-1]))
        log = open(os.path.join(logdir, f"{k}.log"), "w")
        running[k] = subprocess.Popen(["bash", "-c", j["cmd"]], env=env, stdout=log, stderr=subprocess.STDOUT,
                                      cwd="/opt/dlami/nvme/jasonleeeli/projects/UniSS")
        j["_cards"] = [] if j.get("share") else cards
        j["_share_cards"] = cards if j.get("share") and j.get("need_mib", 0) > 0 else []
        held |= set(j["_cards"])
        state[k] = "run"
        for g in j["_cards"]:
            last_start[g] = time.time()
        print(f"{stamp()} start {k} on {cards}", flush=True)
    mem = used_mib()
    busy_share = {g for k2 in running if jobs[k2].get("share") for g in jobs[k2].get("_share_cards", [])}
    idle = [g for g in sorted(mem) if g not in held and g not in holders and g not in busy_share and mem[g] < 1000
            and time.time() - last_start.get(g, 0.0) >= 60]
    share_waiting = any(jobs[k].get("share") for k in unplaced)
    need = min((jobs[k]["gpus"] for k in unplaced if not jobs[k].get("share")), default=None)
    if holders and (share_waiting or (need is not None and len(idle) + len(holders) >= need)):
        print(f"{stamp()} holders off {sorted(holders)} for {unplaced[:3]}", flush=True)   # they make room for a waiting job
        holders_off()
        continue
    if False:   # gpuq6: no holders
        for g in idle:
            r = subprocess.run(["bash", f"{L}/holder_on.sh", str(g)], capture_output=True, text=True)
            if "holder on" in r.stdout:
                holders.add(g)
        if idle:
            print(f"{stamp()} idle cards held: {sorted(holders)}", flush=True)
    time.sleep(10)
    continue
    for k in ready:
        j = jobs[k]
        pool = calm if j["vllm"] else free
        if len(pool) < j["gpus"]:
            continue
        cards = pool[: j["gpus"]]
        free, calm = [g for g in free if g not in cards], [g for g in calm if g not in cards]
        env = dict(os.environ, GPUS=" ".join(map(str, cards)), GPU0=str(cards[0]), GPU1=str(cards[-1]))
        log = open(os.path.join(logdir, f"{k}.log"), "w")
        running[k] = subprocess.Popen(["bash", "-c", j["cmd"]], env=env, stdout=log, stderr=subprocess.STDOUT,
                                      cwd="/opt/dlami/nvme/jasonleeeli/projects/UniSS")
        j["_cards"] = cards
        held |= set(cards)
        state[k] = "run"
        for g in cards:
            last_start[g] = time.time()
        print(f"{stamp()} start {k} on {cards}", flush=True)
    time.sleep(10)
print(f"{stamp()} QUEUE_DONE done={sum(s == 'done' for s in state.values())} fail={sum(s == 'fail' for s in state.values())}",
      flush=True)
# gpuq6: no holder at the end
