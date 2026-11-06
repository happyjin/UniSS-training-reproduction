"""Memory-aware job queue for W3PO (user speed policy 2026-11-06): starts each job as soon as real free memory
(nvidia-smi) allows, with >= HEADROOM GiB left on every card it uses, GPU starts STAGGER s apart; never yields to,
touches or kills other users' jobs.
Reservation keepers (user order 2026-11-06 20:35 UTC, scripts/gpu_keeper.py, one per card in KEEPER_CARDS, state in
<qdir>/keeper/card<k>.state): their held memory counts as available for our jobs; before a start on such a card the
queue asks the keeper (<qdir>/keeper/card<k>.req) to release down to need + 8 GiB real free, waits up to
KEEPER_WAIT s for nvidia-smi to show it (a card running the user's idle workload, see idle_fallback.py, is first
handed back to its keeper with cardctl.Card.to_keeper), starts the job, and for KEEPER_SETTLE s the keeper keeps the job's not yet allocated room free (it tops up the rest).

Jobs are JSON files in <qdir>/jobs/ (added at any time; started in file-name order once their deps are done):
  {"id": str, "cmd": str, "mem_gb": float (per card; 0 = CPU job), "cards": int (default 1),
   "only": [cards allowed] (optional), "deps": [ids], "retries": n (optional, restart a failed job), "exclusive_gb": float (optional: a multi-card job also needs
   at most this much already used on each chosen card, e.g. SFT on lightly loaded cards)}
The command runs as `bash -c` in its own session with CUDA_VISIBLE_DEVICES set, stdout/stderr to <qdir>/logs/<id>.log,
and its exit code is written to <qdir>/state/<id>.rc by the wrapper itself (so a queue restart sees finished jobs).
A job whose dep failed is marked skipped. A multi-card job that is ready but does not fit reserves its allowed cards:
GPU jobs after it in the order are not started there, so the cards drain for it. Memory of a job started < RESERVE_S ago is reserved on its cards.
Stop the queue: touch <qdir>/STOP (running jobs continue).

    setsid nohup python gpuq_w3.py <qdir> > <qdir>/queue.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cardctl  # noqa: E402  (keeper / workload hand-over of our cards)

HEADROOM = 8.6      # GB (8 GiB)
HEADROOM_CARD = {4: 25.0, 5: 25.0, 6: 25.0}   # kenxsong's cards: his processes fluctuate by ~15 GB (cards.json)
STAGGER = 60
RESERVE_S = 300
MAX_CPU = 12
TOTAL_GB = 143.7
KEEPER_CARDS = (0, 1, 2, 3, 7)   # refreshed every loop from <qdir>/cards.json (cardctl.load_cfg; takeover_4_6.py)


def refresh_cfg(q: Path):
    global KEEPER_CARDS, HEADROOM_CARD
    cfg = cardctl.load_cfg(q)
    KEEPER_CARDS, HEADROOM_CARD = cfg["ours"], cfg["headroom"]
KEEPER_WAIT = 150
KEEPER_SETTLE = 180
GB_PER_MIB = 1.048576 / 1024


def keeper_held(q: Path) -> dict:
    """card -> GB held by its keeper (fresh state files only)."""
    res = {}
    for c in KEEPER_CARDS:
        try:
            s = json.loads((q / "keeper" / f"card{c}.state").read_text())
            if time.time() - float(s["time"]) < 20:
                res[c] = float(s["held_mib"]) * GB_PER_MIB
        except Exception:
            pass
    return res


def keeper_req(q: Path, c: int, **kw):
    """kw: release_to_mib + release_until, or pending_mib + pending_since + pending_until (gpu_keeper.py)."""
    tmp = q / "keeper" / f".card{c}.req.tmp"
    tmp.write_text(json.dumps(kw))
    os.replace(tmp, q / "keeper" / f"card{c}.req")


def make_room(q: Path, cards: list, need: float) -> bool:
    """Ask the keepers on `cards` to release until real free >= need + 8 GiB; wait for nvidia-smi to show it."""
    kc = [c for c in cards if c in keeper_held(q)]
    if not kc:
        return True
    want = need + HEADROOM
    for c in kc:
        keeper_req(q, c, release_to_mib=(want + 1.0) / GB_PER_MIB, release_until=time.time() + KEEPER_WAIT)
    t0 = time.time()
    while time.time() - t0 < KEEPER_WAIT:
        mem = free_gb()
        if all(mem[c][0] >= want for c in cards):
            return True
        time.sleep(3)
    for c in kc:   # give up: cancel the request, the keeper may top up again
        keeper_req(q, c)
    return False


def free_gb() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.strip().splitlines():
        i, u, t = [x.strip() for x in line.split(",")]
        res[int(i)] = (float(t) - float(u)) / 1024 * 1.048576, float(u) / 1024 * 1.048576
    return res


def log(q: Path, msg: str):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def main():
    q = Path(sys.argv[1])
    for d in ("jobs", "state", "logs"):
        (q / d).mkdir(parents=True, exist_ok=True)
    started = {}      # id -> (time, cards, mem)
    last_gpu_start = time.time()   # a restart must not start a GPU job right after one started before it
    low_warned = {}
    for f in (q / "state").glob("*.started"):
        started[f.stem] = (0.0, [], 0.0)
    log(q, f"queue up, {len(started)} jobs already started before")
    while not (q / "STOP").exists():
        refresh_cfg(q)
        jobs = []
        for f in sorted((q / "jobs").glob("*.json")):
            try:
                jobs.append(json.loads(f.read_text()))
            except Exception as e:   # a file being written
                continue
        status = {}
        for j in jobs:
            jid = j["id"]
            rc = q / "state" / f"{jid}.rc"
            if (q / "state" / f"{jid}.skipped").exists():
                status[jid] = "failed"
            elif rc.exists():
                status[jid] = "done" if rc.read_text().strip() == "0" else "failed"
                n_fail = len(list((q / "state").glob(f"{jid}.rc.fail*")))
                if status[jid] == "failed" and n_fail < int(j.get("retries", 0)):
                    # e.g. a vLLM memory-profiling assert when a neighbour freed memory: start it again
                    os.replace(rc, q / "state" / f"{jid}.rc.fail{n_fail + 1}")
                    st = q / "state" / f"{jid}.started"
                    if st.exists():
                        os.replace(st, q / "state" / f"{jid}.started.fail{n_fail + 1}")
                    started.pop(jid, None)
                    status[jid] = "pending"
                    log(q, f"retry {jid} (attempt {n_fail + 2})")
            elif jid in started:
                status[jid] = "running"
            else:
                status[jid] = "pending"
        n_cpu = sum(1 for j in jobs if status[j["id"]] == "running" and not j.get("mem_gb"))
        mem = free_gb()
        now = time.time()
        for c, (fr, _) in mem.items():   # a neighbour (or one of ours) grew: flag it for the operator
            if fr < 4.0 and now - low_warned.get(c, 0) > 600:
                low_warned[c] = now
                log(q, f"LOW_MEM card {c}: {fr:.1f} GB free")
        for c, h in keeper_held(q).items():   # our keepers' blocks are available to our jobs
            if c in mem:
                mem[c] = (mem[c][0] + h, mem[c][1] - h)
        wl_cards = {}
        try:
            apps, uu = cardctl.smi_apps(), cardctl.smi_cards()
            for c in KEEPER_CARDS:   # the user's workload on an idle card of ours is ours to take back
                cc = cardctl.Card(q, c)
                if cc.workload_pid() and c in mem:
                    h = cc.workload_mib(uu[c]["uuid"], apps) * GB_PER_MIB
                    wl_cards[c] = cc
                    mem[c] = (mem[c][0] + h, mem[c][1] - h)
        except Exception as e:
            log(q, f"workload view failed: {e}")
        for jid, (t, cards, m) in started.items():
            if status.get(jid) == "running" and now - t < RESERVE_S:
                for c in cards:
                    if c in mem:
                        mem[c] = (mem[c][0] - m, mem[c][1] + m)
        blocked = set()   # cards a waiting multi-card job (earlier in order) is draining: later GPU jobs keep off them
        for j in jobs:
            jid = j["id"]
            if status[jid] != "pending":
                continue
            deps = [status.get(d, "missing") for d in j.get("deps", [])]
            if any(d == "failed" for d in deps):
                (q / "state" / f"{jid}.skipped").write_text("dep failed\n")
                log(q, f"skip {jid} (dependency failed)")
                status[jid] = "failed"
                continue
            if any(d != "done" for d in deps):
                continue
            need = float(j.get("mem_gb", 0))
            env = dict(os.environ)
            cards = []
            if need <= 0:
                if n_cpu >= MAX_CPU:
                    continue
                env["CUDA_VISIBLE_DEVICES"] = ""
                n_cpu += 1
            else:
                if now - last_gpu_start < STAGGER:
                    continue
                allowed = j.get("only", list(mem))
                k = int(j.get("cards", 1))
                ok = [c for c in allowed if c in mem and c not in blocked and mem[c][0] - need >= HEADROOM_CARD.get(c, HEADROOM)
                      and (j.get("exclusive_gb") is None or mem[c][1] <= j["exclusive_gb"])]
                if len(ok) < k:
                    if k > 1:
                        blocked |= set(allowed)
                    continue
                ok.sort(key=lambda c: -mem[c][0]) if k == 1 else ok.sort()
                cards = ok[:k] if k > 1 else ok[:1]
                for c in cards:   # workload -> keeper first (the keeper is up before the workload lets go)
                    if c in wl_cards:
                        with wl_cards[c].lock():
                            log(q, f"card {c}: taking it back from the workload for {jid}")
                            wl_cards[c].to_keeper()
                if not make_room(q, cards, need):
                    log(q, f"keepers could not make room for {jid} on {cards}; retry later")
                    last_gpu_start = time.time()
                    break
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cards))
                for c in cards:
                    mem[c] = (mem[c][0] - need, mem[c][1] + need)
                last_gpu_start = now
            rc = q / "state" / f"{jid}.rc"
            wrapped = f"{j['cmd']}\necho $? > {rc}"
            with open(q / "logs" / f"{jid}.log", "a") as fh:
                subprocess.Popen(["bash", "-c", wrapped], env=env, stdout=fh, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True,
                                 cwd="/opt/dlami/nvme/jasonleeeli/projects/UniSS")
            (q / "state" / f"{jid}.started").write_text(json.dumps({"time": now, "cards": cards}))
            started[jid] = (now, cards, need)
            status[jid] = "running"
            for c in cards:   # the keeper leaves the new job its room until its memory has settled
                if c in KEEPER_CARDS and (q / "keeper").is_dir():
                    t = time.time()
                    keeper_req(q, c, pending_mib=need / GB_PER_MIB, pending_since=t, pending_until=t + KEEPER_SETTLE)
            log(q, f"start {jid} on cards {cards} (need {need} GB)")
            if need > 0:
                break   # one GPU start per stagger period
        time.sleep(15)
    log(q, "STOP file seen; queue exits (running jobs continue)")


if __name__ == "__main__":
    main()
