"""Idle fallback for OUR cards (user rule 2026-11-06 23:20 UTC): a card in {0,1,2,3,7} that has had no W3PO job for
IDLE_S (10 min) while gpuq_w3 has nothing running or runnable for it gets the user's no-checkpoint workload instead of
the keeper, gap-free (cardctl: the workload starts next to the keeper with --tolerate-keeper, the keeper is
stopped once the workload holds its memory); runs detached, independent of the executor. It also repairs: a card with
neither keeper nor workload gets its keeper back; a workload that has not taken over within START_S (e.g. a foreign process
holds the card) is replaced by the keeper. The way back (workload -> keeper -> job)
is done by gpuq_w3 itself before it starts a job on the card.

    setsid nohup python idle_fallback.py <qdir> >> <qdir>/idle_fallback.log 2>&1 < /dev/null &
    python idle_fallback.py <qdir> --dry-run      # one pass: print the decision per card, act on nothing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cardctl import CARDS, Card, log, smi_apps, smi_cards  # noqa: E402

IDLE_S = 600
START_S = 600
TICK = 15


def queue_view(q: Path) -> tuple[set, set]:
    """(cards with a running gpuq job, cards some runnable pending GPU job may use) -- same rules as gpuq_w3."""
    jobs = []
    for f in sorted((q / "jobs").glob("*.json")):
        try:
            jobs.append(json.loads(f.read_text()))
        except Exception:
            continue
    st = q / "state"
    status = {}
    for j in jobs:
        jid = j["id"]
        rc = st / f"{jid}.rc"
        if (st / f"{jid}.skipped").exists():
            status[jid] = "failed"
        elif rc.exists():
            status[jid] = "done" if rc.read_text().strip() == "0" else "failed"
        elif (st / f"{jid}.started").exists():
            status[jid] = "running"
        else:
            status[jid] = "pending"
    running, runnable = set(), set()
    for j in jobs:
        jid = j["id"]
        if not float(j.get("mem_gb", 0)):
            continue
        if status[jid] == "running":
            try:
                running |= set(json.loads((st / f"{jid}.started").read_text()).get("cards", []))
            except Exception:
                pass
        elif status[jid] == "pending" and all(status.get(d) == "done" for d in j.get("deps", [])):
            n_fail = len(list(st.glob(f"{jid}.rc.fail*")))
            if n_fail <= int(j.get("retries", 0)):
                runnable |= set(j.get("only", CARDS))
    return running, runnable


def decide(now: float, last_busy: float, busy: bool, keeper: bool, workload: bool, worker_up: bool,
           wl_age: float, ready: bool = False) -> str:
    """pure decision per card: 'keep' | 'start_keeper' | 'to_workload' | 'drop_keeper' | 'to_keeper'
    (tested in test_idle_fallback)."""
    if workload:
        if keeper:      # gap-free hand-over in progress: the keeper goes once the workload holds its memory
            if ready:
                return "drop_keeper"
            return "to_keeper" if wl_age > START_S else "keep"
        if not worker_up and wl_age > START_S:
            return "to_keeper"
        return "keep"
    if not keeper:
        return "start_keeper"
    if not busy and now - last_busy >= IDLE_S:
        return "to_workload"
    return "keep"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("qdir")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    q = Path(a.qdir)
    cards = {c: Card(q, c) for c in CARDS}
    last_busy = {c: time.time() for c in CARDS}    # a fresh watcher waits a full IDLE_S first
    log(f"idle fallback up (dry_run={a.dry_run}), cards {CARDS}, idle {IDLE_S}s")
    while True:
        now = time.time()
        smi, apps = smi_cards(), smi_apps()
        running, runnable = queue_view(q)
        for c, card in cards.items():
            uuid = smi[c]["uuid"]
            ours = card.our_job_pids(uuid, apps)
            busy = bool(ours) or c in running or c in runnable
            if busy:
                last_busy[c] = now
            wl = card.workload_pid()
            try:
                wl_age = now - json.loads((card.wdir / f"card{c}.json").read_text())["time"] if wl else 0.0
            except Exception:
                wl_age = 0.0
            free_mib = smi[c]["total"] - smi[c]["used"]
            act = decide(now, last_busy[c], busy, bool(card.keeper_pid()), bool(wl),
                         card.workload_started_worker(), wl_age,
                         bool(wl) and card.handover_ready(uuid, free_mib, apps))
            if a.dry_run:
                print(f"card {c}: ours={sorted(ours)} running={c in running} runnable={c in runnable} "
                      f"idle_for={now - last_busy[c]:.0f}s keeper={card.keeper_pid()} workload={wl} -> {act}")
                continue
            if act == "keep":
                continue
            with card.lock():     # re-check under the lock (gpuq may be handing the card back right now)
                running, runnable = queue_view(q)
                if act == "to_workload" and (c in running or c in runnable or card.our_job_pids(uuid)):
                    last_busy[c] = time.time()
                    continue
                log(f"card {c}: {act} (idle {now - last_busy[c]:.0f}s, keeper {card.keeper_pid()}, workload {wl})")
                if act == "start_keeper":
                    card.start_keeper()
                elif act == "to_workload":
                    card.to_workload()
                elif act == "drop_keeper":
                    card.drop_keeper()
                elif act == "to_keeper":
                    card.to_keeper()
        if a.dry_run:
            return
        time.sleep(TICK)


if __name__ == "__main__":
    main()
