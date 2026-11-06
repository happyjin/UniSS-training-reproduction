"""Take-over of cards 4, 5, 6 (user instruction 2026-11-06 23:40 UTC: the user obtained them via K8s; kenxsong's
zh2en TTS pipeline `ktrain_hok_to1000` is still running there and must finish on its own). Detached, root,
independent of the executor:
  trigger: no GPU process on cards 4-6 whose cmdline contains TRIGGER has been seen for GONE_S (2 min);
  action:  <qdir>/cards.json := {"ours": [0..7], "headroom": {}} (gpuq_w3 and idle_fallback re-read it every loop:
           4-6 become normal W3PO cards with the normal 8 GiB headroom, keepers and idle fallback cover them), then
           start the keepers on 4-6 at once (cardctl.Card.start_keeper). kenxsong's other processes still there are
           left alone; the keeper releases whenever free < 12 GiB and yield_guard.py keeps protecting 4-6.
Writes <qdir>/takeover_4_6.done and exits. Touches no process of anyone else.

    setsid nohup python takeover_4_6.py <qdir> >> <qdir>/takeover_4_6.log 2>&1 < /dev/null &
    python takeover_4_6.py <qdir> --dry-run   # one check, prints what it sees, changes nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cardctl import Card, cmdline, log, smi_apps, smi_cards  # noqa: E402

CARDS = (4, 5, 6)
TRIGGER = "ktrain_hok_to1000"
GONE_S = 120
TICK = 10


def pipeline_pids() -> list[int]:
    smi = smi_cards()
    uu = {smi[c]["uuid"] for c in CARDS}
    return sorted({pid for u, pid, _ in smi_apps() if u in uu and TRIGGER in cmdline(pid)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("qdir")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    q = Path(a.qdir)
    done = q / "takeover_4_6.done"
    if done.exists():
        log("already done")
        return
    if a.dry_run:
        print(f"pipeline GPU processes on 4-6: {pipeline_pids()}")
        return
    log(f"take-over watcher up, pid {os.getpid()}: waiting for {TRIGGER} to leave cards {CARDS}")
    gone_since = None
    last = None
    while True:
        p = pipeline_pids()
        if p != last:
            log(f"pipeline GPU processes on 4-6: {p}")
            last = p
        if p:
            gone_since = None
        else:
            gone_since = gone_since or time.time()
            if time.time() - gone_since >= GONE_S:
                break
        time.sleep(TICK)
    tmp = q / ".cards.json.tmp"
    tmp.write_text(json.dumps({"ours": list(range(8)), "headroom": {}, "since": time.time()}))
    os.replace(tmp, q / "cards.json")
    log("cards.json: all 8 cards ours, normal headroom")
    for c in CARDS:
        Card(q, c).start_keeper()
    with open(done, "x") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    log("TAKEOVER_4_6_DONE")


if __name__ == "__main__":
    main()
