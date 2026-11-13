#!/usr/bin/env python
"""--stack-cards (thinker_t_eval_orch.py; user 2026-11-13 ~06:50 / ~06:55 Prague: "都改成叠加跑dev5速查",
"0-3叠加第一个训练的dev5速查，4-7叠加第二个训练的dev5速查"): while FILE exists, this orchestrator's shard decodes
(ids <name>__dec<k>; never <name>__dec_heldout, scorers, talkers, ASR or CPU jobs) go ONLY to the cards listed in FILE
(the (b) training cards 4-7), stacked next to the training:
  one decode per card   no compute process on the card except the training ("pretrain_" in the cmdline) and our
                        placeholders, no pending ledger entry (any owner) on it, none of our own jobs on it
  135 GiB               min(135, total - 5.4) - (max real use over the last 180 s) - (other owners' pending ledger
                        MiB) >= 64 GiB (the --cap135 decode size); re-checked under gpu_launch.lock right before the start
  order                 a group of orchestrators (--stack-group DIR) shares wants.jsonl (append-only): one of them
                        starts a decode on a stack card only when no other live member (want line < 90 s old, ready
                        decodes > 0) has a higher effective rank. --stack-rank: the checkpoint step (newest first), or
                        >= 1e6 (always first) / <= -1e6 (always last). DIR/order = "oldest" flips the checkpoint steps
                        (oldest first) at the next tick; anything else / no file = newest first
Every other job is planned exactly as without the flag. FILE absent (the watchdog renamed it: STACK47_WITHDRAWN) ->
the shard decodes are planned as without the flag again (the cards in FILE stay excluded through <run>/exclude_cards or
the shared exclude file), at the next tick. Default (no --stack-cards): nothing here runs; behaviour unchanged.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
from pathlib import Path

DEC_RE = re.compile(r".*__dec\d+$")
TOP, BOTTOM = 1_000_000, -1_000_000
WANT_TTL_S = 90
HIST_S = 180
TRAIN_PAT = ("pretrain_",)
PLACEHOLDER_PAT = ("scripts/gpu_load/train_mist_workload.py", "gpu_keeper.py")


def is_shard_decode(jid: str) -> bool:
    return bool(DEC_RE.fullmatch(jid))


def read_cards(path: str) -> set[int] | None:
    """cards in FILE, or None when FILE does not exist (stacking off)."""
    try:
        return {int(x) for x in Path(path).read_text().split() if x.strip().isdigit()}
    except OSError:
        return None


def eff_rank(rank: float, order: str) -> float:
    if rank >= TOP or rank <= BOTTOM or order != "oldest":
        return rank
    return -rank


def gate_open(wants: dict, me: str, rank: float, order: str, t: float) -> tuple[bool, str]:
    """wants: owner -> last want record {"rank", "n", "t"}. Open unless another live member with ready decodes has a
    higher effective rank."""
    mine = eff_rank(rank, order)
    for owner, r in wants.items():
        if owner == me or t - float(r.get("t", 0)) > WANT_TTL_S or int(r.get("n", 0)) <= 0:
            continue
        if eff_rank(float(r.get("rank", 0)), order) > mine:
            return False, owner
    return True, ""


def read_wants(path: Path, tail_bytes: int = 262144) -> dict:
    out: dict = {}
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - tail_bytes))
            data = fh.read().decode(errors="ignore").splitlines()
    except OSError:
        return out
    if size > tail_bytes:
        data = data[1:]
    for l in data:
        try:
            r = json.loads(l)
        except ValueError:
            continue
        out[r.get("owner")] = r
    return out


def append_line(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (json.dumps(rec) + "\n").encode())
    finally:
        os.close(fd)


def card_procs() -> dict[int, list[tuple[int, float, str]]]:
    """card -> [(pid, GiB, kind)] for every compute process; kind train | placeholder | other."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                       text=True, timeout=60).stdout
    u2c = {}
    for l in q.strip().split("\n"):
        p = [x.strip() for x in l.split(",")]
        if len(p) == 2 and p[0].isdigit():
            u2c[p[1]] = int(p[0])
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60).stdout
    out: dict[int, list] = {c: [] for c in u2c.values()}
    for l in a.strip().split("\n"):
        p = [x.strip() for x in l.split(",")]
        if len(p) != 3 or not p[0].isdigit() or p[1] not in u2c:
            continue
        pid = int(p[0])
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            cmd = ""
        kind = ("train" if any(x in cmd for x in TRAIN_PAT) else
                "placeholder" if any(x in cmd for x in PLACEHOLDER_PAT) else "other")
        mem = float(p[2]) / 1024 if p[2].replace(".", "").isdigit() else 0.0
        out[u2c[p[1]]].append((pid, mem, kind))
    return out


def card_free(total: float, hist_max_used: float, ledger_other_gib: float, cap_total: float = 135.0,
              margin: float = 5.4) -> float:
    return min(cap_total, total - margin) - hist_max_used - ledger_other_gib


def eligible(c: int, procs: dict, ledger: list[dict], own_cards: set[int], total: float, hist_max_used: float,
             need: float = 64.0) -> tuple[bool, str]:
    """one decode per stack card, within min(135, total - 5.4) by real use."""
    if c in own_cards:
        return False, "own job there"
    oth = [p for p in procs.get(c, []) if p[2] == "other"]
    if oth:
        return False, f"process {oth[0][0]} there"
    if any(r.get("card") == c for r in ledger):
        return False, "pending ledger entry"
    fr = card_free(total, hist_max_used, 0.0)
    if fr < need:
        return False, f"free {fr:.1f} < {need}"
    return True, ""


def apply_stack_cards(o, path: str, rank: float, group: str | None, fo, ledger_path: str) -> None:
    """wrap o.plan / o.launch_gpu (see the module docstring). fo = finalpost_dev_orch."""
    me = o.ledger_owner
    gdir = Path(group) if group else None
    hist: list[tuple[float, dict]] = []
    last_want = {"n": None, "t": 0.0}
    orig_plan, orig_launch = o.plan, o.launch_gpu

    def order() -> str:
        try:
            return (gdir / "order").read_text().strip() if gdir else "newest"
        except OSError:
            return "newest"

    def own_cards() -> set[int]:
        return {j.card for j in o.jobs if j.state == "run" and j.gpu and j.card is not None}

    def want(n: int) -> None:
        if gdir is None:
            return
        t = time.time()
        if n != last_want["n"] or t - last_want["t"] >= 30:
            append_line(gdir / "wants.jsonl", {"owner": me, "rank": rank, "n": n, "t": round(t, 1),
                                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            last_want.update(n=n, t=t)

    def plan(snap, ready):
        t = time.time()
        hist[:] = [(tt, u) for tt, u in hist if t - tt < HIST_S] + [(t, {c: v["used"] for c, v in snap["cards"].items()})]
        cards = read_cards(path)
        if cards is None:
            want(0)
            return orig_plan(snap, ready)
        dec = sorted([j for j in ready if j.kind == "dec" and is_shard_decode(j.id)], key=lambda x: (x.prio, x.order))
        rest = [j for j in ready if j not in dec]
        out = orig_plan(snap, rest) if rest else []
        want(len(dec))
        if not dec:
            return out
        if gdir is not None:
            ok, who = gate_open(read_wants(gdir / "wants.jsonl"), me, rank, order(), t)
            if not ok:
                return out
        procs = card_procs()
        led = [r for r in fo.ledger_pending(ledger_path)]
        own = own_cards() | {c for _, c in out}
        free_cards = []
        for c in sorted(cards):
            if c not in snap["cards"]:
                continue
            hm = max(u.get(c, 0.0) for _, u in hist)
            ok, _ = eligible(c, procs, led, own, snap["cards"][c]["total"], hm)
            if ok:
                free_cards.append((card_free(snap["cards"][c]["total"], hm, 0.0), c))
        for (_, c), j in zip(sorted(free_cards, reverse=True), dec):
            out.append((j, c))
        return out

    def launch_gpu(plan_):
        cards = read_cards(path)
        st = [(j, c) for j, c in plan_ if cards is not None and c in cards and j.kind == "dec" and is_shard_decode(j.id)]
        rest = [(j, c) for j, c in plan_ if (j, c) not in st]
        if rest:
            orig_launch(rest)
        if not st:
            return
        lk = open(fo.LOCK, "a")
        t_lock = time.time()
        while True:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t_lock > 900:
                    o.log("stack: gpu_launch.lock busy for 15 min; skipping this round")
                    lk.close()
                    return
                time.sleep(2)
        try:
            for j, c in st:
                if read_cards(path) is None:
                    o.log(f"stack: {path} gone at re-check; {j.id} not started")
                    break
                snap = o.snapshot()
                procs = card_procs()
                led = fo.ledger_pending(ledger_path)
                hm = max([u.get(c, 0.0) for _, u in hist] + [snap["cards"][c]["used"]])
                ok, why = eligible(c, procs, led, own_cards(), snap["cards"][c]["total"], hm)
                if not ok:
                    o.log(f"stack: skip {j.id} on card {c}: {why} at re-check")
                    continue
                o.log(f"stack: start {j.id} on stack card {c} (rank {rank}, order {order()}; card used "
                      f"{snap['cards'][c]['used']:.1f} GiB, 180 s max {hm:.1f})")
                o.start(j, c)
                if o.ledger:
                    fo.ledger_append(o.ledger, {"owner": o.ledger_owner, "id": j.id, "card": c,
                                                "mib": int(fo.NEED[j.kind] * 1024), "vllm": False, "state": "pending",
                                                "pid": j.sid})
                time.sleep(2)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
            lk.close()

    o.plan, o.launch_gpu = plan, launch_gpu
    o.log(f"--stack-cards {path} (now {sorted(read_cards(path) or []) or 'absent'}): shard decodes only there while it "
          f"exists, one per card, min(135, total - 5.4) by 180 s real use; rank {rank}, group {group}, order {order()}")
