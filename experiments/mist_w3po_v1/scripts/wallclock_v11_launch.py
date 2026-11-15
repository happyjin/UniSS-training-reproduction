#!/usr/bin/env python
"""v11 P1-1 (docs/mist_v11_plan.zh-CN.md §3; mock ACL review M5; prereg reports/mist_w3po_v1/v11_p11_wallclock_prereg.zh-CN.md):
uncontended whole-meeting compute measurement of B+PA and of the method M (stage 2 s3500, wait bias -1).

Meetings (fixed before any run): AMI test, both directions: the three of the 16 meetings at the median of the
zh2en online span (ranks 7-9: IS1009b, TS3003b, EN2002a); AliMeeting test: ranks 10-11 of 20 (R8008_M8014, R8005_M8008).

Jobs, one card each, run only on a card that holds nothing but the no-checkpoint placeholder (stopped by PID right
before the job starts; the guard refills the card afterwards):
  T_<sys>_<set>   the Thinker of <sys> (b: deadline baseline = identity ckpt of the joint s6000 base; m: stage 2 s3500
                  + --wait-bias -1) decoding the chosen meetings with --batch 1 under timed_reads.py (test-decode flags
                  otherwise); --keep-free-gib 16 so nothing else fits on the card.
  K_<sys>_<set>   the PA Talker (zh2en PL-EM export; En->Zh export_i0400) + its CTC scorer on one card, speaking the
                  existing test talker input of <sys> (E2's B+PA run / wb_test's method run) restricted to the chosen
                  meetings, under timed_talker.py (one generate call per block), --keep-free-gib 16.
A co-tenancy monitor samples nvidia-smi every 5 s; a job during which another process (not of the job's own process
tree, > 1 GiB) shared its card is "contended" and is run again (at most 3 attempts; every attempt is kept). Gate: the
first job starts only after this fork's P0-4 test (WB_TEST_S4000_DONE or a FAILED_* file) so it does not measure next
to it. Afterwards wallclock_v11.py per system and set on the cleanest attempt; summary <OUT>/wallclock_summary.json,
marker <OUT>/WALLCLOCK_V11_DONE. Refuses to overwrite the summary; finished jobs (<OUT>/<job>.ok) are not rerun.

    python experiments/mist_w3po_v1/scripts/wallclock_v11_launch.py [--dry]
Run detached: setsid nohup ... > logs/mist_v11/wallclock/launch.log 2>&1 < /dev/null &
Checked with --dry and the unit tests of evaluation/test_wallclock_v11.py only.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
sys.path.insert(0, str(REPO / "experiments/mist_w3po_v1/scripts"))
os.chdir(REPO)
import finalpost_dev_orch as fo  # noqa: E402
import talker_crf_orch as tc  # noqa: E402

OUT = Path("eval_outputs/mist_w3po_v1/wallclock_v11")
LOGD = Path("logs/mist_v11/wallclock")
JL = "logs/mist_w3po_v1/joint"
GATE_DONE = Path(f"{JL}/wb/WB_TEST_S4000_DONE")
GATE_FAILDIR = Path(f"{JL}/wb/test_s4000")
JO = "eval_outputs/mist_w3po_v1/joint"
BASE = f"{JO}/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged"
IDN = f"{JO}/joint_v8h6k_20261113T073658Z/merged_s06000/identity"
CK = "eval_outputs/mist_w3po_v1/crf/train/joint_v2p_20261113T120415Z/ckpt/step_03500"
SPD = "eval_outputs/mist_w3po_v1/speech"
WD = "eval_outputs/mist_w3po_v1/data"
SETS = {
    "ami": {"dir": "zh2en", "win": f"{WD}/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl",
            "meetings": "IS1009b,TS3003b,EN2002a",
            "talker": "eval_outputs/mist_w3po_v1/talker_crf/talker_plem_20261111T001759Z/export/uniss", "lang": ""},
    "ali": {"dir": "zh2en", "win": f"{WD}/windows_test_c2g06_v1/windows_test_online_ali.jsonl",
            "meetings": "R8008_M8014,R8005_M8008",
            "talker": "eval_outputs/mist_w3po_v1/talker_crf/talker_plem_20261111T001759Z/export/uniss", "lang": ""},
    "en": {"dir": "en2zh", "win": f"{WD}/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl",
           "meetings": "IS1009b,TS3003b,EN2002a",
           "talker": "eval_outputs/mist_w3po_v1/talker_crf/talker_plem_en2zh_20261114T045019Z/export_i0400/uniss",
           "lang": "--lang zh "},
}
SYS = {"b": {"tin": "jtest_b{s}_joint_v8h6k_20261113T073658Z_s06000__rlplem",
             "dec": f"-m experiments.mist_w3po_v1.crf.decode_crf_ckpt_finalpost", "pre": f"--crf-ckpt {IDN}"},
       "m": {"tin": "jtest_m{s}_joint_v2p_20261113T120415Z_s03500_wbzm1em1__rlplem",
             "dec": f"-m experiments.mist_w3po_v1.crf.decode_crf_ckpt_waitbias", "pre": f"--wait-bias -1.0 --crf-ckpt {CK}"}}
MSC = "export MODELSCOPE_CACHE=/opt/dlami/nvme/jasonleeeli/cache/modelscope; "
KEEP = 16
PORT0 = 29700
MAX_ATT = 3
FOREIGN_MIB = 1024


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def thinker_cmd(sy: str, s: str, att: int) -> str:
    S, Y = SETS[s], SYS[sy]
    p = OUT / f"T_{sy}_{s}.a{att}"
    target = Y["dec"].split()[-1]
    return (fo.ENV + MSC + f"{fo.PY} -m experiments.mist_w3po_v1.evaluation.timed_reads --target {target} "
            f"--reads-out {p}.reads.jsonl --keep-free-gib {KEEP} -- {Y['pre']} --final-read-post close_detect "
            f"--thinker {BASE} --windows {S['win']} --out {p}.hyp.jsonl --policy wait --lag-ms 2000 --context 0 "
            f"--dir {S['dir']} --batch 1 --max-gib 120 --shard 0 --num-shards 1 --shard-by meeting "
            f"--meetings {S['meetings']} --loop-guard-k 6 --loop-guard-maxp 12 --budget-tps 25 --budget-slack 50 "
            f"> {p}.log 2>&1")


def talker_cmd(sy: str, s: str, att: int, port: int) -> str:
    S = SETS[s]
    p = OUT / f"K_{sy}_{s}.a{att}"
    tin = OUT / f"tin_{sy}_{s}"
    return (fo.ENV + MSC + f"{fo.PY} -m experiments.mist_w3po_v1.evaluation.si_ctc_scorer {S['lang']}--port {port} "
            f"> {p}.scorer.log 2>&1 & SP=$!; " + tc.wait_port(port)
            + f"VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=16 {fo.PYV} -m experiments.mist_w3po_v1.evaluation.timed_talker "
              f"--timing-out {p}.timing.jsonl --keep-free-gib {KEEP} -- --model {S['talker']} --direction {S['dir']} "
              f"--window 1 --output {p}/talker --text-policy file --blocks-file {tin}/blocks.jsonl "
              f"--items-file {tin}/items.jsonl {tc.EVAL_COMMON} --backlog-token --select-k 2 "
              f"--scorer-url http://127.0.0.1:{port}/score --shard 0 --num-shards 1 > {p}.talk.log 2>&1; "
              f"rc=$?; kill $SP 2>/dev/null; wait $SP; exit $rc")


def prep_tin(sy: str, s: str) -> int:
    """the existing test talker input of <sy> restricted to the chosen meetings (new dir; refuses to overwrite)."""
    src = Path(SPD) / SYS[sy]["tin"].format(s=s) / "talker_in"
    dst = OUT / f"tin_{sy}_{s}"
    ms = set(SETS[s]["meetings"].split(","))
    if dst.exists():
        return sum(1 for _ in open(dst / "items.jsonl"))
    dst.mkdir(parents=True)
    n = 0
    for f in ("items.jsonl", "blocks.jsonl"):
        with open(src / f, encoding="utf-8") as fi, open(dst / f, "x", encoding="utf-8") as fh:
            for l in fi:
                if json.loads(l)["id"].split("_on_")[0] in ms:
                    fh.write(l)
                    n += f == "items.jsonl"
    return n


def smi() -> tuple[dict, dict]:
    """card -> [(pid, MiB)], card -> util %."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60).stdout
    u2c, util = {}, {}
    for l in q.strip().split("\n"):
        i, u, ut = [x.strip() for x in l.split(",")]
        u2c[u] = int(i)
        util[int(i)] = float(ut) if ut.replace(".", "").isdigit() else None
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60).stdout
    procs = {c: [] for c in u2c.values()}
    for l in a.strip().split("\n"):
        p = [x.strip() for x in l.split(",")]
        if len(p) == 3 and p[0].isdigit() and p[1] in u2c:
            procs[u2c[p[1]]].append((int(p[0]), float(p[2]) if p[2].replace(".", "").isdigit() else 0.0))
    return procs, util


def free_cards(busy_ours: set) -> list[int]:
    procs, _ = smi()
    wl = tc.workload_pids()
    out = []
    for c, ps in procs.items():
        if c in busy_ours:
            continue
        other = [(p, m) for p, m in ps if p not in set(wl.get(c, []))]
        if not other:
            out.append(c)
    return out


class Job:
    def __init__(self, jid: str, kind: str, sy: str, s: str):
        self.id, self.kind, self.sy, self.s = jid, kind, sy, s
        self.att, self.proc, self.card, self.samples, self.foreign, self.fmax = 0, None, None, 0, 0, 0.0
        self.util, self.results = [], []

    def ok_file(self) -> Path:
        return OUT / f"{self.id}.ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = [Job(f"T_{sy}_{s}", "T", sy, s) for sy in SYS for s in SETS] + \
           [Job(f"K_{sy}_{s}", "K", sy, s) for sy in SYS for s in SETS]
    if a.dry:
        for j in jobs:
            c = thinker_cmd(j.sy, j.s, 1) if j.kind == "T" else talker_cmd(j.sy, j.s, 1, PORT0)
            print(j.id, "|", c[len(fo.ENV):])
        for sy in SYS:
            for s in SETS:
                src = Path(SPD) / SYS[sy]["tin"].format(s=s) / "talker_in"
                print("talker_in", sy, s, src, "exists" if (src / "items.jsonl").exists() else "MISSING")
        for p in (BASE, IDN, CK):
            print(p, "exists" if Path(p).exists() else "MISSING")
        return
    summ = OUT / "wallclock_summary.json"
    if summ.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {summ}")
    OUT.mkdir(parents=True, exist_ok=True)
    for sy in SYS:
        for s in SETS:
            log(f"talker input {sy} {s}: {prep_tin(sy, s)} items")
    while not GATE_DONE.exists() and not list(GATE_FAILDIR.glob("FAILED_*")):
        time.sleep(60)
    log(f"gate passed ({'done' if GATE_DONE.exists() else 'failed'}): P0-4 test over")
    todo = [j for j in jobs if not j.ok_file().exists()]
    running: list[Job] = []
    lock = threading.Lock()
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                procs, util = smi()
                with lock:
                    for j in running:
                        mine = set(tc.tree(j.proc.pid))
                        ps = procs.get(j.card, [])
                        f = [(p, m) for p, m in ps if p not in mine and m > FOREIGN_MIB]
                        j.samples += 1
                        if f:
                            j.foreign += 1
                            j.fmax = max(j.fmax, sum(m for _, m in f))
                        j.util.append(util.get(j.card))
            except Exception as e:  # noqa: BLE001
                log(f"monitor: {e}")
            stop.wait(5)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    port = PORT0
    while todo or running:
        with lock:
            for j in list(running):
                rc = j.proc.poll()
                if rc is None:
                    continue
                running.remove(j)
                res = {"att": j.att, "card": j.card, "rc": rc, "samples": j.samples, "foreign_samples": j.foreign,
                       "foreign_max_mib": j.fmax, "util_mean": (sum(u for u in j.util if u is not None)
                                                                / max(1, len([u for u in j.util if u is not None])))}
                j.results.append(res)
                log(f"{j.id} a{j.att} ended rc={rc} card {j.card} foreign {j.foreign}/{j.samples} (max {j.fmax:.0f} MiB)")
                if rc == 0 and j.foreign == 0:
                    j.ok_file().write_text(json.dumps(j.results) + "\n")
                elif j.att < MAX_ATT:
                    todo.append(j)
                else:
                    (OUT / f"{j.id}.gaveup").write_text(json.dumps(j.results) + "\n")
                    log(f"{j.id}: no clean attempt after {MAX_ATT}; kept all attempts")
        if todo:
            cards = free_cards({j.card for j in running})
            while todo and cards:
                j, c = todo.pop(0), cards.pop(0)
                pids = tc.workload_pids().get(c, [])
                victims = sorted({q for p in pids for q in tc.tree(p)})
                if victims:
                    log(f"card {c}: stop no-checkpoint workload {pids} (tree {victims}) for {j.id}")
                    for p in victims:
                        try:
                            os.kill(p, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    for _ in range(24):
                        procs, _ = smi()
                        if not procs.get(c):
                            break
                        time.sleep(5)
                j.att += 1
                port += 1
                cmd = thinker_cmd(j.sy, j.s, j.att) if j.kind == "T" else talker_cmd(j.sy, j.s, j.att, port)
                with lock:
                    j.card, j.samples, j.foreign, j.fmax, j.util = c, 0, 0, 0.0, []
                    j.proc = subprocess.Popen(["bash", "-c", f"export CUDA_VISIBLE_DEVICES={c}; cd {REPO}; " + cmd],
                                              stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
                    running.append(j)
                log(f"{j.id} a{j.att} started card {c} pid {j.proc.pid}")
        time.sleep(30)
    stop.set()
    log("all jobs over; analysis")
    summary = {"jobs": {j.id: j.results for j in jobs}, "runs": {}}

    def best(jid: str):
        r = [x for x in next(j for j in jobs if j.id == jid).results if x["rc"] == 0]
        return min(r, key=lambda x: x["foreign_samples"]) if r else None

    for sy in SYS:
        for s in SETS:
            bt, bk = best(f"T_{sy}_{s}"), best(f"K_{sy}_{s}")
            if not bt or not bk:
                summary["runs"][f"{sy}_{s}"] = {"error": "a job has no successful attempt"}
                continue
            pt, pk = OUT / f"T_{sy}_{s}.a{bt['att']}", OUT / f"K_{sy}_{s}.a{bk['att']}"
            o = OUT / f"wallclock_{sy}_{s}.json"
            cmd = [fo.PY, "-m", "experiments.mist_w3po_v1.evaluation.wallclock_v11", "--windows", SETS[s]["win"],
                   "--meetings", SETS[s]["meetings"], "--reads", f"{pt}.reads.jsonl", "--talker-in",
                   str(OUT / f"tin_{sy}_{s}"), "--talker-out", f"{pk}/talker", "--talker-timing", f"{pk}.timing.jsonl",
                   "--talker-log", f"{pk}.talk.log", "--out", str(o)]
            r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                                                                             "PYTHONPATH": "."})
            log(f"analysis {sy} {s}: rc={r.returncode} {r.stdout.strip()[-300:]} {r.stderr.strip()[-300:]}")
            summary["runs"][f"{sy}_{s}"] = {"thinker_attempt": bt, "talker_attempt": bk,
                                             "result": json.load(open(o)) if o.exists() else None}
    with open(summ, "x") as fh:
        json.dump(summary, fh, indent=1)
    (OUT / "WALLCLOCK_V11_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    log("WALLCLOCK_V11_DONE")


if __name__ == "__main__":
    main()
