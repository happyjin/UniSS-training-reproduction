"""--mode w3dia (2026-11-07): existing modes must be bit-identical to the advantage.py before w3dia (git 4d2c... copy
passed as argv[1]); w3dia must route z(cpBLEU) to what and when tokens and the w3po RBC advantage to who tokens.
Groups are built from real decodes as in test_pipeline_cpu (base / dropped segment / duplicated / speaker-flipped).

    python -m experiments.mist_w3po_v1.rl.test_w3dia <old advantage.py> <hyp.jsonl> <windows_train_diar.jsonl> [n]
"""
from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np

from experiments.mist_w3po_v1.core.engine import Prompter
from experiments.mist_w3po_v1.rl import advantage as NEW
from experiments.mist_w3po_v1.rl import stamp as S
from experiments.mist_w3po_v1.rl.rollout import load_window
from experiments.mist_w3po_v1.rl.test_pipeline_cpu import traj_from


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def main():
    spec = importlib.util.spec_from_file_location("old_adv", sys.argv[1])
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    hyp, wins = sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    tok = Prompter("zh2en").tok
    W = {w["id"]: w for w in map(json.loads, open(wins, encoding="utf-8"))}
    done = 0
    for rec in map(json.loads, open(hyp, encoding="utf-8")):
        w = W.get(rec["id"])
        if w is None or not w.get("utts") or not rec.get("segments"):
            continue
        units = S.units_of(w["utts"])
        dur = w["dur_ms"] / 1000
        _, post = load_window(w)
        base = traj_from(rec, tok, w)
        nr = len(rec["read_log"])
        first = next((r for r, rd in enumerate(rec["read_log"]) if rd["new"]), None)
        drop = traj_from(rec, tok, w, lambda r, k, ids, bf: [] if r == first else ids)
        dup_ids = tok("".join(rd["new"] for rd in rec["read_log"]), add_special_tokens=False)["input_ids"]
        dup = traj_from(rec, tok, w, lambda r, k, ids, bf: ids + (dup_ids if r == k - 1 else []))
        members = [base, drop, dup]
        creds = [S.credit(t.segs, units, dur, t.wait_reads, [rd.t_ms / 1000 for rd in t.reads], post=post) for t in members]
        aq = [(creds[0].R, creds[0].lat)]
        for mode in ("w3po", "noC1", "noC2", "noC3", "noC4", "dia", "hpo", "seqpo", "hibiki"):
            a = OLD.group_advantages(mode, members, creds, units, dur, aq)
            b = NEW.group_advantages(mode, members, creds, units, dur, aq)
            if a["tok"] != b["tok"]:
                check(f"{w['id']} {mode} unchanged", False)
        d = NEW.group_advantages("w3dia", members, creds, units, dur, aq)
        p = NEW.group_advantages("w3po", members, creds, units, dur, aq)
        zq = NEW.z(np.array([S.seq_cpbleu(t.segs, units) for t in members]))
        for m, t in enumerate(members):
            for r, rd in enumerate(t.reads):
                for k, (wh, who, aw, ah, at) in enumerate(d["tok"][m][r]):
                    if wh:
                        assert abs(aw - zq[m]) < 1e-12, ("when", aw, zq[m])
                    if not who:
                        assert abs(at - zq[m]) < 1e-12, ("what", at, zq[m])
                    else:
                        assert ah == p["tok"][m][r][k][3], ("who", ah, p["tok"][m][r][k][3])
        done += 1
        if done >= n:
            break
    check(f"existing modes bit-identical on {done} groups", True)
    check(f"w3dia: when/what = z(cpBLEU), who = w3po RBC on {done} groups", done > 0)
    print("ALL PASS")


if __name__ == "__main__":
    main()
