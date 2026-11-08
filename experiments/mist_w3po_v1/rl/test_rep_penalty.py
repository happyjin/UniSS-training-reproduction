"""--rep-penalty (2026-11-08): advantage.rep_score on hand-made segments; group_advantages without q_penalty is
bit-identical to the version before (argv[1]); with q_penalty, dia / w3dia use cpBLEU - penalty.
    python -m experiments.mist_w3po_v1.rl.test_rep_penalty <old advantage.py> <hyp.jsonl> <windows_train_diar.jsonl>
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
    seg = lambda t: {"text": t}  # noqa: E731
    check("clean text: 0", NEW.rep_score([seg("we should finish the remote control design today")]) == 0.0)
    loop = NEW.rep_score([seg("I'll discuss it " + "with the " * 50)])
    check(f"loop: high ({loop:.2f})", loop > 0.8)
    dup = NEW.rep_score([seg("the battery should last two years"), seg("the battery should last two years")])
    check(f"re-translated segment: > 0 ({dup:.2f})", dup > 0.3)
    spec = importlib.util.spec_from_file_location("old_adv", sys.argv[1])
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    tok = Prompter("zh2en").tok
    W = {w["id"]: w for w in map(json.loads, open(sys.argv[3], encoding="utf-8"))}
    done = 0
    for rec in map(json.loads, open(sys.argv[2], encoding="utf-8")):
        w = W.get(rec["id"])
        if w is None or not w.get("utts") or not rec.get("segments"):
            continue
        units, dur = S.units_of(w["utts"]), w["dur_ms"] / 1000
        _, post = load_window(w)
        base = traj_from(rec, tok, w)
        dup_ids = tok("".join(rd["new"] for rd in rec["read_log"]), add_special_tokens=False)["input_ids"]
        dupt = traj_from(rec, tok, w, lambda r, k, ids, bf: ids + (dup_ids if r == k - 1 else []))
        members = [base, dupt]
        creds = [S.credit(t.segs, units, dur, t.wait_reads, [rd.t_ms / 1000 for rd in t.reads], post=post) for t in members]
        aq = [(creds[0].R, creds[0].lat)]
        for mode in ("w3po", "noC1", "noC2", "noC3", "noC4", "dia", "w3dia", "hpo", "seqpo", "hibiki"):
            assert OLD.group_advantages(mode, members, creds, units, dur, aq)["tok"] == \
                NEW.group_advantages(mode, members, creds, units, dur, aq)["tok"], mode
        qp = [1.0 * NEW.rep_score(t.segs) for t in members]
        a = NEW.group_advantages("dia", members, creds, units, dur, aq, q_penalty=qp)
        q = np.array([S.seq_cpbleu(t.segs, units) for t in members]) - np.array(qp)
        zq = NEW.z(q)
        assert all(abs(x[4] - zq[m]) < 1e-12 for m in range(2) for r in a["tok"][m] for x in r)
        done += 1
        if done >= 15:
            break
    check(f"no penalty: all modes bit-identical ({done} groups)", done > 0)
    check("dia with penalty uses z(cpBLEU - penalty)", True)
    print("ALL PASS")


if __name__ == "__main__":
    main()
