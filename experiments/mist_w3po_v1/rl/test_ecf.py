"""--ecf-branches (2026-11-09, ECF-GRPO: event-triggered counterfactual flush branches, with --mode dia).
  1. advantage.py: every existing mode (eagrpo included, with and without --rep-penalty) is bit-identical to
     advantage.py before ECF (argv[1]) on real groups;
  2. advantage.py / rollout.py / train_w3po.py differ from their versions before ECF (argv[1..3]) by pure insertions,
     and every inserted trainer line that runs is behind `args.ecf_branches > 0` or the (then empty) ecf dict:
     --ecf-branches 0 (default) is dia (and every other mode) exactly;
  3. ecf_specs on a hand-made group (a long unit and a 0.4 s interjection): only the member that never covers the
     interjection and waits after hearing it is a branch point (read 3.0 s); a final read is never one; the cap keeps
     distinct members;
  4. ecf_apply: the parent's terminal <wait> at the branch read gets exactly lam * clip((q_m - q_b) / max(std q,
     FLOOR), +-sqrt(G - 1)); every other token is unchanged;
  5. branch_rollout with a stub engine on a real window: the parent's reads before r are kept, read r is taken with
     <wait> barred from the parent's committed ids, the later reads are ordinary, the last one final.

    python -m experiments.mist_w3po_v1.rl.test_ecf <advantage.py before> <rollout.py before> <train_w3po.py before> \
        <hyp.jsonl> <windows_train_diar.jsonl> [n]
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.mist_w3po_v1.core.engine import EOS_ID, WAIT_ID, Prompter
from experiments.mist_w3po_v1.rl import advantage as NEW
from experiments.mist_w3po_v1.rl import stamp as S
from experiments.mist_w3po_v1.rl.rollout import branch_rollout, load_window
from experiments.mist_w3po_v1.rl.test_pipeline_cpu import traj_from

MODES = ("w3po", "noC1", "noC2", "noC3", "noC4", "dia", "hpo", "seqpo", "hibiki", "w3dia", "w3dia2", "eagrpo")
RL = Path("experiments/mist_w3po_v1/rl")


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def real_groups(OLD, tok, hyp, wins, n):
    spk_tok = {tok.convert_tokens_to_ids(f"<|spk{k}|>"): k for k in range(1, 9)}
    inv = {k: t for t, k in spk_tok.items()}
    W = {w["id"]: w for w in map(json.loads, open(wins, encoding="utf-8"))}
    done = 0
    for rec in map(json.loads, open(hyp, encoding="utf-8")):
        w = W.get(rec["id"])
        if w is None or not w.get("utts") or not rec.get("segments"):
            continue
        units = S.units_of(w["utts"])
        dur = w["dur_ms"] / 1000
        _, post = load_window(w)
        first = next((r for r, rd in enumerate(rec["read_log"]) if rd["new"]), None)
        members = [traj_from(rec, tok, w),
                   traj_from(rec, tok, w, lambda r, k, ids, bf: [] if r == first else ids),
                   traj_from(rec, tok, w, lambda r, k, ids, bf: [inv[spk_tok[i] % 8 + 1] if i in spk_tok else i for i in ids])]
        creds = [S.credit(t.segs, units, dur, t.wait_reads, [rd.t_ms / 1000 for rd in t.reads], post=post) for t in members]
        aq = [(creds[0].R, creds[0].lat)]
        qp = [0.01 * m for m in range(len(members))]
        for mode in MODES:
            for kw in ({}, {"q_penalty": qp}):
                if kw and mode not in ("dia", "w3dia", "w3dia2", "noC1", "eagrpo"):
                    continue
                ex = {"lam_ev": 6.31, "ev_clip_scaled": True} if mode == "eagrpo" else {}
                a = OLD.group_advantages(mode, members, creds, units, dur, aq, **kw, **ex)
                b = NEW.group_advantages(mode, members, creds, units, dur, aq, **kw, **ex)
                assert a == b, (w["id"], mode, kw)
        done += 1
        if done >= n:
            break
    return done


GUARDS = ("args.ecf_branches > 0", "ecf.get(", "ecf = {}")


def pure_insertion(old_path, new_path, trainer=False):
    a = Path(old_path).read_text().splitlines()
    b = Path(new_path).read_text().splitlines()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ok = True
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag != "insert":
            print("  non-insertion", tag, a[i1:i2][:2], b[j1:j2][:2])
            return False
        hunk = b[j1:j2]
        if trainer:
            first = next(l for l in hunk if l.strip())
            s = first.strip()
            # allowed: argparse options, imports, or a hunk opened by a guard (or a guard-only key tuple + guard)
            if not (s.startswith(("ap.add_argument", "from experiments.", "help=")) or any(g in s for g in GUARDS)
                    or (s.startswith("ECFK =") and any(GUARDS[0] in l for l in hunk[1:2]))):
                print("  unguarded hunk:", s)
                ok = False
    return ok


def hand_group(tok):
    utts = [{"begin_s": 0.0, "end_s": 2.5, "speaker_id": "A", "ref": "the meeting starts now with the budget review",
             "zh": "x"},
            {"begin_s": 2.6, "end_s": 3.0, "speaker_id": "B", "ref": "okay", "zh": "y"}]
    units = S.units_of(utts)
    w = {"id": "hand", "dur_ms": 6000}
    s0 = "<|spk1|><|t0|>the meeting starts now with the budget review<|t31|>"
    logs = {0: [s0, "<|spk2|><|t33|>okay<|t37|>"], 1: [s0, ""], 2: [s0, "<|spk1|><|t33|>okay<|t37|>"]}
    trajs = []
    for g in range(3):
        rl = [{"t": 1000, "new": ""}, {"t": 2000, "new": ""}, {"t": 3000, "new": logs[g][0]}, {"t": 4000, "new": logs[g][1]},
              {"t": 5000, "new": ""}, {"t": 6000, "new": ""}]
        trajs.append(traj_from({"read_log": rl}, tok, w))
    uo = [[0, 1], [0], [0, 1]]
    creds = [SimpleNamespace(unit_of=uo[g]) for g in range(3)]
    ok = True
    sp = NEW.ecf_specs(trajs, creds, units, random.Random(0), 4)
    ok &= sp == [(1, 2, 1)]                                  # member 1, read at 3.0 s (index 2), unit 1
    ok &= trajs[1].reads[2].ids[-1] == WAIT_ID
    # every member misses the interjection: cap 2 keeps two distinct members, in member order
    sp2 = NEW.ecf_specs(trajs, [SimpleNamespace(unit_of=[0])] * 3, units, random.Random(1), 2)
    ok &= len(sp2) == 2 and len({m for m, _, _ in sp2}) == 2 and sp2 == sorted(sp2) and all(r == 2 for _, r, _ in sp2)
    # an interjection ending after the last non-final read: its first read at/after the end is final -> no branch
    late = [{"begin_s": 0.0, "end_s": 2.5, "speaker_id": "A", "ref": "the meeting starts now", "zh": "x"},
            {"begin_s": 5.2, "end_s": 5.6, "speaker_id": "B", "ref": "okay", "zh": "y"}]
    ok &= NEW.ecf_specs(trajs, [SimpleNamespace(unit_of=[0])] * 3, S.units_of(late), random.Random(0), 4) == []
    # a long unit (> 1 s) missed: no branch
    ok &= NEW.ecf_specs(trajs, [SimpleNamespace(unit_of=[1])] * 3, units, random.Random(0), 4) == []
    # ecf_apply
    adv = NEW.group_advantages("dia", trajs, [SimpleNamespace(R=0.5, lat=1.0)] * 3, units, 6.0, [])
    before = json.loads(json.dumps(adv["tok"]))
    q = np.array([S.seq_cpbleu(t.segs, units) for t in trajs])
    qb = float(q[1]) + 0.05
    st = NEW.ecf_apply(adv["tok"], q, sp, [qb], 1.0)
    sig = max(q.std(), NEW.FLOOR)
    want = float(np.clip((q[1] - qb) / sig, -math.sqrt(2), math.sqrt(2)))
    got = adv["tok"][1][2][-1][4] - before[1][2][-1][4]
    ok &= abs(got - want) < 1e-12 and want < 0
    ok &= all(list(map(list, adv["tok"][g][r])) == before[g][r] for g in range(3) for r in range(6) if (g, r) != (1, 2))
    ok &= all(list(adv["tok"][1][2][k]) == before[1][2][k] for k in range(len(before[1][2]) - 1))
    ok &= st["n"] == 1 and st["better"] == 1 and abs(st["dq"] - 0.05) < 1e-12 and abs(st["abs_extra"] - abs(want)) < 1e-12
    # the clip: a huge gap is bounded by sqrt(G - 1)
    adv2 = NEW.group_advantages("dia", trajs, [SimpleNamespace(R=0.5, lat=1.0)] * 3, units, 6.0, [])
    NEW.ecf_apply(adv2["tok"], q, sp, [float(q[1]) - 100.0], 2.0)
    ok &= abs(adv2["tok"][1][2][-1][4] - before[1][2][-1][4] - 2.0 * math.sqrt(2)) < 1e-12
    return ok


class StubEng:
    def __init__(self, tok):
        self.tok = tok
        self.calls = []

    def read(self, items, **kw):
        self.calls.append([(it.t_ms, it.final, it.bar_wait, list(it.committed)) for it in items])
        outs = []
        for it in items:
            ids = [EOS_ID] if (it.final or it.bar_wait) else [WAIT_ID]
            n = len(ids)
            outs.append(SimpleNamespace(ids=ids, when=[True] * n, who=[False] * n, logp=[0.0] * n, logp_wait=[0.0] * n,
                                        logp_who=[0.0] * n))
        return outs


def branch_check(tok, hyp, wins):
    W = {w["id"]: w for w in map(json.loads, open(wins, encoding="utf-8"))}
    for rec in map(json.loads, open(hyp, encoding="utf-8")):
        w = W.get(rec["id"])
        if w is None or len(rec["read_log"]) < 5 or not any(rd["new"] for rd in rec["read_log"][:3]):
            continue
        par = traj_from(rec, tok, w)
        r = 3
        eng = StubEng(tok)
        br = branch_rollout(eng, {w["id"]: w}, [(w["id"], par, r)], temp=1.0, temp_when=1.0, gen=None)[0]
        n = len(par.reads)
        ok = br.kind == "branch" and len(br.reads) == n
        ok &= all(br.reads[k] is par.reads[k] for k in range(r))
        ok &= eng.calls[0] == [(par.reads[r].t_ms, False, True, par.reads[r].before)]
        ok &= all(c[0][2] is False for c in eng.calls[1:]) and eng.calls[-1][0][1] is True
        ok &= [c[0][0] for c in eng.calls] == [rd.t_ms for rd in par.reads[r:]]
        ok &= br.reads[r].before == par.reads[r].before and br.reads[r].ids == [EOS_ID]
        return ok
    return False


def main():
    spec = importlib.util.spec_from_file_location("old_adv", sys.argv[1])
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    tok = Prompter("zh2en").tok
    n = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    done = real_groups(OLD, tok, sys.argv[4], sys.argv[5], n)
    check(f"every existing mode (eagrpo included, +-rep-penalty) bit-identical to advantage.py before ECF on {done} groups",
          done > 0)
    check("advantage.py / rollout.py: pure insertions", pure_insertion(sys.argv[1], RL / "advantage.py")
          and pure_insertion(sys.argv[2], RL / "rollout.py"))
    check("train_w3po.py: pure insertions, every inserted statement behind --ecf-branches > 0 (default 0 = unchanged)",
          pure_insertion(sys.argv[3], RL / "train_w3po.py", trainer=True))
    check("hand-made group: branch points (missed interjection, cap, final read, long unit) and the <wait> credit + clip",
          hand_group(tok))
    check("branch_rollout (stub engine, real window): prefix kept, read r with <wait> barred, later reads ordinary",
          branch_check(tok, sys.argv[4], sys.argv[5]))
    print("ALL PASS")


if __name__ == "__main__":
    main()
