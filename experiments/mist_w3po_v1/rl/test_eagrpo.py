"""--mode eagrpo (2026-11-08, Event-Anchored GRPO).
  1. every existing mode is bit-identical to advantage.py before eagrpo (argv[1]) on real groups (base / dropped read /
     duplicated / speaker-flipped / half-flipped members, with and without --rep-penalty);
  2. lam_ev = 0 gives exactly dia's advantages;
  3. a hand-made group (a long unit and a 0.4 s interjection; rollout 1 misses the interjection, rollout 2 gives it
     the wrong speaker) against hand-computed event credits, for --ev-norm std (first version) and center (default);
  4. (amendment) --ev-norm std is bit-identical to the first eagrpo version (argv[2]); the center clip bounds E;
  5. (tail bound) without --ev-clip-scaled everything is bit-identical to the version before it (argv[3]); with it the
     event term is exactly clip(lam_ev x E, +-sqrt(G - 1)).

    python -m experiments.mist_w3po_v1.rl.test_eagrpo <advantage.py before eagrpo> <first eagrpo advantage.py> \
        <eagrpo advantage.py before --ev-clip-scaled> <hyp.jsonl> <windows_train_diar.jsonl> [n]
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from types import SimpleNamespace

import numpy as np

from experiments.mist_w3po_v1.core.engine import Prompter
from experiments.mist_w3po_v1.rl import advantage as NEW
from experiments.mist_w3po_v1.rl import stamp as S
from experiments.mist_w3po_v1.rl.rollout import load_window
from experiments.mist_w3po_v1.rl.test_pipeline_cpu import traj_from

MODES = ("w3po", "noC1", "noC2", "noC3", "noC4", "dia", "hpo", "seqpo", "hibiki", "w3dia", "w3dia2")


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def real_groups(OLD, EA1, EA2, tok, hyp, wins, n):
    spk_tok = {tok.convert_tokens_to_ids(f"<|spk{k}|>"): k for k in range(1, 9)}
    inv = {k: t for t, k in spk_tok.items()}
    W = {w["id"]: w for w in map(json.loads, open(wins, encoding="utf-8"))}
    done, nz = 0, 0
    for rec in map(json.loads, open(hyp, encoding="utf-8")):
        w = W.get(rec["id"])
        if w is None or not w.get("utts") or not rec.get("segments"):
            continue
        units = S.units_of(w["utts"])
        dur = w["dur_ms"] / 1000
        _, post = load_window(w)
        first = next((r for r, rd in enumerate(rec["read_log"]) if rd["new"]), None)
        dup_ids = tok("".join(rd["new"] for rd in rec["read_log"]), add_special_tokens=False)["input_ids"]
        members = [traj_from(rec, tok, w),
                   traj_from(rec, tok, w, lambda r, k, ids, bf: [] if r == first else ids),
                   traj_from(rec, tok, w, lambda r, k, ids, bf: ids + (dup_ids if r == k - 1 else [])),
                   traj_from(rec, tok, w, lambda r, k, ids, bf: [inv[spk_tok[i] % 8 + 1] if i in spk_tok else i for i in ids]),
                   traj_from(rec, tok, w, lambda r, k, ids, bf: [inv[spk_tok[i] % 8 + 1] if (i in spk_tok and r >= k // 2)
                                                                 else i for i in ids])]
        creds = [S.credit(t.segs, units, dur, t.wait_reads, [rd.t_ms / 1000 for rd in t.reads], post=post) for t in members]
        aq = [(creds[0].R, creds[0].lat)]
        qp = [0.01 * m for m in range(len(members))]
        for mode in MODES:
            for kw in ({}, {"q_penalty": qp}):
                if kw and mode not in ("dia", "w3dia", "w3dia2", "noC1"):
                    continue
                a = OLD.group_advantages(mode, members, creds, units, dur, aq, **kw)
                b = NEW.group_advantages(mode, members, creds, units, dur, aq, **kw)
                assert a == b, (w["id"], mode, kw)
        for kw in ({}, {"q_penalty": qp}):
            d = NEW.group_advantages("dia", members, creds, units, dur, aq, **kw)
            e0 = NEW.group_advantages("eagrpo", members, creds, units, dur, aq, lam_ev=0.0, **kw)
            assert e0["tok"] == d["tok"], (w["id"], "lam_ev 0 != dia", kw)
        for kw in ({}, {"q_penalty": qp}):
            for lam in (0.0, 1.0):
                a = EA1.group_advantages("eagrpo", members, creds, units, dur, aq, lam_ev=lam, **kw)
                b = NEW.group_advantages("eagrpo", members, creds, units, dur, aq, lam_ev=lam, ev_norm="std", **kw)
                assert a["tok"] == b["tok"], (w["id"], "std path != first eagrpo", lam, kw)
        for kw in ({}, {"q_penalty": qp}):
            for extra in ({}, {"ev_norm": "std"}, {"lam_ev": 6.31}):
                a = EA2.group_advantages("eagrpo", members, creds, units, dur, aq, **kw, **extra)
                b = NEW.group_advantages("eagrpo", members, creds, units, dur, aq, **kw, **extra)
                assert a["tok"] == b["tok"], (w["id"], "default != before --ev-clip-scaled", kw, extra)
        bound = math.sqrt(len(members) - 1)
        nc = NEW.group_advantages("eagrpo", members, creds, units, dur, aq, lam_ev=6.31)
        sc = NEW.group_advantages("eagrpo", members, creds, units, dur, aq, lam_ev=6.31, ev_clip_scaled=True)
        zq0 = NEW.z(np.array([S.seq_cpbleu(t.segs, units) for t in members]))
        for m in range(len(members)):
            for r in range(len(members[m].reads)):
                for x, y in zip(nc["tok"][m][r], sc["tok"][m][r]):
                    t = x[4] - zq0[m]
                    want = zq0[m] + min(max(t, -bound), bound)
                    assert abs(y[4] - want) < 1e-9 and abs(y[4] - zq0[m]) <= bound + 1e-9, (t, y[4], want)
        e1 = NEW.group_advantages("eagrpo", members, creds, units, dur, aq)
        assert e1 == NEW.group_advantages("eagrpo", members, creds, units, dur, aq, lam_ev=1.0, ev_norm="center",
                                          ev_clip=1.0), "defaults: lam_ev 1, center, clip 1"
        d = NEW.group_advantages("dia", members, creds, units, dur, aq)
        diff = [abs(x[4] - y[4]) for m in range(len(members)) for r in range(len(members[m].reads))
                for x, y in zip(e1["tok"][m][r], d["tok"][m][r])]
        assert all(math.isfinite(v) for v in diff) and max(diff) <= 1.0 + 1e-12, max(diff)
        nz += any(v > 1e-9 for v in diff)
        done += 1
        if done >= n:
            break
    return done, nz


def hand_group(tok, ev_norm):
    utts = [{"begin_s": 0.0, "end_s": 2.5, "speaker_id": "A", "ref": "the meeting starts now with the budget review",
             "zh": "x"},
            {"begin_s": 2.6, "end_s": 3.0, "speaker_id": "B", "ref": "okay", "zh": "y"}]
    units = S.units_of(utts)
    assert [u["spk"] for u in units] == ["A", "B"]
    w = {"id": "hand", "dur_ms": 6000}
    s0 = "<|spk1|><|t0|>the meeting starts now with the budget review<|t31|>"
    logs = {0: [s0, "<|spk2|><|t33|>okay<|t37|>"], 1: [s0, ""], 2: [s0, "<|spk1|><|t33|>okay<|t37|>"]}
    trajs = []
    for g in range(3):
        rl = [{"t": 1000, "new": ""}, {"t": 2000, "new": ""}, {"t": 3000, "new": logs[g][0]}, {"t": 4000, "new": logs[g][1]},
              {"t": 5000, "new": ""}, {"t": 6000, "new": ""}]
        trajs.append(traj_from({"read_log": rl}, tok, w))
    assert [len(t.segs) for t in trajs] == [2, 1, 2], [t.segs for t in trajs]
    SQ = [[0.8, 0.6], [0.6, 0.0], [0.7, 0.5]]
    uo = [[0, 1], [0], [0, 1]]
    creds = [SimpleNamespace(R=0.5, lat=1.0, SQ=SQ[g], unit_of=uo[g]) for g in range(3)]
    out = NEW.group_advantages("eagrpo", trajs, creds, units, 6.0, [], ev_norm=ev_norm, ev_clip=0.0 if ev_norm == "std" else 1.0)
    zq = NEW.z(np.array([S.seq_cpbleu(t.segs, units) for t in trajs]))

    def nrm(x):
        x = np.array(x, dtype=float)
        return (x - x.mean()) / (1.0 if ev_norm == "center" else max(x.std(), NEW.FLOOR))
    eq0, eq1, em1 = nrm([0.8, 0.6, 0.7]), nrm([0.6, 0.0, 0.5]), nrm([1, 0, 1])
    ec1 = {0: 1.0, 2: -1.0} if ev_norm == "std" else {0: 0.5, 2: -0.5}

    def E(g, r):   # event part of each token of read r of rollout g
        return [(wh, who, x[4] - zq[g]) for (wh, who, _, _, x4), x in zip(out["tok"][g][r], out["tok"][g][r])
                for x4 in [x[4]]]
    ok = True
    # rollout 0, read 3 (4.0 s): <spk2> opens the interjection (when + who): e_q(u1) + e_c(u1); <t33> okay <t37>: e_q(u1);
    # the read's <wait>: the commit of u1 -> e_q(u1)
    r3 = E(0, 3)
    ok &= abs(r3[0][2] - (eq1[0] + ec1[0])) < 1e-9 and r3[0][0] and r3[0][1]
    ok &= all(abs(v - eq1[0]) < 1e-9 for _, _, v in r3[1:-1]) and abs(r3[-1][2] - eq1[0]) < 1e-9 and r3[-1][0]
    # rollout 2 (wrong speaker on the interjection): opening token e_q(u1) + e_c(u1) = eq1[2] - 1
    ok &= abs(E(2, 3)[0][2] - (eq1[2] + ec1[2])) < 1e-9
    # rollout 1 missed u1 (end 3.0 s): <wait> of reads at 3.0 / 4.0 / 5.0 s get e_m(u1) = -sqrt(2); at 3.0 s it is
    # averaged with the commit of u0 in that read (e_q(u0))
    ok &= abs(E(1, 3)[-1][2] - em1[1]) < 1e-9 and abs(E(1, 4)[-1][2] - em1[1]) < 1e-9
    ok &= abs(E(1, 2)[-1][2] - (eq0[1] + em1[1]) / 2) < 1e-9
    ok &= abs(em1[1] - (-math.sqrt(2) if ev_norm == "std" else -2 / 3)) < 1e-9
    # u0 tokens: e_q(u0) on text; its speaker is right everywhere -> e_c(u0) = 0 (the opening token gets e_q(u0) only)
    r2 = E(0, 2)
    ok &= abs(r2[0][2] - eq0[0]) < 1e-9 and all(abs(v - eq0[0]) < 1e-9 for _, _, v in r2[1:])
    # empty reads before any speech: no event
    ok &= all(abs(v) < 1e-12 for g in range(3) for r in (0, 1) for _, _, v in E(g, r))
    # final read: nothing routed
    ok &= all(abs(v) < 1e-12 for g in range(3) for _, _, v in E(g, 5))
    ev = out["stats"]["ev"]
    ok &= ev["units"] == 2 and ev["units_nz"] == 2 and ev["short"] == 3 and ev["short_cov"] == 2
    if ev_norm == "center":   # the clip: with ev_clip 0.3 every |E(t)| <= 0.3 and some token is clipped
        o3 = NEW.group_advantages("eagrpo", trajs, creds, units, 6.0, [], ev_clip=0.3)
        v3 = [abs(x[4] - zq[g]) for g in range(3) for row in o3["tok"][g] for x in row]
        ok &= max(v3) <= 0.3 + 1e-12 and any(abs(v - 0.3) < 1e-12 for v in v3)
    return ok


def main():
    mods = []
    for k, name in ((1, "old_adv"), (2, "ea1_adv"), (3, "ea2_adv")):
        spec = importlib.util.spec_from_file_location(name, sys.argv[k])
        mods.append(importlib.util.module_from_spec(spec))
        spec.loader.exec_module(mods[-1])
    tok = Prompter("zh2en").tok
    n = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    done, nz = real_groups(mods[0], mods[1], mods[2], tok, sys.argv[4], sys.argv[5], n)
    check(f"existing modes (w3dia/w3dia2 included, with and without --rep-penalty) bit-identical on {done} groups", done > 0)
    check(f"eagrpo lam_ev 0 == dia on {done} groups; --ev-norm std == first eagrpo; |E| <= 1 (center); event term "
          f"non-zero in {nz}; default == before --ev-clip-scaled; scaled clip = clip(lam E, +-sqrt(G-1))", done > 0 and nz > 0)
    for ev_norm in ("std", "center"):
        check(f"hand-made group ({ev_norm}): missed interjection, wrong-speaker interjection, commits, empty / final reads"
              + (", clip" if ev_norm == "center" else ""), hand_group(tok, ev_norm))
    print("ALL PASS")


if __name__ == "__main__":
    main()
