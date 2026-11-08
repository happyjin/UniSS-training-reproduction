"""--mode w3dia2 (2026-11-08): every existing mode (w3dia included) must be bit-identical to advantage.py before
w3dia2 (argv[1]); w3dia2 must give what/when tokens exactly w3dia's advantages and who tokens
z(cpBLEU) + who_sc x normalised speaker correctness + who_rbc x normalised RBC.
Groups are built from real decodes as in test_pipeline_cpu (base / dropped read / duplicated / speaker-flipped /
speakers flipped in the second half of the reads).

    python -m experiments.mist_w3po_v1.rl.test_w3dia2 <old advantage.py> <hyp.jsonl> <windows_train_diar.jsonl> [n]
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


def norm(xs):
    flat = np.array([v for x in xs for v in x])
    if len(flat) > 1 and flat.std() > 1e-9:
        mu, sd = flat.mean(), max(flat.std(), NEW.FLOOR)
        return [[(v - mu) / sd for v in x] for x in xs]
    return [[0.0] * len(x) for x in xs]


def main():
    spec = importlib.util.spec_from_file_location("old_adv", sys.argv[1])
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    hyp, wins = sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    tok = Prompter("zh2en").tok
    spk_tok = {tok.convert_tokens_to_ids(f"<|spk{k}|>"): k for k in range(1, 9)}
    inv = {k: t for t, k in spk_tok.items()}
    W = {w["id"]: w for w in map(json.loads, open(wins, encoding="utf-8"))}
    done, sc_var, who_tok = 0, 0, 0
    for rec in map(json.loads, open(hyp, encoding="utf-8")):
        w = W.get(rec["id"])
        if w is None or not w.get("utts") or not rec.get("segments"):
            continue
        units = S.units_of(w["utts"])
        dur = w["dur_ms"] / 1000
        _, post = load_window(w)
        base = traj_from(rec, tok, w)
        first = next((r for r, rd in enumerate(rec["read_log"]) if rd["new"]), None)
        drop = traj_from(rec, tok, w, lambda r, k, ids, bf: [] if r == first else ids)
        dup_ids = tok("".join(rd["new"] for rd in rec["read_log"]), add_special_tokens=False)["input_ids"]
        dup = traj_from(rec, tok, w, lambda r, k, ids, bf: ids + (dup_ids if r == k - 1 else []))
        flip = traj_from(rec, tok, w, lambda r, k, ids, bf: [inv[spk_tok[i] % 8 + 1] if i in spk_tok else i for i in ids])
        # a global speaker permutation is absorbed by pi*: flip only the second half of the reads to vary correctness
        half = traj_from(rec, tok, w, lambda r, k, ids, bf: [inv[spk_tok[i] % 8 + 1] if (i in spk_tok and r >= k // 2) else i
                                                            for i in ids])
        members = [base, drop, dup, flip, half]
        creds = [S.credit(t.segs, units, dur, t.wait_reads, [rd.t_ms / 1000 for rd in t.reads], post=post) for t in members]
        aq = [(creds[0].R, creds[0].lat)]
        qp = [0.01 * m for m in range(len(members))]
        for mode in ("w3po", "noC1", "noC2", "noC3", "noC4", "dia", "hpo", "seqpo", "hibiki", "w3dia"):
            for kw in ({}, {"q_penalty": qp}):
                if mode not in ("dia", "w3dia", "noC1") and kw:
                    continue
                a = OLD.group_advantages(mode, members, creds, units, dur, aq, **kw)
                b = NEW.group_advantages(mode, members, creds, units, dur, aq, **kw)
                check(f"{w['id']} {mode} {sorted(kw)} unchanged", a == b)
        zq = NEW.z(np.array([S.seq_cpbleu(t.segs, units) for t in members]))
        sc = norm([[float(c.unit_of[j] >= 0 and S.speaker_map(t.segs, units).get(s["spk"]) == units[c.unit_of[j]]["spk"])
                    for j, s in enumerate(t.segs)] for t, c in zip(members, creds)])
        rbc = norm([list(c.who) for c in creds])
        sc_var += any(abs(v) > 1e-9 for x in sc for v in x)
        ref = NEW.group_advantages("w3dia", members, creds, units, dur, aq)
        for lam_sc, lam_rbc in ((0.5, 0.0), (0.0, 0.0), (0.0, 1.0), (0.3, 0.7)):
            d = NEW.group_advantages("w3dia2", members, creds, units, dur, aq, who_sc=lam_sc, who_rbc=lam_rbc)
            if (lam_sc, lam_rbc) == (0.5, 0.0):
                assert d == NEW.group_advantages("w3dia2", members, creds, units, dur, aq), "defaults are 0.5 / 0"
            for m, t in enumerate(members):
                for r in range(len(t.reads)):
                    for k, (wh, who, aw, ah, at) in enumerate(d["tok"][m][r]):
                        rw = ref["tok"][m][r][k]
                        assert (wh, who, aw, at) == (rw[0], rw[1], rw[2], rw[4]), ("what/when differ from w3dia", m, r, k)
                        if who:
                            sj = t.tok_seg[r][k]
                            want = zq[m] + (lam_sc * sc[m][sj] + lam_rbc * rbc[m][sj] if sj >= 0 else 0.0)
                            assert abs(ah - want) < 1e-12, ("who", ah, want, lam_sc, lam_rbc)
                            who_tok += 1
        done += 1
        if done >= n:
            break
    check(f"existing modes (w3dia included, with and without --rep-penalty) bit-identical on {done} groups", done > 0)
    check(f"w3dia2: what/when = w3dia, who = z(cpBLEU) + who_sc SC + who_rbc RBC on {done} groups / {who_tok} who tokens",
          who_tok > 0)
    check(f"speaker correctness varies within the group in {sc_var}/{done} groups (half-flip member)", sc_var > 0)
    print("ALL PASS")


if __name__ == "__main__":
    main()
