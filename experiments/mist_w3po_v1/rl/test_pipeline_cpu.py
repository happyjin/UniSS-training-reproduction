"""CPU check of rollout.parse -> stamp.credit -> advantage.group_advantages on real decodes (no model): groups built
from DAgger decodes of train windows (the v8 model's own reads), members = the decode, a copy with one segment's
read dropped, a copy with every segment duplicated in the last read, a copy with the speaker of each segment flipped.

    python -m experiments.mist_w3po_v1.rl.test_pipeline_cpu <hyp.jsonl> <windows_train_diar.jsonl> [n]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from experiments.mist_w3po_v1.core.engine import EOS_ID, WAIT_ID, Prompter
from experiments.mist_w3po_v1.rl import stamp as S
from experiments.mist_w3po_v1.rl.advantage import group_advantages
from experiments.mist_w3po_v1.rl.rollout import Read, Traj, parse


def traj_from(rec, tok, w, mutate=None):
    tr = Traj(w["id"], "policy", "")
    before = []
    n_reads = len(rec["read_log"])
    for r, rd in enumerate(rec["read_log"]):
        ids = tok(rd["new"], add_special_tokens=False)["input_ids"] if rd["new"] else []
        if mutate:
            ids = mutate(r, n_reads, ids, before)
        final = r == n_reads - 1
        blk = ids + [EOS_ID if final else WAIT_ID]
        n = len(blk)
        tr.reads.append(Read(rd["t"], final, list(before), blk, [False] * n, [False] * n, [0.0] * n, [0.0] * n,
                             [0.0] * n, f"{w['id']}|{rd['t']}"))
        before += ids
    parse(tr, tok, w["dur_ms"] / 1000)
    return tr


def main():
    hyp, wins = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    tok = Prompter("zh2en").tok
    W = {w["id"]: w for w in map(json.loads, open(wins, encoding="utf-8"))}
    recs = [r for r in map(json.loads, open(hyp, encoding="utf-8")) if r["id"] in W and r["segments"]][:n]
    spk_tok = {tok.convert_tokens_to_ids(f"<|spk{k}|>"): k for k in range(1, 9)}
    inv = {k: t for t, k in spk_tok.items()}
    agg = {m: [] for m in ("w3po", "noC2", "noC4", "dia", "hibiki", "hpo", "seqpo", "noC1")}
    for rec in recs:
        w = W[rec["id"]]
        units = S.units_of(w["utts"])
        dur = w["dur_ms"] / 1000
        post = np.load(w["post"]["lowlat"]).astype(np.float32)
        base = traj_from(rec, tok, w)
        assert len(base.segs) == len([s for s in rec["segments"] if s["text"].strip()]), (base.segs, rec["segments"])
        for a, b in zip(base.segs, rec["segments"]):
            assert a["spk"] == b["spk"] and abs(a["a"] - b["start_s"]) < 1e-6 and a["text"] == b["text"].strip(), (a, b)
        first_nonempty = next((r for r, rd in enumerate(rec["read_log"]) if rd["new"]), None)
        drop = traj_from(rec, tok, w, lambda r, nr, ids, bf: [] if r == first_nonempty else ids)
        dup_txt = "".join(rd["new"] for rd in rec["read_log"])
        dup = traj_from(rec, tok, w, lambda r, nr, ids, bf: ids + (tok(dup_txt, add_special_tokens=False)["input_ids"]
                                                                   if r == nr - 1 else []))
        flip = traj_from(rec, tok, w, lambda r, nr, ids, bf: [inv[spk_tok[i] % 8 + 1] if i in spk_tok else i for i in ids])
        members = [base, drop, dup, flip]
        creds = [S.credit(t.segs, units, dur, t.wait_reads, [rd.t_ms / 1000 for rd in t.reads], post=post) for t in members]
        for mode in agg:
            adv = group_advantages(mode, members, creds, units, dur, [(creds[0].R, creds[0].lat)])
            for m, t in enumerate(members):
                assert len(adv["tok"][m]) == len(t.reads)
                for r, rd in enumerate(t.reads):
                    assert len(adv["tok"][m][r]) == len(rd.ids)
            agg[mode].append([float(np.mean([x[4] + x[3] + x[2] for rr in adv["tok"][m] for x in rr] or [0]))
                              for m in range(4)])
        print(rec["id"], "R:", [round(c.R, 3) for c in creds], "dup LOO:", [round(d, 3) for d in creds[2].delta][-3:],
              "who(flip):", [round(x, 3) for x in creds[3].who][:4])
    for mode, v in agg.items():
        v = np.array(v)
        print(f"{mode:7s} mean token advantage [base, drop, dup, flip]:", np.round(v.mean(0), 3))
    print("CPU PIPELINE CHECK PASSED")


if __name__ == "__main__":
    main()
