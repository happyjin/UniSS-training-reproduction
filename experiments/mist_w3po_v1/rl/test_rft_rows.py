"""CPU test of rl/build_rft_rows (no model): rows built from real anchor decodes of the pilot (their read blocks have the
same layout as rft_sample's) must round-trip through the tokenizer exactly (prefix -> the committed ids before the read,
target + end token -> the read's block), and the selection rule must follow R / dup / spurious.

    python -m experiments.mist_w3po_v1.rl.test_rft_rows
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.mist_w3po_v1.core.engine import EOS_ID, WAIT_ID
from experiments.mist_w3po_v1.rl.build_rft_rows import keep, rows_of

RUN = Path("eval_outputs/mist_w3po_v1/rl/pilot_w3po_v8")


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def main():
    args = json.loads((RUN / "args.json").read_text())
    from experiments.mist_w3po_v1.core.engine import Prompter
    tok = Prompter("zh2en").tok
    W = {w["id"]: w for w in map(json.loads, open(args["windows"], encoding="utf-8"))}
    n = bad = 0
    for l in list(open(RUN / "anchors" / "rank0.jsonl", encoding="utf-8"))[:40]:
        rec = json.loads(l)
        w = W[rec["id"]]
        reads = rec["trajs"][1]["reads"]          # Lag2 anchor: blocks = new ids + <wait> / <eos>
        rows = rows_of(rec, w, tok, reads)
        for r, rd in zip(rows, reads):
            n += 1
            pre = tok(r["prefix"], add_special_tokens=False)["input_ids"]
            tgt = tok(r["target"], add_special_tokens=False)["input_ids"] + [EOS_ID if r["end"] == "eos" else WAIT_ID]
            if pre != rd["before"] or tgt != rd["ids"]:
                bad += 1
        check(f"{rec['id']}: one row per read, times match", [r["t_ms"] for r in rows] == [rd["t"] for rd in reads])
    print(f"{n} rows, {bad} not round-tripping")
    check("prefix / target round-trip the token ids (>= 99%)", bad <= 0.01 * n)
    g = {"R": 0.4, "dup": 1, "spurious": 0}
    mk = lambda R, d, s: {"greedy": g, "samples": [{"R": R, "dup": d, "spurious": s}], "best": 0}  # noqa: E731
    check("keep: better R, same dup", keep(mk(0.5, 1, 0), 0.0, False))
    check("keep: equal R allowed unless strict", keep(mk(0.4, 1, 0), 0.0, False) and not keep(mk(0.4, 1, 0), 0.0, True))
    check("reject: more duplicates", not keep(mk(0.6, 2, 0), 0.0, False))
    check("reject: more spurious", not keep(mk(0.6, 1, 1), 0.0, False))
    check("reject: below margin", not keep(mk(0.45, 1, 0), 0.1, False))
    print("ALL PASS")


if __name__ == "__main__":
    main()
