"""RFT / RAFT rows from rl/rft_sample.py records (plan branch B2, 2026-11-07): keep a window's best sampled decode when
its training reward is at least the greedy decode's (+ --margin; --strict: strictly above) and it has no more duplicate
and no more spurious segments than the greedy decode; write one SFT row per read of that decode in the stage-0 row
format (experiments/mist_w3po_v1/training/pretrain_thinker_gstar_s0.py: prefix = committed text before the read,
target = the read's new text, end = wait / eos, ctx, post frames after t - LOOKAHEAD zeroed). Refuses to overwrite.

    python -m experiments.mist_w3po_v1.rl.build_rft_rows --samples <dir> --base <merged> --out <rows dir> [--margin 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.core.engine import EOS_ID, LOOKAHEAD_MS, WAIT_ID


def rows_of(rec: dict, w: dict, tok, reads: list[dict]) -> list[dict]:
    out = []
    for rd in reads:
        new = [i for i in rd["ids"] if i not in (EOS_ID, WAIT_ID)]
        t = int(rd["t"])
        out.append({"id": f"{w['id']}|rft|{t}", "audio": w["audio"], "dir": w.get("dir", "zh2en"), "t_ms": t,
                    "dur_ms": w["dur_ms"], "post": w["post"]["lowlat"], "post_until_ms": max(0, t - LOOKAHEAD_MS),
                    "prefix": tok.decode(rd["before"], skip_special_tokens=False),
                    "target": tok.decode(new, skip_special_tokens=False), "end": "eos" if rd["final"] else "wait",
                    "kind": "read", "inst": "gstar", "preset": "lowlat", "ctx": rec.get("ctx", "")})
    return out


def keep(rec: dict, margin: float, strict: bool) -> bool:
    g, b = rec["greedy"], rec["samples"][rec["best"]]
    better = b["R"] > g["R"] + margin if strict else b["R"] >= g["R"] + margin
    return better and b["dup"] <= g["dup"] and b["spurious"] <= g["spurious"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_train_diar.jsonl")
    ap.add_argument("--dir", default="zh2en")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "rows_train.jsonl").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True, exist_ok=True)
    from experiments.mist_w3po_v1.core.engine import Prompter
    tok = Prompter(a.dir).tok      # the training / inference tokenizer (has the speaker and time tokens)
    W = {w["id"]: w for w in map(json.loads, open(a.windows, encoding="utf-8"))}
    n_win = n_keep = n_rows = n_same = 0
    dR = []
    with open(out / "rows_train.jsonl", "x", encoding="utf-8") as fh:
        for f in sorted(Path(a.samples).glob("part_*.jsonl")):
            for l in open(f, encoding="utf-8"):
                try:
                    rec = json.loads(l)
                except json.JSONDecodeError:
                    continue
                n_win += 1
                if not keep(rec, a.margin, a.strict):
                    continue
                n_keep += 1
                dR.append(rec["samples"][rec["best"]]["R"] - rec["greedy"]["R"])
                n_same += [r["ids"] for r in rec["best_reads"]] == [r["ids"] for r in rec["greedy_reads"]]
                for r in rows_of(rec, W[rec["id"]], tok, rec["best_reads"]):
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_rows += 1
    summ = {"windows": n_win, "kept": n_keep, "rows": n_rows, "kept_identical_to_greedy": n_same,
            "mean_dR_kept": sum(dR) / max(len(dR), 1), "args": vars(a)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
