"""Robust line, combination helpers (CPU; reports/mist_w3po_v1/robust_combo_plan_20261120.zh-CN.md).

  assemble --root <arm root> --dir <d>   NCWI merged fallback: the per-alpha parts <root>/parts/thinker_<d>_a<alpha>/
            blocks_<d>_w1.jsonl -> <root>/thinker_<d>/blocks_<d>_w1.jsonl in S1 item order; every S1 item exactly once,
            from the part of its own alpha* (wave16/eval_ncwi_merged/parts/items_<d>_a<alpha>.jsonl, = wave16_tools groups).
  groups --dir <d>                       print wave 16's merged-fallback groups for one direction ("<d> <alpha> <model>
            <items file> <n>", from wave16/groups.txt); exit 2 if wave 16 used the LoRA path (NCWI_ROOT = eval_ncwi).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
W16 = REPO / os.environ.get("COMBO_W16", "eval_outputs/mist_w3po_v1/wave16")   # override only for DRY_RUN tests
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def groups(d: str) -> list[list[str]]:
    rows = [ln.split() for ln in open(W16 / "groups.txt", encoding="utf-8") if ln.strip()]
    return [r for r in rows if r[0] == d]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("assemble", "groups"))
    ap.add_argument("--root", default="")
    ap.add_argument("--dir", required=True, choices=("en2zh", "zh2en"))
    a = ap.parse_args()
    d = a.dir
    if a.cmd == "groups":
        nr = (W16 / "NCWI_ROOT").read_text().strip()
        if nr == "eval_ncwi":
            raise SystemExit(2)
        assert nr == "eval_ncwi_merged", nr
        for r in groups(d):
            print(" ".join(r))
        return
    root = Path(a.root)
    out = root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    order = [r["id"] for r in jl(S1 / "data" / f"items_{d}.jsonl")]
    got = {}
    for _, al, _model, items, n in groups(d):
        want = [r["id"] for r in jl(REPO / items)]
        assert len(want) == int(n), (al, len(want), n)
        rows = jl(root / "parts" / f"thinker_{d}_a{al}" / f"blocks_{d}_w1.jsonl")
        assert [r["id"] for r in rows] == want, f"part a{al}: ids differ from its items file"
        for r in rows:
            assert r["id"] not in got, r["id"]
            r["alpha"] = float(al)
            got[r["id"]] = r
    assert set(got) == set(order), (len(got), len(order))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for i in order:
            fh.write(json.dumps(got[i], ensure_ascii=False) + "\n")
    print(f"assembled {len(order)} rows -> {out}")


if __name__ == "__main__":
    main()
