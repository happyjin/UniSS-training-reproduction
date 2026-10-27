"""Fluency plan v3, recipe 2: Thinker oracle-prefix rows from the likelihood alignment (build_align_oracle.py).

For an aligned item and every read t: the prefix is the oracle's units whose write time is before t (the previous
read), the target the units that become writable at t, then the wait marker -- or the rest and <|im_end|> at the read
that hears the end. Every prefix and target is a prefix of the oracle. These rows replace the item's dump rows; the
other items keep their v1 rows (build_thinker_oracle_prefix.py, the same seed), and a share of offline rows (whole
audio, whole oracle) is kept as in v1.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_align_rows --align <align.jsonl> --delta 1.0 \
        --base data/si_thinker_oracle_prefix_v1 --out data/si_thinker_oracle_prefix_v6_align
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--align", required=True)
    ap.add_argument("--delta", default="1.0")
    ap.add_argument("--base", default="data/si_thinker_oracle_prefix_v1")
    ap.add_argument("--items", default="data/unist_simul_thinker_v1/items_{d}.jsonl")
    ap.add_argument("--max-rows", type=int, default=240000)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    audio = {}
    for d in ("en2zh", "zh2en"):
        for line in open(args.items.format(d=d), encoding="utf-8"):
            r = json.loads(line)
            audio[r["id"]] = (r["audio"], int(r["dur_ms"]))
    aligned, rows = set(), []
    for line in open(args.align, encoding="utf-8"):
        a = json.loads(line)
        zh = a["dir"] == "en2zh"
        U, D = a["units"], a["delay_ms"][args.delta]
        join = (lambda us: "".join(us)) if zh else (lambda us: " ".join(us))
        aligned.add(a["id"])
        prev = 0
        for t in a["reads_ms"]:
            last = t == a["reads_ms"][-1]
            k0 = sum(1 for x in D if x <= prev) if prev else 0
            k1 = len(U) if last else sum(1 for x in D if x <= t)
            prefix, tgt = join(U[:k0]), join(U[k0:k1])
            if not zh and prefix and tgt:
                tgt = " " + tgt
            rows.append({"id": a["id"], "audio": audio[a["id"]][0], "dir": a["dir"], "t_ms": t, "dur_ms": audio[a["id"]][1],
                         "prefix": prefix, "target": tgt, "end": "eos" if last else "wait", "kind": "read"})
            prev = t
        rows.append({"id": a["id"], "audio": audio[a["id"]][0], "dir": a["dir"], "t_ms": audio[a["id"]][1],
                     "dur_ms": audio[a["id"]][1], "prefix": "", "target": join(U), "end": "eos", "kind": "offline"})
    base = [json.loads(l) for name in ("rows_train.jsonl", "rows_val.jsonl")
            for l in open(Path(args.base) / name, encoding="utf-8")]
    keep = [r for r in base if r["id"] not in aligned]
    rng = random.Random(args.seed)
    rng.shuffle(keep)
    rows = rows + keep[: max(0, args.max_rows - len(rows))]
    rng.shuffle(rows)
    out.mkdir(parents=True)
    is_val = lambda r: zlib.crc32(r["id"].encode()) % args.val_every == 0
    tr, va = [r for r in rows if not is_val(r)], [r for r in rows if is_val(r)]
    for name, part in (("rows_train.jsonl", tr), ("rows_val.jsonl", va)):
        with open(out / name, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    n_al = sum(r["id"] in aligned for r in rows)
    stats = {"aligned_items": len(aligned), "rows": len(rows), "train": len(tr), "val": len(va), "aligned_rows": n_al,
             "wait_share": round(sum(r["end"] == "wait" for r in rows) / len(rows), 3),
             "empty_wait_share": round(sum(r["end"] == "wait" and not r["target"] for r in rows) / len(rows), 3),
             "args": {k: v for k, v in vars(args).items() if k != "out"}}
    (out / "README.md").write_text("Plan v3 recipe 2 rows (data_clean/build_align_rows.py).\n\n" + json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
