"""sst_hsmm_v1 prereg change 40-F4: add the talker lookahead ("look", not spoken) to an HSC thinker blocks file, offline.

For every non-final block written at read t, look = the next --units units of that read's draft after everything committed
so far (= si_thinker_stream's LA2 rule: draft[committed: committed + look_units]); the final block gets none. The draft is
the stored drafts[t]["draft"] (raw-view d0, the content that is committed). Asserts that the committed prefix is a prefix
of that draft by unit COUNT (the thinker slices drafts by index; a unit the next continuation re-tokenises, e.g. "did" ->
"did.", differs in text, counted as prefix_text_mismatch). Refuses to overwrite.

    python -m experiments.sst_hsmm_v1.p3tl_look --blocks <in.jsonl> --direction en2zh --units 3 --out <out.jsonl>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import join, units  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--direction", choices=("en2zh", "zh2en"), required=True)
    ap.add_argument("--units", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refuse to overwrite {out}")
    zh = a.direction.endswith("zh")
    n_blk = n_look = n_empty = n_mis = 0
    rows = []
    for r in map(json.loads, open(a.blocks, encoding="utf-8")):
        drafts = {int(d["t"]): d["draft"] for d in r["drafts"]}
        nc = 0
        for b in r["blocks"]:
            nc += len(units(b["text"], zh))
            if b["final"]:
                continue
            n_blk += 1
            d = units(drafts[int(b["write_ms"])], zh)
            assert len(d) >= nc, (r["id"], b["write_ms"])
            n_mis += int(units("".join(
                x["text"] for x in r["blocks"][: r["blocks"].index(b) + 1]), zh) != d[:nc])
            look = d[nc: nc + a.units]
            b["look"] = join(look, zh) if look else ""
            n_look += int(bool(look))
            n_empty += int(not look)
        rows.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(rows), "nonfinal_blocks": n_blk, "with_look": n_look, "empty_look": n_empty, "prefix_text_mismatch": n_mis}))


if __name__ == "__main__":
    main()
