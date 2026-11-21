"""E10 v3 records: baseline B's two training stages on B's own files, in LiveInterpret's chunk-as-turn format.

  --stage o2    B's O2 rows (e2e_gstar_bidir_v2/rows_train.jsonl, 67,795 rows: offline G-STAR + single-speaker), one
                record per row, every text at the last step; decision labels masked (e10.v3), speaker stream laid
                out offline (the row's offline posterior, as B).
  --stage joint B's joint rows (e2e_gstar_stream_bidir_v2/rows_train.jsonl). Its `read` rows are the reads of B's
                streaming windows; a window's reads become ONE streaming record whose writes are B's deadline
                chunks (exactly as E10 v1, build_li_records.py), with decision loss and the lowlat speaker stream.
                Its offline / sd rows are taken one record per unique row, decision masked. Directions are then
                balanced by cycling (B's joint file is balanced the same way).
Validation (both stages): B's validation windows as streaming records (what the in-training eval measures).

Unlike v1/v2 there is no --offline-share: the mix is B's own (joint: ~48 % streaming windows, ~43 % offline,
~8 % single-speaker by window).

    cd third_party/liveinterpret_training_pipelines
    PYTHONPATH=/opt/dlami/nvme/jasonleeeli/projects/UniSS $PY e10_multispk/build_li_records_v3.py --stage o2 \
        --out e10_multispk/data/li_v3_o2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_li_records as V1  # noqa: E402
from e10.common import N_SLOT, parse_pieces, spk_token  # noqa: E402

UNISS = V1.UNISS
ROWS = {"o2": "e2e_gstar_bidir_v2/rows_train.jsonl", "joint": "e2e_gstar_stream_bidir_v2/rows_train.jsonl"}


def v3rec(rid, d, audio, dur, src, writes, post, inst, kind, layout, mask):
    r = V1.record(rid, d, audio, dur, src, writes, post, inst, kind)
    r["e10"]["v3"] = {"layout": layout, "mask_decision": mask, "post": post}
    return r


def row_record(r, st):
    """B offline / sd row -> one masked record (None if unusable)."""
    dur = int(r["dur_ms"])
    kind = "sd" if r.get("inst") == "sd" else "offline"
    if not V1.MIN_MS <= dur <= V1.MAX_MS:
        st[f"skip_{kind}_duration"] += 1
        return None
    if kind == "offline":
        pieces = [spk_token(k) + t for k, t in parse_pieces(r["target"]) if 1 <= k <= N_SLOT]
        if not pieces or not r.get("post"):
            st["skip_offline_empty"] += 1
            return None
        rec = v3rec(f"offline:{r['id']}", r["dir"], r["audio"], dur, "", V1.last_step_writes(dur, pieces),
                    r["post"], "gstar", "offline", "offline", True)
    else:
        t = (r.get("target") or "").strip()
        if not t:
            st["skip_sd_empty"] += 1
            return None
        rec = v3rec(f"sd:{r['id']}", r["dir"], r["audio"], dur, "", V1.last_step_writes(dur, [t]), None, "sd",
                    "sd", "none", True)
    st[f"{kind}_records"] += 1
    return rec


def stream_records(root, d, split, st):
    out = []
    gen = {"en2zh": V1.en2zh_windows, "zh2en": V1.zh2en_windows}[d]
    for w, chunks, src in gen(root, split, st):
        dur = int(w["dur_ms"])
        if not V1.MIN_MS <= dur <= V1.MAX_MS or not chunks or any(not 1 <= k <= N_SLOT for k, _, _ in chunks):
            st["skip_stream"] += 1
            continue
        writes = V1.stream_writes(dur, chunks)
        st["stream_records"] += 1
        out.append(v3rec(w["id"], d, w["audio"], dur, src, writes, w["post"]["lowlat"], "gstar", "stream",
                         "stream", False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["o2", "joint"], required=True)
    ap.add_argument("--data", default=str(UNISS / "data/ami_multispk_si_v1"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: records per direction")
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(a.seed)
    stats: dict[str, Counter] = defaultdict(Counter)
    train = {"en2zh": [], "zh2en": []}
    seen = set()
    for line in open(root / ROWS[a.stage], encoding="utf-8"):
        r = json.loads(line)
        if a.stage == "joint" and r.get("kind") != "offline":
            continue                      # read rows: rebuilt per window below
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        d = r["dir"]
        if a.limit and len(train[d]) >= a.limit:
            continue
        rec = row_record(r, stats[f"{d}/train"])
        if rec is not None:
            train[d].append(rec)
    if a.stage == "joint":
        for d in ("en2zh", "zh2en"):
            s = stream_records(root, d, "train", stats[f"{d}/train"])
            train[d] += s[: a.limit] if a.limit else s
    val = {d: stream_records(root, d, "validation", stats[f"{d}/validation"]) for d in ("en2zh", "zh2en")}
    if a.limit:
        val = {d: v[: max(8, a.limit // 4)] for d, v in val.items()}
    if a.stage == "joint":                # balance the directions by cycling (as B's joint rows)
        n_max = max(len(v) for v in train.values())
        for d, L in train.items():
            stats[f"{d}/train"]["unique"] = len(L)
            cyc = []
            while len(cyc) < n_max:
                cyc += rng.sample(L, len(L))
            train[d] = cyc[:n_max]
    for d, L in train.items():
        rng.shuffle(L)
        stats[f"{d}/train"]["written"] = len(L)
        rd = out / "records" / d / "train" / "records"
        rd.mkdir(parents=True)
        with open(rd / "part-00000.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in L)
        (rd / "_manifest.json").write_text(json.dumps({"direction": d, "split": "train", "chunk_ms": V1.BASE_CHUNK_MS,
                                                       "num_records": len(L),
                                                       "parts": [{"file": "part-00000.jsonl", "num_records": len(L)}]}))
        vd = out / "val"
        vd.mkdir(parents=True, exist_ok=True)
        with open(vd / f"{d}_val.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in val[d])
        (vd / f"{d}_val_keys.json").write_text(json.dumps({"direction": d, "val_keys": [], "audio_root": str(root)}))
    kinds = {d: dict(Counter(r["e10"]["kind"] for r in L)) for d, L in train.items()}
    summary = {"stage": a.stage, "stats": {k: dict(v) for k, v in stats.items()}, "train_kinds": kinds,
               "val": {d: len(v) for d, v in val.items()}, "args": vars(a)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
