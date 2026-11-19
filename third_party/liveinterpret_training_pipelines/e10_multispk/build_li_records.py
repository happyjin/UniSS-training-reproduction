"""E10 data: baseline B's AMI training data -> LiveInterpret S5 `si-rw-m1-v0` records (one per window).

Reads only (never writes) B's files under data/ami_multispk_si_v1 and reuses B's own labels:
  * En->Zh windows: e2e_gstar_stream_v1/windows_{train,validation}_ref.jsonl, whose `stream_target.chunks` are B's
    deadline chunks (punctuation chunks committed 1 s after their last source word, lowlat speaker slots);
  * Zh->En windows: e2e_gstar_zh2en_v1/windows_*_ref.jsonl, chunks rebuilt with B's own `utt_chunks` exactly as
    build_ami_gstar_stream_rows_zh2en.py does (no source word times: proportional ends, lag 1 s, slot_map lowlat);
  * offline whole-window rows and single-speaker (`inst: sd`) rows: drawn from B's final streaming rows files
    (e2e_gstar_stream_v1 / e2e_gstar_stream_zh2en_v1 rows_train.jsonl), --offline-share / --sd-share of the
    streaming records per direction (B: 0.2 / 0.15 of its read rows).

Trajectory (640 ms base chunks, the S5 granularity): a chunk committed at c ms is written at the first step whose
source end is >= c, i.e. B's read rule at 640 ms instead of 1 s. Every chunk text is prefixed with its speaker tag
(e10.common.SPK_TOKENS). Offline and sd records write everything at the last step (sd: no tag, no posterior).

B-form training is Thinker-only (SI_SPEECH_LOSS_WEIGHT=0, the Talker never runs), so `target_codec_codes` holds a
placeholder: one frame per WRITE, only to satisfy the loader's span-tiling checks.

Directions are balanced like B's joint rows (480k per direction): each direction's train list is cycled through
fresh permutations to the larger direction's size (--no-balance to disable).

    cd third_party/liveinterpret_training_pipelines
    PYTHONPATH=/opt/dlami/nvme/jasonleeeli/projects/UniSS $PY e10_multispk/build_li_records.py \
        --out e10_multispk/data/li_s5_records_v1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e10.common import BASE_CHUNK_MS, N_SLOT, instruction, parse_pieces, spk_token  # noqa: E402

UNISS = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
TGT = {"en2zh": "zh", "zh2en": "en"}
MIN_MS = 500                      # S4StreamingDataset.min_audio_sec
MAX_MS = 30000


def steps_for(dur_ms: int, writes: dict[int, list[str]]) -> list[dict]:
    """640 ms steps over [0, dur]; writes: step index -> pieces written at that step."""
    n = max(1, math.ceil(dur_ms / BASE_CHUNK_MS))
    last_w = max(writes) if writes else -1
    steps, cursor = [], 0
    for i in range(n):
        end = min((i + 1) * BASE_CHUNK_MS, dur_ms)
        w = writes.get(i)
        steps.append({"source_start_ms": i * BASE_CHUNK_MS, "source_end_ms": end,
                      "action": "write" if w else "wait", "text_delta": " ".join(w) if w else "",
                      "source_codec_spans": [[cursor, cursor + 1]] if w else [], "commit_ms": end,
                      "source_stream_done": i == n - 1, "translation_eos": i == last_w})
        cursor += bool(w)
    return steps


def record(rid, direction, audio, dur_ms, src_text, writes, post, inst, kind) -> dict:
    steps = steps_for(dur_ms, writes)
    n_w = sum(s["action"] == "write" for s in steps)
    sep = "" if TGT[direction] == "zh" else " "
    return {"schema_version": "si-rw-m1-v0", "record_id": rid, "source_audio": audio, "source_duration_ms": dur_ms,
            "source_transcript": src_text, "target_lang": TGT[direction],
            "target_text": sep.join(s["text_delta"] for s in steps if s["text_delta"]),
            "target_codec_codes": [0] * n_w,
            "trajectory": {"base_chunk_ms": BASE_CHUNK_MS, "codec": {"spec": "placeholder_thinker_only",
                                                                       "frame_ms": 40.0},
                           "raw_target_num_frames": n_w, "steps": steps},
            "e10": {"dir": direction, "inst": inst, "kind": kind, "post": post,
                    "instruction": instruction(TGT[direction], inst == "gstar")}}


def stream_writes(dur_ms: int, chunks: list[tuple[int, str, int]]) -> dict[int, list[str]]:
    """chunks [(slot, text, commit_ms)] in commit order -> {step: [tagged pieces]} (B's rule at 640 ms)."""
    out = defaultdict(list)
    n = max(1, math.ceil(dur_ms / BASE_CHUNK_MS))
    for k, text, commit in chunks:
        i = min(n - 1, max(0, math.ceil(min(commit, dur_ms) / BASE_CHUNK_MS) - 1))
        out[i].append(spk_token(k) + text)
    return dict(out)


def last_step_writes(dur_ms: int, pieces: list[str]) -> dict[int, list[str]]:
    return {max(1, math.ceil(dur_ms / BASE_CHUNK_MS)) - 1: pieces} if pieces else {}


def en2zh_windows(root: Path, split: str, stats: Counter):
    for w in map(json.loads, open(root / "e2e_gstar_stream_v1" / f"windows_{split}_ref.jsonl", encoding="utf-8")):
        chunks = []
        for c in w["stream_target"]["chunks"]:
            for k, text in parse_pieces(c["piece"]):
                chunks.append((k, text, int(c["commit_ms"])))
        src = " ".join(u["en"] for u in w["utts"] if not u.get("hesitation"))
        yield w, chunks, src


def zh2en_windows(root: Path, split: str, stats: Counter, lag_ms: int = 1000):
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import utt_chunks
    smap = json.loads((root / "e2e_gstar_zh2en_v1" / "slot_map.json").read_text(encoding="utf-8"))["lowlat"]
    for w in map(json.loads, open(root / "e2e_gstar_zh2en_v1" / f"windows_{split}_ref.jsonl", encoding="utf-8")):
        m, dur = w["meeting"], w["dur_ms"]
        chunks = []
        for u in w["utts"]:
            if u["hesitation"] or not u["ref"].strip():
                continue
            uu = {"begin_time": w["start_s"] + u["begin_s"], "end_time": w["start_s"] + u["end_s"], "zh": u["ref"]}
            for c in utt_chunks(uu, [], lag_ms, w["start_s"]):
                chunks.append((smap[m][u["speaker_id"]], c["text"].strip(), min(c["commit_ms"], dur),
                               c["src_start_ms"]))
        chunks.sort(key=lambda c: (c[2], c[3]))
        src = " ".join(u["zh"] for u in w["utts"] if not u.get("hesitation"))
        yield w, [c[:3] for c in chunks if c[1]], src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(UNISS / "data/ami_multispk_si_v1"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--offline-share", type=float, default=0.2)
    ap.add_argument("--sd-share", type=float, default=0.15)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke: windows per direction and split")
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()

    root, out = Path(a.data), Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(a.seed)
    stats: dict[str, Counter] = defaultdict(Counter)
    recs: dict[tuple[str, str], list[dict]] = {}
    rows_file = {"en2zh": root / "e2e_gstar_stream_v1" / "rows_train.jsonl",
                 "zh2en": root / "e2e_gstar_stream_zh2en_v1" / "rows_train.jsonl"}
    gen = {"en2zh": en2zh_windows, "zh2en": zh2en_windows}
    for d in ("en2zh", "zh2en"):
        for split in ("train", "validation"):
            st = stats[f"{d}/{split}"]
            L = []
            for w, chunks, src in gen[d](root, split, st):
                if a.limit and len(L) >= a.limit:
                    break
                st["windows"] += 1
                dur = int(w["dur_ms"])
                if not MIN_MS <= dur <= MAX_MS:
                    st["skip_duration"] += 1
                    continue
                if not chunks:
                    st["skip_no_chunk"] += 1
                    continue
                if any(not 1 <= k <= N_SLOT for k, _, _ in chunks):
                    st["skip_bad_slot"] += 1
                    continue
                writes = stream_writes(dur, chunks)
                st["chunks"] += len(chunks)
                st["write_steps"] += len(writes)
                st["multi_speaker_steps"] += sum(len({p.split("|>")[0] for p in ps}) > 1 for ps in writes.values())
                L.append(record(w["id"], d, w["audio"], dur, src, writes, w["post"]["lowlat"], "gstar", "stream"))
            recs[(d, split)] = L
        # offline / sd rows from B's final streaming rows (train only)
        pool = defaultdict(list)
        for r in map(json.loads, open(rows_file[d], encoding="utf-8")):
            if r.get("inst") == "gstar" and r.get("kind") == "offline":
                pool["offline"].append(r)
        for r in map(json.loads, open(rows_file["en2zh"], encoding="utf-8")):   # B's sd pool, both directions
            if r.get("inst") == "sd" and r.get("dir") == d:
                pool["sd"].append(r)
        n = len(recs[(d, "train")])
        st = stats[f"{d}/train"]
        for kind, share in (("offline", a.offline_share), ("sd", a.sd_share)):
            take = rng.sample(pool[kind], min(len(pool[kind]), round(share * n)))
            for r in take:
                dur = int(r["dur_ms"])
                if not MIN_MS <= dur <= MAX_MS:
                    st[f"skip_{kind}_duration"] += 1
                    continue
                if kind == "offline":
                    pieces = [spk_token(k) + t for k, t in parse_pieces(r["target"]) if 1 <= k <= N_SLOT]
                    post, inst = r["post"], "gstar"
                else:
                    pieces, post, inst = [r["target"].strip()], None, "sd"
                if not pieces or not pieces[0]:
                    st[f"skip_{kind}_empty"] += 1
                    continue
                recs[(d, "train")].append(record(f"{kind}:{r['id']}", d, r["audio"], dur, "",
                                                 last_step_writes(dur, pieces), post, inst, kind))
                st[f"{kind}_records"] += 1
            st[f"{kind}_pool"] = len(pool[kind])
    # balance the directions like B's joint rows (cycle permutations of the smaller direction)
    n_max = max(len(recs[(d, "train")]) for d in ("en2zh", "zh2en"))
    for d in ("en2zh", "zh2en"):
        L = recs[(d, "train")]
        st = stats[f"{d}/train"]
        st["train_records_unique"] = len(L)
        if not a.no_balance and len(L) < n_max:
            cyc = []
            while len(cyc) < n_max:
                cyc += rng.sample(L, len(L))
            L = cyc[:n_max]
        rng.shuffle(L)
        recs[(d, "train")] = L
        st["train_records_written"] = len(L)

    for d in ("en2zh", "zh2en"):
        rd = out / "records" / d / "train" / "records"
        rd.mkdir(parents=True)
        with open(rd / "part-00000.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs[(d, "train")])
        (rd / "_manifest.json").write_text(json.dumps({"direction": d, "split": "train", "chunk_ms": BASE_CHUNK_MS,
                                                       "num_records": len(recs[(d, "train")]),
                                                       "parts": [{"file": "part-00000.jsonl",
                                                                  "num_records": len(recs[(d, "train")])}]}))
        vd = out / "val"
        vd.mkdir(parents=True, exist_ok=True)
        with open(vd / f"{d}_val.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs[(d, "validation")])
        # AMI validation meetings are disjoint from the train meetings: nothing to exclude by source text
        (vd / f"{d}_val_keys.json").write_text(json.dumps({"direction": d, "val_keys": [],
                                                           "audio_root": str(root)}))
    summary = {"stats": {k: dict(v) for k, v in stats.items()}, "args": vars(a),
               "records": {f"{d}/{s}": len(v) for (d, s), v in recs.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
