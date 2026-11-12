"""Hy-SimulMT-30B-A3B on RealSI sentences through SimulInfer 0.5.18's own policy (registration:
paper/baselines_run_plan_hysimulmt_v1.md, tracked copy reports/s2st_baselines_v1/baselines_run_plan_hysimulmt_v1.md).

Each RealSI sentence (si_infer.realsi_items, verbatim below) is one simulinfer.translation.VLLMSession, driven exactly
as SimulInfer's file upload path (app.translate): push(all samples) then finish(). Settings = SimulInfer's Settings
defaults + the [inference] values of its deploy config configs/deploy/target_only.toml + the grid point (chunk_ms,
st_wait_chunks). Computation-unaware: every event's end_ms (source ms read when the model was called) is its commit
time. Writes <out>/events_<dir>.jsonl (one row per sentence, raw events without token-id arrays), <out>/blocks_<dir>.jsonl
(talker blocks, plan §3) and <out>/meta_<dir>.json; refuses existing files.

    venvs/hysimulmt/bin/python experiments/baselines_hysimulmt_v1/hy_realsi.py --direction en2zh --point hy-std-w4 \
        --url http://127.0.0.1:18816/v1 --out eval_outputs/baselines_hysimulmt_v1/thinker/hy-std-w4 [--concurrency 48]
    ... --blocks-only --out <dir>   (rebuild blocks_<dir>.jsonl from events_<dir>.jsonl into blocks_<dir>.rebuilt.jsonl)
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import glob
import json
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
REALSI = REPO / "data/raw/realsi/RealSI_data/data"
GRID = {   # plan §2 + amendment 1 (§7): the two native modes only (chunk_ms after the fixed 1600 ms first chunk, st_wait_chunks)
    "hy-std-w4": (1600, 4), "hy-sc-w4": (800, 4)}
# SimulInfer configs/deploy/target_only.toml [inference] (the translation-only model's deployment config)
DEPLOY_INFERENCE = dict(first_chunk_ms=1600, no_repeat_ngram_size=3, st_no_repeat_ngram_size=3,
                        model_logits_processors=True, no_chinese_target=True, simulst_crop_speech=False,
                        st_crop_speech=True)
SIMUL_DIR = {"en2zh": "en-zh", "zh2en": "zh-en"}
ALNUM = re.compile(r"[A-Za-z0-9]")


def realsi_items(direction: str) -> list[dict]:
    """experiments/uniss_qwen3_omni_v1/evaluation/si_infer.py realsi_items, verbatim."""
    items = []
    for f in sorted(glob.glob(str(REALSI / direction / "json" / "*.json"))):
        j = json.load(open(f, encoding="utf-8"))
        for k, s in enumerate(j["segment"]):
            subs = [u for u in s.get("utterance", []) if u.get("sub_trg_text", "").strip()]
            if not subs or not s.get("trg_text", "").strip():
                continue
            items.append({"id": f"{j['vid']}_{k:03d}", "wav": str(REALSI / direction / "wav" / f"{j['vid']}.wav"),
                          "start": int(s["start_time"]), "end": int(s["end_time"]), "src_text": s["src_text"],
                          "trg_text": s["trg_text"], "subs": [(int(u["sub_end_time"]), u["sub_trg_text"].strip())
                                                              for u in subs]})
    return items


def settings_for(point: str, url: str):
    from simulinfer.config import Model, Settings
    chunk_ms, wait = GRID[point]
    model = Model(backend="vllm", base_url=url, model="Hy-SimulMT-30B-A3B")
    return Settings(**DEPLOY_INFERENCE, chunk_ms=chunk_ms, st_wait_chunks=wait, models={"target_only": model}), model


def blocks_of(events: list[dict], dur_ms: int, tgt_en: bool) -> tuple[list[dict], list[dict]]:
    """plan §3: one block per event with committed text, write_ms = the event's end_ms. Returns (blocks in the talker's
    format {write_ms, text, final}, per-block diagnostics)."""
    out, diag, prev_raw = [], [], ""
    for e in events:
        if e.get("event") not in ("chunk", "flush"):
            continue
        raw = e.get("text") or ""
        if not raw.strip():
            continue
        text, mid_word = raw.strip(), False
        if tgt_en and out:
            mid_word = bool(ALNUM.match(prev_raw[-1]) and ALNUM.match(raw[0]))
            text = text if mid_word else " " + text
        assert int(e["end_ms"]) <= dur_ms, (e["end_ms"], dur_ms)
        out.append({"write_ms": int(e["end_ms"]), "text": text, "final": False})
        diag.append({"event": e.get("event"), "recovery": e.get("recovery"), "trigger": e.get("recovery_trigger"),
                     "mid_word": mid_word})
        prev_raw = raw
    if out:
        out[-1]["final"] = True
    for a, b in zip(out, out[1:]):
        assert a["write_ms"] <= b["write_ms"], (a, b)
    return out, diag


SLIM_DROP = ("generated_token_ids", "st_generated_token_ids")


async def run_one(it: dict, wav: np.ndarray, settings, model, direction: str, sem: asyncio.Semaphore) -> dict:
    from simulinfer.translation import VLLMSession
    async with sem:
        t0 = time.time()
        s = VLLMSession(model, settings, "target_only", SIMUL_DIR[direction])
        events, err = [], None
        try:
            await s.prepare()
            async for e in s.push(wav):
                events.append(e)
            async for e in s.finish():
                events.append(e)
        except Exception as exc:   # recorded, never silently dropped
            err = f"{type(exc).__name__}: {exc}"
        finally:
            await s.close()
    slim = [{k: v for k, v in e.items() if k not in SLIM_DROP} for e in events]
    done = next((e for e in events if e.get("event") == "done"), None)
    return {"id": it["id"], "dur_ms": len(wav) // 16, "events": slim, "full_text": done["full_text"] if done else None,
            "error": err, "wall_s": round(time.time() - t0, 2)}


async def main_async(a) -> None:
    from simulinfer.translation import ChatAPI
    settings, model = settings_for(a.point, a.url)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev_path, bl_path, meta_path = out / f"events_{a.direction}.jsonl", out / f"blocks_{a.direction}.jsonl", out / f"meta_{a.direction}.json"
    for p in (ev_path, bl_path, meta_path):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    api = ChatAPI(model, settings, "target_only", SIMUL_DIR[a.direction])   # SimulInfer's own preflight (preflight.check)
    try:
        check = await api.check()
        check["no_repeat_processors"] = await api.check_repetition_processors()
    finally:
        await api.close()
    print("preflight", json.dumps(check, ensure_ascii=False), flush=True)
    items = realsi_items(a.direction)
    if a.limit:
        items = items[: a.limit]
    cache, wavs = {}, {}
    for it in items:   # si_infer's audio: the talk wav (16 kHz, channels averaged) cut at start / end
        if it["wav"] not in cache:
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            assert sr == 16000, it["wav"]
            cache = {it["wav"]: x.mean(axis=1)}
        wavs[it["id"]] = np.ascontiguousarray(cache[it["wav"]][it["start"] * 16: it["end"] * 16], dtype=np.float32)
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()
    tasks = [asyncio.create_task(run_one(it, wavs[it["id"]], settings, model, a.direction, sem)) for it in items]
    rows, n_done = {}, 0
    for fut in asyncio.as_completed(tasks):
        r = await fut
        rows[r["id"]] = r
        n_done += 1
        if n_done % 50 == 0:
            print(f"{n_done}/{len(items)} sentences, {time.time() - t0:.0f} s", flush=True)
    errors = [r for r in rows.values() if r["error"]]
    with open(ev_path, "x", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(rows[it["id"]], ensure_ascii=False) + "\n")
    write_blocks(items, rows, a.direction, bl_path)
    meta = {"point": a.point, "grid": GRID[a.point], "direction": a.direction, "url": a.url, "n_items": len(items),
            "n_errors": len(errors), "errors": [(r["id"], r["error"]) for r in errors][:20],
            "settings": {k: v for k, v in dataclasses.asdict(settings).items() if k not in ("models", "model_endpoints")},
            "preflight": check, "wall_s": round(time.time() - t0, 1),
            "versions": {n: version(n) for n in ("simulinfer", "simulst", "vllm", "torch", "transformers", "fireredvad")},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
    with open(meta_path, "x", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print("DONE", json.dumps({k: meta[k] for k in ("point", "direction", "n_items", "n_errors", "wall_s")}), flush=True)
    if errors:
        sys.exit(2)


def write_blocks(items: list[dict], rows: dict, direction: str, path: Path) -> None:
    """written under a temporary name, then renamed: talker workers wait for the final name to appear."""
    tgt_en = direction == "zh2en"
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "x", encoding="utf-8") as fh:
        for it in items:
            r = rows[it["id"]]
            bl, diag = blocks_of(r["events"], r["dur_ms"], tgt_en) if not r["error"] else ([], [])
            fh.write(json.dumps({"id": it["id"], "translation": r["full_text"] or "", "dur_ms": r["dur_ms"],
                                 "blocks": bl, "diag": diag}, ensure_ascii=False) + "\n")
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    tmp.rename(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction", choices=sorted(SIMUL_DIR), required=True)
    ap.add_argument("--point", choices=sorted(GRID))
    ap.add_argument("--url", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="first n sentences only (plumbing checks)")
    ap.add_argument("--blocks-only", action="store_true")
    a = ap.parse_args()
    if a.blocks_only:
        out = Path(a.out)
        rows = {r["id"]: r for r in map(json.loads, open(out / f"events_{a.direction}.jsonl", encoding="utf-8"))}
        items = [{"id": i} for i in rows]
        write_blocks(items, rows, a.direction, out / f"blocks_{a.direction}.rebuilt.jsonl")
        return
    if not (a.point and a.url):
        ap.error("--point and --url are required")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
