"""E8 (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §1 S2): X-Translator (official code + weights,
released config `xt-default`: Qwen3-ASR-1.7B + LMT-60-8B + X-Voice 0.4B Stage2, CAM++ speaker prompts) on our ONLINE
multi-speaker test windows, one window = one independent X-Translator session.

The per-session logic is the RealSI driver's, imported unchanged: experiments/baselines_xtranslator_v1/xt_realsi.py
Runner.process (X-Translator's own TranslationSession, 512-sample frames, ASR request chunks processed at once,
SpeakerPromptTracker feed + boundary flush, browser "stop" + finalize at the window end, every translation item
translated and synthesised at once with write_ms = source ms received; computation not counted) and
Runner.translate_item (_tts_loop's body: an MT or TTS exception drops the segment's audio, as the session does).
Only the input unit differs: the window's `audio` (16 kHz) instead of a RealSI sentence cut from a talk.

One addition, infrastructure only: a connection failure to one of OUR servers (requests ConnectionError: refused /
reset / aborted) is not a model failure; the shard then exits with code 3 WITHOUT writing that window's row, so the
queue can resume it on healthy servers (--resume skips the ids already written). Every other exception follows E7:
the window is recorded with `error` and scores as empty.

Per window one row of <out>/part_<shard>.jsonl:
  segs      every translation item in emission order: text (LMT output, stripped), source_text (the ASR text it
            translates), write_ms (source ms received when the ASR final that produced it was emitted), dur_ms and wav
            (the X-Voice audio, 48 kHz PCM16 -> 16 kHz with soxr as in RealSI; dur 0 / wav null when there is none),
            mt_error / tts_error, X-Translator's own speaker_id / source_start_ms / source_end_ms / clone prompt
            (recorded only; the scoring uses the E7 rules, build_e8.py)
  counters  boundary_flushes / speaker_events / asr_chunks;  error, elapsed_s, config
The scorer formats (speaker relabelling, streams, queues): build_e8.py (CPU).

    PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=<c> $V/bin/python run_e8.py --windows <online windows> --direction zh2en|en2zh \
        --out <raw dir> --asr-port P --tts-port P --mt-port P --shard k --num-shards n [--ids a,b] [--limit N] [--resume]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/baselines_xtranslator_v1"))
import xt_realsi as XR  # noqa: E402  (Runner, import_xtranslate, LANGS, SR: the RealSI driver, unchanged)

XT_CONFIG = XR.XT_SRC / "config.json"
CAMPPLUS = "/opt/dlami/nvme/jasonleeeli/models/xtranslator/campplus"
DIRS = {"en2zh": ("en", "zh"), "zh2en": ("zh", "en")}
SR = XR.SR


class ServerDown(BaseException):
    """a connection failure to one of our servers; BaseException so the session's `except Exception` does not record
    it as a dropped segment."""


def is_conn_error(e: BaseException) -> bool:
    import requests
    return isinstance(e, requests.exceptions.ConnectionError)


def write_config(direction: str, asr: int, mt: int, tts: int, run: Path, path: Path) -> dict:
    """experiments/baselines_xtranslator_v1/make_config.py for point xt-default, verbatim fields: the released config
    with only the service URLs, the direction, the local CAM++ copy and the storage dirs changed."""
    c = json.load(open(XT_CONFIG, encoding="utf-8"))
    c["asr"]["qwen3_asr_url"] = f"http://127.0.0.1:{asr}/v1/recognize"
    c["translation"]["lmt_url"] = f"http://127.0.0.1:{mt}/v1/chat/completions"
    c["tts"]["xvoice_tts_url"] = f"http://127.0.0.1:{tts}/synthesize"
    c["translation"]["source_lang"], c["translation"]["target_lang"] = DIRS[direction]
    c["tts"]["speaker_embedding_model"] = CAMPPLUS
    c["storage"] = {"session_audio_dir": str(run.resolve() / "sessions"), "log_dir": str(run.resolve() / "logs")}
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(c, fh, ensure_ascii=False, indent=2)
    return c


class E8Runner(XR.Runner):
    def session_for(self, item_id: str):
        sess = super().session_for(item_id)
        for svc, name in ((sess.mt_service, "translate"), (sess.tts_service, "synthesize")):
            f = getattr(svc, name)

            def wrapped(*a, _f=f, _n=name, **k):
                try:
                    return _f(*a, **k)
                except Exception as e:  # noqa: BLE001
                    if is_conn_error(e):
                        raise ServerDown(f"{_n}: {type(e).__name__}: {e}") from e
                    raise
            setattr(svc, name, wrapped)
        return sess


def fresh(p: Path) -> Path:
    """never overwrite (a crash between the wav and the row writes leaves an orphan wav)."""
    k, q = 0, p
    while q.exists():
        k += 1
        q = p.with_name(f"{p.stem}-r{k}{p.suffix}")
    return q


def make_record(w: dict, wav: np.ndarray, res: dict, err, out: Path, tts_sr: int, cfg: dict, elapsed: float) -> dict:
    segs = []
    for k, s in enumerate(res["segs"]):
        txt = (s.get("translated") or "").strip()
        pcm = s.get("pcm") or b""
        row = {"text": txt, "source_text": s.get("source_text", ""), "write_ms": int(round(s["write_ms"])),
               "dur_ms": 0, "wav": None, "mt_error": s.get("mt_error"), "tts_error": s.get("tts_error"),
               "speaker_id": s.get("speaker_id", ""), "clone_audio_path": s.get("clone_audio_path", ""),
               "xt_source_start_ms": s.get("source_start_ms"), "xt_source_end_ms": s.get("source_end_ms"),
               "segment_index": s.get("segment_index"), "segment_count": s.get("segment_count"),
               "mt_s": s.get("mt_s"), "tts_s": s.get("tts_s"), "tts_samples": len(pcm) // 2}
        if pcm:
            a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            a16 = soxr.resample(a, tts_sr, SR).astype(np.float32) if tts_sr != SR else a
            p = fresh(out / "wav" / f"{w['id']}_{k:03d}.wav")
            sf.write(p, a16, SR)
            row["wav"], row["dur_ms"] = str(p), int(round(len(a16) / 16))
        segs.append(row)
    return {"id": w["id"], "meeting": w["meeting"], "dur_ms": int(w["dur_ms"]), "src_samples": int(len(wav)),
            "segs": segs, "counters": res.get("counters", {}), "session_id": res.get("session_id", ""),
            "elapsed_s": round(elapsed, 3), "error": err, "config": cfg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--direction", choices=sorted(DIRS), required=True)
    ap.add_argument("--out", required=True, help="raw dir of the set (shared by its shards)")
    ap.add_argument("--asr-port", type=int, required=True)
    ap.add_argument("--tts-port", type=int, required=True)
    ap.add_argument("--mt-port", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--ids", default="", help="comma-separated window ids (smoke tests only)")
    ap.add_argument("--limit", type=int, default=0, help="first N windows of the shard (smoke tests only)")
    ap.add_argument("--resume", action="store_true", help="append to an existing part file, skipping its ids")
    a = ap.parse_args()

    out = Path(a.out)
    for d in ("wav", "sessions", "logs", "configs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    part = out / f"part_{a.shard:02d}.jsonl"
    done = set()
    if part.exists():
        if not a.resume:
            sys.exit(f"refuse: {part} exists (use --resume)")
        for l in open(part, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except json.JSONDecodeError:
                pass
    tag = f"shard{a.shard:02d}_a{a.asr_port}_t{a.tts_port}_m{a.mt_port}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"
    cfg_path = out / "configs" / f"{tag}.json"
    cj = write_config(a.direction, a.asr_port, a.mt_port, a.tts_port, out, cfg_path)
    os.environ["XTRANSLATE_CONFIG"] = str(cfg_path.resolve())
    os.environ["XTRANSLATE_LOG_DIR"] = str((out / "logs" / f"xtranslate_shard{a.shard:02d}").resolve())
    os.environ["XTRANSLATE_SESSION_AUDIO_DIR"] = str((out / "sessions" / f"shard{a.shard:02d}").resolve())
    S, Settings = XR.import_xtranslate(REPO / "eval_outputs/mist_w3po_v1/e8_xtranslator/_pkg")
    for h in logging.getLogger("xtranslate").handlers:   # full INFO log in the file, only warnings on stderr
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.WARNING)
    settings = Settings.from_env()
    assert (settings.source_lang, settings.target_lang) == DIRS[a.direction], (settings.source_lang, settings.target_lang)
    assert settings.paraformer_chunk_size == 2560 and settings.asr_provider == "qwen3", settings
    cfg = {"system": "X-Translator (Qwen3-ASR-1.7B + LMT-60-8B + X-Voice 0.4B Stage2)", "point": "xt-default",
           "direction": a.direction, "code": "X-Translator dc6a59a, X-Voice 817bcdd, qwen-asr 0.0.6, vllm 0.14.0",
           "config_file": str(cfg_path), "asr_request_chunk_samples": settings.paraformer_chunk_size,
           "asr_server_chunk_size_sec": 0.6, "frame_samples": XR.FRAME, "driver": "xt_realsi.Runner (unchanged)",
           "registration": "reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md"}
    assert cj["translation"]["max_tokens"] == 256

    W = [json.loads(l) for l in open(a.windows, encoding="utf-8")]
    if a.ids:
        want = set(a.ids.split(","))
        W = [w for w in W if w["id"] in want]
    items = W[a.shard:: a.num_shards]
    if a.limit:
        items = items[: a.limit]
    runner = E8Runner(S, settings, out / "sessions")
    print(f"[e8 {time.strftime('%H:%M:%S')}] dir={a.direction} shard={a.shard}/{a.num_shards} windows={len(items)} "
          f"done={len(done)} asr={settings.qwen3_asr_url} mt={settings.lmt_url} tts={settings.xvoice_tts_url}", flush=True)
    t_all = time.time()
    n_new = 0
    with open(part, "a", encoding="utf-8") as fh:
        for k, w in enumerate(items):
            if w["id"] in done:
                continue
            x, sr = sf.read(w["audio"], dtype="float32", always_2d=True)
            assert sr == SR, w["audio"]
            wav = x.mean(axis=1)
            t0 = time.perf_counter()
            err = None
            try:
                res = runner.process(w, wav)
            except ServerDown as e:
                print(f"[e8] SERVER_DOWN at {w['id']}: {e}", flush=True)
                sys.exit(3)
            except Exception as e:  # noqa: BLE001
                if is_conn_error(e):
                    print(f"[e8] SERVER_DOWN (asr) at {w['id']}: {type(e).__name__}: {e}", flush=True)
                    sys.exit(3)
                err = traceback.format_exc()[-2000:]
                res = {"segs": [], "counters": {}, "session_id": ""}
                print(f"[e8] ERROR {w['id']}\n{err}", flush=True)
            rec = make_record(w, wav, res, err, out, settings.tts_sample_rate, cfg, time.perf_counter() - t0)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_new += 1
            if n_new % 25 == 1 or a.ids or a.limit:
                print(f"[e8 {time.strftime('%H:%M:%S')}] {k + 1}/{len(items)} {w['id']} dur={w['dur_ms']} "
                      f"segs={len(rec['segs'])} spoken={sum(1 for s in rec['segs'] if s['wav'])} "
                      f"flush={rec['counters'].get('boundary_flushes')} t={rec['elapsed_s']:.1f}s err={bool(err)}",
                      flush=True)
    print(f"[e8 {time.strftime('%H:%M:%S')}] E8_SHARD_DONE {part} {len(items)} windows ({n_new} new) in "
          f"{time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
