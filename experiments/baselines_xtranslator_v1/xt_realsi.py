"""X-Translator on RealSI sentence level, simulated source clock (R4, paper/baselines_run_plan_xtranslator_v1.md).

One RealSI sentence = one X-Translator session. The session is X-Translator's own TranslationSession
(backend/session.py, commit dc6a59a) built with a no-op websocket; its own methods do the work
(_handle_audio, _process_asr_chunk -> Qwen3StreamingRecognizer, _emit_asr_output, _flush_asr_finalize,
_reset_asr_turn, _prompt_fields_for_source_span -> SpeakerPromptTracker, mt_service, tts_service). Only the asyncio /
websocket orchestration is replaced by a synchronous loop on a simulated source clock (computation not counted):

  * audio enters in the browser's native 512-sample frames (frontend/main.jsx ScriptProcessor 512, 16 kHz PCM16);
  * per frame: session._handle_audio (ASR buffer), then SpeakerPromptTracker.feed (as _speaker_loop); a speaker change
    runs _request_asr_boundary_flush's logic (pending audio up to the request, then _flush_asr_finalize);
  * whenever the ASR buffer holds one request chunk (paraformer_chunk_size samples) it is processed at once
    (_next_asr_chunk without backlog trimming: the simulated ASR never falls behind);
  * at the sentence end the browser's "stop" event: remaining audio, then _flush_asr_finalize;
  * every translation item the session enqueues is translated and synthesised at once (_tts_loop's body, same calls,
    same failure handling), with write_ms = source ms received at that moment;
  * playback: a segment starts at max(write_ms, end of the previous one) (the paper's queue model).

Output rows follow R1's SeamlessStreaming runner (experiments/s2st_baselines_v1/seamless/run_seamless.py make_record).
"""
from __future__ import annotations

import argparse
import asyncio
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

SR = 16000
FRAME = 512   # frontend ScriptProcessor buffer
LANGS = {"en2zh": ("eng", "cmn", "en", "zh"), "zh2en": ("cmn", "eng", "zh", "en")}
XT_SRC = Path("/opt/dlami/nvme/jasonleeeli/research_sources/baselines_20261112/X-Translator")


class NoopWebSocket:
    async def accept(self):
        return None

    async def send_text(self, _payload):
        return None

    async def send_bytes(self, _payload):
        return None


def import_xtranslate(pkg_root: Path):
    """X-Translator's backend uses package-relative imports (main.py imports it as xtranslate.backend)."""
    pkg_root.mkdir(parents=True, exist_ok=True)
    link = pkg_root / "xtranslate"
    if not link.exists():
        link.symlink_to(XT_SRC)
    assert link.resolve() == XT_SRC.resolve(), link
    sys.path.insert(0, str(pkg_root))
    from xtranslate.backend import session as S  # noqa: E402
    from xtranslate.backend.config import Settings  # noqa: E402
    from xtranslate.backend.services import pcm_bytes_to_float32  # noqa: F401,E402
    return S, Settings


def to_pcm16(x: np.ndarray) -> bytes:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16).tobytes()


class Runner:
    def __init__(self, S, settings, sessions_dir: Path):
        self.S = S
        self.settings = settings
        self.sessions_dir = sessions_dir
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def run(self, coro):
        return self.loop.run_until_complete(coro)

    def translate_item(self, sess, item: dict) -> dict:
        """_tts_loop's body for one item (session.py 1353-1465), synchronous."""
        seg = {"source_text": item.get("source_text", item["text"]), "mt_input": item["text"],
               "speaker_id": item.get("speaker_id", ""), "clone_audio_path": item.get("clone_audio_path") or "",
               "segment_index": item.get("segment_index", 1), "segment_count": item.get("segment_count", 1),
               "source_start_ms": item.get("source_start_ms"), "source_end_ms": item.get("source_end_ms")}
        t0 = time.perf_counter()
        try:
            translated = sess.mt_service.translate(item["text"], sess.state.source_lang, sess.state.target_lang)
        except Exception as exc:  # session: logger.error + continue (segment dropped)
            seg.update(mt_error=repr(exc), translated="", pcm=b"")
            return seg
        seg["translated"] = translated
        seg["mt_s"] = round(time.perf_counter() - t0, 3)
        clone = item.get("clone_audio_path") or None
        if sess.speaker_prompt_tracker is None:   # not the default (speaker_prompt_enabled true)
            clone = self.run(sess.audio_store.export_recent_clone(self.settings.clone_window_seconds))
        t1 = time.perf_counter()
        try:
            pcm = sess.tts_service.synthesize(translated, clone)
        except Exception as exc:  # session: logger.error + continue (no audio for this segment)
            seg.update(tts_error=repr(exc)[:500], pcm=b"")
            return seg
        seg["tts_s"] = round(time.perf_counter() - t1, 3)
        seg["pcm"] = pcm
        return seg

    def session_for(self, item_id: str):
        sess = self.S.TranslationSession(NoopWebSocket(), self.settings)
        sess._speaker_audio_queue = None   # the tracker is fed below, as _speaker_loop does
        tr = sess.speaker_prompt_tracker
        # efficiency only: reuse the loaded CAM++ pipeline instead of reloading it for every sentence's session
        if tr is not None and getattr(self, "_campplus", None) is not None and hasattr(tr.embedding_backend, "_pipeline"):
            tr.embedding_backend._pipeline = self._campplus
        return sess

    def keep_pipeline(self, sess) -> str:
        tr = sess.speaker_prompt_tracker
        if tr is None:
            return "none"
        if getattr(tr.embedding_backend, "_pipeline", None) is not None:
            self._campplus = tr.embedding_backend._pipeline
        return tr.embedding_backend.name

    def process(self, it: dict, wav: np.ndarray) -> dict:
        S, st = self.S, self.settings
        sess = self.session_for(it["id"])
        self.run(sess._handle_text(json.dumps({"action": "session_config", "source_lang": st.source_lang,
                                               "target_lang": st.target_lang, "sample_rate": SR})))
        chunk_bytes = sess._asr_chunk_bytes
        pcm = to_pcm16(wav)
        segs: list[dict] = []
        counters = {"boundary_flushes": 0, "speaker_events": 0, "asr_chunks": 0}
        received = [0]   # bytes of source audio received so far = the simulated clock

        def drain_items():
            while not sess.translation_queue.empty():
                item = sess.translation_queue.get_nowait()
                if not item:
                    continue
                write_ms = received[0] / 2 / SR * 1000
                seg = self.translate_item(sess, item)
                seg["write_ms"] = write_ms
                segs.append(seg)

        def process_bytes(b: bytes):
            counters["asr_chunks"] += 1
            self.run(sess._process_asr_chunk(b))
            drain_items()

        def take(n: int) -> bytes:
            b = bytes(sess._asr_pending_audio[:n])
            del sess._asr_pending_audio[:n]
            return b

        def boundary_flush():
            """_request_asr_boundary_flush + the ASR loop's handling of it (session.py 460-480, 527-535, 558-618)."""
            pending = len(sess._asr_pending_audio)
            if sess._current_audio_bytes <= 0 and pending <= 0:
                sess._reset_asr_turn()
                return
            counters["boundary_flushes"] += 1
            sess._asr_boundary_flush_requested = True
            sess._asr_boundary_flush_bytes = pending
            while sess._asr_pending_audio and sess._asr_boundary_flush_bytes > 0:
                n = min(len(sess._asr_pending_audio), chunk_bytes, sess._asr_boundary_flush_bytes)
                sess._asr_boundary_flush_bytes = max(0, sess._asr_boundary_flush_bytes - n)
                process_bytes(take(n))
            self.run(sess._flush_asr_finalize())
            sess._asr_boundary_flush_requested = False
            sess._asr_boundary_flush_bytes = None
            drain_items()

        for off in range(0, len(pcm), FRAME * 2):
            frame = pcm[off: off + FRAME * 2]
            received[0] += len(frame)
            self.run(sess._handle_audio(frame))
            if sess.speaker_prompt_tracker is not None:
                events = self.run(sess.speaker_prompt_tracker.feed(frame))
                for ev in events:
                    counters["speaker_events"] += 1
                    sess._forced_prompt_speaker_id = ev.previous_speaker_id   # _handle_speaker_change_event
                    boundary_flush()
            while len(sess._asr_pending_audio) >= chunk_bytes:
                process_bytes(take(chunk_bytes))
        # the browser's stop event, then the ASR loop's flush (session.py 398-403, 516-541)
        self.run(sess._handle_text(json.dumps({"event": "stop"})))
        while sess._asr_pending_audio:
            process_bytes(take(min(len(sess._asr_pending_audio), chunk_bytes)))
        self.run(sess._flush_asr_finalize())
        drain_items()
        sess._closed = True
        counters["embedding_backend"] = self.keep_pipeline(sess)
        return {"segs": segs, "counters": counters, "session_id": sess.state.session_id}


def make_record(it: dict, wav: np.ndarray, res: dict, err, out: Path, src_lang: str, tgt: str, tts_sr: int,
                cfg: dict, elapsed: float) -> dict:
    name = it["id"].replace("/", "__")

    def fresh(p: Path) -> Path:   # never overwrite (a crash between wav and row writes leaves an orphan wav)
        k = 0
        q = p
        while q.exists():
            k += 1
            q = p.with_name(f"{p.stem}-r{k}{p.suffix}")
        return q

    src_path = fresh(out / "src" / f"{name}.wav")
    sf.write(src_path, wav, SR)
    X = len(wav) // 16
    blocks, spoken, unspoken, end_prev = [], [], [], 0.0
    joiner = "" if tgt == "cmn" else " "
    texts_all = []
    for s in res["segs"]:
        txt = (s.get("translated") or "").strip()
        if txt:
            texts_all.append(txt)
        pcm = s.get("pcm") or b""
        if not txt or not pcm:
            if txt:
                unspoken.append(txt)
            continue
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        audio16 = soxr.resample(audio, tts_sr, SR).astype(np.float32) if tts_sr != SR else audio
        dur = len(audio16) / 16
        start = max(s["write_ms"], end_prev)
        end_prev = start + dur
        text = (joiner if blocks and joiner else "") + txt
        blocks.append({"write_ms": int(round(s["write_ms"])), "start_ms": int(round(start)), "dur_ms": int(round(dur)),
                       "text": text})
        spoken.append(audio16)
    speech = np.concatenate(spoken) if spoken else np.zeros(320, dtype=np.float32)
    total = int(max(X, end_prev)) + 300
    right = np.zeros(total * 16, dtype=np.float32)
    for b, a in zip(blocks, spoken):
        a0 = b["start_ms"] * 16
        right[a0: a0 + len(a)] += a[: len(right) - a0]
    sim_path = fresh(out / "wav" / f"{name}_simul.wav")
    speech_path = fresh(out / "wav" / f"{name}_speech.wav")
    sf.write(sim_path, right, SR)
    sf.write(speech_path, speech, SR)
    seg_meta = [{k: v for k, v in s.items() if k != "pcm"} | {"tts_samples": len(s.get("pcm") or b"") // 2}
                for s in res["segs"]]
    return {"id": it["id"], "mode": "si", "name": it["id"], "src_lang": src_lang, "tgt_lang": tgt,
            "transcription_ref": it["src_text"], "translation_ref": it["trg_text"],
            "translated_text": joiner.join(texts_all).strip(), "audio_path": str(sim_path),
            "speech_path": str(speech_path), "audio_duration_seconds": round(len(right) / SR, 3),
            "source_audio_path": str(src_path), "source_duration_ms": X, "error": err,
            "start_offset_ms": blocks[0]["start_ms"] if blocks else None,
            "end_offset_ms": int(end_prev - X) if blocks else None,
            "blocks": blocks, "unspoken_text": joiner.join(unspoken),
            "n_segments": len(res["segs"]), "n_tts_fail": sum(1 for s in res["segs"] if s.get("tts_error")),
            "n_mt_fail": sum(1 for s in res["segs"] if s.get("mt_error")),
            "asr_source_text": "".join(s["source_text"] for s in res["segs"]) if tgt == "eng" else
            " ".join(s["source_text"] for s in res["segs"]),
            "counters": res["counters"], "segments": seg_meta, "elapsed_s": round(elapsed, 3), "config": cfg}


def load_wav(it: dict, cache: dict) -> np.ndarray:
    if it["wav"] not in cache:
        x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
        assert sr == SR, it["wav"]
        cache.clear()
        cache[it["wav"]] = x.mean(axis=1)
    return cache[it["wav"]][it["start"] * 16: it["end"] * 16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction", choices=sorted(LANGS), required=True)
    ap.add_argument("--items", required=True, help="items_<dir>.jsonl from si_infer.realsi_items()")
    ap.add_argument("--config", required=True, help="the point's X-Translator config.json (written by make_configs.py)")
    ap.add_argument("--point", required=True)
    ap.add_argument("--out", required=True, help="the run's realsi_<dir> folder")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated ids (smoke tests)")
    a = ap.parse_args()

    out = Path(a.out)
    for d in ("src", "wav", "sessions", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    os.environ["XTRANSLATE_CONFIG"] = str(Path(a.config).resolve())
    os.environ["XTRANSLATE_LOG_DIR"] = str((out / "logs" / f"xtranslate_shard{a.shard}").resolve())
    os.environ["XTRANSLATE_SESSION_AUDIO_DIR"] = str((out / "sessions" / f"shard{a.shard:02d}").resolve())
    S, Settings = import_xtranslate(out.parent.parent / "_pkg")
    xt_logger = logging.getLogger("xtranslate")
    for h in xt_logger.handlers:   # keep the full INFO log in the file, only warnings on stderr
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.WARNING)
    settings = Settings.from_env()
    src_lang, tgt, xs, xt = LANGS[a.direction]
    assert (settings.source_lang, settings.target_lang) == (xs, xt), (settings.source_lang, settings.target_lang)
    assert str(settings.session_audio_dir).startswith(str(out.resolve())), settings.session_audio_dir
    cfg = {"system": "X-Translator (Qwen3-ASR-1.7B + LMT-60-8B + X-Voice 0.4B Stage2)", "point": a.point,
           "direction": a.direction, "code": "X-Translator dc6a59a, X-Voice 817bcdd, qwen-asr 0.0.6, vllm 0.14.0",
           "config_json": json.load(open(a.config, encoding="utf-8")),
           "asr_request_chunk_samples": settings.paraformer_chunk_size,
           "asr_server_chunk_size_sec": float(os.environ.get("XT_ASR_CHUNK_SIZE_SEC", "0.6")),
           "registration": "paper/baselines_run_plan_xtranslator_v1.md"}

    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    if a.ids:
        want = set(a.ids.split(","))
        items = [x for x in items if x["id"] in want]
    items = items[a.shard::a.num_shards]
    if a.limit:
        items = items[: a.limit]
    part = out / f"part_{a.shard:02d}.jsonl"
    done = set()
    if part.exists():
        done = {json.loads(l)["id"] for l in open(part, encoding="utf-8")}
    runner = Runner(S, settings, out / "sessions")
    cache: dict = {}
    print(f"[xt] point={a.point} dir={a.direction} shard={a.shard}/{a.num_shards} items={len(items)} done={len(done)}"
          f" asr={settings.qwen3_asr_url} mt={settings.lmt_url} tts={settings.xvoice_tts_url}", flush=True)
    with open(part, "a", encoding="utf-8") as fh:
        for k, it in enumerate(items):
            if it["id"] in done:
                continue
            wav = load_wav(it, cache)
            t0 = time.perf_counter()
            err = None
            try:
                res = runner.process(it, wav)
            except Exception:
                err = traceback.format_exc()[-2000:]
                res = {"segs": [], "counters": {}, "session_id": ""}
                print(f"[xt] ERROR {it['id']}\n{err}", flush=True)
            rec = make_record(it, wav, res, err, out, src_lang, tgt, settings.tts_sample_rate, cfg,
                              time.perf_counter() - t0)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[xt] {k + 1}/{len(items)} {it['id']} X={rec['source_duration_ms']} segs={rec['n_segments']} "
                  f"blocks={len(rec['blocks'])} first={rec['start_offset_ms']} tts_fail={rec['n_tts_fail']} "
                  f"t={rec['elapsed_s']:.1f}s err={bool(err)}", flush=True)
    print("[xt] SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
