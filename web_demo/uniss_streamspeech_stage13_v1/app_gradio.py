"""No-login public Gradio UI for the Stage09-12 research pipeline."""

from __future__ import annotations

import argparse
import html
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Iterator

import gradio as gr
import numpy as np
import soundfile as sf

from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.config import Stage09Config
from experiments.uniss_streamspeech_ctc_v1.stage11_streaming_audio.config import Stage11Config
from experiments.uniss_streamspeech_ctc_v1.stage11_streaming_audio.engine import (
    Stage11Engine,
    Stage11Event,
    Stage11Result,
    Stage11Update,
)
from web_demo.streaming_s2st_r2_v1.audio_io import (
    SAMPLE_RATE,
    normalize_uploaded_audio,
    resample_mono,
)

from .config import Stage13Config
from .registry import BrowserState, Registry


DIRECTIONS = {"英文 → 中文": "eng->cmn", "中文 → 英文": "cmn->eng"}
CONFIG: Stage13Config | None = None
ENGINE: Stage11Engine | None = None
REGISTRY = Registry()
GPU_LOCK = threading.Lock()


def event_dict(event: Stage11Event) -> dict[str, object]:
    return event.__dict__.copy()


def timeline_html(trace: list[dict[str, object]] | None) -> str:
    values = trace or []
    if not values:
        return "<div class='empty'>等待 Stage09 CTC WAIT/WRITE 事件。</div>"
    rows = []
    for value in values[-80:]:
        action = str(value.get("policy_action", "WAIT"))
        rejected = html.escape(str(value.get("semantic_rejected_reason") or ""))
        qwen = html.escape(str(value.get("qwen_text_delta") or ""))
        rows.append(
            "<tr>"
            f"<td>{int(value.get('index', 0))}</td>"
            f"<td>{float(value.get('source_end_ms', 0)):.0f} ms</td>"
            f"<td class='{action.lower()}'>{action}</td>"
            f"<td>{qwen}</td><td>{int(value.get('audio_samples', 0)) / SAMPLE_RATE:.2f}s</td>"
            f"<td>{rejected}</td></tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr><th>#</th><th>Source</th>"
        "<th>Policy</th><th>Qwen delta</th><th>Audio</th><th>Reject</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def result_status(result: Stage11Result) -> str:
    fallback = (
        f"  \n⚠️ **Final offline safety fallback used**：`{result.fallback_reason}`"
        if result.fallback_used
        else ""
    )
    return (
        f"**完成 · Research-only**  \n"
        f"First WRITE：`{result.first_write_ms} ms` · "
        f"First audio NCA/CA：`{result.first_audio_nca_ms} / {result.first_audio_ca_ms} ms`  \n"
        f"在线有效/拒绝 WRITE：`{result.valid_audio_writes}/{result.rejected_writes}` · "
        f"处理 RTF：`{result.wall_seconds / max(result.source_seconds, 1e-6):.2f}`"
        f"{fallback}"
    )


def _engine() -> Stage11Engine:
    if ENGINE is None:
        raise RuntimeError("Stage13 engine has not been initialized")
    return ENGINE


def _new_session(direction_label: str, request_dir: Path):
    if CONFIG is None:
        raise RuntimeError("Stage13 config has not been initialized")
    return _engine().new_session(
        direction=DIRECTIONS[direction_label],
        speaker_tokens=CONFIG.fixed_speaker_tokens(),
        request_dir=request_dir,
    )


def stream_upload(
    audio_path: str | None,
    direction: str,
    trace: list[dict[str, object]] | None,
) -> Iterator[tuple[object, ...]]:
    if not audio_path:
        raise gr.Error("请先上传音频")
    assert CONFIG is not None
    current = list(trace or [])
    request_dir = CONFIG.output_root / "upload" / uuid.uuid4().hex
    normalized = request_dir.parent / f".{request_dir.name}.input.wav"
    metadata = normalize_uploaded_audio(
        audio_path,
        normalized,
        max_upload_bytes=CONFIG.max_upload_bytes,
        min_audio_seconds=CONFIG.min_audio_seconds,
        max_audio_seconds=CONFIG.max_audio_seconds,
    )
    waveform, _ = sf.read(normalized, dtype="float32")
    normalized.unlink(missing_ok=True)
    with GPU_LOCK:
        session = _new_session(direction, request_dir)
        chunk_samples = CONFIG.chunk_ms * 16
        try:
            for start in range(0, len(waveform), chunk_samples):
                end = min(len(waveform), start + chunk_samples)
                for update in session.push(waveform[start:end], final=end == len(waveform)):
                    if update.event is not None:
                        current.append(event_dict(update.event))
                    if update.result is None:
                        yield (
                            session.runtime.source_transcription,
                            session.runtime.committed_translation,
                            update.translation,
                            (SAMPLE_RATE, update.audio_chunk) if update.audio_chunk.size else None,
                            None, None, None, None, None,
                            update.status,
                            timeline_html(current),
                            current,
                        )
                    else:
                        result = update.result
                        yield (
                            result.transcription,
                            result.ctc_translation,
                            result.translation,
                            None,
                            result.timeline_audio_path,
                            result.stereo_audio_path,
                            result.translation_audio_path,
                            result.result_json_path,
                            result.stereo_audio_path,
                            result_status(result),
                            timeline_html(current),
                            current,
                        )
        except Exception as exc:
            raise gr.Error(f"Stage13上传推理失败：{type(exc).__name__}: {exc}") from exc


def _micro_outputs(state: BrowserState, update: Stage11Update, trace):
    result = update.result
    return (
        (SAMPLE_RATE, update.audio_chunk) if update.audio_chunk.size else None,
        state.session.runtime.source_transcription if result is None else result.transcription,
        state.session.runtime.committed_translation if result is None else result.ctc_translation,
        update.translation if result is None else result.translation,
        update.status if result is None else result_status(result),
        timeline_html(trace),
        state.session_id,
        trace,
        result.timeline_audio_path if result else None,
        result.stereo_audio_path if result else None,
        result.translation_audio_path if result else None,
        result.result_json_path if result else None,
    )


def microphone_step(audio_chunk, direction, session_id, trace):
    assert CONFIG is not None
    current = list(trace or [])
    with GPU_LOCK:
        if session_id:
            state = REGISTRY.get(session_id)
            if state.direction != direction:
                raise gr.Error("录音会话期间不能切换翻译方向")
        else:
            request = CONFIG.output_root / "microphone" / uuid.uuid4().hex
            state = REGISTRY.create(direction, _new_session(direction, request))
        if audio_chunk is None:
            return
        rate, audio = audio_chunk
        values = resample_mono(audio, int(rate))
        state.samples += len(values)
        if state.samples / SAMPLE_RATE > CONFIG.max_microphone_seconds:
            raise gr.Error("麦克风录音超过时长限制")
        for update in state.session.push(values, final=False):
            if update.event is not None:
                current.append(event_dict(update.event))
            yield _micro_outputs(state, update, current)


def microphone_finalize(direction, session_id, trace):
    if not session_id:
        raise gr.Error("没有可结束的麦克风会话")
    current = list(trace or [])
    with GPU_LOCK:
        state = REGISTRY.get(session_id)
        for update in state.session.push([], final=True):
            if update.event is not None:
                current.append(event_dict(update.event))
            yield _micro_outputs(state, update, current)
        REGISTRY.discard(session_id)


def reset_microphone(session_id):
    REGISTRY.discard(session_id)
    return (None, "", "", "", "等待录音。", timeline_html([]), None, [], None, None, None, None)


def build_demo(config: Stage13Config, engine: Stage11Engine) -> gr.Blocks:
    global CONFIG, ENGINE
    CONFIG, ENGINE = config, engine
    css = """
    .warning{border:1px solid #f59e0b;background:#fffbeb;padding:12px;border-radius:12px}
    .model{border:1px solid #60a5fa;background:#eff6ff;padding:12px;border-radius:12px}
    .scroll{max-height:320px;overflow:auto}.scroll table{width:100%;border-collapse:collapse}
    .scroll th,.scroll td{padding:6px;border-bottom:1px solid #ddd}.write{color:#15803d;font-weight:700}.wait{color:#a16207}
    .empty{padding:12px;color:#64748b;border:1px dashed #cbd5e1}
    """
    title = "UniSS-Stream Stage13 Simultaneous S2ST Research Demo"
    with gr.Blocks(title=title, css=css) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(
            f"**模型链**：`{config.model_label}`  \n固定目标音色 · 160 ms source chunk · 无需登录",
            elem_classes=["model"],
        )
        gr.Markdown(
            "⚠️ 这是未过质量门的研究体验版。EN→ZH 实测首音频 NCA 880 ms、CA 5.16 s；"
            "ZH→EN 当前通常在 final 使用离线安全回退。页面会显示所有拒绝和 fallback。",
            elem_classes=["warning"],
        )
        with gr.Tab("上传音频"):
            upload_trace = gr.State([])
            with gr.Row():
                with gr.Column():
                    source = gr.Audio(label="源音频", sources=["upload"], type="filepath")
                    direction = gr.Radio(list(DIRECTIONS), value="英文 → 中文", label="方向")
                    run = gr.Button("开始160 ms同传回放", variant="primary")
                    status = gr.Markdown("等待上传。")
                with gr.Column():
                    asr = gr.Textbox(label="Causal CTC ASR", lines=2)
                    ctc_translation = gr.Textbox(label="CTC policy target", lines=2)
                    qwen_translation = gr.Textbox(label="Qwen增量翻译", lines=3)
                    live_audio = gr.Audio(label="当前目标音频chunk", streaming=True, autoplay=True)
            timeline_audio = gr.Audio(label="WAIT对齐目标时间线", type="filepath")
            stereo_player = gr.Audio(label="双声道：左源语音 / 右翻译语音", type="filepath")
            timeline = gr.HTML(timeline_html([]))
            with gr.Row():
                target_file = gr.File(label="连续目标WAV")
                json_file = gr.File(label="事件JSON")
                stereo_file = gr.File(label="立体声WAV")
            run.click(
                stream_upload,
                inputs=[source, direction, upload_trace],
                outputs=[asr, ctc_translation, qwen_translation, live_audio, timeline_audio, stereo_player, target_file, json_file, stereo_file, status, timeline, upload_trace],
                concurrency_limit=1,
                api_name="stage13_stream_upload",
            )
        with gr.Tab("麦克风边录边传"):
            mic_id = gr.State(None)
            mic_trace = gr.State([])
            mic = gr.Audio(label="麦克风流", sources=["microphone"], streaming=True, type="numpy")
            mic_direction = gr.Radio(list(DIRECTIONS), value="英文 → 中文", label="方向")
            with gr.Row():
                finalize = gr.Button("停止并final flush")
                reset = gr.Button("清空会话")
            mic_live = gr.Audio(label="实时翻译音频", streaming=True, autoplay=True)
            mic_asr = gr.Textbox(label="Causal CTC ASR")
            mic_ctc = gr.Textbox(label="CTC policy target")
            mic_qwen = gr.Textbox(label="Qwen增量翻译")
            mic_status = gr.Markdown("等待录音。")
            mic_timeline = gr.HTML(timeline_html([]))
            mic_timeline_audio = gr.Audio(label="完成后的WAIT时间线", type="filepath")
            mic_stereo = gr.Audio(label="完成后的左源右译立体声", type="filepath")
            mic_target = gr.File(label="目标WAV")
            mic_json = gr.File(label="事件JSON")
            mic_outputs = [mic_live, mic_asr, mic_ctc, mic_qwen, mic_status, mic_timeline, mic_id, mic_trace, mic_timeline_audio, mic_stereo, mic_target, mic_json]
            mic.stream(
                microphone_step,
                inputs=[mic, mic_direction, mic_id, mic_trace],
                outputs=mic_outputs,
                stream_every=config.chunk_ms / 1000.0,
                time_limit=config.max_microphone_seconds,
                concurrency_limit=1,
            )
            mic.stop_recording(
                microphone_finalize,
                inputs=[mic_direction, mic_id, mic_trace],
                outputs=mic_outputs,
                concurrency_limit=1,
            )
            finalize.click(microphone_finalize, inputs=[mic_direction, mic_id, mic_trace], outputs=mic_outputs, concurrency_limit=1)
            reset.click(reset_microphone, inputs=[mic_id], outputs=mic_outputs)
    return demo


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def launch(argv=None):
    args = parse_args(argv)
    config = Stage13Config.from_env()
    config.validate()
    stage09 = Stage09Config(device=config.device)
    stage11 = Stage11Config(output_root=config.output_root)
    engine = Stage11Engine(stage09, stage11)
    engine.load()
    demo = build_demo(config, engine).queue(default_concurrency_limit=1, max_size=config.queue_size)
    launched = demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        prevent_thread_lock=True,
        allowed_paths=[str(config.output_root.resolve())],
        blocked_paths=[str(config.repo_root / name) for name in ("checkpoints", "data", "pretrained_models")],
        show_api=False,
        quiet=False,
    )
    local_url = str(launched[1])
    public_url = str(launched[2]) if launched[2] else None
    config.demo_root.joinpath("public_url.txt").write_text((public_url or "") + "\n", encoding="utf-8")
    config.demo_root.joinpath("access_info.json").write_text(
        json.dumps(
            {
                "local_url": local_url,
                "public_url": public_url,
                "auth_mode": "public_no_login",
                "model": config.model_label,
                "warning": "research-only; quality and subsecond CA gates failed",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"LOCAL_URL={local_url}", flush=True)
    print(f"PUBLIC_URL={public_url or ''}", flush=True)
    if args.share and not public_url:
        raise RuntimeError("Gradio share requested but no public URL was returned")
    demo.block_thread()
    return local_url, public_url


if __name__ == "__main__":
    launch()
