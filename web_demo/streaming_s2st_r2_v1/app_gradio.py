"""No-login Gradio UI for the isolated R2 pseudo-streaming S2ST engine."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Iterator

import gradio as gr
import numpy as np

from .audio_io import write_json
from .config import StreamingDemoConfig
from .engine import StreamingDemoEngine, StreamingResult, StreamingUpdate
from .session_manager import SessionRegistry

ENGINE: StreamingDemoEngine | None = None
REGISTRY: SessionRegistry | None = None


def event_timeline_html(updates: list[dict[str, object]] | None) -> str:
    events = updates or []
    if not events:
        return "<div class='timeline-empty'>等待 WAIT/WRITE 事件。</div>"
    rows = []
    for event in events[-40:]:
        action = str(event.get("action", "")).upper()
        css_class = "write" if action == "WRITE" else "wait"
        text = html.escape(str(event.get("generated_text", "")))
        forced = html.escape(
            str(
                event.get("quality_rejected_reason")
                or event.get("forced_reason")
                or ""
            )
        )
        rows.append(
            "<tr>"
            f"<td>{int(event.get('event_index', 0)) + 1}</td>"
            f"<td>{float(event.get('source_end_ms', 0.0)):.0f} ms</td>"
            f"<td><span class='action-chip {css_class}'>{action}</span></td>"
            f"<td>{text}</td><td>{forced}</td>"
            "</tr>"
        )
    return (
        "<div class='timeline-scroll'><table class='timeline-table'>"
        "<thead><tr><th>#</th><th>Source</th><th>Action</th>"
        "<th>Text delta</th><th>Recovery</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _append_event(
    trace: list[dict[str, object]] | None, update: StreamingUpdate
) -> list[dict[str, object]]:
    values = list(trace or [])
    if update.event is not None:
        values.append(update.event.__dict__.copy())
    return values


def _audio_value(update: StreamingUpdate):
    if update.audio_chunk.size:
        return 16_000, np.asarray(update.audio_chunk, dtype=np.float32)
    return None


def format_final_status(result: StreamingResult) -> str:
    fallback = (
        f"  \n音频安全回退：`Phase3 full198 Quality` · 原因：`{result.fallback_reason}`"
        if result.fallback_used
        else ""
    )
    return (
        f"**完成** · `{result.model_label}`  \n"
        f"模式：`{result.mode}`  \n"
        f"源音频：{result.source_duration_seconds:.2f}s · "
        f"目标音频：{result.translation_duration_seconds:.2f}s · "
        f"服务器处理：{result.total_seconds:.2f}s  \n"
        f"First WRITE：{result.first_write_ms if result.first_write_ms is not None else 'N/A'} ms · "
        f"First audio：{result.first_audio_ms if result.first_audio_ms is not None else 'N/A'} ms · "
        f"forced={result.forced_actions} · recovery={result.structural_recoveries}"
        f"{fallback}"
    )


def stream_upload_request(
    audio_path: str | None,
    direction: str,
    trace: list[dict[str, object]] | None,
) -> Iterator[tuple[object, ...]]:
    if not audio_path:
        raise gr.Error("请先上传音频")
    if ENGINE is None:
        raise gr.Error("Streaming engine 尚未初始化")
    current_trace = list(trace or [])
    try:
        for update in ENGINE.stream_upload(audio_path, direction=direction):
            current_trace = _append_event(current_trace, update)
            if update.result is None:
                yield (
                    update.translation,
                    _audio_value(update),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    update.status,
                    event_timeline_html(current_trace),
                    current_trace,
                )
                continue
            result = update.result
            yield (
                result.translation,
                None,
                result.timeline_audio_path,
                result.aligned_stereo_path,
                result.translation_audio_path,
                result.aligned_stereo_path,
                result.result_json_path,
                result.translation_audio_path,
                format_final_status(result),
                event_timeline_html(current_trace),
                current_trace,
            )
    except Exception as exc:
        if "CUDA out of memory" in str(exc):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        raise gr.Error(f"R2 streaming 推理失败：{type(exc).__name__}: {exc}") from exc


def _registry_session(session_id: str | None):
    if REGISTRY is None:
        raise gr.Error("Session registry 尚未初始化")
    if session_id:
        return REGISTRY.get(session_id)
    return REGISTRY.create()


def microphone_step(
    audio_chunk: tuple[int, np.ndarray] | None,
    direction: str,
    session_id: str | None,
    trace: list[dict[str, object]] | None,
) -> Iterator[tuple[object, ...]]:
    if ENGINE is None:
        raise gr.Error("Streaming engine 尚未初始化")
    session = _registry_session(session_id)
    current_trace = list(trace or [])
    try:
        for update in ENGINE.process_microphone(
            session,
            audio_chunk,
            direction=direction,
            is_final=False,
        ):
            current_trace = _append_event(current_trace, update)
            frontend = update.frontend or {}
            frontend_text = (
                f" · candidate={frontend.get('candidate_tokens', 0)}"
                f" · committed={frontend.get('committed_tokens', 0)}"
                f" · revisions={frontend.get('revision_events', 0)}"
                if frontend
                else ""
            )
            yield (
                _audio_value(update),
                update.translation,
                f"{update.status}{frontend_text}",
                event_timeline_html(current_trace),
                session.session_id,
                current_trace,
                None,
                None,
                None,
                None,
                None,
            )
    except Exception as exc:
        raise gr.Error(f"麦克风流式处理失败：{type(exc).__name__}: {exc}") from exc


def microphone_finalize(
    direction: str,
    session_id: str | None,
    trace: list[dict[str, object]] | None,
) -> Iterator[tuple[object, ...]]:
    if ENGINE is None:
        raise gr.Error("Streaming engine 尚未初始化")
    if not session_id:
        raise gr.Error("没有可结束的麦克风会话")
    session = _registry_session(session_id)
    current_trace = list(trace or [])
    try:
        for update in ENGINE.process_microphone(
            session,
            None,
            direction=direction,
            is_final=True,
        ):
            current_trace = _append_event(current_trace, update)
            result = update.result
            yield (
                _audio_value(update),
                update.translation,
                format_final_status(result) if result else update.status,
                event_timeline_html(current_trace),
                session.session_id,
                current_trace,
                result.timeline_audio_path if result else None,
                result.aligned_stereo_path if result else None,
                result.translation_audio_path if result else None,
                result.aligned_stereo_path if result else None,
                result.result_json_path if result else None,
            )
    except Exception as exc:
        raise gr.Error(f"麦克风 final flush 失败：{type(exc).__name__}: {exc}") from exc


def reset_microphone(session_id: str | None):
    if session_id and REGISTRY is not None:
        try:
            REGISTRY.get(session_id).cancel()
        except KeyError:
            pass
        REGISTRY.discard(session_id)
    return (
        None,
        None,
        "等待点击录音。",
        event_timeline_html([]),
        None,
        [],
        None,
        None,
        None,
        None,
        None,
    )


def build_demo(
    config: StreamingDemoConfig,
    engine: StreamingDemoEngine,
    registry: SessionRegistry,
) -> gr.Blocks:
    global ENGINE, REGISTRY
    ENGINE = engine
    REGISTRY = registry
    css = """
    .model-card {border:1px solid #bfdbfe;border-radius:14px;padding:14px;background:#eff6ff}
    .boundary-card {border:1px solid #fed7aa;border-radius:14px;padding:12px;background:#fff7ed}
    .timeline-scroll {max-height:300px;overflow:auto;border:1px solid #e2e8f0;border-radius:10px}
    .timeline-table {width:100%;border-collapse:collapse;font-size:.85rem}
    .timeline-table th,.timeline-table td {padding:7px;border-bottom:1px solid #e2e8f0;text-align:left}
    .action-chip {display:inline-block;padding:2px 7px;border-radius:999px;font-weight:700}
    .action-chip.wait {background:#fef3c7;color:#92400e}.action-chip.write {background:#dcfce7;color:#166534}
    .timeline-empty {color:#64748b;padding:12px;border:1px dashed #cbd5e1;border-radius:10px}
    """
    title = "UniSS R2 Streaming Speech-to-Speech from jasonleeeli(李琎) Intern"
    sync_js = """() => {
      const left = document.querySelector('#upload-source audio');
      const right = document.querySelector('#upload-target audio');
      if (!left || !right) { return []; }
      left.pause(); right.pause(); left.currentTime = 0; right.currentTime = 0;
      Promise.allSettled([left.play(), right.play()]);
      return [];
    }"""
    with gr.Blocks(title=title, css=css) as demo:
        gr.Markdown(f"# {title}\n上传音频回放或使用麦克风体验当前最佳 R2 同传策略。")
        gr.Markdown(
            f"**固定主模型**：`{config.model_label}`　"
            "**目标生成**：R2 WAIT/WRITE + Streaming BiCodec　"
            "**公网**：公开、无需注册或登录",
            elem_classes=["model-card"],
        )
        gr.Markdown(
            "⚠️ 当前源前端是 pseudo-streaming：上传默认先完整提取 source token；"
            "麦克风使用 WhisperVQ 累计前缀重编码，不是 causal encoder。",
            elem_classes=["boundary-card"],
        )
        with gr.Tab("上传音频 / Upload replay"):
            upload_trace = gr.State([])
            with gr.Row():
                with gr.Column():
                    upload_source = gr.Audio(
                        label="源音频 / Source",
                        sources=["upload"],
                        type="filepath",
                        format=None,
                        elem_id="upload-source",
                    )
                    upload_direction = gr.Radio(
                        ["中文 → 英文", "英文 → 中文"],
                        value="中文 → 英文",
                        label="翻译方向",
                    )
                    upload_button = gr.Button("开始 R2 同传回放", variant="primary")
                    upload_status = gr.Markdown("等待上传音频。")
                with gr.Column():
                    upload_translation = gr.Textbox(
                        label="增量翻译文本", lines=4, interactive=False
                    )
                    upload_live_audio = gr.Audio(
                        label="生成中的目标音频 chunk",
                        streaming=True,
                        autoplay=True,
                        interactive=False,
                    )
                    upload_target = gr.Audio(
                        label="同传时间线 / Translation with WAIT silence",
                        type="filepath",
                        autoplay=False,
                        interactive=False,
                        elem_id="upload-target",
                    )
                    sync_button = gr.Button("从 0 秒同步播放左右音频")
            gr.Markdown(
                "### 🎧 双声道延迟对比\n"
                "建议佩戴耳机：左声道播放源语言；右声道按照实际 WAIT/WRITE 时间线播放翻译语言。"
                "开头和 WRITE 之间的静音就是模型等待延迟。手机或单声道扬声器可能自动混音。"
            )
            upload_stereo_player = gr.Audio(
                label="同步双声道播放（左=源语言，右=翻译语言）",
                type="filepath",
                autoplay=False,
                interactive=False,
                show_download_button=True,
            )
            upload_timeline = gr.HTML(event_timeline_html([]))
            with gr.Row():
                upload_raw_audio = gr.File(label="连续目标语音 WAV")
                upload_stereo = gr.File(label="同步双声道 WAV（左源/右译）")
                upload_json = gr.File(label="事件与指标 JSON")
                upload_download = gr.File(label="目标语音下载")
            upload_button.click(
                stream_upload_request,
                inputs=[upload_source, upload_direction, upload_trace],
                outputs=[
                    upload_translation,
                    upload_live_audio,
                    upload_target,
                    upload_stereo_player,
                    upload_raw_audio,
                    upload_stereo,
                    upload_json,
                    upload_download,
                    upload_status,
                    upload_timeline,
                    upload_trace,
                ],
                concurrency_limit=1,
                api_name="stream_upload_r2",
            )
            sync_button.click(fn=None, js=sync_js)

        with gr.Tab("麦克风 / Online prefix pseudo-streaming"):
            mic_session_id = gr.State(None)
            mic_trace = gr.State([])
            gr.Markdown(
                "点击录音即会解锁浏览器音频播放。建议佩戴耳机；首个稳定 source token "
                "通常约 4 秒，首段翻译语音可能需要 6–10 秒。"
            )
            with gr.Row():
                with gr.Column():
                    microphone = gr.Audio(
                        label="边录边传 / Streaming microphone",
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                    )
                    mic_direction = gr.Radio(
                        ["中文 → 英文", "英文 → 中文"],
                        value="中文 → 英文",
                        label="翻译方向（录音期间不能更改）",
                    )
                    with gr.Row():
                        mic_finalize_button = gr.Button("停止后手动 final flush")
                        mic_reset_button = gr.Button("清空会话")
                    mic_status = gr.Markdown("等待点击录音。")
                with gr.Column():
                    mic_translation = gr.Textbox(
                        label="增量翻译文本", lines=4, interactive=False
                    )
                    mic_stream_audio = gr.Audio(
                        label="实时目标语音 / Streaming translation",
                        streaming=True,
                        autoplay=True,
                        interactive=False,
                    )
                    mic_timeline_audio = gr.Audio(
                        label="完成后的同传时间线", type="filepath", interactive=False
                    )
                    mic_stereo_player = gr.Audio(
                        label="完成后的双声道延迟对比（左=源语言，右=翻译语言）",
                        type="filepath",
                        autoplay=False,
                        interactive=False,
                        show_download_button=True,
                    )
                    gr.Markdown(
                        "佩戴耳机播放双声道结果，可以直接听出从左耳源语音到右耳翻译语音的延迟。"
                    )
            mic_timeline = gr.HTML(event_timeline_html([]))
            with gr.Row():
                mic_raw_audio = gr.File(label="连续目标语音 WAV")
                mic_stereo = gr.File(label="同步双声道 WAV")
                mic_json = gr.File(label="事件与指标 JSON")
            mic_outputs = [
                mic_stream_audio,
                mic_translation,
                mic_status,
                mic_timeline,
                mic_session_id,
                mic_trace,
                mic_timeline_audio,
                mic_stereo_player,
                mic_raw_audio,
                mic_stereo,
                mic_json,
            ]
            microphone.stream(
                microphone_step,
                inputs=[microphone, mic_direction, mic_session_id, mic_trace],
                outputs=mic_outputs,
                stream_every=config.chunk_ms / 1000.0,
                time_limit=config.microphone_max_audio_seconds,
                concurrency_limit=1,
            )
            microphone.stop_recording(
                microphone_finalize,
                inputs=[mic_direction, mic_session_id, mic_trace],
                outputs=mic_outputs,
                concurrency_limit=1,
            )
            mic_finalize_button.click(
                microphone_finalize,
                inputs=[mic_direction, mic_session_id, mic_trace],
                outputs=mic_outputs,
                concurrency_limit=1,
            )
            mic_reset_button.click(
                reset_microphone,
                inputs=[mic_session_id],
                outputs=mic_outputs,
                concurrency_limit=1,
            )
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--public-url-file", default=str(StreamingDemoConfig().demo_root / "public_url.txt")
    )
    parser.add_argument(
        "--access-info-file", default=str(StreamingDemoConfig().demo_root / "access_info.json")
    )
    return parser.parse_args(argv)


def write_access_files(
    *,
    public_url_file: str | Path,
    access_info_file: str | Path,
    local_url: str,
    public_url: str | None,
    config: StreamingDemoConfig,
) -> None:
    public_path = Path(public_url_file)
    temporary = public_path.with_suffix(public_path.suffix + ".tmp")
    temporary.write_text((public_url or "") + "\n", encoding="utf-8")
    os.replace(temporary, public_path)
    access_path = Path(access_info_file)
    write_json(
        access_path,
        {
            "local_url": local_url,
            "public_url": public_url,
            "auth_mode": "public_no_login",
            "username": None,
            "password": None,
            "model": config.model_label,
            "mode": "R2 pseudo-streaming upload + microphone prefix",
        },
    )
    os.chmod(access_path, 0o600)


def launch(argv: list[str] | None = None) -> tuple[str, str | None]:
    args = parse_args(argv)
    config = StreamingDemoConfig.from_env()
    config.validate()
    engine = StreamingDemoEngine(config)
    engine.load()
    registry = SessionRegistry(
        config.output_root,
        config.microphone_max_audio_seconds,
        limit=16,
    )
    demo = build_demo(config, engine, registry).queue(
        default_concurrency_limit=1,
        max_size=config.queue_max_size,
    )
    launched = demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=None,
        prevent_thread_lock=True,
        allowed_paths=[str(config.output_root.resolve())],
        blocked_paths=[
            str((config.repo_root / "checkpoints").resolve()),
            str((config.repo_root / "data").resolve()),
            str((config.repo_root / "pretrained_models").resolve()),
        ],
        show_api=False,
        quiet=False,
    )
    local_url = str(launched[1])
    share_url = str(launched[2]) if launched[2] else None
    write_access_files(
        public_url_file=args.public_url_file,
        access_info_file=args.access_info_file,
        local_url=local_url,
        public_url=share_url,
        config=config,
    )
    print(f"LOCAL_URL={local_url}", flush=True)
    print(f"PUBLIC_URL={share_url or ''}", flush=True)
    print("AUTH_MODE=public_no_login", flush=True)
    if args.share and not share_url:
        raise RuntimeError("Gradio share did not return a public URL")
    demo.block_thread()
    return local_url, share_url


if __name__ == "__main__":
    try:
        launch()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"FATAL={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
