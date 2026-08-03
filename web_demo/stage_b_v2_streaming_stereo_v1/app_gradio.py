"""Public no-login Gradio UI for Student-v2 causal streaming stereo S2ST."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gradio as gr

from web_demo.streaming_s2st_r2_v1 import app_gradio as legacy
from web_demo.streaming_s2st_r2_v1.audio_io import write_json
from web_demo.streaming_s2st_r2_v1.session_manager import SessionRegistry

from .config import StudentV2StreamingConfig
from .engine import StudentV2StreamingEngine


def format_student_final_status(result) -> str:
    payload: dict[str, object] = {}
    try:
        payload = json.loads(Path(result.result_json_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    latency = payload.get("latency_metrics", {})
    latency = latency if isinstance(latency, dict) else {}

    def _ms(name: str) -> str:
        value = latency.get(name)
        return "N/A" if value is None else f"{float(value):.0f} ms"

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
        "First WRITE（源时间线/NCA）："
        f"{_ms('first_write_source_timeline_nca_ms')} · "
        "First audio 放置点（NCA）："
        f"{_ms('first_audio_timeline_placement_nca_ms')}  \n"
        "First WRITE 决策可用（CA估算）："
        f"{_ms('first_write_decision_ca_estimate_ms')} · "
        "First audio 服务端就绪（CA估算）："
        f"{_ms('first_audio_ready_ca_estimate_ms')}  \n"
        "CA估算尚不含浏览器回调、网络和播放缓冲 · "
        f"forced={result.forced_actions} · recovery={result.structural_recoveries}"
        f"{fallback}"
    )


def build_demo(
    config: StudentV2StreamingConfig,
    engine: StudentV2StreamingEngine,
    registry: SessionRegistry,
) -> gr.Blocks:
    legacy.ENGINE = engine
    legacy.REGISTRY = registry
    legacy.format_final_status = format_student_final_status
    css = """
    .model-card {border:1px solid #a7f3d0;border-radius:14px;padding:14px;background:#ecfdf5}
    .boundary-card {border:1px solid #fde68a;border-radius:14px;padding:12px;background:#fffbeb}
    .timeline-scroll {max-height:300px;overflow:auto;border:1px solid #e2e8f0;border-radius:10px}
    .timeline-table {width:100%;border-collapse:collapse;font-size:.85rem}
    .timeline-table th,.timeline-table td {padding:7px;border-bottom:1px solid #e2e8f0;text-align:left}
    .action-chip {display:inline-block;padding:2px 7px;border-radius:999px;font-weight:700}
    .action-chip.wait {background:#fef3c7;color:#92400e}.action-chip.write {background:#dcfce7;color:#166534}
    .timeline-empty {color:#64748b;padding:12px;border:1px dashed #cbd5e1;border-radius:10px}
    """
    title = "UniSS Student-v2 Causal Streaming Stereo from jasonleeeli(李琎) Intern"
    sync_js = """() => {
      const left = document.querySelector('#student-v2-source audio');
      const right = document.querySelector('#student-v2-target audio');
      if (!left || !right) { return []; }
      left.pause(); right.pause(); left.currentTime = 0; right.currentTime = 0;
      Promise.allSettled([left.play(), right.play()]);
      return [];
    }"""
    with gr.Blocks(title=title, css=css) as demo:
        gr.Markdown(
            f"# {title}\n"
            "使用最新 Stage-B-v2 prefix-80 Student 作为因果源语音前端，"
            "复用已验证的 R2 WAIT/WRITE 与 Streaming BiCodec。"
        )
        gr.Markdown(
            f"**固定链路**：`{config.model_label}`　"
            "**Student原生几何**：`160 ms chunk + 80 ms lookahead`　"
            "**R2决策周期**：`640 ms`　**公网**：无需注册或登录",
            elem_classes=["model-card"],
        )
        gr.Markdown(
            "⚠️ Student前端是真因果流式并使用Emformer cache；R2策略仍按旧网站训练分布每640ms决策。"
            "当前Student target agreement约29.3%，correct-stable 320/480ms只覆盖31.25%的验证样本。"
            "页面用于真实试听和延迟审计，不代表质量门已通过；First audio必须看CA估算，不能只看NCA放置点。",
            elem_classes=["boundary-card"],
        )
        with gr.Tab("上传音频 / Causal replay"):
            upload_trace = gr.State([])
            with gr.Row():
                with gr.Column():
                    upload_source = gr.Audio(
                        label="源音频 / Source",
                        sources=["upload"],
                        type="filepath",
                        format=None,
                        elem_id="student-v2-source",
                    )
                    upload_direction = gr.Radio(
                        ["中文 → 英文", "英文 → 中文"],
                        value="中文 → 英文",
                        label="翻译方向",
                    )
                    upload_button = gr.Button("开始 Student-v2 同传回放", variant="primary")
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
                        label="同传时间线 / Translation with real WAIT silence",
                        type="filepath",
                        interactive=False,
                        elem_id="student-v2-target",
                    )
                    sync_button = gr.Button("从0秒同步播放源音频和翻译时间线")
            gr.Markdown(
                "### 🎧 左右声道真实延迟\n"
                "佩戴耳机：左声道是源语音，右声道按R2实际WRITE时刻播放翻译语音；"
                "右声道开头的静音就是可听延迟。"
            )
            upload_stereo_player = gr.Audio(
                label="双声道播放（左=源语言，右=翻译语言）",
                type="filepath",
                interactive=False,
                show_download_button=True,
            )
            upload_timeline = gr.HTML(legacy.event_timeline_html([]))
            with gr.Row():
                upload_raw_audio = gr.File(label="连续目标语音 WAV")
                upload_stereo = gr.File(label="同步双声道 WAV")
                upload_json = gr.File(label="事件与指标 JSON")
                upload_download = gr.File(label="目标语音下载")
            upload_button.click(
                legacy.stream_upload_request,
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
                api_name="stream_upload_student_v2",
            )
            sync_button.click(fn=None, js=sync_js)

        with gr.Tab("麦克风 / True causal frontend"):
            mic_session_id = gr.State(None)
            mic_trace = gr.State([])
            gr.Markdown(
                "浏览器每640ms把新增PCM交给后端；Student内部按160ms分块并保留Emformer cache。"
                "当前仍需收集约3.2秒源语音获得目标音色，因此首段翻译音频不会等于320ms前端延迟。"
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
                        label="完成后的双声道对比（左=源语言，右=翻译语言）",
                        type="filepath",
                        interactive=False,
                        show_download_button=True,
                    )
            mic_timeline = gr.HTML(legacy.event_timeline_html([]))
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
                legacy.microphone_step,
                inputs=[microphone, mic_direction, mic_session_id, mic_trace],
                outputs=mic_outputs,
                stream_every=config.chunk_ms / 1000.0,
                time_limit=config.microphone_max_audio_seconds,
                concurrency_limit=1,
            )
            microphone.stop_recording(
                legacy.microphone_finalize,
                inputs=[mic_direction, mic_session_id, mic_trace],
                outputs=mic_outputs,
                concurrency_limit=1,
            )
            mic_finalize_button.click(
                legacy.microphone_finalize,
                inputs=[mic_direction, mic_session_id, mic_trace],
                outputs=mic_outputs,
                concurrency_limit=1,
            )
            mic_reset_button.click(
                legacy.reset_microphone,
                inputs=[mic_session_id],
                outputs=mic_outputs,
                concurrency_limit=1,
            )
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--public-url-file",
        default=str(StudentV2StreamingConfig().demo_root / "public_url.txt"),
    )
    parser.add_argument(
        "--access-info-file",
        default=str(StudentV2StreamingConfig().demo_root / "access_info.json"),
    )
    return parser.parse_args(argv)


def write_access_files(
    *,
    public_url_file: str | Path,
    access_info_file: str | Path,
    local_url: str,
    public_url: str | None,
    config: StudentV2StreamingConfig,
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
            "student_checkpoint": str(config.student_checkpoint_path),
            "mode": "Student-v2 causal frontend + R2 WAIT/WRITE + streaming BiCodec",
        },
    )
    os.chmod(access_path, 0o600)


def launch(argv: list[str] | None = None) -> tuple[str, str | None]:
    args = parse_args(argv)
    config = StudentV2StreamingConfig.from_env()
    config.validate()
    engine = StudentV2StreamingEngine(config)
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
