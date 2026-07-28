"""Gradio UI for the frozen Phase3 Quality-only offline S2ST engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gradio as gr

from .config import DemoConfig
from .inference_engine import InferenceResult, Phase3QualityEngine

ENGINE: Phase3QualityEngine | None = None


def format_status(result: InferenceResult) -> str:
    warning_text = "无" if not result.warnings else "；".join(result.warnings)
    return (
        f"**完成** · 模型：`{result.model_label}` · 模式：`{result.mode}`  \n"
        f"输入：{result.input_duration_seconds:.2f}s · 输出：{result.output_duration_seconds:.2f}s · "
        f"推理：{result.total_seconds:.2f}s  \n"
        f"警告：{warning_text}"
    )


def append_history(
    history: list[dict[str, str]] | None, result: InferenceResult
) -> list[dict[str, str]]:
    updated = list(history or [])
    updated.extend(
        [
            {
                "role": "user",
                "content": (
                    f"🎙️ **输入语音**：{result.input_duration_seconds:.2f}s  \n"
                    f"方向：{result.direction}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"**源语音转写 / Source transcription**  \n{result.transcription or '⚠️ 未生成'}"
                    f"\n\n**翻译文本 / Translation**  \n{result.translation or '⚠️ 未生成'}"
                    f"\n\n🔊 翻译语音已显示在本消息上方的播放器中；"
                    f"如果浏览器阻止自动播放，请点击播放按钮或下载 WAV。"
                ),
            },
        ]
    )
    return updated


def translate_request(
    audio_path: str | None,
    direction: str,
    use_silence_chunking: bool,
    history: list[dict[str, str]] | None,
    progress=gr.Progress(),  # noqa: B008 - Gradio injects Progress through this default.
):
    if not audio_path:
        raise gr.Error("请先录音或上传音频")
    if ENGINE is None:
        raise gr.Error("Phase3 inference engine 尚未初始化")

    def notify(fraction: float, message: str) -> None:
        progress(fraction, desc=message)

    try:
        result = ENGINE.translate(
            audio_path,
            direction=direction,
            use_silence_chunking=use_silence_chunking,
            progress=notify,
        )
    except Exception as exc:
        if "CUDA out of memory" in str(exc) and hasattr(ENGINE, "device"):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        raise gr.Error(f"Phase3 Quality 推理失败：{type(exc).__name__}: {exc}") from exc
    return (
        result.transcription,
        result.translation,
        result.output_audio_path,
        result.output_audio_path,
        result.result_json_path,
        format_status(result),
        append_history(history, result),
    )


def clear_outputs():
    return None, "", "", None, None, None, "等待输入音频。", []


def build_demo(config: DemoConfig, engine: Phase3QualityEngine) -> gr.Blocks:
    global ENGINE
    ENGINE = engine
    css = """
    .model-card {border: 1px solid #dbeafe; border-radius: 12px; padding: 12px; background: #eff6ff;}
    .footer-note {font-size: 0.88rem; color: #64748b;}
    """
    page_title = "Offline Speech-to-Speech from jasonleeeli(李琎) Intern"
    with gr.Blocks(title=page_title, css=css) as demo:
        gr.Markdown(
            f"# {page_title}\n"
            "录音或上传一句中文/英文语音，查看模型自身ASR转写、翻译文本，并播放生成的翻译语音。"
        )
        gr.Markdown(
            f"**固定模型**：`{config.model_label}`　 **固定模式**：`Quality`　 "
            "**类型**：`Offline / Non-simultaneous`",
            elem_classes=["model-card"],
        )
        history = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                input_audio = gr.Audio(
                    label="输入语音 / Record or upload",
                    sources=["microphone", "upload"],
                    type="filepath",
                    # Keep browser microphone WebM/Opus files unchanged here.
                    # audio_io.py decodes them with the bundled imageio-ffmpeg;
                    # forcing Gradio to WAV would require a system ffprobe.
                    format=None,
                )
                direction = gr.Radio(
                    choices=["中文 → 英文", "英文 → 中文"],
                    value="中文 → 英文",
                    label="翻译方向",
                )
                use_chunking = gr.Checkbox(
                    value=True,
                    label="长音频按静音切段（推荐）",
                )
                with gr.Row():
                    run_button = gr.Button("开始翻译", variant="primary")
                    clear_button = gr.Button("清空")
                status = gr.Markdown("等待输入音频。")
            with gr.Column(scale=1):
                transcription = gr.Textbox(
                    label="源语音转写 / Source transcription",
                    lines=4,
                    interactive=False,
                )
                translation = gr.Textbox(
                    label="翻译文本 / Translation",
                    lines=4,
                    interactive=False,
                )
                output_audio = gr.Audio(
                    label="翻译语音 / Generated speech",
                    type="filepath",
                    format="wav",
                    interactive=False,
                    autoplay=True,
                    show_download_button=True,
                    show_share_button=False,
                )
                output_audio_download = gr.File(label="下载翻译语音 WAV")
                result_json = gr.File(label="下载本轮JSON")
        chatbot = gr.Chatbot(
            label="本次浏览器会话",
            type="messages",
            height=360,
        )
        gr.Markdown(
            "本页面使用离线Phase3模型：收到完整语句或静音切段后才生成，不是实时同声传译。"
            "上传音频和结果默认在24小时后清理。",
            elem_classes=["footer-note"],
        )
        run_button.click(
            fn=translate_request,
            inputs=[input_audio, direction, use_chunking, history],
            outputs=[
                transcription,
                translation,
                output_audio,
                output_audio_download,
                result_json,
                status,
                chatbot,
            ],
            api_name="translate_phase3_quality",
        ).then(lambda value: value, inputs=[chatbot], outputs=[history])
        clear_button.click(
            fn=clear_outputs,
            outputs=[
                input_audio,
                transcription,
                translation,
                output_audio,
                output_audio_download,
                result_json,
                status,
                chatbot,
            ],
        ).then(list, outputs=[history])
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--auth-user", default=os.environ.get("UNISS_DEMO_AUTH_USER", "uniss")
    )
    parser.add_argument(
        "--auth-password", default=os.environ.get("UNISS_DEMO_AUTH_PASSWORD")
    )
    parser.add_argument(
        "--public-url-file", default=str(DemoConfig().demo_root / "public_url.txt")
    )
    parser.add_argument(
        "--access-info-file", default=str(DemoConfig().demo_root / "access_info.json")
    )
    return parser.parse_args(argv)


def launch(argv: list[str] | None = None) -> tuple[str, str | None]:
    args = parse_args(argv)
    config = DemoConfig.from_env()
    config.validate()
    if args.share and not args.auth_password:
        raise ValueError("Public Gradio launch requires UNISS_DEMO_AUTH_PASSWORD")
    engine = Phase3QualityEngine(config)
    engine.load()
    demo = build_demo(config, engine).queue(default_concurrency_limit=1, max_size=8)
    auth = (args.auth_user, args.auth_password) if args.auth_password else None
    launched = demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        prevent_thread_lock=True,
        allowed_paths=[str(config.output_root.resolve())],
        blocked_paths=[
            str(config.repo_root / "checkpoints"),
            str(config.repo_root / "data"),
            str(config.repo_root / "pretrained_models"),
        ],
        show_api=False,
        quiet=False,
    )
    local_url = str(launched[1])
    share_url = str(launched[2]) if launched[2] else None
    public_path = Path(args.public_url_file)
    public_path.write_text((share_url or "") + "\n", encoding="utf-8")
    access_path = Path(args.access_info_file)
    access_path.write_text(
        json.dumps(
            {
                "local_url": local_url,
                "public_url": share_url,
                "username": args.auth_user if auth else None,
                "password": args.auth_password if auth else None,
                "model": config.model_label,
                "mode": "Quality",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.chmod(access_path, 0o600)
    print(f"LOCAL_URL={local_url}", flush=True)
    print(f"PUBLIC_URL={share_url or ''}", flush=True)
    if args.share and not share_url:
        raise RuntimeError("Gradio share tunnel did not return a public URL")
    demo.block_thread()
    return local_url, share_url


if __name__ == "__main__":
    try:
        launch(sys.argv[1:])
    except KeyboardInterrupt:
        pass
