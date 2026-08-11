#!/usr/bin/env python3
"""Public Gradio UI for the isolated pilot15 true-input-streaming runtime."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from typing import Iterator

import gradio as gr
import numpy as np

from web_demo.streaming_s2st_r2_v1.audio_io import SAMPLE_RATE, write_json

from .config import DemoConfig
from .engine import StreamUpdate, TrueSubsecondStreamingEngine


ENGINE: TrueSubsecondStreamingEngine | None = None


def _event_payload(update: StreamUpdate) -> dict[str, object] | None:
    return asdict(update.event) if update.event is not None else None


def run_upload(
    audio_path: str | None,
    direction: str,
    chunk_ms: int,
) -> Iterator[tuple]:
    if not audio_path:
        raise gr.Error("请先上传或录制源音频。")
    if ENGINE is None:
        raise gr.Error("模型尚未加载完成。")
    try:
        for update in ENGINE.stream(
            audio_path, direction=direction, decision_chunk_ms=int(chunk_ms)
        ):
            live = (
                (SAMPLE_RATE, np.asarray(update.audio_chunk, dtype=np.float32))
                if len(update.audio_chunk)
                else None
            )
            if update.result is None:
                yield (
                    update.translation,
                    live,
                    None,
                    None,
                    None,
                    None,
                    f"**{update.status}**",
                    update.progress * 100.0,
                    _event_payload(update),
                )
                continue
            result = update.result
            first_write = (
                f"{result.first_write_source_ms} ms"
                if result.first_write_source_ms is not None
                else "N/A"
            )
            first_audio = (
                f"{result.first_useful_audio_source_ms} ms"
                if result.first_useful_audio_source_ms is not None
                else "N/A"
            )
            status = (
                f"**完成 · iter_{result.selected_iteration:07d} · "
                f"{result.decision_chunk_ms} ms · "
                f"{'质量门通过' if result.quality_passed else '质量门失败'}**  \n"
                f"First WRITE={first_write} · First useful audio={first_audio} · "
                f"RTF={result.rtf:.3f}  \n"
                f"源音频={result.source_duration_seconds:.2f}s · "
                f"翻译语音={result.translation_duration_seconds:.2f}s · "
                f"覆盖率={result.translation_coverage_ratio:.1%} · "
                f"forced={result.forced_writes} · natural={result.natural_writes}  \n"
                f"质量问题={', '.join(result.quality_failures) if result.quality_failures else '无'} · "
                f"counterfactual frontend revisions={result.committed_revision_violations}"
            )
            playable = result.translation_duration_seconds > 0
            yield (
                result.committed_translation,
                live,
                result.translation_path if playable else None,
                result.timeline_path if playable else None,
                result.stereo_path if playable else None,
                result.result_path,
                status,
                100.0,
                result.to_dict(),
            )
    except Exception as exc:
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc


def build_demo(config: DemoConfig, engine: TrueSubsecondStreamingEngine) -> gr.Blocks:
    global ENGINE
    ENGINE = engine
    css = """
    .hero {max-width: 1050px; margin: auto; text-align: center;}
    .notice {border-left: 5px solid #d97706; padding: 12px 16px; background: #fff7ed;}
    .truth {border-left: 5px solid #047857; padding: 12px 16px; background: #ecfdf5;}
    """
    with gr.Blocks(title="UniSS True Streaming S2ST · pilot15", css=css) as demo:
        gr.Markdown(
            "# True Streaming Speech-to-Speech from jasonleeeli（李琎）Intern\n"
            "### Phase3 v4 + repaired pilot15 causal adapter / action / safe-commit",
            elem_classes=["hero"],
        )
        gr.Markdown(
            "**真实数据可见性约束**：推理核心每次只接收已经到达的 PCM；WhisperVQ 使用与训练缓存一致的 "
            "160 ms chunk + 80 ms bounded right context；Qwen 使用 append-only KV cache；BiCodec "
            "只解码已经生成的 semantic block。前 3.2 秒是 observed-only VAD speaker warm-up；"
            "其后模型不会接收未来 PCM。",
            elem_classes=["truth"],
        )
        gr.Markdown(
            "⚠️ 当前网页的上传/普通麦克风组件会在文件准备完成后进行严格实时回放模拟；它不是浏览器 "
            "WebRTC 边录边传。支持最长 305 秒，因此 5 分钟文件走同一条 streaming 状态机，不再切成 "
            "18–30 秒独立离线窗口。修复版禁止 support=0 的未监督 forced audio；如果质量门失败，"
            "页面会明确报告并拒绝播放乱码，而不会把噪声伪装成同传结果。",
            elem_classes=["notice"],
        )
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Audio(
                    label="源音频（上传或录制，最长 5 分钟）",
                    sources=["upload", "microphone"],
                    type="filepath",
                    format=None,
                )
                direction = gr.Radio(
                    ["英文 → 中文", "中文 → 英文"],
                    value="英文 → 中文",
                    label="翻译方向",
                )
                chunk = gr.Radio(
                    [320, 480, 640],
                    value=config.decision_chunk_ms,
                    label="决策间隔（ms）",
                )
                run = gr.Button("开始真正 streaming 回放推理", variant="primary")
                progress = gr.Slider(
                    0,
                    100,
                    value=0,
                    step=0.1,
                    interactive=False,
                    label="源流进度（%）",
                )
                status = gr.Markdown("等待音频。")
            with gr.Column(scale=1):
                translation = gr.Textbox(
                    label="不可回滚提交的翻译文本",
                    lines=8,
                    interactive=False,
                )
                live_audio = gr.Audio(
                    label="刚刚产生的 streaming 翻译语音块（自动播放）",
                    streaming=True,
                    autoplay=True,
                    type="numpy",
                    interactive=False,
                )
                continuous = gr.Audio(
                    label="完整翻译语音（移除等待空白）",
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                )
                timeline = gr.Audio(
                    label="真实输出时间线（保留 READ/计算等待）",
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                )
        gr.Markdown("## 🎧 延迟直观试听：左声道源语音，右声道按实际 WRITE/PCM 时刻播放翻译语音")
        stereo = gr.Audio(
            label="Stereo：左=源语言，右=翻译语言",
            type="filepath",
            interactive=False,
            show_download_button=True,
        )
        with gr.Row():
            metrics = gr.JSON(label="当前事件 / 最终完整 streaming 指标")
            result_file = gr.File(label="完整 result.json")
        run.click(
            run_upload,
            inputs=[source, direction, chunk],
            outputs=[
                translation,
                live_audio,
                continuous,
                timeline,
                stereo,
                result_file,
                status,
                progress,
                metrics,
            ],
            concurrency_limit=1,
            api_name="true_subsecond_streaming_s2st",
        )
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7868)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def launch(argv: list[str] | None = None) -> tuple[str, str | None]:
    args = parse_args(argv)
    config = DemoConfig.from_env()
    config.validate_assets(require_export=True)
    engine = TrueSubsecondStreamingEngine(config)
    engine.load()
    demo = build_demo(config, engine).queue(
        default_concurrency_limit=1, max_size=2
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
        max_file_size="160mb",
    )
    local_url = str(launched[1])
    public_url = str(launched[2]) if launched[2] else None
    (config.demo_root / "public_url.txt").write_text(
        (public_url or "") + "\n", encoding="utf-8"
    )
    write_json(
        config.demo_root / "access_info.json",
        {
            "local_url": local_url,
            "public_url": public_url,
            "auth_mode": "public_no_login",
            "model": (
                "Phase3 v4 + repaired true-subsecond pilot15 "
                f"iter_{int(engine.manifest['selected_iteration']):07d}"
            ),
            "maximum_audio_seconds": config.max_audio_seconds,
            "decision_chunks_ms": [320, 480, 640],
            "stereo": "left=source,right=translation-at-emission-time",
            "true_input_streaming": True,
            "browser_webrtc_live": False,
        },
    )
    print(f"LOCAL_URL={local_url}", flush=True)
    print(f"PUBLIC_URL={public_url or ''}", flush=True)
    if args.share and not public_url:
        raise RuntimeError("Gradio did not return a public share URL")
    demo.block_thread()
    return local_url, public_url


if __name__ == "__main__":
    try:
        launch()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"FATAL={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
