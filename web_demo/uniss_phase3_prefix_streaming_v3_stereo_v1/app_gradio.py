#!/usr/bin/env python3
"""Gradio UI for 320/480/640 ms Phase3 prefix-streaming stereo listening."""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path
from typing import Iterator

import gradio as gr
import numpy as np

from experiments.evaluation.uniss_phase3_prefix_streaming_v3_inference_v1.streaming_engine import (
    EngineConfig,
    PrefixStreamingEngine,
    StreamUpdate,
)
from web_demo.streaming_s2st_r2_v1.audio_io import write_json

from .config import DemoConfig


ENGINE: PrefixStreamingEngine | None = None


def timeline_html(trace: list[dict[str, object]] | None) -> str:
    rows = []
    for event in (trace or [])[-60:]:
        action = str(event.get("action", "")).upper()
        css = "write" if action == "WRITE" else "wait"
        rows.append(
            "<tr>"
            f"<td>{int(event.get('index', 0)) + 1}</td>"
            f"<td>{float(event.get('source_end_ms', 0.0)):.0f} ms</td>"
            f"<td><span class='chip {css}'>{action}</span></td>"
            f"<td>{int(event.get('new_glm_tokens', 0))}</td>"
            f"<td>{int(event.get('new_text_tokens', 0))}</td>"
            f"<td>{int(event.get('semantic_tokens', 0))}</td>"
            f"<td>{html.escape(str(event.get('committed_text', '')))}</td>"
            "</tr>"
        )
    if not rows:
        return "<div class='empty'>等待流式事件。</div>"
    return (
        "<div class='scroll'><table><thead><tr><th>#</th><th>Source</th>"
        "<th>Action</th><th>+GLM</th><th>+Text</th><th>+Semantic</th>"
        f"<th>Committed text</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _status(update: StreamUpdate, chunk_ms: int) -> str:
    if update.result is None:
        return f"**{chunk_ms} ms** · {update.status}"
    result = update.result
    return (
        f"**完成 · iter_0008000 · {chunk_ms} ms**  \n"
        f"Source={result.source_duration_seconds:.2f}s · Target={result.translation_duration_seconds:.2f}s · "
        f"RTF={result.rtf:.3f}  \n"
        f"First WRITE={result.first_write_source_ms if result.first_write_source_ms is not None else 'N/A'} ms · "
        f"First audio={result.first_audio_source_ms if result.first_audio_source_ms is not None else 'N/A'} ms · "
        f"AL={result.al_ms if result.al_ms is not None else 'N/A'} ms · AP={result.ap if result.ap is not None else 'N/A'}"
    )


def run_upload(
    audio_path: str | None,
    direction: str,
    chunk_ms: int,
    trace: list[dict[str, object]] | None,
) -> Iterator[tuple[object, ...]]:
    if not audio_path:
        raise gr.Error("请先上传或录制源音频")
    if ENGINE is None:
        raise gr.Error("推理引擎尚未初始化")
    current = list(trace or [])
    try:
        for update in ENGINE.stream(
            audio_path, direction=direction, chunk_ms=int(chunk_ms)
        ):
            if update.event is not None:
                current.append(update.event.__dict__.copy())
            if update.result is None:
                live = (
                    (16_000, np.asarray(update.audio_chunk, dtype=np.float32))
                    if update.audio_chunk.size
                    else None
                )
                yield (
                    update.translation,
                    live,
                    None,
                    None,
                    None,
                    _status(update, int(chunk_ms)),
                    timeline_html(current),
                    current,
                )
                continue
            result = update.result
            yield (
                result.translation,
                result.translation_path,
                result.timeline_path,
                result.stereo_path,
                result.result_path,
                _status(update, int(chunk_ms)),
                timeline_html(current),
                current,
            )
    except Exception as exc:
        if "CUDA out of memory" in str(exc):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        raise gr.Error(f"流式推理失败：{type(exc).__name__}: {exc}") from exc


def build_demo(config: DemoConfig, engine: PrefixStreamingEngine) -> gr.Blocks:
    global ENGINE
    ENGINE = engine
    css = """
    .model {border:1px solid #bfdbfe;background:#eff6ff;border-radius:12px;padding:12px}
    .notice {border:1px solid #fed7aa;background:#fff7ed;border-radius:12px;padding:12px}
    .scroll {max-height:320px;overflow:auto;border:1px solid #e2e8f0;border-radius:10px}
    table {width:100%;border-collapse:collapse;font-size:.84rem}
    th,td {padding:6px;border-bottom:1px solid #e2e8f0;text-align:left}
    .chip {padding:2px 7px;border-radius:999px;font-weight:700}
    .chip.wait {background:#fef3c7;color:#92400e}.chip.write {background:#dcfce7;color:#166534}
    .empty {padding:12px;color:#64748b;border:1px dashed #cbd5e1;border-radius:10px}
    """
    title = "UniSS Phase3 Prefix-Streaming Stereo from jasonleeeli(李琎) Intern"
    with gr.Blocks(title=title, css=css) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(
            "**固定模型**：full198 prefix-streaming v3 validation 最优 `iter_0008000`（LoRA + 原 Phase3 base）。",
            elem_classes=["model"],
        )
        gr.Markdown(
            "⚠️ 这是训练定义一致的 source-prefix pseudo-streaming：WhisperVQ 对累计音频前缀重复编码，"
            "不是 causal Whisper。320/480/640ms 是前缀更新间隔；实际 First audio 由 WAIT/WRITE 策略决定，"
            "因此更小 chunk 不保证每条语音都更早 WRITE。",
            elem_classes=["notice"],
        )
        trace = gr.State([])
        with gr.Row():
            with gr.Column():
                source = gr.Audio(
                    label="源音频（上传或麦克风录制）",
                    sources=["upload", "microphone"],
                    type="filepath",
                    format=None,
                )
                direction = gr.Radio(
                    ["中文 → 英文", "英文 → 中文"],
                    value="中文 → 英文",
                    label="翻译方向",
                )
                chunk = gr.Radio(
                    [320, 480, 640], value=480, label="累计前缀更新间隔（ms）"
                )
                run = gr.Button("开始流式推理并生成双声道", variant="primary")
                status = gr.Markdown("等待音频。")
            with gr.Column():
                translation = gr.Textbox(label="稳定提交的翻译文本", lines=5, interactive=False)
                live = gr.Audio(
                    label="最近生成的右声道语音块（完成后为连续目标语音）",
                    autoplay=True,
                    interactive=False,
                )
                timeline = gr.Audio(
                    label="翻译时间线（包含 WAIT 静音）", type="filepath", interactive=False
                )
        gr.Markdown(
            "## 🎧 左右声道延迟试听\n佩戴耳机：左声道始终是原语音，右声道按模型实际 First audio 时间开始播放翻译语音。"
        )
        stereo = gr.Audio(
            label="双声道：左=源语言，右=翻译语言",
            type="filepath",
            interactive=False,
            show_download_button=True,
        )
        events = gr.HTML(timeline_html([]))
        result_file = gr.File(label="完整事件与延迟指标 JSON")
        run.click(
            run_upload,
            inputs=[source, direction, chunk, trace],
            outputs=[translation, live, timeline, stereo, result_file, status, events, trace],
            concurrency_limit=1,
            api_name="prefix_streaming_stereo",
        )
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def launch(argv: list[str] | None = None) -> tuple[str, str | None]:
    args = parse_args(argv)
    config = DemoConfig.from_env()
    config.validate()
    engine = PrefixStreamingEngine(
        EngineConfig(
            adapter_dir=config.adapter_dir,
            speech_tokenizer_dir=config.speech_tokenizer_dir,
            output_root=config.output_root,
            device=config.device,
            chunk_ms=480,
            max_audio_seconds=config.max_audio_seconds,
        )
    )
    engine.load()
    demo = build_demo(config, engine).queue(
        default_concurrency_limit=1, max_size=config.queue_max_size
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
    public_url = str(launched[2]) if launched[2] else None
    Path(config.demo_root / "public_url.txt").write_text(
        (public_url or "") + "\n", encoding="utf-8"
    )
    write_json(
        config.demo_root / "access_info.json",
        {
            "local_url": local_url,
            "public_url": public_url,
            "auth_mode": "public_no_login",
            "model": "full198 prefix-streaming v3 iter_0008000",
            "chunks_ms": [320, 480, 640],
            "stereo": "left=source,right=translation",
            "pseudo_streaming": True,
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
