#!/usr/bin/env python3
"""Public Gradio UI for five-minute bounded-window Phase3 inference."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Iterator, Sequence

import gradio as gr
import torch

from experiments.evaluation.uniss_phase3_prefix_streaming_v3_inference_v1.streaming_engine import (
    EngineConfig,
    PrefixStreamingEngine,
)
from web_demo.streaming_s2st_r2_v1.audio_io import write_json

from .config import LongFormDemoConfig
from .engine import BoundedLongFormEngine, LongFormWindowRecord


ENGINE: BoundedLongFormEngine | None = None


def window_table_html(records: Sequence[LongFormWindowRecord] | None) -> str:
    if not records:
        return "<div class='empty'>窗口结果将在推理完成后显示。</div>"
    rows: list[str] = []
    for record in records:
        css = "ok" if record.status == "completed" else "bad"
        first = (
            f"{record.first_audio_global_ms / 1000.0:.2f}s"
            if record.first_audio_global_ms is not None
            else "N/A"
        )
        rtf = f"{record.rtf:.3f}" if record.rtf is not None else "N/A"
        rows.append(
            "<tr>"
            f"<td>{record.index}</td>"
            f"<td>{record.source_start_seconds:.1f}–{record.source_end_seconds:.1f}s</td>"
            f"<td><span class='chip {css}'>{html.escape(record.status)}</span></td>"
            f"<td>{record.depth}</td>"
            f"<td>{first}</td>"
            f"<td>{rtf}</td>"
        )
        rows[-1] += (
            f"<td>{html.escape(record.translation[:160])}</td>"
            "</tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr><th>#</th><th>Source</th>"
        "<th>Status</th><th>Retry depth</th><th>Global first audio</th>"
        f"<th>RTF</th><th>Translation</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _final_status(result) -> str:
    first = (
        f"{result.first_audio_global_ms / 1000.0:.2f}s"
        if result.first_audio_global_ms is not None
        else "N/A"
    )
    return (
        f"**完成 · iter_0008000 · {result.chunk_ms}ms**  \n"
        f"Source={result.source_duration_seconds:.2f}s · "
        f"Target={result.translation_duration_seconds:.2f}s · "
        f"Windows={result.completed_windows}/{result.planned_windows}  \n"
        f"Processing={result.processing_seconds:.1f}s · RTF={result.rtf:.3f} · "
        f"Global first audio={first} · Retries={result.retry_windows}"
    )


def run_upload(
    audio_path: str | None, direction: str, chunk_ms: int
) -> Iterator[tuple[object, ...]]:
    if not audio_path:
        raise gr.Error("请先上传或录制源音频")
    if ENGINE is None:
        raise gr.Error("长音频推理引擎尚未初始化")
    try:
        for update in ENGINE.run(
            audio_path, direction=direction, chunk_ms=int(chunk_ms)
        ):
            if update.result is None:
                yield (
                    update.translation,
                    None,
                    None,
                    None,
                    None,
                    f"**进度 {update.progress * 100.0:.1f}%** · {update.status}",
                    update.progress * 100.0,
                    window_table_html(None),
                )
                continue
            result = update.result
            yield (
                result.translation,
                result.translation_path,
                result.timeline_path,
                result.stereo_path,
                result.result_path,
                _final_status(result),
                100.0,
                window_table_html(result.records),
            )
    except Exception as exc:
        if "CUDA out of memory" in str(exc) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(f"5分钟有界窗口推理失败：{type(exc).__name__}: {exc}") from exc


def build_demo(config: LongFormDemoConfig, engine: BoundedLongFormEngine) -> gr.Blocks:
    global ENGINE
    ENGINE = engine
    css = """
    .model {border:1px solid #bfdbfe;background:#eff6ff;border-radius:12px;padding:12px}
    .notice {border:1px solid #fed7aa;background:#fff7ed;border-radius:12px;padding:12px}
    .scope {border:1px solid #ddd6fe;background:#f5f3ff;border-radius:12px;padding:12px}
    .scroll {max-height:360px;overflow:auto;border:1px solid #e2e8f0;border-radius:10px}
    table {width:100%;border-collapse:collapse;font-size:.84rem}
    th,td {padding:6px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}
    .chip {padding:2px 7px;border-radius:999px;font-weight:700}
    .chip.ok {background:#dcfce7;color:#166534}.chip.bad {background:#fee2e2;color:#991b1b}
    .empty {padding:12px;color:#64748b;border:1px dashed #cbd5e1;border-radius:10px}
    """
    title = "UniSS Phase3 Five-Minute Bounded-Window S2ST from jasonleeeli(李琎) Intern"
    with gr.Blocks(title=title, css=css) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(
            "**固定模型**：full198 prefix-streaming v3 validation 最优 `iter_0008000`，"
            "与短音频Demo使用完全相同的Phase3 base和LoRA。",
            elem_classes=["model"],
        )
        gr.Markdown(
            "**长音频方式**：输入按语音低能量位置切成18–30秒有界窗口；每个窗口运行原有"
            "320/480/640ms prefix-streaming，再把翻译语音放到不可重叠的全局时间线。"
            "单个窗口失败时自动二分重试。",
            elem_classes=["scope"],
        )
        gr.Markdown(
            "⚠️ 最长接受305秒（为5分钟编码误差留5秒容差）。当前网页在上传/录制完成后才开始"
            "模拟逐chunk可见性，因此这是有界重算的 pseudo-streaming 演示，不是浏览器实时麦克风传输，"
            "也不是因果encoder cache。窗口间不重叠，避免重复语音；跨窗口上下文可能弱于整句offline。",
            elem_classes=["notice"],
        )
        with gr.Row():
            with gr.Column():
                source = gr.Audio(
                    label="源音频（最长5分钟；上传或麦克风录制）",
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
                    [320, 480, 640], value=480, label="窗口内前缀更新间隔（ms）"
                )
                run = gr.Button("开始5分钟有界窗口推理", variant="primary")
                progress = gr.Slider(
                    0,
                    100,
                    value=0,
                    step=0.1,
                    label="窗口完成进度（%）",
                    interactive=False,
                )
                status = gr.Markdown("等待音频。")
            with gr.Column():
                translation = gr.Textbox(
                    label="按窗口稳定提交的完整翻译文本", lines=12, interactive=False
                )
                target = gr.Audio(
                    label="连续翻译语音（移除WAIT静音）",
                    type="filepath",
                    interactive=False,
                )
                timeline = gr.Audio(
                    label="全局翻译时间线（包含WAIT和排队延迟）",
                    type="filepath",
                    interactive=False,
                )
        gr.Markdown(
            "## 🎧 五分钟左右声道试听\n佩戴耳机：左声道为完整源音频；右声道按照每个窗口"
            "实际可用时间和上一段目标音频结束时间开始播放。"
        )
        stereo = gr.Audio(
            label="双声道：左=源语言，右=翻译语言",
            type="filepath",
            interactive=False,
            show_download_button=True,
        )
        result_file = gr.File(label="完整窗口、时延和错误审计JSON")
        table = gr.HTML(window_table_html(None))
        run.click(
            run_upload,
            inputs=[source, direction, chunk],
            outputs=[
                translation,
                target,
                timeline,
                stereo,
                result_file,
                status,
                progress,
                table,
            ],
            concurrency_limit=1,
            api_name="phase3_bounded_longform",
        )
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7867)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def launch(argv: list[str] | None = None) -> tuple[str, str | None]:
    args = parse_args(argv)
    config = LongFormDemoConfig.from_env()
    config.validate_assets()
    base = PrefixStreamingEngine(
        EngineConfig(
            adapter_dir=config.adapter_dir,
            speech_tokenizer_dir=config.speech_tokenizer_dir,
            output_root=config.output_root / "window_runs",
            device=config.device,
            chunk_ms=480,
            max_upload_bytes=config.max_upload_bytes,
            max_audio_seconds=config.maximum_window_seconds + 0.05,
        )
    )
    engine = BoundedLongFormEngine(config, base_engine=base)
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
    (config.demo_root / "public_url.txt").write_text(
        (public_url or "") + "\n", encoding="utf-8"
    )
    write_json(
        config.demo_root / "access_info.json",
        {
            "local_url": local_url,
            "public_url": public_url,
            "auth_mode": "public_no_login",
            "model": "full198 prefix-streaming v3 iter_0008000",
            "maximum_audio_seconds": config.max_audio_seconds,
            "source_windows_seconds": [
                config.minimum_window_seconds,
                config.maximum_window_seconds,
            ],
            "chunks_ms": [320, 480, 640],
            "stereo": "left=source,right=translation",
            "bounded_window": True,
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
