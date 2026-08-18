"""Public Gradio frontend for Stage-B E2 causal latency diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
import torchaudio

from training.simul_uniss.policy_tokenizer import PolicyTokenizer
from training.simul_uniss.subsecond_v1.model import CausalAudioStudentV2, StageBModelConfig
from training.simul_uniss.subsecond_v1.streaming import CausalStudentStreamingSession


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/simul_uniss_subsecond_v1/stage_b_pilot_15shard_vectorized_v2/best.pt"
)
DEFAULT_TOKENIZER = (
    REPO_ROOT
    / "data/processed/simul_uniss_v1/bootstrap_15shard/policy_tokenizer/policy_8k.model"
)


class E2Engine:
    def __init__(self, checkpoint: Path, tokenizer: Path, device: str) -> None:
        self.checkpoint_path = checkpoint.resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        value = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        config = StageBModelConfig.from_dict(value["model_config"])
        self.model = CausalAudioStudentV2(config).to(self.device).eval()
        self.model.load_state_dict(value["model"], strict=True)
        del value
        self.tokenizer = PolicyTokenizer(tokenizer)
        self.lock = threading.Lock()

    @staticmethod
    def _mono_float(audio: tuple[int, np.ndarray]) -> tuple[int, torch.Tensor]:
        sample_rate, samples = audio
        value = np.asarray(samples)
        if value.ndim == 2:
            value = value.astype(np.float32).mean(axis=1)
        if np.issubdtype(value.dtype, np.integer):
            scale = float(max(abs(np.iinfo(value.dtype).min), np.iinfo(value.dtype).max))
            value = value.astype(np.float32) / scale
        else:
            value = value.astype(np.float32)
        value = np.nan_to_num(value, copy=False)
        peak = float(np.max(np.abs(value))) if value.size else 0.0
        if peak > 1.0:
            value /= peak
        return int(sample_rate), torch.from_numpy(value.copy())

    def run(self, audio: tuple[int, np.ndarray] | None, wait_k: int) -> tuple[str, str, str, str]:
        if audio is None:
            raise gr.Error("请先上传音频或完成麦克风录音。")
        sample_rate, waveform = self._mono_float(audio)
        if waveform.numel() < max(400, sample_rate // 10):
            raise gr.Error("音频过短，请至少提供约0.1秒语音。")
        if sample_rate != self.model.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sample_rate, self.model.config.sample_rate
            ).squeeze(0)
        chunk_ms = self.model.config.segment_frames * 40
        right_ms = self.model.config.right_context_frames * 40
        chunk_samples = round(chunk_ms * self.model.config.sample_rate / 1000)
        with self.lock:
            session = CausalStudentStreamingSession(self.model)
            for start in range(0, waveform.numel(), chunk_samples):
                end = min(waveform.numel(), start + chunk_samples)
                session.feed(waveform[start:end], final=end == waveform.numel())

        first = session.glm_emissions[0] if session.glm_emissions else None
        stable = next(
            (item for item in session.glm_emissions if item.stability_probability >= 0.5),
            None,
        )
        scheduled_ms = wait_k * chunk_ms

        def write_times(item: Any) -> tuple[float | None, float | None]:
            if item is None:
                return None, None
            nca = max(float(scheduled_ms), float(item.nca_ms))
            return nca, max(nca, float(item.ca_ms))

        first_nca, first_ca = write_times(first)
        stable_nca, stable_ca = write_times(stable)
        source_piece_ids = [item.token_id for item in session.source_emissions]
        transcription = self.tokenizer.processor.decode(source_piece_ids).strip()
        glm_ids = [item.token_id for item in session.glm_emissions]
        under_one_second = stable_ca is not None and stable_ca < 1000.0
        status = (
            f"### E2 frontend结果：{'低于1秒' if under_one_second else '未达到1秒或没有稳定token'}\n\n"
            f"- 原生结构：chunk `{chunk_ms} ms`，right context `{right_ms} ms`，wait-k `{wait_k}`\n"
            f"- First predicted GLM：NCA `{first.nca_ms:.1f} ms` / CA `{first.ca_ms:.1f} ms`\n"
            if first is not None
            else "### E2 frontend结果：没有发出GLM token\n\n"
        )
        if first is not None:
            status += (
                f"- fixed wait-k First WRITE：NCA `{first_nca:.1f} ms` / CA `{first_ca:.1f} ms`\n"
                f"- stability≥0.5 First WRITE：NCA `{stable_nca:.1f} ms` / CA `{stable_ca:.1f} ms`\n"
                if stable is not None
                else "- stability≥0.5 First WRITE：无\n"
            )
            status += (
                f"- Active RTF `{session.active_rtf:.4f}`；末尾计算backlog `{session.final_backlog_ms:.1f} ms`\n"
                "- 注意：这是因果前端/First WRITE诊断，不是Qwen+BiCodec端到端翻译首音频延迟。"
            )
        timeline = [event.to_dict() for event in session.chunk_events]
        token_payload = {
            "glm_token_count": len(glm_ids),
            "glm_token_ids": glm_ids,
            "source_ctc_piece_count": len(source_piece_ids),
            "source_ctc_piece_ids": source_piece_ids,
        }
        return (
            status,
            transcription or "（Source CTC未解码出可显示文本）",
            json.dumps(token_payload, ensure_ascii=False, indent=2),
            json.dumps(timeline, ensure_ascii=False, indent=2),
        )


def build_app(engine: E2Engine) -> gr.Blocks:
    with gr.Blocks(title="UniSS Subsecond E2 Diagnostic") as demo:
        gr.Markdown(
            "# UniSS 真流式亚秒 E2 诊断\n"
            "上传音频或录音后，系统按真实 `160 ms` PCM chunk、causal log-Mel 和 Emformer cache 增量执行。"
            "本页面验证 frontend/First WRITE；当前阶段尚未连接完整翻译语音生成链路。"
        )
        with gr.Row():
            audio = gr.Audio(
                label="源语音（上传或麦克风；录音结束后进行真实chunk回放）",
                sources=["upload", "microphone"],
                type="numpy",
            )
            wait_k = gr.Radio([2, 3], value=2, label="fixed wait-k（chunk数）")
        run = gr.Button("开始 E2 真流式延迟诊断", variant="primary")
        status = gr.Markdown()
        transcription = gr.Textbox(label="Source CTC 转写诊断", lines=3)
        with gr.Accordion("GLM / Source token IDs", open=False):
            tokens = gr.Code(language="json")
        with gr.Accordion("逐chunk computation-aware时间线", open=False):
            timeline = gr.Code(language="json")
        run.click(
            engine.run,
            inputs=[audio, wait_k],
            outputs=[status, transcription, tokens, timeline],
        )
        gr.Markdown(
            "质量限制：当前 Stage B checkpoint 的独立GLM agreement质量门未通过；"
            "页面显示 `<1 s` 时，只能说明frontend延迟下界，不代表翻译质量或端到端Useful First Audio已经合格。"
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--device", default=os.environ.get("UNISS_E2_DEVICE", "cuda:0"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = E2Engine(Path(args.checkpoint), Path(args.tokenizer), args.device)
    app = build_app(engine)
    print(f"CHECKPOINT={engine.checkpoint_path}", flush=True)
    print("SCOPE=frontend_e2_diagnostic_not_end_to_end_audio", flush=True)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
