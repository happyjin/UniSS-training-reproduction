"""Student-v2 source frontend plugged into the audited R2/BiCodec web engine."""

from __future__ import annotations

import json
from pathlib import Path

from web_demo.streaming_s2st_r2_v1.audio_io import write_json
from web_demo.streaming_s2st_r2_v1.engine.streaming_pipeline import (
    LiveEngineState,
    StreamingDemoEngine,
)

from training.simul_uniss.subsecond_v2.validate_stage_b_latent import load_model

from .config import StudentV2StreamingConfig
from .student_frontend import (
    StudentV2SpeechTokenizerAdapter,
    StudentV2StreamingFrontend,
)


class StudentV2StreamingEngine(StreamingDemoEngine):
    def __init__(self, config: StudentV2StreamingConfig):
        super().__init__(config)
        self.student_model = None
        self.student_checkpoint: dict[str, object] | None = None

    def load(self) -> None:
        if self.student_model is not None and self.loaded:
            return
        super().load()
        assert self.speech_tokenizer is not None
        self.student_model, self.student_checkpoint = load_model(
            self.config.student_checkpoint_path,
            self.device,
            None,
            None,
        )
        self.speech_tokenizer = StudentV2SpeechTokenizerAdapter(
            self.speech_tokenizer,
            self.student_model,
        )

    def _patch_result_metadata(self, result, *, mode: str, interface: str) -> None:
        result.mode = mode
        result_path = Path(result.result_json_path)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        first_write_event = next(
            (event for event in payload.get("events", []) if event.get("action") == "write"),
            None,
        )
        first_audio_event = next(
            (
                event
                for event in payload.get("events", [])
                if int(event.get("audio_samples", 0)) > 0
            ),
            None,
        )

        def _ca_ms(event, *, include_generation: bool) -> float | None:
            if event is None:
                return None
            milliseconds = float(event["source_end_ms"])
            milliseconds += float(event.get("action_seconds", 0.0)) * 1000.0
            if include_generation:
                milliseconds += float(event.get("write_seconds", 0.0)) * 1000.0
                milliseconds += float(event.get("codec_seconds", 0.0)) * 1000.0
            return milliseconds

        payload.update(
            {
                "mode": mode,
                "source_frontend": (
                    "Stage-B-v2 prefix-80 cached causal Emformer Student"
                ),
                "prefix_revision_events": 0,
                "student_frontend": {
                    "interface": interface,
                    "feed_ms": self.config.frontend_feed_ms,
                    "right_context_ms": self.config.frontend_right_context_ms,
                    "r2_policy_tick_ms": self.config.chunk_ms,
                },
                "latency_metrics": {
                    "first_write_source_timeline_nca_ms": result.first_write_ms,
                    "first_audio_timeline_placement_nca_ms": result.first_audio_ms,
                    "first_write_decision_ca_estimate_ms": _ca_ms(
                        first_write_event, include_generation=False
                    ),
                    "first_audio_ready_ca_estimate_ms": _ca_ms(
                        first_audio_event, include_generation=True
                    ),
                    "ca_estimate_excludes_ms": "browser capture callback, network and playback buffering",
                    "warning": (
                        "legacy first_audio_ms is timeline placement, not wall-clock audio availability"
                    ),
                },
            }
        )
        write_json(result_path, payload)

    def stream_upload(self, *args, **kwargs):
        for update in super().stream_upload(*args, **kwargs):
            if update.result is not None:
                self._patch_result_metadata(
                    update.result,
                    mode="evaluation-compatible Student-v2 causal replay",
                    interface="uploaded complete waveform replayed through causal frontend",
                )
            yield update

    def _new_live_state(self, session, direction: str) -> LiveEngineState:
        assert isinstance(self.config, StudentV2StreamingConfig)
        assert isinstance(self.speech_tokenizer, StudentV2SpeechTokenizerAdapter)
        request_dir = session.ensure_request_dir()
        state = LiveEngineState(
            direction=direction,
            request_dir=request_dir,
            frontend=StudentV2StreamingFrontend(
                self.speech_tokenizer,
                feed_ms=self.config.frontend_feed_ms,
            ),
            next_boundary_samples=int(
                round(self.config.chunk_ms * 16_000 / 1000.0)
            ),
        )
        session.engine_state = state
        return state

    def process_microphone(self, *args, **kwargs):
        updates = super().process_microphone(*args, **kwargs)
        for update in updates:
            update.status = update.status.replace(
                "Online pseudo-streaming", "Online causal Student-v2 streaming"
            ).replace(
                "稳定 WhisperVQ 前缀", "Student-v2 因果 source token"
            )
            if update.result is not None:
                self._patch_result_metadata(
                    update.result,
                    mode="online cached causal Student-v2 streaming",
                    interface="live append-only microphone PCM",
                )
        return updates
