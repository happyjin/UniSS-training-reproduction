import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.simultaneous_streaming.stage4_aggregate import (
    common_comparisons,
    flatten_common,
    parse_args,
)
from evaluation.simultaneous_streaming.stage4_metrics import (
    playback_metrics,
    policy_metrics,
    token_latency_metrics,
)
from evaluation.simultaneous_streaming.stage4_streaming_decode import (
    boundary_metrics,
    completed_decode_summary,
    decode_streaming_row,
)
from evaluation.simultaneous_streaming.stage4_streaming_generate import (
    GenerationState,
    account_prompt_length,
    normalized_write_tail,
    parse_write_tokens,
    source_chunk_tokens,
    stage4_header,
)
from evaluation.simultaneous_streaming.stage4_streaming_generate import (
    parse_args as parse_generation_args,
)
from training import constants_uniss as c


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(str(value) for value in ids)


class Stage4StreamingTest(unittest.TestCase):
    def test_generation_write_bias_defaults_to_zero_and_is_configurable(self):
        required = [
            "--model",
            "model",
            "--schedules",
            "schedules.jsonl",
            "--output-dir",
            "output",
            "--rank",
            "0",
            "--world-size",
            "1",
        ]
        self.assertEqual(parse_generation_args(required).write_logit_bias, 0.0)
        self.assertEqual(
            parse_generation_args(
                [*required, "--write-logit-bias", "0.3"]
            ).write_logit_bias,
            0.3,
        )

    def test_completed_decode_resume_returns_saved_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = {"rank": 0, "decoded": 12, "failed": 0}
            (output_dir / "decode_summary.rank000.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (output_dir / "DECODE_COMPLETE.rank000").write_text(
                "complete\n", encoding="utf-8"
            )
            self.assertEqual(completed_decode_summary(output_dir, 0, True), summary)
            self.assertIsNone(completed_decode_summary(output_dir, 0, False))

    def test_prompt_header_and_source_chunk_match_training_format(self):
        schedule = {"tgt_lang": "eng", "speaker_tokens": list(range(32))}
        header = stage4_header(schedule)
        self.assertEqual(
            header[:5],
            [
                c.TOKEN_TASK_STREAMING_S2ST,
                c.TOKEN_STREAMING_MODE,
                c.TOKEN_DYNAMIC_MODE,
                c.TOKEN_ENG,
                c.speed_token_id(1.0),
            ],
        )
        chunk = source_chunk_tokens({"source_glm": [1, 2]})
        self.assertEqual(
            chunk,
            [
                c.TOKEN_START_GLM,
                c.glm_semantic_id(1),
                c.glm_semantic_id(2),
                c.TOKEN_END_GLM,
            ],
        )

    def test_parse_and_normalize_write(self):
        ids = [
            c.TOKEN_ENG,
            c.TOKEN_START_CONTENT,
            10,
            11,
            c.TOKEN_END_CONTENT,
            c.TOKEN_START_SEMANTIC,
            c.bicodec_semantic_id(3),
            c.bicodec_semantic_id(4),
            c.TOKEN_END_SEMANTIC,
        ]
        parsed = parse_write_tokens(ids, FakeTokenizer())
        self.assertEqual(parsed["text_ids"], [10, 11])
        self.assertEqual(parsed["semantic_values"], [3, 4])
        self.assertEqual(normalized_write_tail(parsed, "eng"), ids)

    def test_free_running_context_boundary_is_recorded(self):
        state = GenerationState(index=0, schedule={}, prompt_ids=list(range(18_001)))
        account_prompt_length(state, 18_000)
        self.assertEqual(state.max_prompt_tokens, 18_001)
        self.assertTrue(state.training_context_exceeded)

    def sample_row(self):
        return {
            "source_glm_length": 4,
            "reference_target_text_length": 2,
            "source_duration_ms_proxy": 1920,
            "forced_action_count": 0,
            "structural_recovery_count": 0,
            "event_trace": [
                {
                    "reference_action": "wait",
                    "action": "wait",
                    "source_glm_end": 1,
                    "source_end_ms": 640,
                    "source_is_final": False,
                    "action_request_seconds": 0.01,
                    "write_request_seconds": None,
                    "codec_seconds": None,
                    "audio_samples": 0,
                },
                {
                    "reference_action": "write",
                    "action": "write",
                    "source_glm_end": 4,
                    "source_end_ms": 1920,
                    "source_is_final": True,
                    "generated_text_ids": [1, 2],
                    "action_request_seconds": 0.01,
                    "write_request_seconds": 0.03,
                    "codec_seconds": 0.02,
                    "audio_samples": 16000,
                },
            ],
        }

    def test_policy_and_token_latency(self):
        row = self.sample_row()
        policy = policy_metrics(row)
        self.assertEqual(policy["binary_accuracy"], 1.0)
        self.assertTrue(policy["final_flush_success"])
        latency = token_latency_metrics(row)
        self.assertEqual(latency["first_write_ms_proxy"], 1920.0)
        self.assertGreaterEqual(latency["al_glm_tokens_proxy"], 0.0)

    def test_playback_latency_includes_compute(self):
        result = playback_metrics(self.sample_row())
        self.assertEqual(result["num_audio_chunks"], 1)
        self.assertAlmostEqual(result["start_offset_nca_ms"], 1920.0)
        self.assertGreater(result["start_offset_ca_ms"], result["start_offset_nca_ms"])
        self.assertGreater(result["rtf_generated_audio"], 0.0)

    def test_streaming_decode_and_boundary_metrics(self):
        row = self.sample_row()
        row["speaker_tokens"] = list(range(32))
        row["event_trace"][1]["generated_semantic_values"] = list(range(10))

        def fake_decode(_speaker, semantics):
            return np.repeat(np.asarray(semantics, dtype=np.float32) / 10.0, 320)

        waveform, traces, summary = decode_streaming_row(
            row,
            decode=fake_decode,
            sample_rate=16000,
            semantic_rate=50.0,
            left_context_tokens=50,
            holdback_tokens=0,
            overlap_ms=0.0,
        )
        self.assertEqual(len(waveform), 3200)
        self.assertEqual(traces[-1]["audio_samples"], 3200)
        self.assertEqual(summary["audio_chunks"], 1)
        metrics = boundary_metrics(np.ones(320), np.zeros(320), 16000)
        self.assertAlmostEqual(metrics["amplitude_jump"], 1.0)
        self.assertEqual(metrics["click"], 1.0)

    def test_flatten_common_metrics_accepts_streaming_mode(self):
        values = flatten_common(
            {
                "text_bleu": {
                    "groups": {
                        "streaming_stage4:cmn->eng": {"score": 12.5},
                    }
                },
                "slc": {
                    "groups": {
                        "streaming_stage4:cmn->eng": {"slc_0_2": 0.4, "slc_0_4": 0.7},
                    }
                },
            }
        )
        self.assertEqual(values[("text_bleu", "streaming_stage4", "cmn->eng")], 12.5)
        self.assertEqual(values[("slc_0_4", "streaming_stage4", "cmn->eng")], 0.7)

    def test_report_cli_accepts_test_split_and_later_gpus(self):
        args = parse_args(
            [
                "report",
                "--run-dir",
                "/tmp/run",
                "--results",
                "/tmp/results.jsonl",
                "--offline-phase3-root",
                "/tmp/offline",
                "--output-json",
                "/tmp/aggregate.json",
                "--report",
                "/tmp/report.md",
                "--expected-records",
                "23",
                "--gpu-ids",
                "4,5,6,7",
                "--split-label",
                "test",
            ]
        )
        self.assertEqual(args.gpu_ids, "4,5,6,7")
        self.assertEqual(args.split_label, "test")

    def test_report_cli_accepts_stage6_identity(self):
        args = parse_args(
            [
                "report",
                "--run-dir",
                "/tmp/run",
                "--results",
                "/tmp/results.jsonl",
                "--offline-phase3-root",
                "/tmp/offline",
                "--output-json",
                "/tmp/aggregate.json",
                "--report",
                "/tmp/report.md",
                "--expected-records",
                "23",
                "--stage-label",
                "Stage6",
                "--stage-iteration",
                "1189",
                "--stage-description",
                "joint low-LR refinement",
                "--streaming-mode",
                "streaming_stage6",
            ]
        )
        self.assertEqual(args.stage_label, "Stage6")
        self.assertEqual(args.stage_iteration, 1189)
        self.assertEqual(args.streaming_mode, "streaming_stage6")

    def test_common_comparison_filters_requested_stage_mode(self):
        streaming = {
            "text_bleu": {
                "groups": {
                    "streaming_stage4:cmn->eng": {"score": 1.0},
                    "streaming_stage6:cmn->eng": {"score": 3.0},
                }
            }
        }
        offline = {
            "text_bleu": {
                "groups": {
                    "quality:cmn->eng": {"score": 4.0},
                    "performance:cmn->eng": {"score": 2.0},
                }
            }
        }
        rows = common_comparisons(
            streaming,
            offline,
            streaming_mode="streaming_stage6",
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["streaming_value"] == 3.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
