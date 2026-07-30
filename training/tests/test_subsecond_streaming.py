from __future__ import annotations

import unittest

import torch

from training.simul_uniss.subsecond_v1.model import (
    CausalAudioStudentV2,
    StageBModelConfig,
    greedy_ctc_tokens,
)
from training.simul_uniss.subsecond_v1.streaming import CausalStudentStreamingSession


class SubsecondStreamingTest(unittest.TestCase):
    def _model(self, right: int = 1) -> CausalAudioStudentV2:
        torch.manual_seed(7)
        return CausalAudioStudentV2(
            StageBModelConfig(
                policy_vocab_size=32,
                hidden_size=32,
                num_layers=2,
                num_heads=4,
                ffn_dim=64,
                dropout=0.0,
                n_mels=16,
                segment_frames=2,
                right_context_frames=right,
                left_context_frames=8,
            )
        ).eval()

    def test_incremental_projection_matches_offline_complete_stacks(self) -> None:
        model = self._model()
        waveform = torch.randn(1, 8000)
        offline = model.infer_waveform(waveform)
        session = CausalStudentStreamingSession(model, synchronize_cuda=False)
        for start in range(0, waveform.shape[1], 1600):
            end = min(waveform.shape[1], start + 1600)
            session.feed(waveform[0, start:end], final=end == waveform.shape[1])
        offline_tokens = [
            value - 1 for value in greedy_ctc_tokens(offline["teacher_glm_logits"])
        ]
        streaming_tokens = [event.token_id for event in session.glm_emissions]
        self.assertEqual(streaming_tokens, offline_tokens)
        self.assertEqual(
            session.summary()["output_frames"], offline["teacher_glm_logits"].shape[1]
        )

    def test_causal_session_reports_monotonic_timestamps(self) -> None:
        model = self._model(right=0)
        session = CausalStudentStreamingSession(model, synchronize_cuda=False)
        waveform = torch.randn(6400)
        for start in range(0, waveform.numel(), 1280):
            end = min(waveform.numel(), start + 1280)
            session.feed(waveform[start:end], final=end == waveform.numel())
        source_times = [event.source_end_ms for event in session.chunk_events]
        ca_times = [event.computation_end_ms for event in session.chunk_events]
        self.assertEqual(source_times, sorted(source_times))
        self.assertEqual(ca_times, sorted(ca_times))
        self.assertGreater(session.summary()["output_frames"], 0)
        self.assertGreaterEqual(session.active_rtf, 0.0)

    def test_feed_after_final_is_rejected(self) -> None:
        session = CausalStudentStreamingSession(self._model(), synchronize_cuda=False)
        session.feed(torch.zeros(1600), final=True)
        with self.assertRaises(RuntimeError):
            session.feed(torch.zeros(1600))


if __name__ == "__main__":
    unittest.main()
