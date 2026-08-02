from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from training.simul_uniss.subsecond_v2.stage_b_latent_model import (
    LatentCausalAudioStudent,
    LatentStageBModelConfig,
)
from web_demo.stage_b_v2_streaming_stereo_v1.student_frontend import (
    LatentStudentStreamingSession,
    StudentV2StreamingFrontend,
)


class StudentV2FrontendTest(unittest.TestCase):
    def test_cached_pcm_session_matches_full_causal_inference(self) -> None:
        torch.manual_seed(7)
        config = LatentStageBModelConfig(
            policy_vocab_size=16,
            codebook_size=32,
            codebook_dim=8,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            ffn_dim=16,
            dropout=0.0,
            n_mels=8,
            left_context_frames=8,
        )
        model = LatentCausalAudioStudent(config, torch.randn(32, 8)).eval()
        waveform = torch.randn(1, 16_000) * 0.02
        with torch.inference_mode():
            output = model.infer_waveform(waveform)
            length = int(output["token_lengths"][0])
            expected = model.quantize(output["glm_latent"][:, :length]).reshape(-1).tolist()
        session = LatentStudentStreamingSession(model, synchronize_cuda=False)
        chunk = 2_560
        for start in range(0, waveform.shape[-1], chunk):
            end = min(waveform.shape[-1], start + chunk)
            session.feed(waveform[0, start:end], final=end == waveform.shape[-1])
        self.assertEqual(session.tokens, expected)
        self.assertGreater(len(session.events), 1)
        self.assertGreater(len(session.tokens), 0)

    def test_frontend_exposes_legacy_finalizer_revision_contract(self) -> None:
        torch.manual_seed(11)
        config = LatentStageBModelConfig(
            policy_vocab_size=16,
            codebook_size=32,
            codebook_dim=8,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            ffn_dim=16,
            dropout=0.0,
            n_mels=8,
            left_context_frames=8,
        )
        model = LatentCausalAudioStudent(config, torch.randn(32, 8)).eval()
        frontend = StudentV2StreamingFrontend(SimpleNamespace(model=model))
        self.assertIs(frontend.committer, frontend)
        self.assertEqual(frontend.committer.revision_events, 0)


if __name__ == "__main__":
    unittest.main()
