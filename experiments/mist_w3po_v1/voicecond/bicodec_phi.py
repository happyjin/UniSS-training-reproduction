"""Shared BiCodec global-code access for the voice-condition ideas (reports/mist_w3po_v1/game_voice_condition_ideas_plan.zh-CN.md §1.1).

phi(y): the continuous PRE-BOUND code u in R^{32x6} (= ResidualFSQ.project_in of the Perceiver output), computed exactly as
BiCodecTokenizer.tokenize does for the global tokens: volume normalisation (audio_volume_normalize), first 6 s (tiled if
shorter), mel (model.mel_transformer), ECAPA features -> Perceiver -> project_in. tokens(u) applies the FSQ layer (bound +
round) and returns the 32 global token ids identical to encode_wav_to_tokens(...)[:32] (checked by selftest()).
codes(u) gives the normalised FSQ codes q in {-1,-.5,0,.5}^{32x6}; d_vector(ids) the decoder condition (affine in q).

    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi("cuda:0"); u = P.phi_wave(wav16k)            # (32, 6) float32 numpy
    ids = P.tokens(u)                                            # (32,) int64
    python -m experiments.mist_w3po_v1.voicecond.bicodec_phi --selftest <wav> [<wav> ...]   # venvs/vllm_omni_030 python
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]


class BiCodecPhi:
    def __init__(self, device: str = "cuda:0", model_dir: str | None = None):
        sys.path.insert(0, str(REPO))
        from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
        from uniss.speech_tokenizer.bicodec.utils.audio import audio_volume_normalize
        self.device = torch.device(device)
        self.tok = BiCodecTokenizer(model_dir or str(REPO / "pretrained_models/UniSS/bicodec"), device=self.device)
        self.model = self.tok.model.eval()
        self.se = self.model.speaker_encoder
        self.rq = self.se.quantizer
        self.fsq = self.rq.layers[0]
        self._vn = audio_volume_normalize
        sr = int(self.tok.config["sample_rate"])
        hop = int(self.tok.config["latent_hop_length"])
        self.ref_len = int(sr * self.tok.config["ref_segment_duration"]) // hop * hop

    def ref_clip(self, wav: np.ndarray, normalize: bool = True) -> np.ndarray:
        w = np.asarray(wav, dtype=np.float64)
        if normalize:
            w = self._vn(w)
        if len(w) < self.ref_len:
            w = np.tile(w, 1 + self.ref_len // max(1, len(w)))
        return w[: self.ref_len]

    @torch.no_grad()
    def phi_wave(self, wav, normalize: bool = True) -> np.ndarray:
        """one waveform (16 kHz) or a list of them -> u (32,6) or (B,32,6)"""
        single = isinstance(wav, np.ndarray) and wav.ndim == 1
        ws = [wav] if single else list(wav)
        ref = torch.from_numpy(np.stack([self.ref_clip(w, normalize) for w in ws])).float().to(self.device)
        u = self._u_from_ref(ref).cpu().numpy().astype(np.float32)
        return u[0] if single else u

    def phi_path(self, path: str) -> np.ndarray:
        """exactly the tokenizer's file path: load_audio(sr, volume_normalize) -> ref clip -> u"""
        from uniss.speech_tokenizer.bicodec.utils.audio import load_audio
        w = load_audio(path, sampling_rate=int(self.tok.config["sample_rate"]),
                       volume_normalize=bool(self.tok.config["volume_normalize"]))
        return self.phi_wave(w, normalize=False)

    def _u_from_ref(self, ref: torch.Tensor) -> torch.Tensor:
        mel = self.model.mel_transformer(ref).squeeze(1)                 # (B, n_mels, T)
        _, feats = self.se.speaker_encoder(mel.transpose(1, 2), True)
        x = self.se.perceiver_sampler(feats.transpose(1, 2)).transpose(1, 2)   # (B, 128, 32) channel-first
        x = x.transpose(1, 2)                                               # (B, 32, 128)
        return self.rq.project_in(x)                                        # (B, 32, 6) pre-bound u

    @torch.no_grad()
    def tokens(self, u) -> np.ndarray:
        t = torch.as_tensor(np.asarray(u), dtype=torch.float32, device=self.device)
        single = t.ndim == 2
        t = t[None] if single else t
        _, idx = self.fsq(t)
        idx = idx.reshape(idx.shape[0], -1).long().cpu().numpy()
        return idx[0] if single else idx

    @torch.no_grad()
    def bounded(self, u) -> np.ndarray:
        t = torch.as_tensor(np.asarray(u), dtype=torch.float32, device=self.device)
        return self.fsq.bound(t).cpu().numpy()

    @torch.no_grad()
    def codes(self, ids) -> np.ndarray:
        t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.device)
        return self.fsq.indices_to_codes(t).cpu().numpy()

    @torch.no_grad()
    def d_vector(self, ids) -> np.ndarray:
        t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.device)
        t = t[None] if t.ndim == 1 else t
        return self.se.detokenize(t[:, :, None] if t.ndim == 2 else t).cpu().numpy()

    def selftest(self, paths: list[str]) -> bool:
        ok = True
        for p in paths:
            ref = np.asarray(self.tok.encode_wav_to_tokens(p).flatten().tolist()[:32])
            mine = self.tokens(self.phi_path(p))
            same = bool((ref == mine).all())
            ok &= same
            print(f"{'OK ' if same else 'DIFF'} {p} mismatches={int((ref != mine).sum())}")
        return ok


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest":
        P = BiCodecPhi("cuda:0" if torch.cuda.is_available() else "cpu")
        sys.exit(0 if P.selftest(sys.argv[2:]) else 1)
    print(__doc__)
