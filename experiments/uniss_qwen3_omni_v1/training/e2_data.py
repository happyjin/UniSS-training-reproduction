"""E2's data: UniST S2ST rows, fully shuffled, with the source's pitch and energy.

Rows come from stage 2's audio pool -- the 969,266 UniST utterances whose
source audio is local, both directions -- through Stage2Dataset's order: a
real permutation per pass, and with UNISS_RESHUFFLE_EACH_EPOCH=1 a fresh one
for every pass after the first. Each row carries what the E2 chain reads:
the source waveform (for the Thinker and for prosody), the translation, the
source's GLM-4 content tokens and 32 BiCodec global tokens, and the target's
BiCodec semantic codes in the source speaker's voice.

Prosody is measured here, in the dataloader workers, at the GLM-4 token
rate (12.5 a second, one feature row per content token) so it lands on the
positions UniSS already reads the source at.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.uniss_omni_s2st_v1.training.stage2_dataset import Stage2Dataset

PROSODY_DIM = 4
HOP = 160          # 10 ms at 16 kHz
FRAME = 1024


def prosody(wave: np.ndarray, rows: int, sr: int = 16_000) -> np.ndarray:
    """(rows, 4): voiced log-F0 z-score, voiced share, log-energy z-score,
    and the utterance's mean log-F0 relative to 150 Hz (its register).

    Voicing is by energy, as in the evaluation's pitch probe; YIN's own
    aperiodicity is noisier on this data.
    """
    import librosa

    if rows <= 0:
        return np.zeros((0, PROSODY_DIM), dtype=np.float32)
    wave = np.asarray(wave, dtype=np.float32)
    if wave.size < FRAME:
        wave = np.pad(wave, (0, FRAME - wave.size))
    f0 = librosa.yin(wave, fmin=60, fmax=500, sr=sr, frame_length=FRAME, hop_length=HOP)
    rms = librosa.feature.rms(y=wave, frame_length=FRAME, hop_length=HOP)[0][: f0.size]
    f0 = f0[: rms.size]
    positive = rms[rms > 0]
    voiced = rms > 0.3 * float(np.median(positive)) if positive.size else np.zeros_like(rms, dtype=bool)
    lf0 = np.log2(np.clip(f0, 40, 1000))
    if voiced.sum() >= 3:
        mu, sd = float(lf0[voiced].mean()), float(lf0[voiced].std()) + 1e-3
    else:
        mu, sd = float(np.log2(150.0)), 1.0
    z = np.where(voiced, (lf0 - mu) / sd, 0.0)
    energy = np.log(rms + 1e-5)
    energy = (energy - energy.mean()) / (energy.std() + 1e-3)
    out = np.zeros((rows, PROSODY_DIM), dtype=np.float32)
    for i, frames in enumerate(np.array_split(np.arange(rms.size), rows)):
        if frames.size == 0:
            continue
        v = voiced[frames]
        out[i, 0] = float(z[frames][v].mean()) if v.any() else 0.0
        out[i, 1] = float(v.mean())
        out[i, 2] = float(energy[frames].mean())
    out[:, 3] = mu - np.log2(150.0)
    return out


class E2Dataset(Stage2Dataset):
    """Stage2Dataset's order over the audio pool; E2's row and batch shape.

    Given a processor and the inference instruction (set by the trainer once
    the model is built, before the workers fork), collate also encodes the
    Thinker's side of the batch, so the mel extraction runs in the workers.
    """

    processor = None
    instruction = None
    # light: no Thinker input and no prosody -- for runs without the added
    # inputs, where UniSS alone trains; the source's duration then comes
    # from the index when it carries one, so no audio is decoded at all.
    light = False
    # "released": UniST's source_glm; "teacher": the index's re-tokenised GLM.
    glm_source = "released"

    def __getitem__(self, index: int) -> dict:
        step, position = divmod(index, self.global_batch)
        step = min(step, len(self.schedule) - 1)
        cursor = self.offsets[step] + position
        audio_row = self._row("audio", cursor)
        row = self.corpus.audio_sample(audio_row)
        rows = self.corpus.audio_rows
        i = audio_row % len(rows)
        glm = rows.teacher_glm(i) if self.glm_source == "teacher" else row["source_glm"]
        source_ms = getattr(rows, "source_ms", None)
        return {
            "audio": row["audio"],
            "translation": row["target_text"],
            "source_lang": row["source_lang"],
            "target_lang": row["target_lang"],
            "codes": row["codes"],
            "globals": row["globals"],
            "source_glm": glm,
            "source_seconds": None if source_ms is None else float(source_ms[i]) / 1000,
        }

    def collate(self, samples: list[dict]) -> dict:
        glm = [np.asarray(s["source_glm"], dtype=np.int64) for s in samples]
        if self.light and all(s.get("source_seconds") is not None for s in samples):
            return {
                "thinker": None, "waveforms": None, "prosody": None,
                "translation": [s["translation"] for s in samples],
                "source_lang": [s["source_lang"] for s in samples],
                "target_lang": [s["target_lang"] for s in samples],
                "codes": [torch.as_tensor(np.asarray(s["codes"], dtype=np.int64)) for s in samples],
                "globals": torch.stack([torch.as_tensor(np.asarray(s["globals"], dtype=np.int64)) for s in samples]),
                "source_glm": [torch.as_tensor(g) for g in glm],
                "source_seconds": torch.tensor([s["source_seconds"] for s in samples], dtype=torch.float32),
            }
        waves = [self.corpus.waveform(s["audio"])[0] for s in samples]
        thinker = None
        if self.processor is not None and not self.light:
            from experiments.uniss_qwen3_omni_v1.training.e2_batch import LANG3, encode_thinker
            thinker = encode_thinker(self.processor, self.instruction, [LANG3[s["target_lang"]] for s in samples],
                                     waves, [s["translation"] for s in samples])
        return {
            "thinker": thinker,
            "waveforms": waves,
            "translation": [s["translation"] for s in samples],
            "source_lang": [s["source_lang"] for s in samples],
            "target_lang": [s["target_lang"] for s in samples],
            "codes": [torch.as_tensor(np.asarray(s["codes"], dtype=np.int64)) for s in samples],
            "globals": torch.stack([torch.as_tensor(np.asarray(s["globals"], dtype=np.int64)) for s in samples]),
            "source_glm": [torch.as_tensor(g) for g in glm],
            "prosody": None if self.light else [torch.as_tensor(prosody(w, g.size)) for w, g in zip(waves, glm)],
            "source_seconds": torch.tensor([w.size / 16_000 for w in waves], dtype=torch.float32),
        }
