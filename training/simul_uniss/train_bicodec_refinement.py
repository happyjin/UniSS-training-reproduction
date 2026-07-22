"""Chunk-boundary refinement for BiCodec prenet/decoder without changing the base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class BiCodecChunkDataset(Dataset):
    def __init__(self, manifest_path: str | Path, chunk_tokens: int = 80) -> None:
        self.path = Path(manifest_path)
        self.chunk_tokens = chunk_tokens
        self.offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"{self.path} contains no records")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            item = json.loads(handle.readline().decode("utf-8"))
        semantic = [int(token) for token in item["target_bicodec"]]
        if not semantic:
            raise ValueError("target_bicodec must be non-empty")
        length = min(self.chunk_tokens, len(semantic))
        max_start = len(semantic) - length
        start = 0 if max_start == 0 else (index * 104729) % (max_start + 1)
        end = start + length
        waveform, sample_rate = torchaudio.load(item["target_audio"])
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_start = start * 320
        sample_end = min(waveform.shape[-1], end * 320)
        reference = waveform[:1, sample_start:sample_end].squeeze(0)
        return {
            "semantic": torch.tensor(semantic[start:end], dtype=torch.long),
            "global": torch.tensor(item["bicodec_global"], dtype=torch.long),
            "reference": reference,
        }


def collate_bicodec(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    semantic_lengths = torch.tensor([len(item["semantic"]) for item in batch], dtype=torch.long)
    audio_lengths = torch.tensor([len(item["reference"]) for item in batch], dtype=torch.long)
    semantic = torch.zeros(len(batch), int(semantic_lengths.max()), dtype=torch.long)
    reference = torch.zeros(len(batch), int(audio_lengths.max()), dtype=torch.float32)
    global_tokens = torch.stack([item["global"] for item in batch])
    for index, item in enumerate(batch):
        semantic[index, : len(item["semantic"])] = item["semantic"]
        reference[index, : len(item["reference"])] = item["reference"]
    return {
        "semantic": semantic,
        "semantic_lengths": semantic_lengths,
        "global": global_tokens,
        "reference": reference,
        "audio_lengths": audio_lengths,
    }


def multi_resolution_stft_loss(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    losses = []
    for n_fft, hop in ((256, 64), (512, 128), (1024, 256)):
        if prediction.shape[-1] < n_fft:
            continue
        window = torch.hann_window(n_fft, device=prediction.device)
        pred = torch.stft(prediction, n_fft, hop, window=window, return_complex=True)
        ref = torch.stft(reference, n_fft, hop, window=window, return_complex=True)
        losses.append(F.l1_loss(torch.log1p(pred.abs()), torch.log1p(ref.abs())))
    return torch.stack(losses).mean() if losses else prediction.new_zeros(())


def boundary_loss(prediction: torch.Tensor, reference: torch.Tensor, boundary_samples: int = 1600) -> torch.Tensor:
    length = min(boundary_samples, prediction.shape[-1], reference.shape[-1])
    if length < 2:
        return prediction.new_zeros(())
    pred_start = prediction[..., :length]
    ref_start = reference[..., :length]
    pred_end = prediction[..., -length:]
    ref_end = reference[..., -length:]
    value = F.l1_loss(pred_start, ref_start) + F.l1_loss(pred_end, ref_end)
    derivative = F.l1_loss(torch.diff(pred_start), torch.diff(ref_start))
    derivative = derivative + F.l1_loss(torch.diff(pred_end), torch.diff(ref_end))
    return value + derivative


def differentiable_detokenize(model, semantic: torch.Tensor, global_tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        z_q = model.quantizer.detokenize(semantic)
        d_vector = model.speaker_encoder.detokenize(global_tokens.unsqueeze(1))
    x = model.prenet(z_q, d_vector)
    x = x + d_vector.unsqueeze(-1)
    return model.decoder(x).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bicodec-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--chunk-tokens", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from uniss.speech_tokenizer.bicodec.models.bicodec import BiCodec

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = BiCodec.load_from_checkpoint(Path(args.bicodec_checkpoint)).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.prenet, model.decoder):
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    dataset = BiCodecChunkDataset(args.manifest, args.chunk_tokens)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_bicodec)
    writer = SummaryWriter(args.tensorboard_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    while step < args.max_steps:
        for batch in loader:
            step += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = differentiable_detokenize(model, batch["semantic"], batch["global"])
            common = min(prediction.shape[-1], batch["reference"].shape[-1])
            prediction = prediction[..., :common]
            reference = batch["reference"][..., :common]
            waveform_loss = F.l1_loss(prediction, reference)
            stft_loss = multi_resolution_stft_loss(prediction, reference)
            edge_loss = boundary_loss(prediction, reference)
            total = waveform_loss + stft_loss + 0.5 * edge_loss
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if step % args.log_interval == 0 or step == 1:
                writer.add_scalar("stage5_refine/total_loss", float(total), step)
                writer.add_scalar("stage5_refine/waveform_loss", float(waveform_loss), step)
                writer.add_scalar("stage5_refine/stft_loss", float(stft_loss), step)
                writer.add_scalar("stage5_refine/boundary_loss", float(edge_loss), step)
                writer.add_scalar("stage5_refine/grad_norm", float(grad_norm), step)
                writer.flush()
                print(
                    json.dumps(
                        {
                            "step": step,
                            "total": float(total),
                            "waveform": float(waveform_loss),
                            "stft": float(stft_loss),
                            "boundary": float(edge_loss),
                        }
                    ),
                    flush=True,
                )
            if step % args.save_interval == 0 or step == args.max_steps:
                torch.save(
                    {
                        "prenet": model.prenet.state_dict(),
                        "decoder": model.decoder.state_dict(),
                        "step": step,
                        "args": vars(args),
                    },
                    output_dir / "bicodec_streaming_refinement.pt",
                )
            if step >= args.max_steps:
                break
    writer.close()
    print(json.dumps({"status": "complete", "step": step}))


if __name__ == "__main__":
    main()
