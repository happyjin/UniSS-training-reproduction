"""Storage abstraction for autoregressively generated codec frames."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch


class CodecFrameBuffer:
    """Store ``[main, residual...]`` frames in lists or a CUDA tensor."""

    def __init__(
        self,
        max_frames: int,
        residual_groups: int,
        *,
        device: torch.device,
        use_gpu: bool,
    ) -> None:
        self.max_frames = int(max_frames)
        self.residual_groups = int(residual_groups)
        self.device = torch.device(device)
        self.use_gpu = bool(use_gpu)
        if self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if self.residual_groups < 1:
            raise ValueError("residual_groups must be positive")
        if self.use_gpu and self.device.type != "cuda":
            raise ValueError("GPU codec buffer requires a CUDA device")
        self._length = 0
        if self.use_gpu:
            self._storage = torch.empty(
                self.max_frames,
                self.residual_groups + 1,
                dtype=torch.long,
                device=self.device,
            )
            self._main_codes = None
            self._residual_codes = None
        else:
            self._storage = None
            self._main_codes: list[int] = []
            self._residual_codes: list[list[int]] = []

    def __len__(self) -> int:
        return self._length

    def append(
        self, main_code: torch.Tensor, residual_codes: torch.Tensor
    ) -> None:
        """Append one single-utterance codec frame."""
        if self._length >= self.max_frames:
            raise IndexError("codec frame buffer capacity exceeded")
        main = torch.as_tensor(main_code, device=self.device, dtype=torch.long)
        residual = torch.as_tensor(
            residual_codes, device=self.device, dtype=torch.long
        )
        if main.numel() != 1:
            raise ValueError("main_code must contain exactly one token")
        if residual.numel() != self.residual_groups:
            raise ValueError(
                f"residual_codes must contain {self.residual_groups} tokens"
            )
        if self.use_gpu:
            assert self._storage is not None
            self._storage[self._length, 0] = main.reshape(-1)[0]
            self._storage[self._length, 1:] = residual.reshape(-1)
        else:
            assert self._main_codes is not None
            assert self._residual_codes is not None
            self._main_codes.append(int(main.reshape(-1)[0].item()))
            self._residual_codes.append(
                [int(value) for value in residual.reshape(-1).cpu().tolist()]
            )
        self._length += 1

    def main_history(self) -> torch.LongTensor:
        """Return main-code history as ``[1, frames]`` on ``self.device``."""
        if self.use_gpu:
            assert self._storage is not None
            return self._storage[: self._length, 0].unsqueeze(0)
        assert self._main_codes is not None
        return torch.tensor(
            [self._main_codes], dtype=torch.long, device=self.device
        )

    def last_main(self) -> Union[int, torch.LongTensor]:
        """Return a host scalar in list mode or a device tensor in GPU mode."""
        if self._length == 0:
            raise IndexError("codec frame buffer is empty")
        if self.use_gpu:
            assert self._storage is not None
            return self._storage[self._length - 1, 0].reshape(1)
        assert self._main_codes is not None
        return self._main_codes[-1]

    def window(self, start: int = 0) -> torch.LongTensor:
        """Return frames ``[start:, main+residual]`` on ``self.device``."""
        start = int(start)
        if start < 0 or start > self._length:
            raise IndexError("codec frame window start is out of range")
        if self.use_gpu:
            assert self._storage is not None
            return self._storage[start : self._length]
        assert self._main_codes is not None
        assert self._residual_codes is not None
        rows = [
            [main, *residual]
            for main, residual in zip(
                self._main_codes[start:], self._residual_codes[start:]
            )
        ]
        if not rows:
            return torch.empty(
                0,
                self.residual_groups + 1,
                dtype=torch.long,
                device=self.device,
            )
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def to_numpy(self) -> np.ndarray:
        """Transfer all frames to one contiguous host array."""
        if self._length == 0:
            return np.empty((0, self.residual_groups + 1), dtype=np.int64)
        if self.use_gpu:
            assert self._storage is not None
            return (
                self._storage[: self._length]
                .detach()
                .cpu()
                .contiguous()
                .numpy()
            )
        assert self._main_codes is not None
        assert self._residual_codes is not None
        return np.asarray(
            [
                [main, *residual]
                for main, residual in zip(
                    self._main_codes, self._residual_codes
                )
            ],
            dtype=np.int64,
        )


__all__ = ["CodecFrameBuffer"]
