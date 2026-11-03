#!/usr/bin/env python3
"""The established trainer, with the tensor sharing strategy set for workers.

This experiment raises the dataloader worker count from the 0 that every
continuation launcher in this lineage hardcoded to the 8 that
`run_e2e_megatron.sh` has always defaulted to, because the parent run logged 47%
mean GPU utilisation with 46% of samples at zero.

`runtime_dataset.__getitem__` reopens the file per access and holds no handle
across calls, so worker processes are safe from that angle -- but the first
attempt still died at startup with `RuntimeError: received 0 items of ancdata`.
That is not the open-file limit, which is already 1,048,576 here; it is
PyTorch's default `file_descriptor` sharing strategy, which passes one
descriptor per shared tensor over the worker socket and exhausts the per-message
ancillary-data budget when a batch carries many tensors.  `file_system` shares
through /dev/shm names instead and has no such per-message limit.

The strategy has to be set in the parent process before any worker forks, and it
cannot be set from the environment, so this entrypoint sets it and then hands
over to the established `main` unchanged.  No objective, weight or data path is
touched: this file exists only so that `--num-workers` can be non-zero.
"""

from __future__ import annotations

import torch.multiprocessing as multiprocessing

import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron as trainer

SHARING_STRATEGY = "file_system"


def install() -> str:
    available = multiprocessing.get_all_sharing_strategies()
    if SHARING_STRATEGY not in available:
        raise SystemExit(
            f"{SHARING_STRATEGY} sharing is unavailable; have {sorted(available)}"
        )
    multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
    return multiprocessing.get_sharing_strategy()


def main() -> None:
    active = install()
    print(f'{{"tensor_sharing_strategy": "{active}"}}', flush=True)
    trainer.main()


if __name__ == "__main__":
    main()
