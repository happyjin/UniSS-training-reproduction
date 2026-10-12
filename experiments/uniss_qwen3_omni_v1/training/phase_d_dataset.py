"""Ordering for Phase D: a true shuffle that does not pay for it twice.

Code lengths run from 4 to 600 against a median of 243, and a micro batch
pads to its longest member. Drawing uniformly, the measured arithmetic is:

    pure permutation     46.8% of the compute is real, the rest padding
    length-sorted        88.4%

Stage 1 answered that by sorting the *whole pool* by length and shuffling
the resulting chunks, which keeps batches homogeneous but fixes their
membership -- the same 384 rows travel together every epoch.

This does it the other way round. The pool is permuted in full, so which
rows share a batch is uniformly random and different every epoch; then
each global batch is sorted internally before Megatron deals it to the
ranks, so each rank's micro batch is length-homogeneous. The only thing
that is not random is which rank receives which lengths, and gradients are
summed across ranks, so that is not a statistical property at all.

It also avoids the cost stage 1 recorded for its own bucketing. Sorting
the whole pool makes every row in an optimiser step nearly the same
length, and stage 1's note says dev sat about 0.19 worse for six
consecutive evaluations after that change. Here the global batch is a
uniform draw, so a step still spans the whole length distribution; only
the assignment within it is ordered.

This relies on ``MegatronPretrainingSampler`` (``--dataloader-type
single``) handing out contiguous aligned ranges, which is the same
assumption stage 1's bucketing already made.
"""

from __future__ import annotations

import numpy as np

from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import OmniTtsDataset


def shuffled_sorted_order(
    lengths: np.ndarray, *, global_batch: int, seed: int
) -> np.ndarray:
    """Permute everything, then sort inside each global batch."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(lengths.size)
    usable = (order.size // global_batch) * global_batch
    if usable == 0:
        raise ValueError(
            f"{lengths.size} rows cannot fill one global batch of {global_batch}"
        )
    head = order[:usable].reshape(-1, global_batch)
    # argsort on the lengths of each row of the batch, applied to the
    # indices: within a batch the shortest go to rank 0 and the longest to
    # the last rank, so no rank waits on a neighbour's padding.
    inside = np.argsort(lengths[head], axis=1, kind="stable")
    head = np.take_along_axis(head, inside, axis=1).reshape(-1)
    # The remainder would be a short final batch; Megatron's sampler drops
    # a partial batch anyway, so it is left out rather than left ragged.
    return head


class PhaseDDataset(OmniTtsDataset):
    """``OmniTtsDataset`` with the ordering above instead of bucketing."""

    def __init__(self, rows, *, global_batch: int, epochs: int = 1,
                 seed: int = 20261012, **kwargs):
        kwargs.pop("bucket", None)
        kwargs.pop("micro_batch", None)
        super().__init__(rows, bucket=False, epochs=1, seed=seed, **kwargs)
        # super() built an identity order over the rows that survived the
        # text-length filter; those are the only rows that may be drawn.
        kept = np.asarray(self.order)
        if hasattr(rows, "code_lengths"):
            lengths = np.asarray(rows.code_lengths())[kept]
        else:
            lengths = np.asarray([len(rows[int(i)]["codes"]) for i in kept])
        orders = []
        for epoch in range(epochs):
            local = shuffled_sorted_order(
                lengths, global_batch=global_batch, seed=seed + epoch)
            orders.append(kept[local])
        self.order = np.concatenate(orders)
        self.global_batch = int(global_batch)
        self.epochs_built = int(epochs)
