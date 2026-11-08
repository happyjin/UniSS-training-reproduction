"""Decode-time loop guard (2026-11-08; inference only, off by default). The dev tcpBLEU swings of up to 8 BLEU between two
decodes of the same model come from a few windows that fall into repetition loops ("with the with the ...", thousands
of words). Two rules on the ids a read is about to commit:

  periodic run   the first position i where some period p (1..max_p tokens) repeats >= k times in a row
                 (ids[i:i+p] == ids[i+p:i+2p] == ...): the block is cut after the first copy (i + p)
  budget         the window's committed ids may not exceed tokens_per_s * audio seconds + slack; the excess is cut

`guard(new, committed, k, max_p, budget)` returns the (possibly shortened) new ids and the reason ("" = untouched).
"""
from __future__ import annotations


def first_periodic_run(ids: list[int], k: int, max_p: int) -> tuple[int, int]:
    """(i, p) of the earliest run of >= k consecutive copies of a p-token pattern, or (-1, 0)."""
    n = len(ids)
    for i in range(n):
        for p in range(1, max_p + 1):
            if i + k * p > n:
                break
            pat = ids[i:i + p]
            if all(ids[i + j * p:i + (j + 1) * p] == pat for j in range(1, k)):
                return i, p
    return -1, 0


def guard(new: list[int], committed: list[int], k: int, max_p: int, budget: int | None) -> tuple[list[int], str]:
    reason = ""
    if k > 0:
        # look across the boundary too: a loop that started in earlier reads continues in this one
        tail = committed[-k * max_p:] if committed else []
        seq = tail + new
        i, p = first_periodic_run(seq, k, max_p)
        if i >= 0:
            cut = max(0, i + p - len(tail))
            new, reason = new[:cut], f"loop p={p}"
    if budget is not None and len(committed) + len(new) > budget:
        new, reason = new[: max(0, budget - len(committed))], (reason + " budget").strip()
    return new, reason
