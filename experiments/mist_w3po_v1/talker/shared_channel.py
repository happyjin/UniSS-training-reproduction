"""One shared output channel per meeting (reports/mist_w3po_v1/talker_crf_prereg_v2.zh-CN.md, experiment 2): every
speaker's translated speech plays on ONE channel (one interpreter voice). Shared by the training rows (talker_rows
--scope meeting), the precheck (precheck_plcrf select-meeting) and inference (si_infer --backlog-scope meeting), so the
three use the same timeline:

  * a block of stream k (= window x speaker) is available at a = window start + speech_latency_lit's online protocol
    (max(write_ms, open_detect_ms); a block written at/after the window end also >= close_detect_ms), in ms;
  * a meeting's blocks play FIFO in (a, stream id, block index) order -- the order speech_latency_lit's online_1ch
    schedule uses -- each from max(a, end of whatever was playing);
  * the shared backlog of a block = max(0, channel end - a) before it plays; its token bin is si_infer.backlog_bin's
    (0.5 s bins, capped at 7).

CPU only, no heavy imports.
"""
from __future__ import annotations

from collections import defaultdict


def availability(w: dict, write_ms: float) -> float:
    """speech_latency_lit.online / talker_rows.availability (window-relative ms)."""
    c = max(write_ms, w["open_detect_ms"])
    if write_ms >= w["dur_ms"]:
        c = max(c, w["close_detect_ms"])
    return float(c)


def abs_avail_ms(w: dict, write_ms: float) -> float:
    """meeting-absolute availability (ms)."""
    return w["start_s"] * 1000.0 + availability(w, write_ms)


def window_of(item_id: str) -> str:
    return item_id.split("__S")[0]


def backlog_bin(backlog_ms: float) -> int:
    """si_infer.backlog_bin's binning of a backlog: 0.5 s bins, capped at 7."""
    return min(7, int(max(0, backlog_ms) / 500))


class SharedChannel:
    """one output channel: blocks are played in the order play() is called, each from max(a, end)."""

    def __init__(self):
        self.end = None

    def backlog(self, a: float) -> float:
        return 0.0 if self.end is None else max(0.0, self.end - a)

    def play(self, a: float, dur_ms: float) -> float:
        """-> the block's start."""
        st = a if self.end is None else max(a, self.end)
        self.end = st + dur_ms
        return st


def meeting_queues(items: list[dict], wins: dict) -> dict:
    """meeting -> FIFO list of (a_ms, item index, block index) over the items' blocks (item: {id, blocks[{write_ms}]});
    stream ties broken by item id then block index, as speech_latency_lit."""
    q = defaultdict(list)
    for n, it in enumerate(items):
        w = wins[window_of(it["id"])]
        for r, b in enumerate(it["blocks"]):
            q[w["meeting"]].append((abs_avail_ms(w, b["write_ms"]), it["id"], r, n))
    return {m: [(a, n, r) for a, _, r, n in sorted(L, key=lambda x: (x[0], x[1], x[2]))] for m, L in sorted(q.items())}


def meeting_shard_of(items: list[dict], wins: dict, num_shards: int) -> dict:
    """meeting -> shard: longest-processing-time first over the meetings' block counts (a meeting is sequential on the
    shared channel, so it stays whole in one shard); deterministic (ties by meeting name)."""
    nb = defaultdict(int)
    for it in items:
        nb[wins[window_of(it["id"])]["meeting"]] += len(it["blocks"])
    load = [0] * num_shards
    out = {}
    for m in sorted(nb, key=lambda m: (-nb[m], m)):
        k = min(range(num_shards), key=lambda k: (load[k], k))
        out[m] = k
        load[k] += nb[m]
    return out


def shared_backlogs(items: list[dict], wins: dict, durs: dict) -> dict:
    """(item index, block index) -> shared backlog (ms) before the block plays, given each block's duration
    durs[(item index, block index)] in ms (e.g. the A' rule path's chosen candidate)."""
    out = {}
    for m, L in meeting_queues(items, wins).items():
        ch = SharedChannel()
        for a, n, r in L:
            out[(n, r)] = ch.backlog(a)
            ch.play(a, durs[(n, r)])
    return out
