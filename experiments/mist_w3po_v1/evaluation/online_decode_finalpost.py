"""online_decode.py + --final-read-post (LOG 2026-11-09 19:35 UTC; reports/mist_w3po_v1/sched_precheck_v8.zh-CN.md §8):
at a window's FINAL read only, the diarizer posteriors P are visible up to min(t, close_detect_ms - 1.04 s)
(window-relative; t = the final read time = the window duration) instead of t - 1.04 s.

Why it is causal and costs no latency: the online scorer stamps the final read's output at max(write, close_detect_ms)
(reports/gstar_si_v8/overlap_score.py:141-144), i.e. the final read happens at the close detection, and a lowlat P
frame is known engine.LOOKAHEAD_MS (1.04 s) after its end (Engine.audio_embeds' own rule). On dev, 93.6 % of the online
windows have close_detect - 1.04 s >= duration (all in-window frames visible; the last, partial 80 ms frame that
straddles the window end stays masked, as min(t, ...) caps the cut at t).

  --final-read-post off            (default) exactly online_decode.main: nothing is patched, argv passed unchanged
  --final-read-post close_detect   Engine.read is wrapped; each item with final=True and full=False is replaced by a
                                   copy whose post is zeroed from frame cut // 80 on (the engine's own indexing),
                                   cut = max(t - LOOKAHEAD_MS, min(t, close_detect_ms - LOOKAHEAD_MS)), with full=True
                                   so audio_embeds applies no second mask. Non-final reads, the audio, the prompt ids
                                   and every other argument are unchanged (test_online_decode_finalpost.py).
All other arguments go to online_decode unchanged (--windows is also read here, for close_detect_ms per window id;
the window id is the item's akey "<id>|<t>").

    python -m experiments.mist_w3po_v1.evaluation.online_decode_finalpost --final-read-post close_detect \
        <online_decode arguments>
"""
from __future__ import annotations

import dataclasses
import json
import sys

from experiments.mist_w3po_v1.core import engine as E
from experiments.mist_w3po_v1.evaluation import online_decode as od

MODES = ("off", "close_detect")
STATS = {"final": 0, "extended": 0, "all_in_window": 0}


def split_argv(argv: list[str]) -> tuple[str, list[str]]:
    """-> (mode, the remaining argv in order); accepts '--final-read-post X' and '--final-read-post=X'."""
    mode, rest, i = "off", [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--final-read-post":
            if i + 1 >= len(argv):
                raise SystemExit("--final-read-post needs a value")
            mode, i = argv[i + 1], i + 2
            continue
        if a.startswith("--final-read-post="):
            mode, i = a.split("=", 1)[1], i + 1
            continue
        rest.append(a)
        i += 1
    if mode not in MODES:
        raise SystemExit(f"--final-read-post must be one of {MODES}, got {mode!r}")
    return mode, rest


def windows_arg(argv: list[str]) -> str:
    for i, a in enumerate(argv):
        if a == "--windows" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--windows="):
            return a.split("=", 1)[1]
    raise SystemExit("--windows is required")


def load_close_detect(path: str) -> dict[str, int]:
    return {w["id"]: int(w["close_detect_ms"]) for w in map(json.loads, open(path, encoding="utf-8"))}


def final_cut_ms(t_ms: int, close_detect_ms: int) -> int:
    """the P visibility cut of a final read at t (window-relative ms); never earlier than the engine's t - 1.04 s."""
    return max(0, t_ms - E.LOOKAHEAD_MS, min(t_ms, close_detect_ms - E.LOOKAHEAD_MS))


def transform(items: list, cd: dict[str, int]) -> list:
    """final, non-oracle items -> copies with P masked at final_cut_ms and full=True; all others: the same objects."""
    out = []
    for it in items:
        if it.final and not it.full:
            wid, t = it.akey.rsplit("|", 1)
            assert int(t) == it.t_ms, (it.akey, it.t_ms)
            cut = final_cut_ms(it.t_ms, cd[wid])
            post = it.post.copy()
            post[cut // 80:] = 0.0
            STATS["final"] += 1
            STATS["extended"] += int(cut // 80 > max(0, it.t_ms - E.LOOKAHEAD_MS) // 80)
            STATS["all_in_window"] += int(cut >= it.t_ms)
            it = dataclasses.replace(it, post=post, full=True)
        out.append(it)
    return out


def install(cd: dict[str, int]):
    """wrap Engine.read (class attribute, shared by online_decode's import); returns the original for tests."""
    orig = E.Engine.read

    def read(self, items, *a, **k):
        return orig(self, transform(items, cd), *a, **k)

    read.__wrapped__ = orig
    E.Engine.read = read
    return orig


def main() -> None:
    mode, rest = split_argv(sys.argv[1:])
    if mode == "close_detect":
        install(load_close_detect(windows_arg(rest)))
        print(f"final-read-post close_detect: final reads see P up to min(t, close_detect - {E.LOOKAHEAD_MS} ms)",
              flush=True)
    sys.argv = [sys.argv[0]] + rest
    od.main()
    if mode == "close_detect":
        print(f"FINALPOST_STATS {json.dumps(STATS)}", flush=True)


if __name__ == "__main__":
    main()
