"""W3PO rollouts (plan §3.1, §5): G sampled continuations of the wait policy per window, every read of every member
batched together (members of one window share the audio encoding at each read: engine akey), plus the same-model
fixed-lag anchors of FLAD (C3: greedy, adapters off, policy lag, cached per window).

A trajectory keeps, per read, everything needed to recompute its log-probabilities by teacher forcing (prompt =
audio up to t, ctx, committed ids before the read; block = the generated ids) and to route credit:
  segments   parsed from the committed ids exactly like engine.Parser (a speaker token opens a segment, the first time
             token is its start, the second closes it; an unclosed segment is closed by its length at the next speaker
             token, or at the window's end), each with its read index r, commit time t (s) and its token positions
  tok_seg    per read, per block token: the segment index it belongs to (-1: none)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch

from experiments.mist_w3po_v1.core.engine import (
    EOS_ID, NEW0, NEW_END, SPK0, WAIT_ID, Engine, ReadItem, lag_cut, read_times)


@dataclass
class Read:
    t_ms: int
    final: bool
    before: list            # committed ids before this read
    ids: list               # generated block (policy) / committed block + terminator (anchor)
    when: list
    who: list
    logp: list
    logp_wait: list
    logp_who: list
    akey: str


@dataclass
class Traj:
    wid: str
    kind: str               # "policy" or "anchor<lag ms>"
    ctx: str
    reads: list = field(default_factory=list)
    segs: list = field(default_factory=list)
    tok_seg: list = field(default_factory=list)
    wait_reads: list = field(default_factory=list)
    stop_reason: list = field(default_factory=list)


def load_window(w: dict, post_key: str = "lowlat"):
    x, sr = sf.read(w["audio"], dtype="float32", always_2d=True)
    assert sr == 16000
    return x.mean(1), np.load(w["post"][post_key]).astype(np.float32)


def parse(tr: Traj, tok, dur_s: float) -> None:
    """Fill tr.segs / tr.tok_seg from tr.reads (engine.Parser semantics)."""
    segs, tok_seg = [], []
    cur = None   # {"spk", "a", "text_ids", "toks", "r", "open"}

    def close(end_s, r, t):
        text = tok.decode(cur["text_ids"], skip_special_tokens=False).strip()
        if text:
            segs.append({"spk": cur["spk"], "a": cur["a"], "b": end_s, "text": text, "r": r, "t": t,
                         "toks": cur["toks"]})
            return len(segs) - 1
        return -1

    pending = []   # token positions of the current segment, to relabel when it closes
    for r, rd in enumerate(tr.reads):
        ts = []
        t = rd.t_ms / 1000
        for p, i in enumerate(rd.ids):
            ts.append(-1)
            if SPK0 <= i < NEW_END:
                if cur is not None and cur["open"]:
                    k = close(round(cur["a"] + max(0.5, 0.25 * len(tok.decode(cur["text_ids"]).strip())), 2), r, t)
                    for (rr, pp) in pending:
                        (tok_seg[rr] if rr < r else ts)[pp] = k
                cur = {"spk": i - SPK0 + 1, "a": None, "text_ids": [], "toks": [], "open": False}
                pending = [(r, p)]
            elif NEW0 <= i < SPK0 and cur is not None:
                tm = round((i - NEW0) * 0.08, 2)
                pending.append((r, p))
                if cur["a"] is None:
                    cur["a"], cur["open"] = tm, True
                elif cur["open"]:
                    k = close(tm, r, t)
                    for (rr, pp) in pending:
                        (tok_seg[rr] if rr < r else ts)[pp] = k
                    cur, pending = None, []
            elif i not in (EOS_ID, WAIT_ID) and cur is not None:
                cur["text_ids"].append(i)
                pending.append((r, p))
        tok_seg.append(ts)
    if cur is not None and cur["open"] and tr.reads:
        r = len(tr.reads) - 1
        k = close(dur_s, r, tr.reads[-1].t_ms / 1000)
        for (rr, pp) in pending:
            tok_seg[rr][pp] = k
    for s in segs:
        s.pop("toks", None)
    tr.segs, tr.tok_seg = segs, tok_seg
    tr.wait_reads = [r for r, rd in enumerate(tr.reads) if not rd.final and rd.ids and rd.ids[-1] == WAIT_ID]


@torch.no_grad()
def rollout(eng: Engine, windows: list[dict], ctxs: dict, G: int, *, temp: float, temp_when: float,
            gen: torch.Generator, max_new: int = 300, cache: dict | None = None) -> dict:
    """G sampled wait-policy trajectories per window. Returns {wid: [Traj] * G}."""
    data = {w["id"]: load_window(w) for w in windows}
    trajs, states = {}, []
    for w in windows:
        trajs[w["id"]] = []
        for g in range(G):
            tr = Traj(w["id"], "policy", ctxs.get(w["id"], ""))
            trajs[w["id"]].append(tr)
            states.append({"w": w, "tr": tr, "committed": [], "times": read_times(w["dur_ms"]), "r": 0})
    while True:
        live = [s for s in states if s["r"] < len(s["times"])]
        if not live:
            break
        items = []
        for s in live:
            t = s["times"][s["r"]]
            wav, post = data[s["w"]["id"]]
            items.append(ReadItem(wav=wav, post=post, t_ms=t, final=s["r"] == len(s["times"]) - 1,
                                  committed=s["committed"], ctx=s["tr"].ctx, akey=f"{s['w']['id']}|{t}"))
        outs = eng.read(items, max_new=max_new, temp=temp, temp_when=temp_when, gen=gen, record=True, cache=cache)
        for s, it, o in zip(live, items, outs):
            s["tr"].reads.append(Read(it.t_ms, it.final, list(s["committed"]), o.ids, o.when, o.who, o.logp,
                                      o.logp_wait, o.logp_who, it.akey))
            s["committed"] = s["committed"] + [i for i in o.ids if i not in (EOS_ID, WAIT_ID)]
            s["r"] += 1
    for w in windows:
        for tr in trajs[w["id"]]:
            parse(tr, eng.tok, w["dur_ms"] / 1000)
    return trajs


@torch.no_grad()
def anchors(eng: Engine, windows: list[dict], ctxs: dict, lags_ms: list[int], max_new: int = 300,
            cache: dict | None = None) -> dict:
    """Greedy fixed-lag decodes (gstar_si_stream_online_v6s --policy lag) per window and lag. The committed part of
    each read is stored as the block a wait-policy model would emit for it: new segments + <wait> (<eos> at the end)."""
    data = {w["id"]: load_window(w) for w in windows}
    out, states = {}, []
    for w in windows:
        out[w["id"]] = []
        for lag in lags_ms:
            tr = Traj(w["id"], f"anchor{lag}", ctxs.get(w["id"], ""))
            out[w["id"]].append(tr)
            states.append({"w": w, "tr": tr, "lag": lag, "committed": [], "times": read_times(w["dur_ms"]), "r": 0})
    while True:
        live = [s for s in states if s["r"] < len(s["times"])]
        if not live:
            break
        items = []
        for s in live:
            t = s["times"][s["r"]]
            wav, post = data[s["w"]["id"]]
            items.append(ReadItem(wav=wav, post=post, t_ms=t, final=s["r"] == len(s["times"]) - 1,
                                  committed=s["committed"], ctx=s["tr"].ctx, akey=f"{s['w']['id']}|{t}",
                                  bar_wait=True, stop_ms=t - s["lag"]))
        outs = eng.read(items, max_new=max_new, cache=cache)
        for s, it, o in zip(live, items, outs):
            body = [i for i in o.ids if i not in (EOS_ID, WAIT_ID)]
            new = body if it.final else body[: lag_cut(body, it.t_ms, s["lag"])]
            block = new + [EOS_ID if it.final else WAIT_ID]
            n = len(block)
            s["tr"].reads.append(Read(it.t_ms, it.final, list(s["committed"]), block, [False] * n, [False] * n,
                                      [0.0] * n, [0.0] * n, [0.0] * n, it.akey))
            s["committed"] = s["committed"] + new
            s["r"] += 1
    for w in windows:
        for tr in out[w["id"]]:
            parse(tr, eng.tok, w["dur_ms"] / 1000)
    return out


def branch_rollout(eng: Engine, windows: dict, specs: list, *, temp: float, temp_when: float,
                   gen: torch.Generator, max_new: int = 300, cache: dict | None = None) -> list:
    """--ecf-branches (2026-11-09): counterfactual "flush now" branches. specs: [(wid, parent Traj, r)]; each branch
    copies the parent's reads before r (and so its committed ids and ctx), takes read r with <wait> barred (the
    engine's bar_wait: say everything heard so far) and then samples the remaining reads as usual. Returns the branch
    Trajs (kind "branch"), parsed, in the order of specs. Not called unless --ecf-branches > 0."""
    data = {wid: load_window(windows[wid]) for wid in {s[0] for s in specs}}
    out, states = [], []
    for wid, par, r in specs:
        tr = Traj(wid, "branch", par.ctx)
        tr.reads = list(par.reads[:r])
        committed = list(par.reads[r].before)
        out.append(tr)
        states.append({"w": windows[wid], "tr": tr, "committed": committed,
                       "times": [rd.t_ms for rd in par.reads], "r": r, "r0": r})
    while True:
        live = [s for s in states if s["r"] < len(s["times"])]
        if not live:
            break
        items = []
        for s in live:
            t = s["times"][s["r"]]
            wav, post = data[s["w"]["id"]]
            items.append(ReadItem(wav=wav, post=post, t_ms=t, final=s["r"] == len(s["times"]) - 1,
                                  committed=s["committed"], ctx=s["tr"].ctx, akey=f"{s['w']['id']}|{t}",
                                  bar_wait=s["r"] == s["r0"]))
        outs = eng.read(items, max_new=max_new, temp=temp, temp_when=temp_when, gen=gen, record=True, cache=cache)
        for s, it, o in zip(live, items, outs):
            s["tr"].reads.append(Read(it.t_ms, it.final, list(s["committed"]), o.ids, o.when, o.who, o.logp,
                                      o.logp_wait, o.logp_who, it.akey))
            s["committed"] = s["committed"] + [i for i in o.ids if i not in (EOS_ID, WAIT_ID)]
            s["r"] += 1
    for tr, s in zip(out, states):
        parse(tr, eng.tok, s["w"]["dur_ms"] / 1000)
    return out
