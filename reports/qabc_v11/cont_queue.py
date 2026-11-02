"""QABC v11 (prereg_v6): the talk-level clock shared by the continuous-clock runs.

A talk is a list of sentences in order, each (id, offset_ms on the talk clock, dur_ms). RealSI's talks are its
recordings (sentence start times as given); a dev pseudo-talk is a run of consecutive spontaneous-dev utterances
placed back to back with GAP_MS between them (non-test long audio for choosing the throughput control).

Tracked queue (what a policy sees when it decides, and what drives the Talker's speed shift): one Talker per talk;
a block committed at talk time t with n units adds n * RATE * exp(-gamma * shift) seconds behind
max(busy, t); q at time t is max(0, busy - t). Unlike v10 it is carried from one sentence to the next.

Throughput control (the same for every policy): a block written while q >= 0.5 s asks the Talker for
shift = min(S_MAX, round(K * q)) speed steps more than its block pace (si_infer_cont --speed-policy tracked).
CONTROLS names the candidates prereg_v6 chooses from on the dev pseudo-talks.

Real timeline (after the Talker has spoken): every block of a talk starts at the later of its commit and the end of
the block before it, with its real duration; continuous LAAL times every unit at its block's start on the talk clock
minus its sentence's offset (v10_talk.py's definition).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RATE = {"en2zh": 0.191, "zh2en": 0.339}        # s per unit, the SDR queue rate of v4-v10
CONTROLS = {"none": (0.0, 0), "fr": (3.0, 6), "strong": (3.0, 12)}
GAMMA0 = 0.017                                  # speed gained per step (FR, appendix of v10); refitted on dev
GAP_MS = 300
PER_TALK_S = 270                                # dev pseudo-talks of about 4.5 minutes
SPONT = {d: REPO / f"data/si_spont_v1/dev_{d}_ref.jsonl" for d in ("en2zh", "zh2en")}


def units(text: str, zh: bool) -> list[str]:
    return [c for c in text if not c.isspace()] if zh else text.split()


def shift_of(q_s: float, ctrl: str) -> int:
    k, smax = CONTROLS[ctrl]
    return min(smax, int(round(k * q_s))) if smax and q_s >= 0.5 else 0


def unit_s(direction: str, shift: int, gamma: float) -> float:
    return RATE[direction] * math.exp(-gamma * shift)


def realsi_talks(direction: str) -> list[tuple[str, list[tuple[str, int, int]]]]:
    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items
    talks = {}
    for it in realsi_items(direction):
        talks.setdefault(Path(it["wav"]).stem, []).append((it["id"], it["start"], it["end"] - it["start"]))
    return [(k, sorted(v, key=lambda x: x[1])) for k, v in sorted(talks.items())]


def pseudo_talks(direction: str) -> list[tuple[str, list[tuple[str, int, int]]]]:
    rows = [json.loads(l) for l in open(SPONT[direction], encoding="utf-8")]
    talks, cur, t = [], [], 0
    for r in rows:
        cur.append((r["id"], t, int(r["dur_ms"])))
        t += int(r["dur_ms"]) + GAP_MS
        if t >= PER_TALK_S * 1000:
            talks.append((f"pseudo{len(talks):02d}", cur))
            cur, t = [], 0
    if cur:
        if talks and t < PER_TALK_S * 500:   # a short remainder joins the last talk
            name, last = talks[-1]
            end = last[-1][1] + last[-1][2] + GAP_MS
            talks[-1] = (name, last + [(i, end + o, d) for i, o, d in cur])
        else:
            talks.append((f"pseudo{len(talks):02d}", cur))
    return talks


def carry(blocks: dict, talks, direction: str, ctrl: str, gamma: float) -> dict:
    """Annotate every block (in place) with the tracked queue q_ms at its commit and its shift; returns per-talk stats."""
    zh = direction.endswith("zh")
    out = {}
    for talk, sents in talks:
        busy = -1e18
        qs = []
        for sid, off, _ in sents:
            for b in blocks.get(sid, []):
                t = off + b["write_ms"]
                q = max(0.0, busy - t) / 1000
                sh = shift_of(q, ctrl)
                b["q_ms"], b["shift"] = int(round(q * 1000)), sh
                n = len(units(b["text"], zh))
                if n:
                    busy = max(busy, t) + n * unit_s(direction, sh, gamma) * 1000
                qs.append(q)
        out[talk] = {"q_mean": float(np.mean(qs)) if qs else 0.0}
    return out


def timeline(results: dict, talks, direction: str):
    """results: id -> si_infer record (blocks with write_ms, dur_ms, text). Returns per-sentence block starts
    (talk clock), and per-talk wait / drift / tail / share of time with > 2 s unplayed."""
    starts, per_talk = {}, {}
    for talk, sents in talks:
        rows, free = [], -1e18
        for sid, off, dur in sents:
            r = results.get(sid)
            if r is None:
                continue
            for k, b in enumerate(r["blocks"]):
                c = off + b["write_ms"]
                s = max(c, free)
                free = s + b.get("dur_ms", 0)
                starts[(sid, k)] = s
                rows.append((c, s, b.get("dur_ms", 0)))
        if not rows:
            continue
        end_src = max(off + dur for _, off, dur in sents)
        commits = np.array([c for c, _, _ in rows], float)
        st = np.array([s for _, s, _ in rows], float)
        du = np.array([d for _, _, d in rows], float)
        w = (st - commits) / 1000
        tt = commits / 60000
        ts = np.arange(rows[0][0], free, 100.0)
        ends = st + du
        unplayed = [du[commits <= t].sum() - np.clip(np.minimum(ends, t) - st, 0, None)[commits <= t].sum() for t in ts]
        per_talk[talk] = {"wait": float(w.mean()), "wait95": float(np.percentile(w, 95)),
                          "drift": float(np.polyfit(tt, w, 1)[0]) if len(rows) > 2 and tt.std() > 0 else 0.0,
                          "tail": max(0.0, (free - end_src) / 1000), "backlog2s": float(np.mean(np.array(unplayed) > 2000)),
                          "speech_over_source": float(du.sum() / max(1.0, end_src - sents[0][1])),
                          "minutes": (end_src - sents[0][1]) / 60000}
    return starts, per_talk


def laal_cont(results: dict, talks, direction: str, starts: dict) -> dict:
    """id -> continuous LAAL (ms): every unit at its block's start on the talk clock minus the sentence offset."""
    sys.path.insert(0, str(REPO / "paper/bayes_simul_s2st_v10/scripts"))
    import endpoints as ep
    unit = ep.CJK if direction.endswith("zh") else ep.EN
    out = {}
    for _, sents in talks:
        for sid, off, dur in sents:
            r = results.get(sid)
            if r is None:
                continue
            ds = [starts[(sid, k)] - off for k, b in enumerate(r["blocks"]) for _ in unit.findall(b["text"])]
            out[sid] = ep._laal(ds, dur, len(unit.findall(r["translation_ref"]))) if ds else float("nan")
    return out


def read_blocks(path) -> dict:
    return {r["id"]: r["blocks"] for r in map(json.loads, open(path, encoding="utf-8"))}


def write_blocks(src_path, blocks: dict, out_path) -> int:
    """The blocks file at src_path with its blocks replaced by the annotated ones; refuses to overwrite."""
    n = 0
    with open(out_path, "x", encoding="utf-8") as fh:
        for r in map(json.loads, open(src_path, encoding="utf-8")):
            if r["id"] in blocks:
                r["blocks"] = blocks[r["id"]]
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    return n
