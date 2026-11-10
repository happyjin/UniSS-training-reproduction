"""W3PO stage 0 SFT rows for Zh->En (docs/mist_grpo_plan.zh-CN.md §4), on the v8 recipe of
experiments/uniss_qwen3_omni_v1/data_clean/build_ami_gstar_stream_rows_zh2en.py (deadline labels: each English
reference cut at punctuation into chunks, chunk j ends at the fraction of the utterance's duration covering its share of
characters, committed later; a row per read every 1 s with the posteriors finalised by then, the chunks committed
before as the prefix, the new ones then <wait>, or the rest then <eos> at the window's end; empty reads kept with
p 0.5; + 20 % whole-window offline rows and 15 % single-speaker rows of the read-row count). The read-row budget is
the v8 count (--n-read, 130,560) whatever the mix, so every stage-0 variant trains on the same number of rows per pass.

Switches (each one an S0 item, so leave-one-out ablations use the same builder):
  --jitter   S0-1  commit = evidence + max(0, U(-0.5, 1.0)) s per chunk (not before the evidence; non-decreasing
                   within an utterance) instead of evidence + 1.0 s
  --dagger   S0-2  files of online_decode.py outputs on train windows (the v8 model's own reads): --dagger-share of
                   the read rows take the model's committed text before a read as the prefix and, as the target, the
                   gold chunks due by then that the prefix does not cover (coverage per reference utterance: the
                   characters of the model segments whose time span best matches it, collar 0.5 s), in commit order
  --diar     S0-3  half of the gold read rows from the test-rule diarizer windows on train meetings
                   (windows_train_diar.jsonl, padding 0.2 / 0.3 s), half from the v8 reference windows
  --context  S0-4  rows carry "ctx": the last k in {1, 2} (p 0.45 each; none p 0.1) reference utterances of the meeting
                   that end before the window starts, as engine.ctx_string would render the previous window's output;
                   DAgger rows on whole decoded meetings (online window ids) use the model's own previous window
                   output; DAgger rows on sampled reference windows the gold context
  --winmap   S0-5  speaker tokens are the window-level Hungarian slots (build_windows_zh2en.window_slot_map on the
                   window's own low-latency posteriors) instead of the whole-meeting map

    python -m experiments.mist_w3po_v1.data.build_s0_rows_zh2en --out <dir> [--jitter] [--dagger f ...] [--diar] \
        [--context] [--winmap]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.core.engine import ctx_string
from experiments.mist_w3po_v1.data.build_windows_zh2en import window_slot_map
from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import q
from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import utt_chunks

LOOKAHEAD_MS = 1040
SEG_RE = re.compile(r"<\|spk(\d+)\|><\|t(\d+)\|>(.*?)<\|t(\d+)\|>", re.S)


def parse_segments(text: str) -> tuple[list[dict], bool]:
    """Closed segments of a committed reply text, and whether the text ends inside an open segment."""
    segs = [{"spk": int(m.group(1)), "a": int(m.group(2)) * 0.08, "b": int(m.group(4)) * 0.08, "text": m.group(3).strip()}
            for m in SEG_RE.finditer(text)]
    tail = SEG_RE.sub("", text)
    return segs, bool(re.search(r"<\|spk\d+\|>", tail))


def chunks_of_window(w: dict, slot_of: dict, lag_ms: int, rng: random.Random | None, jitter: bool) -> list[dict]:
    """Deadline chunks of a window (v8 recipe), with the S0-1 jitter when asked."""
    dur = w["dur_ms"]
    chunks = []
    for ui, u in enumerate(w["utts"]):
        if u["hesitation"] or not u["ref"].strip():
            continue
        uu = {"begin_time": w["start_s"] + u["begin_s"], "end_time": w["start_s"] + u["end_s"], "zh": u["ref"]}
        prev_commit = 0
        for c in utt_chunks(uu, [], lag_ms, w["start_s"]):
            if jitter:
                c["commit_ms"] = c["src_end_ms"] + round(1000 * max(0.0, rng.uniform(-0.5, 1.0)))
                c["commit_ms"] = max(c["commit_ms"], prev_commit)
            prev_commit = c["commit_ms"]
            c["commit_ms"] = min(c["commit_ms"], dur)
            c["utt"] = ui
            c["text"] = c["text"].strip()
            c["piece"] = (f"<|spk{slot_of[u['speaker_id']]}|><|t{q(c['src_start_ms'] / 1000)}|>{c['text']}"
                          f"<|t{q(c['src_end_ms'] / 1000)}|>")
            chunks.append(c)
    chunks.sort(key=lambda c: (c["commit_ms"], c["src_start_ms"]))
    return chunks


def covered_chunks(w: dict, chunks: list[dict], segs: list[dict]) -> set:
    """Indices of the gold chunks a committed prefix covers (see the module doc, S0-2)."""
    utts = w["utts"]
    cov = defaultdict(int)
    for s in segs:
        best, bu = 0.0, None
        for ui, u in enumerate(utts):
            ov = min(s["b"] + 0.5, u["end_s"]) - max(s["a"] - 0.5, u["begin_s"])
            if ov > best:
                best, bu = ov, ui
        if bu is not None:
            cov[bu] += len(s["text"])
    done = set()
    acc = defaultdict(int)
    for j, c in sorted(enumerate(chunks), key=lambda jc: (jc[1]["utt"], jc[1]["src_start_ms"])):
        n = len(c["text"])
        if acc[c["utt"]] + 0.5 * n <= cov[c["utt"]]:
            done.add(j)
        acc[c["utt"]] += n
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-src", default="data/ami_multispk_si_v1/e2e_gstar_zh2en_v1")
    ap.add_argument("--diar-windows", default="eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_train_diar.jsonl")
    ap.add_argument("--sd-from", default="data/ami_multispk_si_v1/e2e_gstar_stream_v1/rows_train.jsonl")
    ap.add_argument("--utts", default="data/ami_multispk_si_v1/zh_mix_v8/utts")
    ap.add_argument("--en-norm", default="data/ami_multispk_si_v1/en_norm_v1.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-read", type=int, default=130560, help="read rows (v8: 130,560)")
    ap.add_argument("--keep-empty", type=float, default=0.5)
    ap.add_argument("--offline-share", type=float, default=0.2)
    ap.add_argument("--sd-share", type=float, default=0.15)
    ap.add_argument("--lag", type=float, default=1.0)
    ap.add_argument("--jitter", action="store_true")
    ap.add_argument("--dagger", nargs="*", default=[])
    ap.add_argument("--dagger-windows", nargs="*", default=[], help="window files the DAgger decodes ran on")
    ap.add_argument("--dagger-share", type=float, default=0.3)
    ap.add_argument("--diar", action="store_true")
    ap.add_argument("--context", action="store_true")
    ap.add_argument("--winmap", action="store_true")
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    src = Path(args.ref_src)
    sess = json.loads((src / "slot_map.json").read_text(encoding="utf-8"))["lowlat"]
    lag_ms = round(args.lag * 1000)
    stats = defaultdict(int)

    # reference utterances per meeting, for the S0-4 gold context
    en_norm = {r["record_id"]: r["en"] for r in map(json.loads, open(args.en_norm, encoding="utf-8"))}
    meet_utts = {}

    def utts_of(m):
        if m not in meet_utts:
            us = []
            for u in map(json.loads, open(Path(args.utts) / f"{m}.jsonl", encoding="utf-8")):
                if u["hes"] or not u.get("placed", True) or any(f.startswith("excluded:") for f in u.get("flags", [])):
                    continue
                ref = en_norm.get(u["record_id"], "").strip()
                if ref and u["speaker_id"] in sess.get(m, {}):
                    us.append((u["zh_end"], u["speaker_id"], ref))
            meet_utts[m] = sorted(us)
        return meet_utts[m]

    def gold_ctx(w):
        if not args.context:
            return ""
        x = rng.random()
        k = 0 if x < 0.1 else (1 if x < 0.55 else 2)
        prev = [u for u in utts_of(w["meeting"]) if u[0] <= w["start_s"]][-k:] if k else []
        return ctx_string([{"spk": sess[w["meeting"]][s], "text": t} for _, s, t in prev], k)

    def slots(w):
        m = w["meeting"]
        if not args.winmap:
            return sess[m]
        post = np.load(w["post"]["lowlat"]).astype(np.float32)
        wm = window_slot_map(w["utts"], post, sess[m])
        stats["winmap_changed"] += sum(wm[s] != sess[m][s] for s in wm)
        stats["winmap_spk"] += len(wm)
        return {**sess[m], **wm}

    def read_rows(w, chunks, ctx, kind):
        rows = []
        dur = w["dur_ms"]
        reads = list(range(1000, dur, 1000)) + [dur]
        prev = 0
        for t in reads:
            last = t >= dur
            done = [c for c in chunks if c["commit_ms"] <= prev]
            new = [c for c in chunks if c["commit_ms"] > prev and (last or c["commit_ms"] <= t)]
            target = "".join(c["piece"] for c in new)
            if target or last or rng.random() < args.keep_empty:
                rows.append({"id": f"{w['id']}|{kind}|{t}", "audio": w["audio"], "dir": "zh2en", "t_ms": t,
                             "dur_ms": dur, "post": w["post"]["lowlat"], "post_until_ms": max(0, t - LOOKAHEAD_MS),
                             "prefix": "".join(c["piece"] for c in done), "target": target,
                             "end": "eos" if last else "wait", "kind": kind, "inst": "gstar", "preset": "lowlat",
                             "ctx": ctx})
            prev = t
        return rows

    # gold read rows
    ref_w = [json.loads(l) for l in open(src / "windows_train_ref.jsonl", encoding="utf-8")]
    pools = {"ref": [], "diar": []}
    for w in ref_w:
        pools["ref"] += read_rows(w, chunks_of_window(w, slots(w), lag_ms, rng, args.jitter), gold_ctx(w), "read")
    if args.diar:
        for w in map(json.loads, open(args.diar_windows, encoding="utf-8")):
            pools["diar"] += read_rows(w, chunks_of_window(w, slots(w), lag_ms, rng, args.jitter), gold_ctx(w), "read_diar")
    for k, v in pools.items():
        stats[f"pool_{k}"] = len(v)

    # DAgger rows (S0-2)
    dag = []
    if args.dagger:
        wins = {}
        for f in args.dagger_windows:
            for w in map(json.loads, open(f, encoding="utf-8")):
                wins[w["id"]] = w
        recs = {}
        for f in args.dagger:
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                recs[r["id"]] = r
        prev_out = {}   # meeting -> windows in order, for the model's own previous-window context
        for wid, r in recs.items():
            if wid in wins:
                prev_out.setdefault(wins[wid]["meeting"], []).append((wins[wid]["start_s"], r["segments"]))
        for m in prev_out:
            prev_out[m].sort(key=lambda x: x[0])
        for wid, r in recs.items():
            w = wins.get(wid)
            if w is None or not w.get("utts"):
                stats["dagger_no_window"] += 1
                continue
            chunks = chunks_of_window(w, slots(w), lag_ms, rng, args.jitter)
            ctx = ""
            if args.context and "_on_" in wid:   # whole decoded meetings: the model's own previous window
                before = [s for st, s in prev_out.get(w["meeting"], []) if st < w["start_s"]]
                k = 1 if rng.random() < 0.5 else 2
                ctx = ctx_string(before[-1], k) if before else ""
            elif args.context:                    # sampled reference windows: the gold context
                ctx = gold_ctx(w)
            prefix = ""
            for rd in r["read_log"]:
                t = rd["t"]
                last = t >= w["dur_ms"]
                segs, open_tail = parse_segments(prefix)
                if not open_tail:
                    cov = covered_chunks(w, chunks, segs)
                    new = [c for j, c in enumerate(chunks) if j not in cov and (last or c["commit_ms"] <= t)]
                    target = "".join(c["piece"] for c in new)
                    if target or last or rng.random() < args.keep_empty:
                        dag.append({"id": f"{wid}|dagger|{t}", "audio": w["audio"], "dir": "zh2en", "t_ms": t,
                                    "dur_ms": w["dur_ms"], "post": w["post"]["lowlat"],
                                    "post_until_ms": max(0, t - LOOKAHEAD_MS), "prefix": prefix, "target": target,
                                    "end": "eos" if last else "wait", "kind": "dagger", "inst": "gstar",
                                    "preset": "lowlat", "ctx": ctx})
                        stats["dagger_prefix_nonempty"] += bool(prefix)
                        stats["dagger_prefix_differs"] += prefix != "".join(c["piece"] for c in chunks
                                                                            if c["commit_ms"] <= t - 1000)
                else:
                    stats["dagger_open_tail_skipped"] += 1
                prefix += rd["new"]
        stats["pool_dagger"] = len(dag)

    n = args.n_read
    n_dag = min(len(dag), round(args.dagger_share * n)) if dag else 0
    n_gold = n - n_dag
    if args.diar:
        n_diar = min(len(pools["diar"]), n_gold // 2)
        take = rng.sample(pools["ref"], min(len(pools["ref"]), n_gold - n_diar)) + rng.sample(pools["diar"], n_diar)
    else:
        take = rng.sample(pools["ref"], min(len(pools["ref"]), n_gold))
    take += rng.sample(dag, n_dag)
    stats["read_rows"] = len(take)
    stats["read_rows_dagger"] = n_dag
    off = [r for r in map(json.loads, open(src / "rows_train.jsonl", encoding="utf-8"))
           if r.get("inst") == "gstar" and r.get("preset") == "offline"]
    sd = [r for r in map(json.loads, open(args.sd_from, encoding="utf-8")) if r.get("inst") == "sd"]
    rows = take + rng.sample(off, min(len(off), round(args.offline_share * len(take)))) \
        + rng.sample(sd, min(len(sd), round(args.sd_share * len(take))))
    rng.shuffle(rows)
    stats["rows"] = len(rows)
    stats["rows_with_ctx"] = sum(bool(r.get("ctx")) for r in rows)
    stats["empty_target_read_rows"] = sum(1 for r in take if not r["target"])
    out.mkdir(parents=True)
    with open(out / "rows_train.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    summ = {"stats": dict(stats), "args": vars(args)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
