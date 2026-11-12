"""Scheme 1' data (crf_novel_ideas.zh-CN.md §10.2-§10.3, CPU): per AMI train window of the v8 recipe, the fixed target
stream S, the reads, the admissibility bounds, the lags and the heuristic (v8 label) schedule, for the schedule lattice
(crf/schedule_lattice.py).

v8 = thinker_gstar_s2dl_zh2en_3k_20261030, trained (reports/gstar_si_paper_v1/run_v5_train.sh:38) on
data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1/rows_train.jsonl, built by
experiments/uniss_qwen3_omni_v1/data_clean/build_ami_gstar_stream_rows_zh2en.py from the windows
e2e_gstar_zh2en_v1/windows_train_ref.jsonl. The chunks below are that builder's lines 57-67 verbatim (lag 1.0 s,
slot map "lowlat"), so S = the chunk pieces in (commit_ms, src_start_ms) order and the heuristic schedule is the
builder's read loop (lines 70-86: read t commits the chunks with prev < commit_ms <= t, everything left at the last
read). Every v8 read row of the window is checked against this reconstruction (prefix and target strings, end token).

Per window (one jsonl line):
  S_ids       token ids of S = concatenated per-piece tokenisation (asserted equal to the joint tokenisation, so any
              chunk-boundary split tokenises the same), with the trainer's tokenizer conventions (Qwen3-Omni tokenizer
              + the 251 time / 8 speaker tokens at 151676.., pretrain_thinker_gstar.processor)
  kind        per token: 0 speaker, 1 start time, 2 text, 3 end time; chunk / unit per token; wstart (text token that
              starts a word, for block word counts)
  t_ms        read times (v8: every 1 s and the window end)
  lo, hi      inclusive read bounds per token:
                speaker + start time token: first read with t_r >= its start time (the time token's value, q * 80 ms)
                text tokens: first read with t_r >= a_u (the unit's start); end time token: t_r >= its value
                hi (all tokens of unit u): first read with t_r >= b_u + D_max (4 s), else the last read
              (a token without such a read gets the last read)
  tau_ms      per token: the end of its chunk's source span (src_end_ms); lag[r, j] = (t_ms[r] - tau_ms[j]) / 1000 s
              (stored factorised: lag_matrix() materialises it)
  beta_heur   committed token count after each read under the v8 labels; lat_heur = Lat(beta_heur)
  units       per unit: a_ms, b_ms, slot, overlap with other speakers' utterances of the window (s)

summary.json: counts, distributions, L_ref = mean Lat(heur), bound violations of beta_heur, the v8-row check, and the
lattice run on every window with random log-probs (feasibility, CPU time per window, single thread).

    python -m experiments.mist_w3po_v1.crf.build_sched_windows --out eval_outputs/mist_w3po_v1/crf/sched_v1 \
        --workers 32

--split validation (additive, default train = unchanged): the same recipe on windows_validation_ref.jsonl (AMI dev
meetings), checked against v8's own validation rows:
    python -m experiments.mist_w3po_v1.crf.build_sched_windows --split validation \
        --v8-rows data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1/rows_validation.jsonl \
        --out eval_outputs/mist_w3po_v1/crf/sched_dev_v1

En->Zh (additive, 2026-11-12, reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §2.2; every default unchanged):
the same recipe on the en2zh v8 data (thinker_gstar_s2dl_20261029b was trained on e2e_gstar_stream_v1/rows_train.jsonl,
built by uniss_qwen3_omni_v1/data_clean/build_ami_gstar_stream_rows.py): its reference windows, the Chinese text,
its flat lowlat slot map, chunk deadlines from the AMI word times (utt_chunks with load_words, as that builder), the
chunk text unstripped in the piece (as that builder), unit lengths in characters (zh_len). Every v8 read row is checked
as for zh2en, and the chunks against the windows' own stream_target (stream_target_match / _mismatch):
    python -m experiments.mist_w3po_v1.crf.build_sched_windows --src data/ami_multispk_si_v1/e2e_gstar_stream_v1 \
        --v8-rows data/ami_multispk_si_v1/e2e_gstar_stream_v1/rows_train.jsonl --text-key zh --slot-key "" \
        --words-manual data/ami_multispk_si_v1/annotations/ami_public_manual_1.6.2.zip --piece-strip 0 --units char \
        --out eval_outputs/mist_w3po_v1/crf/sched_en2zh_v1
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1"
V8_ROWS = "data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1/rows_train.jsonl"
TOKENIZER = "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
NEW0, N_TIME, N_SLOT = 151676, 251, 8
SPK0 = NEW0 + N_TIME
NEW_END = SPK0 + N_SLOT
K_SPK, K_TA, K_TEXT, K_TB = 0, 1, 2, 3
# en2zh options (main): their defaults reproduce the zh2en build bit for bit and are left out of summary.json's args
NEW_DEFAULTS = {"text_key": "ref", "slot_key": "lowlat", "words_manual": "", "piece_strip": 1, "units": "word"}

_TOK = None


def tokenizer(path: str = TOKENIZER):
    """The trainer's tokenizer (pretrain_thinker_gstar.processor: Qwen3-Omni tokenizer + the new tokens), loaded
    without the processor so that no accelerator is probed."""
    global _TOK
    if _TOK is None:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

        tok = Qwen2Tokenizer.from_pretrained(path)
        tok.add_tokens([f"<|t{i}|>" for i in range(N_TIME)] + [f"<|spk{k}|>" for k in range(1, N_SLOT + 1)],
                       special_tokens=True)
        assert tok.convert_tokens_to_ids("<|t0|>") == NEW0 and tok.convert_tokens_to_ids("<|spk1|>") == SPK0
        _TOK = tok
    return _TOK


def v8_chunks(w: dict, smap_m: dict, lag_ms: int, text_key: str = "ref", words: dict | None = None,
              strip: bool = True) -> list[dict]:
    """build_ami_gstar_stream_rows_zh2en.py lines 57-67 (+ the unit index, which the sort does not see). en2zh
    (build_ami_gstar_stream_rows.py lines 103-110): text_key "zh", words = the meeting's AMI word times per speaker,
    strip False."""
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import q
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import utt_chunks

    dur = w["dur_ms"]
    chunks = []
    for ui, u in enumerate(w["utts"]):
        if u["hesitation"] or not u[text_key].strip():
            continue
        uu = {"begin_time": w["start_s"] + u["begin_s"], "end_time": w["start_s"] + u["end_s"], "zh": u[text_key]}
        for c in utt_chunks(uu, [] if words is None else words.get(u["speaker_id"], []), lag_ms, w["start_s"]):
            c["commit_ms"] = min(c["commit_ms"], dur)
            txt = c["text"].strip() if strip else c["text"]
            c["piece"] = (f"<|spk{smap_m[u['speaker_id']]}|><|t{q(c['src_start_ms'] / 1000)}|>{txt}"
                          f"<|t{q(c['src_end_ms'] / 1000)}|>")
            c["utt"] = ui
            c["slot"] = smap_m[u["speaker_id"]]
            c["q_start"], c["q_end"] = q(c["src_start_ms"] / 1000), q(c["src_end_ms"] / 1000)
            chunks.append(c)
    chunks.sort(key=lambda c: (c["commit_ms"], c["src_start_ms"]))
    return chunks


def v8_reads(dur: int, win_ms: int = 1000) -> list[int]:
    """build_ami_gstar_stream_rows_zh2en.py line 70 (= core/engine.read_times)."""
    return list(range(win_ms, dur, win_ms)) + [dur]


def lag_matrix(rec: dict) -> np.ndarray:
    """[R, N] lag in seconds of a window record."""
    return (np.asarray(rec["t_ms"], np.float64)[:, None] - np.asarray(rec["tau_ms"], np.float64)[None, :]) / 1000.0


def first_read_at(t_ms: list[int], x: float) -> int:
    """Index of the first read with t >= x, else the last read."""
    return min(bisect.bisect_left(t_ms, x), len(t_ms) - 1)


def build_window(w: dict, smap_m: dict, args, tok, words: dict | None = None) -> tuple[dict, dict]:
    dur = w["dur_ms"]
    o = {k: getattr(args, k, v) for k, v in NEW_DEFAULTS.items()}
    chunks = v8_chunks(w, smap_m, round(args.lag * 1000), o["text_key"], words, bool(o["piece_strip"]))
    t_ms = v8_reads(dur)
    R = len(t_ms)
    ids, kind, chunk_of, unit_of, wstart = [], [], [], [], []
    for ci, c in enumerate(chunks):
        p = tok(c["piece"], add_special_tokens=False)["input_ids"]
        assert len(p) >= 3 and SPK0 <= p[0] < NEW_END and NEW0 <= p[1] < SPK0 and NEW0 <= p[-1] < SPK0, c["piece"]
        assert all(not (NEW0 <= i < NEW_END) for i in p[2:-1]), c["piece"]
        c["tok0"], c["tok1"] = len(ids), len(ids) + len(p)
        ids += p
        kind += [K_SPK, K_TA] + [K_TEXT] * (len(p) - 3) + [K_TB]
        chunk_of += [ci] * len(p)
        unit_of += [c["utt"]] * len(p)
        strs = tok.convert_ids_to_tokens(p[2:-1])
        wstart += [0, 0] + [int(k == 0 or s.startswith("Ġ")) for k, s in enumerate(strs)] + [0]
    S = "".join(c["piece"] for c in chunks)
    joint = tok(S, add_special_tokens=False)["input_ids"] if S else []
    assert joint == ids, f"joint tokenisation differs from per-piece: {w['id']}"
    N = len(ids)
    # heuristic schedule (the v8 read loop): after read t, the chunks with commit_ms <= t; all at the last read
    beta = []
    for r, t in enumerate(t_ms):
        last = r == R - 1
        beta.append(N if last else sum(c["tok1"] - c["tok0"] for c in chunks if c["commit_ms"] <= t))
    assert all(a <= b for a, b in zip(beta, beta[1:]))
    # units
    utts = w["utts"]
    units = {}
    for c in chunks:
        ui = c["utt"]
        if ui not in units:
            u = utts[ui]
            a, b = u["begin_s"], u["end_s"]
            ov = sum(max(0.0, min(b, v["end_s"]) - max(a, v["begin_s"])) for v in utts
                     if v["speaker_id"] != u["speaker_id"])
            units[ui] = {"utt": ui, "a_ms": round(a * 1000), "b_ms": round(b * 1000), "slot": c["slot"],
                         "speaker_id": u["speaker_id"], "ov_s": round(ov, 3),
                         "words": len(u[o["text_key"]].split()) if o["units"] == "word" else zh_len(u[o["text_key"]])}
    # bounds
    lo, hi, req, tau = [], [], [], []
    for j in range(N):
        c = chunks[chunk_of[j]]
        un = units[c["utt"]]
        k = kind[j]
        rq = (c["q_start"] * 80 if k in (K_SPK, K_TA) else un["a_ms"] if k == K_TEXT else c["q_end"] * 80)
        req.append(rq)
        lo.append(first_read_at(t_ms, rq))
        hi.append(first_read_at(t_ms, un["b_ms"] + args.dmax * 1000))
        tau.append(c["src_end_ms"])
    # heuristic commit read per token, violations, Lat(heur)
    r_of = np.zeros(N, np.int64)
    prev = 0
    for r, b in enumerate(beta):
        r_of[prev:b] = r
        prev = b
    lo_a, hi_a = np.array(lo, np.int64), np.array(hi, np.int64)
    viol_lo = int((r_of < lo_a).sum()) if N else 0
    viol_hi = int((r_of > hi_a).sum()) if N else 0
    lat_heur = float(np.mean((np.array(t_ms)[r_of] - np.array(tau)) / 1000.0)) if N else 0.0
    rec = {"id": w["id"], "meeting": w["meeting"], "start_s": w["start_s"], "dur_ms": dur, "audio": w["audio"],
           "post": w["post"]["lowlat"], "t_ms": t_ms, "S": S, "S_ids": ids, "kind": kind, "chunk": chunk_of,
           "unit": unit_of, "wstart": wstart, "lo": lo, "hi": hi, "req_ms": req, "tau_ms": tau,
           "beta_heur": beta, "lat_heur": lat_heur, "n_tok": N, "n_read": R,
           "seg_bounds": [c["tok1"] for c in chunks],
           "chunks": [{k: c[k] for k in ("piece", "text", "slot", "utt", "src_start_ms", "src_end_ms", "commit_ms",
                                         "q_start", "q_end", "tok0", "tok1")} for c in chunks],
           "units": [units[k] for k in sorted(units)]}
    info = {"viol_lo": viol_lo, "viol_hi": viol_hi, "chunks": chunks}
    return rec, info


def check_v8_rows(chunks: list[dict], t_ms: list[int], rows: list[dict]) -> dict:
    """Every v8 read row of the window must be (prefix, target, end) of the reconstruction at its read."""
    out = defaultdict(int)
    at = {}
    prev = 0
    for t in t_ms:
        last = t >= t_ms[-1]
        done = "".join(c["piece"] for c in chunks if c["commit_ms"] <= prev)
        new = "".join(c["piece"] for c in chunks if c["commit_ms"] > prev and (last or c["commit_ms"] <= t))
        at[t] = (done, new, "eos" if last else "wait")
        prev = t
    for r in rows:
        exp = at.get(r["t_ms"])
        if exp is None:
            out["v8_rows_unknown_read"] += 1
        elif (r["prefix"], r["target"], r["end"]) != exp:
            out["v8_rows_mismatch"] += 1
        else:
            out["v8_rows_match"] += 1
        # the committed history of the next read is this read's prefix + target (prefix property)
    return dict(out)


def dummy_lattice(rec: dict, W: int | None, rng: np.random.Generator) -> dict:
    """The lattice on the window's real shapes with random log-probs: feasibility and CPU time."""
    import torch

    from experiments.mist_w3po_v1.crf import schedule_lattice as sl

    R, N = rec["n_read"], rec["n_tok"]
    tok_lp = -rng.exponential(1.0, (R, N))
    end_lp = np.log(rng.uniform(1e-3, 1.0, (R, N + 1)))
    lat = sl.Lattice.single(tok_lp, end_lp, rec["lo"], rec["hi"], lag_matrix(rec), W=W)
    t0 = time.perf_counter()
    mg = sl.marginals(lat, 1.0)
    t1 = time.perf_counter()
    st, _ = sl.viterbi(lat, 1.0)
    t2 = time.perf_counter()
    heur = torch.tensor([rec["beta_heur"]])
    return {"finite": bool(mg["feasible"][0]) and bool(torch.isfinite(mg["logZ"][0])),
            "heur_ok": bool(sl.blocks_ok(lat, heur)[0]),
            "fb_s": t1 - t0, "vit_s": t2 - t1,
            "lat_check": abs(float(sl.schedule_lat(lat, heur)[0]) - rec["lat_heur"]) < 1e-9}


def _init(threads: int):
    import torch

    torch.set_num_threads(threads)
    tokenizer()


def zh_len(s: str) -> int:
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import zh_len as z

    return z(s)


def run_shard(job: tuple) -> dict:
    k, wins, smap, v8rows, args, *rest = job
    words = rest[0] if rest else None        # en2zh: meeting -> speaker -> AMI word times (--words-manual)
    tok = tokenizer()
    rng = np.random.default_rng(1000 + k)
    st = defaultdict(int)
    dist = defaultdict(list)
    path = Path(args.out) / f"shard_{k:03d}.jsonl"
    with open(path, "x", encoding="utf-8") as fh:
        for w in wins:
            rec, info = build_window(w, smap[w["meeting"]], args, tok,
                                     None if words is None else words.get(w["meeting"], {}))
            if "stream_target" in w:          # en2zh windows carry the v8 builder's own chunks
                ok = [(c["piece"], c["commit_ms"]) for c in info["chunks"]] == \
                    [(c["piece"], c["commit_ms"]) for c in w["stream_target"]["chunks"]]
                st["stream_target_match" if ok else "stream_target_mismatch"] += 1
            for key, v in check_v8_rows(info["chunks"], rec["t_ms"], v8rows.get(w["id"], [])).items():
                st[key] += v
            st["windows"] += 1
            st["v8_windows_with_rows"] += w["id"] in v8rows
            st["tokens"] += rec["n_tok"]
            st["reads"] += rec["n_read"]
            st["viol_lo_tokens"] += info["viol_lo"]
            st["viol_hi_tokens"] += info["viol_hi"]
            st["viol_windows"] += bool(info["viol_lo"] or info["viol_hi"])
            st["empty_S_windows"] += rec["n_tok"] == 0
            st["chunks"] += len(rec["chunks"])
            st["units"] += len(rec["units"])
            st["ov_units"] += sum(u["ov_s"] > 0 for u in rec["units"])
            st["time_tokens_capped"] += sum((c["q_end"] == N_TIME - 1) for c in rec["chunks"])
            b = [0] + rec["beta_heur"]
            blens = [y - x for x, y in zip(b, b[1:])]
            st["empty_reads"] += sum(x == 0 for x in blens)
            st["empty_reads_nonlast"] += sum(x == 0 for x in blens[:-1])
            st["reads_nonlast"] += len(blens) - 1
            dist["N"].append(rec["n_tok"])
            dist["R"].append(rec["n_read"])
            dist["blk"] += [x for x in blens if x > 0]
            cks = [0] + [sum(1 for c in rec["chunks"] if c["tok1"] <= bb) for bb in rec["beta_heur"]]
            dist["blk_chunks"] += [y - x for x, y in zip(cks, cks[1:]) if y > x]
            dist["lat_heur"].append(rec["lat_heur"])
            dist["lat_heur_w"].append(rec["n_tok"])
            if rec["n_tok"]:
                d = dummy_lattice(rec, None, rng)
                st["dummy_infeasible"] += not d["finite"]
                st["dummy_heur_inadmissible"] += not d["heur_ok"]
                st["dummy_lat_mismatch"] += not d["lat_check"]
                dist["fb_s"].append(d["fb_s"])
                dist["vit_s"].append(d["vit_s"])
                if args.wcap:
                    dc = dummy_lattice(rec, args.wcap, rng)
                    st["dummy_infeasible_wcap"] += not dc["finite"]
                    st["dummy_heur_inadmissible_wcap"] += not dc["heur_ok"]
                    dist["fb_s_wcap"].append(dc["fb_s"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"stats": dict(st), "dist": dict(dist)}


def q_stats(x) -> dict:
    x = np.asarray(x, np.float64)
    if not len(x):
        return {}
    return {"n": int(len(x)), "mean": float(x.mean()), "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max()), "min": float(x.min())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--v8-rows", default=V8_ROWS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lag", type=float, default=1.0, help="v8 deadline lag, s")
    ap.add_argument("--dmax", type=float, default=4.0, help="D_max, s")
    ap.add_argument("--wcap", type=int, default=128, help="also time the lattice with this block cap (0: skip)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="first n windows only (smoke test)")
    ap.add_argument("--split", choices=["train", "validation"], default="train",
                    help="windows_<split>_ref.jsonl of --src (validation: pass v8's rows_validation.jsonl as --v8-rows)")
    ap.add_argument("--text-key", default="ref", help="utterance field of the target text (en2zh: zh)")
    ap.add_argument("--slot-key", default="lowlat", help="key of slot_map.json ('' = the file is the map itself, en2zh)")
    ap.add_argument("--words-manual", default="", help="AMI manual zip: chunk deadlines from the word times (en2zh)")
    ap.add_argument("--piece-strip", type=int, default=1, choices=[0, 1], help="strip the chunk text (en2zh: 0)")
    ap.add_argument("--units", default="word", choices=["word", "char"], help="unit length (en2zh: char = zh_len)")
    args = ap.parse_args()
    if args.workers > 64:
        raise SystemExit("at most 64 workers")
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    src = Path(args.src)
    smap = json.loads((src / "slot_map.json").read_text(encoding="utf-8"))
    smap = smap[args.slot_key] if args.slot_key else smap
    words_all = None
    if args.words_manual:
        from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import load_words

        words_all = {m: {s: list(v) for s, v in d.items()} for m, d in load_words(Path(args.words_manual)).items()}
    wins = [json.loads(l) for l in open(src / f"windows_{args.split}_ref.jsonl", encoding="utf-8")]
    assert all(w["split"] == args.split for w in wins)
    if args.limit:
        wins = wins[: args.limit]
    ids = {w["id"] for w in wins}
    v8rows = defaultdict(list)
    n_v8_read = 0
    for l in open(args.v8_rows, encoding="utf-8"):
        r = json.loads(l)
        if r.get("kind") != "read":
            continue
        n_v8_read += 1
        wid = r["id"].split("|")[0]
        if wid in ids:
            v8rows[wid].append(r)
    out.mkdir(parents=True)
    shards = [wins[i: i + args.shard_size] for i in range(0, len(wins), args.shard_size)]
    jobs = [(k, sh, {m: smap[m] for m in {w["meeting"] for w in sh}}, {w["id"]: v8rows.get(w["id"], []) for w in sh},
             args) for k, sh in enumerate(shards)]
    if words_all is not None:
        jobs = [j + ({m: words_all.get(m, {}) for m in {w["meeting"] for w in j[1]}},) for j in jobs]
    print(f"{len(wins)} windows, {len(shards)} shards, {min(args.workers, len(shards))} workers, "
          f"{n_v8_read} v8 read rows", flush=True)
    import multiprocessing as mp

    res = []
    with mp.get_context("spawn").Pool(min(args.workers, len(shards)), initializer=_init, initargs=(1,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_shard, jobs)):
            res.append(r)
            print(f"shard {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    st = defaultdict(int)
    dist = defaultdict(list)
    for r in res:
        for k, v in r["stats"].items():
            st[k] += v
        for k, v in r["dist"].items():
            dist[k] += v
    lat = np.array(dist["lat_heur"])
    wN = np.array(dist["lat_heur_w"], np.float64)
    feas = wN > 0
    summary = {
        "stats": dict(st),
        "v8_read_rows_total": n_v8_read,
        "N": q_stats(dist["N"]), "R": q_stats(dist["R"]),
        "heur_block_tokens_nonempty": q_stats(dist["blk"]),
        "heur_block_chunks_nonempty": q_stats(dist["blk_chunks"]),
        "share_empty_reads": st["empty_reads"] / max(st["reads"], 1),
        "share_empty_reads_nonlast": st["empty_reads_nonlast"] / max(st["reads_nonlast"], 1),
        "L_ref_mean_lat_heur_windows": float(lat[feas].mean()) if feas.any() else None,
        "L_ref_token_weighted": float((lat[feas] * wN[feas]).sum() / wN[feas].sum()) if feas.any() else None,
        "lat_heur": q_stats(lat[feas]),
        "dummy_fb_s": q_stats(dist["fb_s"]), "dummy_viterbi_s": q_stats(dist["vit_s"]),
        "dummy_fb_s_wcap": q_stats(dist.get("fb_s_wcap", [])),
        "args": {k: v for k, v in vars(args).items() if not (k == "split" and v == "train")
                 and not (k in NEW_DEFAULTS and v == NEW_DEFAULTS[k])},   # as before at default
        "elapsed_s": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"SCHED_WINDOWS_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
