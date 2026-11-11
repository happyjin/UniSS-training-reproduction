"""T0: write-timing labels from the Thinker's own teacher-forced argmax agreement.

Pre-registration: reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md §T0 (commit 18ab3bee, before any T0 result).
Results: reports/mist_w3po_v1/thinker_crf_t_results_v1.zh-CN.md.

Everything except the per-piece statistic is B-2's (crf/precheck_b2_earliest_write.py, imported, unchanged): the A'
Thinker loaded with the explicit 2-card layer map, the targets (S0-3 windows, <= 4-word pieces, kappa = S_prop's label
order, S = all pieces), reads every 0.5 s plus the final read, ONE teacher-forced forward of S per read (audio <= t_r,
posteriors zeroed after t_r - 1.04 s, ctx "", no perturbation, <wait> barred before the softmax at the final read).

  a_j(r)  the number of piece j's TEXT tokens whose argmax prediction (full vocabulary, after the final-read <wait> bar)
          equals the reference token. A piece is <|spk k|><|t a|>text<|t b|>: its first (speaker), second (start time)
          and last (end time) tokens are excluded (asserted by token id ranges).
  e_j     the smallest r with a_j(r') >= a_j(R) for every r' >= r (R = the final read): stable at the full-context
          agreement level from r on.
          a_j(R) = 0 (no text token agrees even with the full context; "stable" holds trivially from the first read):
          e_j := u_j, the S_prop read (the first 0.5 s read at or after source end + 1.0 s, else the final read). The
          literal reading (e_j = the first read) is reported, not used.
  then    the running max within each speaker in kappa order (B.cummax_by_speaker) and the online clamps (B.online).

Final read posteriors: dev = B-2's final_cut_ms(t, close_detect); train (no detector times) = t // 80, the v2'
training rows' --final-read-full-post 1.

Stages (every output refuses to overwrite):
  dev5 (targets = B-2's cpu/targets.jsonl, copied by reference):
    python -m experiments.mist_w3po_v1.crf.t0_agree_labels --score --root <new dev root> --targets \\
        eval_outputs/mist_w3po_v1/crf/b2_earliest_write_dev5/cpu/targets.jsonl --shard k/n   (2 visible cards)
    python -m experiments.mist_w3po_v1.crf.t0_agree_labels --analyse --root <dev root>
  train (the windows of the T1/T2 3000-step plan):
    python -m experiments.mist_w3po_v1.crf.t0_agree_labels --cpu-train --root <new train root>
    python -m experiments.mist_w3po_v1.crf.t0_agree_labels --score --split train --root <train root> --shard k/n
    python -m experiments.mist_w3po_v1.crf.t0_agree_labels --labels --root <train root>
  tests: PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_t0_agree_labels
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B

NEW0, N_TIME, N_SLOT = 151676, 251, 8
SPK0 = NEW0 + N_TIME
NEW_END = SPK0 + N_SLOT
GRID = 500
SRC_TRAIN = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1"
SCHED_V1 = "eval_outputs/mist_w3po_v1/crf/sched_v1"
PLAN_SEED, PLAN_STEPS, PLAN_ROWS = 20261109, 3000, 64
SCHEDS = ("S_cur", "S_prop", "S_lik", "T0")


# ------------------------------------------------------------------------------------------------ statistic
def text_spans(rec: dict) -> list[tuple[int, int]]:
    """[(a, b)) token span of each piece's text tokens: tok0 + 2 .. tok1 - 1 (asserted structure)."""
    ids = rec["S_ids"]
    out = []
    for p in rec["pieces"]:
        a, b = p["tok0"], p["tok1"]
        assert b - a >= 4, p["piece"]
        assert SPK0 <= ids[a] < NEW_END and NEW0 <= ids[a + 1] < SPK0 and NEW0 <= ids[b - 1] < SPK0, p["piece"]
        assert all(not (NEW0 <= t < NEW_END) for t in ids[a + 2: b - 1]), p["piece"]
        out.append((a + 2, b - 1))
    return out


def agree_counts(hit: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """hit [R, N] bool (argmax == reference per read and token) -> a [R, J] int."""
    return np.stack([hit[:, a:b].sum(1) for a, b in spans], 1).astype(np.int64) if spans else np.zeros((len(hit), 0),
                                                                                                         np.int64)


def earliest_stable(a: np.ndarray, u: list[int] | None = None, literal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """a [R, J] -> (e [J], zero [J] bool): e_j = min{r : a_j(r') >= a_j(R) for all r' >= r}; where a_j(R) = 0 (zero),
    e_j = u_j (literal: the first read)."""
    R, J = a.shape
    suf = np.minimum.accumulate(a[::-1], axis=0)[::-1]          # suf[r] = min_{r' >= r} a[r']
    ok = suf >= a[-1][None, :]
    e = ok.argmax(0).astype(np.int64)                            # ok[R - 1] is always True
    zero = a[-1] == 0
    if not literal:
        assert u is not None and len(u) == J
        e = np.where(zero, np.asarray(u, np.int64), e)
    return e, zero


def prop_reads(rec: dict) -> list[int]:
    """u_j: S_prop's read of each piece (B.schedule's S_prop index)."""
    return [B.first_read(rec["t_ms"], p["src_end_ms"] + B.LAG_MS) for p in rec["pieces"]]


def t0_events(rec: dict, e: np.ndarray) -> list[dict]:
    """B.schedule's S_lik path (running max per speaker, online clamps) with e as the raw read."""
    return B.schedule(rec, "S_lik", [int(x) for x in e])


# ------------------------------------------------------------------------------------------------ scoring (GPU)
def make_item(rec: dict, r: int, wav, post, split: str):
    """B.make_item for dev; train: the final read's posteriors visible up to t // 80 (--final-read-full-post 1)."""
    if split == "dev" or r < len(rec["t_ms"]) - 1:
        return B.make_item(rec, r, wav, post)
    from experiments.mist_w3po_v1.core.engine import ReadItem

    t = rec["t_ms"][r]
    p = post.copy()
    p[t // 80:] = 0.0
    return ReadItem(wav=wav, post=p, t_ms=t, final=True, committed=list(rec["S_ids"]), ctx="",
                    akey=f"{rec['id']}|{t}|fp", full=True)


def score_batch(eng, part: list, wavs: dict, split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """B.score_batch's forward, returning per (rec, r) (log p of the reference tokens [m], argmax == reference [m])."""
    import torch

    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD
    from experiments.mist_w3po_v1.rl.rollout import load_window

    its = []
    for rec, r in part:
        if rec["id"] not in wavs:
            wavs[rec["id"]] = load_window({"audio": rec["audio"], "post": {"lowlat": rec["post"]}})
        its.append(make_item(rec, r, *wavs[rec["id"]], split))
    ms = [len(rec["S_ids"]) for rec, _ in part]
    out = []
    with torch.no_grad():
        ae = eng.audio_embeds(its, eng.emb_layer.weight.dtype)
        prompts = [eng.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(its, ae)]
        Bn, L, M = len(its), max(len(p) for p in prompts), max(ms)
        ids = torch.full((Bn, L), eng.tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((Bn, L), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.as_tensor(p)
            mask[i, L - len(p):] = 1
        ids, mask = ids.to(eng.dev), mask.to(eng.dev)
        emb = eng.emb_layer(ids)
        emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
        h = eng.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                              return_dict=True).last_hidden_state[:, L - M - 1: L - 1]
        for i, ((rec, r), it) in enumerate(zip(part, its)):
            m = ms[i]
            lg = eng.logits(h[i, M - m:]).to(eng.dev)                       # [m, V] float32
            if it.final:
                lg[:, B.WAIT_ID] = -1e9
            tgt = torch.as_tensor(rec["S_ids"], dtype=torch.long, device=lg.device)
            lp = lg.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(lg, -1)
            hit = lg.argmax(-1) == tgt
            out.append((lp.double().cpu().numpy(), hit.cpu().numpy()))
            del lg
        del h, emb
    return out


def run_score(args) -> None:
    import torch

    from experiments.mist_w3po_v1.crf.decode_crf_ckpt import apply_ckpt

    root = Path(args.root)
    k, n = B.parse_shard(args.shard)
    out = root / (f"score_{k}of{n}" + (f"_limit{args.limit}" if args.limit else ""))
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    tpath = Path(args.targets) if args.targets else root / "cpu" / "targets.jsonl"
    recs = [json.loads(l) for l in open(tpath, encoding="utf-8")]
    if args.split == "dev":
        assert all(r["close_detect_ms"] is not None for r in recs)
    else:
        assert all(r["close_detect_ms"] is None for r in recs)
    recs = B.shard_of(recs, k, n)
    if args.limit:
        recs = recs[: args.limit]
    out.mkdir(parents=True)
    (out / "targets_path.txt").write_text(str(tpath) + "\n")
    t_start = time.time()
    if args.mem_cap_gib > 0:
        for i in range(torch.cuda.device_count()):
            tot = torch.cuda.get_device_properties(i).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gib * 2**30 / tot), i)
    eng = B.load_engine(args.thinker, "zh2en")
    info = apply_ckpt(eng, args.ckpt)
    print("device map: " + json.dumps(sorted(Counter(str(p.device) for p in eng.thinker.parameters()).items()))
          + f" audio_tower {next(eng.thinker.audio_tower.parameters()).device} engine {eng.dev}", flush=True)
    print("crf ckpt applied " + json.dumps(info), flush=True)
    load_s = time.time() - t_start
    bad = [r["id"] for r in recs if eng.tok(r["S"], add_special_tokens=False)["input_ids"] != r["S_ids"]
           and r.get("joint_tok_equal", True)]
    if bad:
        raise SystemExit(f"tokenizer mismatch between the targets and the engine: {bad[:5]}")
    spans = {rec["id"]: text_spans(rec) for rec in recs}
    reqs = [(rec, r) for rec in recs for r in range(len(rec["t_ms"]))]
    bl = B.batches(reqs, args.batch, args.tok_budget)
    print(f"shard {k}/{n}: {len(recs)} windows, {len(reqs)} reads, {len(bl)} batches", flush=True)
    res = {rec["id"]: [None] * len(rec["t_ms"]) for rec in recs}
    left = {rec["id"]: len(rec["t_ms"]) for rec in recs}
    by_id = {rec["id"]: rec for rec in recs}
    wavs = {}
    t1 = time.time()
    with open(out / "scores.jsonl", "x", encoding="utf-8") as fh:
        for b, part in enumerate(bl):
            for (rec, r), (lp, hit) in zip(part, score_batch(eng, part, wavs, args.split)):
                res[rec["id"]][r] = (lp, hit)
                left[rec["id"]] -= 1
                if left[rec["id"]] == 0:
                    rr = by_id[rec["id"]]
                    lp_ = np.stack([x[0] for x in res[rec["id"]]])
                    hit_ = np.stack([x[1] for x in res[rec["id"]]])
                    a = agree_counts(hit_, spans[rec["id"]])
                    s_text = np.stack([lp_[:, x:y].mean(1) for x, y in spans[rec["id"]]], 1)
                    row = {"id": rec["id"], "t_ms": rr["t_ms"], "a": a.tolist(), "n_text": [y - x for x, y in
                                                                                             spans[rec["id"]]],
                           "s_text": np.round(s_text, 5).tolist()}
                    if args.split == "dev":
                        row["hit_final"] = hit_[-1].astype(int).tolist()
                    fh.write(json.dumps(row) + "\n")
                    res[rec["id"]] = None
                    wavs.pop(rec["id"], None)
            if (b + 1) % 50 == 0:
                fh.flush()
                el = time.time() - t1
                print(f"{b + 1}/{len(bl)} batches, {el:.0f}s, eta {el / (b + 1) * (len(bl) - b - 1):.0f}s", flush=True)
    assert all(v == 0 for v in left.values())
    peak = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]
    st = {"windows": len(recs), "reads": len(reqs), "batches": len(bl), "load_s": round(load_s, 1),
          "scoring_s": round(time.time() - t1, 1), "wall_s": round(time.time() - t_start, 1),
          "cards": torch.cuda.device_count(), "peak_alloc_gib": peak, "ckpt": info, "args": vars(args)}
    (out / "done.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(f"T0_SCORED {out} {json.dumps(st)}", flush=True)


def load_scores(root: Path) -> tuple[dict, list]:
    scores, done = {}, []
    for d in sorted(root.glob("score_*of*")):
        if "_limit" in d.name or not (d / "done.json").exists():
            continue
        done.append(json.loads((d / "done.json").read_text()))
        for l in open(d / "scores.jsonl", encoding="utf-8"):
            r = json.loads(l)
            assert r["id"] not in scores, r["id"]
            scores[r["id"]] = r
    return scores, done


# ------------------------------------------------------------------------------------------------ dev analysis (CPU)
def window_labels(rec: dict, sc: dict, literal: bool = False) -> dict:
    a = np.asarray(sc["a"], np.int64)
    assert a.shape == (len(rec["t_ms"]), len(rec["pieces"])), rec["id"]
    assert sc["t_ms"] == rec["t_ms"]
    u = prop_reads(rec)
    e, zero = earliest_stable(a, u, literal)
    return {"a": a, "u": u, "e_raw": e, "zero": zero}


def run_analyse(args) -> None:
    from experiments.mist_w3po_v1.evaluation import metric_headroom as M   # puts reports/gstar_si_v8 on sys.path
    from meeting_score import universe

    root = Path(args.root)
    out = root / "analysis"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    tpath = Path((sorted(root.glob("score_*of*"))[0] / "targets_path.txt").read_text().strip())
    recs = [json.loads(l) for l in open(tpath, encoding="utf-8")]
    b2 = Path(args.b2_root)
    scores, done = load_scores(root)
    lik = {}
    for d in sorted(b2.glob("score_*of*")):
        if "_limit" in d.name or not (d / "done.json").exists():
            continue
        for l in open(d / "scores.jsonl", encoding="utf-8"):
            r = json.loads(l)
            lik[r["id"]] = np.asarray(r["s"], np.float64)
    miss = [r["id"] for r in recs if r["id"] not in scores or r["id"] not in lik]
    if miss:
        raise SystemExit(f"{len(miss)} windows without scores, e.g. {miss[:5]}")
    variants = {"T0": False, "T0_literal": True}
    ev, P = {}, []
    for rec in recs:
        lab = window_labels(rec, scores[rec["id"]])
        lab_lit = window_labels(rec, scores[rec["id"]], literal=True)
        e_lik = B.earliest(lik[rec["id"]])
        ev[rec["id"]] = {"S_cur": B.schedule(rec, "S_cur"), "S_prop": B.schedule(rec, "S_prop"),
                         "S_lik": B.schedule(rec, "S_lik", e_lik), "T0": t0_events(rec, lab["e_raw"]),
                         "T0_literal": t0_events(rec, lab_lit["e_raw"])}
        for j, p in enumerate(rec["pieces"]):
            u = rec["members"][p["utt"]]
            npc = sum(1 for x in rec["pieces"] if x["utt"] == p["utt"])
            t0e, lke, lte, pr = (ev[rec["id"]][k][j] for k in ("T0", "S_lik", "T0_literal", "S_prop"))
            P.append({"id": rec["id"], "j": j, "utt": u["record_id"], "n_pieces_utt": npc, "last": p["last"],
                      "n_words": p["n_words"], "src_start_ms": p["src_start_ms"], "src_end_ms": p["src_end_ms"],
                      "utt_end_ms": round(u["end_s"] * 1000), "R": len(rec["t_ms"]),
                      "a_final": int(lab["a"][-1, j]), "n_text": int(scores[rec["id"]]["n_text"][j]),
                      "zero": bool(lab["zero"][j]), "e_raw": int(lab["e_raw"][j]), "e": t0e["read"], "t": t0e["t"],
                      "c": t0e["c"], "final": t0e["final"], "e_lik": lke["read"], "c_lik": lke["c"],
                      "c_lit": lte["c"], "c_prop": pr["c"], "u": pr["read"]})
    W, _ = B.load_dev(args.win)
    RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
    sc = M.scored_utts(W, RW)
    U = universe(M.DR, {w["meeting"] for w in W})
    names = list(SCHEDS) + ["T0_literal"]
    lat = {nm: [] for nm in names}
    reg, unit_rows, complete, n_missing = [], [], set(), 0
    for m in sorted(U):
        units = M.assign(m, [], sc, U[m])["units"]
        per = {}
        for nm in names:
            words_of = defaultdict(list)
            for rec in recs:
                if rec["meeting"] != m:
                    continue
                for e in ev[rec["id"]][nm]:
                    rid = rec["members"][e["utt"]]["record_id"]
                    words_of[rid].append((e["n_words"], rec["start_s"] + e["c"] / 1000))
            per[nm] = B.unit_laal(units, words_of)
        n_missing += len(per["S_cur"]["missing"])
        for uid in sorted(per["S_cur"]["laal"]):
            complete.update(units[uid]["utts"])
            reg.append(per["S_cur"]["laal"][uid][1])
            row = {"meeting": m, "unit": uid, "reg": reg[-1]}
            for nm in names:
                lat[nm].append(per[nm]["laal"][uid][0])
                row[nm] = round(per[nm]["laal"][uid][0], 4)
            unit_rows.append(row)
    reg = np.asarray(reg)
    tab = {nm: {"all": B.q(np.asarray(lat[nm])), "ov": B.q(np.asarray(lat[nm])[reg == "ov"]),
                "1spk": B.q(np.asarray(lat[nm])[reg == "1spk"])} for nm in names}
    nl = [p for p in P if p["n_pieces_utt"] >= 2 and not p["last"]]
    share = {k: float(np.mean([p[c] < p["utt_end_ms"] for p in nl])) for k, c in
             (("T0", "c"), ("T0_literal", "c_lit"), ("S_lik", "c_lik"), ("S_prop", "c_prop"))}
    go1 = share["T0"] >= B.GO_SHARE
    go2 = np.mean(lat["T0"]) <= np.mean(lat["S_prop"])
    lag = {}
    for name, sel in (("all", lambda p: True), ("last", lambda p: p["last"]), ("nonlast", lambda p: not p["last"])):
        ps = [p for p in P if sel(p)]
        lag[name] = {"T0_label": B.q([(p["c"] - p["src_end_ms"]) / 1000 for p in ps]),
                     "T0_raw_read": B.q([(p["t"] - p["src_end_ms"]) / 1000 for p in ps]),
                     "S_lik_label": B.q([(p["c_lik"] - p["src_end_ms"]) / 1000 for p in ps]),
                     "S_prop_label": B.q([(p["c_prop"] - p["src_end_ms"]) / 1000 for p in ps])}
    fr = {}
    for nm in names:
        tot = fin = tot_u = fin_u = 0
        for rec in recs:
            for e in ev[rec["id"]][nm]:
                tot += e["n_words"]
                fin += e["n_words"] * e["final"]
                if rec["members"][e["utt"]]["record_id"] in complete:
                    tot_u += e["n_words"]
                    fin_u += e["n_words"] * e["final"]
        fr[nm] = {"all_words": round(fin / tot, 4), "unit_words": round(fin_u / max(tot_u, 1), 4)}
    d_lik = np.asarray([p["e"] - p["e_lik"] for p in P])
    gpu = {"shards": len(done), "wall_s": [d["wall_s"] for d in done], "scoring_s": [d["scoring_s"] for d in done],
           "reads": [d["reads"] for d in done], "cards_per_shard": [d["cards"] for d in done],
           "gpu_card_hours": round(sum(d["wall_s"] * d["cards"] for d in done) / 3600, 3),
           "reads_per_card_second": round(sum(d["reads"] for d in done)
                                          / sum(d["scoring_s"] * d["cards"] for d in done), 3),
           "peak_alloc_gib": [d["peak_alloc_gib"] for d in done]}
    res = {"go": {"line1_share_nonlast_before_utt_end": round(share["T0"], 4), "line1_ok": bool(go1),
                  "line1_n_nonlast": len(nl), "line2_T0_laal": tab["T0"]["all"]["mean"],
                  "line2_S_prop_laal": tab["S_prop"]["all"]["mean"], "line2_ok": bool(go2),
                  "GO": bool(go1 and go2)},
           "laal": tab, "share_nonlast_before_utt_end": {k: round(v, 4) for k, v in share.items()},
           "pieces": len(P), "zero_agreement_pieces": int(sum(p["zero"] for p in P)),
           "zero_agreement_share": round(float(np.mean([p["zero"] for p in P])), 4),
           "T0_final_read_pieces": round(float(np.mean([p["final"] for p in P])), 4),
           "T0_first_read_pieces": round(float(np.mean([p["e"] == 0 for p in P])), 4),
           "final_read_word_share": fr, "implied_lag_s": lag,
           "T0_minus_S_lik_reads": {"mean": round(float(d_lik.mean()), 3), "p10": float(np.percentile(d_lik, 10)),
                                    "p50": float(np.median(d_lik)), "p90": float(np.percentile(d_lik, 90)),
                                    "equal": round(float(np.mean(d_lik == 0)), 4),
                                    "later": round(float(np.mean(d_lik > 0)), 4)},
           "label_before_piece_src_start": round(float(np.mean([p["t"] < p["src_start_ms"] for p in P])), 4),
           "label_before_piece_src_start_after_clamp": round(float(np.mean([p["c"] < p["src_start_ms"]
                                                                             for p in P])), 4),
           "units": {"complete": len(unit_rows), "incomplete": n_missing},
           "text_tokens_final_read_agree": round(float(sum(p["a_final"] for p in P) / sum(p["n_text"] for p in P)), 4),
           "gpu": gpu, "args": vars(args)}
    out.mkdir(parents=True)
    with open(out / "results.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(out / "pieces.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in P)
    with open(out / "units.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in unit_rows)
    print(json.dumps(res, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ train targets (CPU)
def plan_windows(durs_by_id: dict, seed: int = PLAN_SEED, steps: int = PLAN_STEPS, rows: int = PLAN_ROWS) -> list[str]:
    """The window ids the trainer's plan_steps touches when every window has the 0.5 s reads (sorted by id, as
    train_crf_sft.load_windows)."""
    from experiments.mist_w3po_v1.crf.train_crf_sft import plan_steps

    ids = sorted(durs_by_id)
    Rs = [len(B.read_times(durs_by_id[i], GRID)) for i in ids]
    plan = plan_steps(Rs, seed, steps, rows)
    return sorted({ids[i] for s in plan for i in s})


def chunk_key(c: dict) -> tuple:
    return c["utt"], c["text"].strip(), c["src_start_ms"], c["src_end_ms"], int(c["slot"]), c["commit_ms"]


def train_target(w: dict, smap_m: dict, tok, sched_rec: dict | None) -> dict:
    """B.build_window on a windows_train_ref window (members = its utts; no detector times); asserts the punctuation
    chunks equal sched_v1's (text, source span, slot, commit) and the window fields."""
    ww = {"id": w["id"], "meeting": w["meeting"], "start_s": w["start_s"], "dur_ms": w["dur_ms"], "audio": w["audio"],
          "post": {"lowlat": w["post"]["lowlat"]}, "open_detect_ms": None, "close_detect_ms": None}
    rec = B.build_window(ww, w["utts"], smap_m, tok)
    rec["split"] = "train"
    if sched_rec is not None:
        assert sched_rec["dur_ms"] == rec["dur_ms"] and sched_rec["audio"] == rec["audio"], w["id"]
        assert sched_rec["post"] == rec["post"], w["id"]
        assert sorted(map(chunk_key, rec["chunks"])) == sorted(map(chunk_key, sched_rec["chunks"])), w["id"]
    return rec


def run_cpu_train(args) -> None:
    from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer

    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {root}")
    t0 = time.time()
    src = Path(SRC_TRAIN)
    smap = json.loads((src / "slot_map.json").read_text(encoding="utf-8"))["lowlat"]
    W = {w["id"]: w for w in map(json.loads, open(src / "windows_train_ref.jsonl", encoding="utf-8"))}
    sched = {}
    for f in sorted(Path(SCHED_V1).glob("shard_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            sched[r["id"]] = {k: r[k] for k in ("dur_ms", "audio", "post", "chunks")}
    assert set(sched) == set(W), "sched_v1 and windows_train_ref differ"
    used = plan_windows({i: w["dur_ms"] for i, w in W.items()}, args.plan_seed, args.plan_steps, args.plan_rows)
    tok = tokenizer()
    root.mkdir(parents=True)
    (root / "cpu").mkdir()
    st = Counter()
    with open(root / "cpu" / "targets.jsonl", "x", encoding="utf-8") as fh:
        for i in used:
            rec = train_target(W[i], smap[W[i]["meeting"]], tok, sched[i])
            assert rec["pieces"], i
            st["windows"] += 1
            st["pieces"] += len(rec["pieces"])
            st["chunks"] += len(rec["chunks"])
            st["reads"] += len(rec["t_ms"])
            st["S_tokens"] += len(rec["S_ids"])
            st["joint_tok_mismatch"] += not rec["joint_tok_equal"]
            st["audio_ms"] += rec["dur_ms"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summ = {"all_windows": len(W), "used_windows": len(used), **st,
            "plan": {"seed": args.plan_seed, "steps": args.plan_steps, "rows_per_step": args.plan_rows, "grid_ms": GRID},
            "cpu_s": round(time.time() - t0, 1)}
    (root / "cpu" / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1), flush=True)


def run_labels(args) -> None:
    """Per train window: u (S_prop reads), e_raw (T0, before the running max), e (after it), zero flags."""
    root = Path(args.root)
    out = root / "labels.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    scores, done = load_scores(root)
    st = Counter()
    with open(out, "x", encoding="utf-8") as fh:
        for l in open(root / "cpu" / "targets.jsonl", encoding="utf-8"):
            rec = json.loads(l)
            lab = window_labels(rec, scores[rec["id"]])
            e = B.cummax_by_speaker(lab["e_raw"], [p["slot"] for p in rec["pieces"]])
            R = len(rec["t_ms"])
            st["windows"] += 1
            st["pieces"] += len(e)
            st["zero"] += int(lab["zero"].sum())
            st["final"] += sum(x == R - 1 for x in e)
            st["first"] += sum(x == 0 for x in e)
            st["le_u"] += sum(x <= u for x, u in zip(e, lab["u"]))
            st["gt_u_plus1"] += sum(x > u + 1 for x, u in zip(e, lab["u"]))
            fh.write(json.dumps({"id": rec["id"], "t_ms": rec["t_ms"], "u": lab["u"],
                                 "e_raw": lab["e_raw"].tolist(), "e": e, "zero": lab["zero"].astype(int).tolist()})
                     + "\n")
    summ = dict(st)
    summ.update({"gpu_card_hours": round(sum(d["wall_s"] * d["cards"] for d in done) / 3600, 3),
                 "reads": sum(d["reads"] for d in done), "shards": len(done)})
    (root / "labels_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--cpu-train", action="store_true")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--root", required=True)
    ap.add_argument("--targets", default="", help="dev: B-2's cpu/targets.jsonl (default <root>/cpu/targets.jsonl)")
    ap.add_argument("--split", choices=["dev", "train"], default="dev")
    ap.add_argument("--b2-root", default="eval_outputs/mist_w3po_v1/crf/b2_earliest_write_dev5")
    ap.add_argument("--win", default=B.WIN)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--thinker", default=B.THINKER)
    ap.add_argument("--ckpt", default=B.CKPT)
    ap.add_argument("--mem-cap-gib", type=float, default=56.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tok-budget", type=int, default=12000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plan-seed", type=int, default=PLAN_SEED)
    ap.add_argument("--plan-steps", type=int, default=PLAN_STEPS)
    ap.add_argument("--plan-rows", type=int, default=PLAN_ROWS)
    args = ap.parse_args()
    if args.cpu_train:
        run_cpu_train(args)
    if args.score:
        run_score(args)
    if args.analyse:
        run_analyse(args)
    if args.labels:
        run_labels(args)


if __name__ == "__main__":
    main()
