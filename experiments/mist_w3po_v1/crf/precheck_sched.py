"""Scheme 1' precheck (crf_novel_ideas.zh-CN.md §10.5, §11): the v8 posterior over commit schedules on 500 train
windows, compared with the heuristic (v8 label) schedule. GPU for the scoring pass only; NOT run by the CPU stage.

1. Scoring (needs the model; v8 = thinker_gstar_s2dl_zh2en_3k_20261030 merged): per window of sched_v1
   (crf/build_sched_windows.py) and per read r, ONE teacher-forced forward with the engine's own prompt and model code
   (core/engine.Engine.from_merged, Engine.audio_embeds, Prompter.ids; batching as Engine.read: left padding + mask):
       prompt = audio <= t_r (+ speaker stream, posteriors zeroed after t_r - 1.04 s) + ctx "" (v8 protocol) + full S
   Causal attention gives, at the position after S^{<b}, both log p(S_b | .) (tok_lp[r, b]) and log p(e_r | .)
   (end_lp[r, b]; e_r = <wait>, <eos> at the last read). At the last read <wait> is barred before the softmax, as
   Engine.read does at decode time (--no-final-bar-wait: plain softmax). Saved per window (logprobs/*.npz) so that
   step 2 can be rerun on CPU (--analysis-only).
2. Analysis (CPU or GPU, float64): lam* by bisection so that the mean E_q[Lat] over the windows equals L_ref (the mean
   Lat of the heuristic schedule on the same windows; --lref train: the whole train set's, from sched_v1/summary.json);
   MAP (Viterbi) and 16 FFBS samples per window at lam*; versus the heuristic:
     C1  share of units (reference utterances; chunk level also reported) whose commit read (the read that commits
         their last token) differs; < 10 % -> STOP (1' would hardly change the labels)
     C2  mean commit-time shift of the chunks of overlapped units (MAP, and E_q); earlier than -0.3 s -> STOP (S0 risk)
     C3  share of non-empty blocks with <= 2 words (MAP, FFBS samples) vs the heuristic; a rise > 5 points -> flag
         "add g" (the threshold is this script's, the plan only says "clearly higher")
   plus lam = 0 (the model's own preference), block lengths, empty reads, mid-chunk boundaries, <= 1 s units.
   Both boundary modes are analysed: "any" (the exact model: a block may end at any token) and "chunk" (block ends
   restricted to chunk boundaries, end_lp = -inf elsewhere). W = --W tokens (default 128 > the heuristic max 77).

    # plan only (no model, nothing written; also runs the analysis on random log-probs for 20 windows)
    python -m experiments.mist_w3po_v1.crf.precheck_sched --out <new dir> --dry-run
    # the real run (GPU, ~2 cards at --max-gib 40)
    python -m experiments.mist_w3po_v1.crf.precheck_sched --out eval_outputs/mist_w3po_v1/crf/precheck_v8_500
    # analysis again from saved log-probs (CPU), into a new dir
    python -m experiments.mist_w3po_v1.crf.precheck_sched --analysis-only <scored dir> --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

SCHED = "eval_outputs/mist_w3po_v1/crf/sched_v1"
V8 = "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged"
WAIT_ID, EOS_ID = 151649, 151645


def load_records(sched: str, n: int, seed: int) -> list[dict]:
    recs = []
    for f in sorted(Path(sched).glob("shard_*.jsonl")):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    recs = [r for r in recs if r["n_tok"] > 0]
    recs.sort(key=lambda r: r["id"])
    return random.Random(seed).sample(recs, min(n, len(recs)))


def lag_matrix(rec: dict) -> np.ndarray:
    from experiments.mist_w3po_v1.crf.build_sched_windows import lag_matrix as lm

    return lm(rec)


# ------------------------------------------------------------------------------------------------ scoring (GPU)
def score_window(eng, rec: dict, batch: int, final_bar_wait: bool) -> tuple[np.ndarray, np.ndarray]:
    """tok_lp [R, N], end_lp [R, N+1] of one window, one teacher-forced forward per read."""
    import torch

    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD, ReadItem
    from experiments.mist_w3po_v1.rl.rollout import load_window

    S = list(rec["S_ids"])
    N, R = len(S), rec["n_read"]
    wav, post = load_window({"audio": rec["audio"], "post": {"lowlat": rec["post"]}})
    items = [ReadItem(wav=wav, post=post, t_ms=t, final=r == R - 1, committed=S, ctx="", akey=f"{rec['id']}|{t}")
             for r, t in enumerate(rec["t_ms"])]
    tok_lp = np.zeros((R, N), np.float64)
    end_lp = np.zeros((R, N + 1), np.float64)
    S_t = torch.as_tensor(S, device=eng.dev)
    emb_dtype = eng.emb_layer.weight.dtype
    for s0 in range(0, R, batch):
        its = items[s0: s0 + batch]
        with torch.no_grad():
            ae = eng.audio_embeds(its, emb_dtype)
            prompts = [eng.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(its, ae)]
            B, L = len(its), max(len(p) for p in prompts)
            ids = torch.full((B, L), eng.tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((B, L), dtype=torch.long)
            for i, p in enumerate(prompts):                       # Engine.read: left padding
                ids[i, L - len(p):] = torch.as_tensor(p)
                mask[i, L - len(p):] = 1
            ids, mask = ids.to(eng.dev), mask.to(eng.dev)
            emb = eng.emb_layer(ids)
            emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
            h = eng.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                                  return_dict=True).last_hidden_state[:, L - N - 1:]     # positions predicting S_0..e
            lg = eng.logits(h).to(eng.dev)                                              # [B, N+1, V] float32
            for i, it in enumerate(its):
                if it.final and final_bar_wait:
                    lg[i, :, WAIT_ID] = -1e9                                            # Engine.read line 246
            lsm = torch.log_softmax(lg, -1)
            tl = lsm[:, :N].gather(-1, S_t[None, :, None].expand(B, N, 1))[..., 0]
            el = torch.stack([lsm[i, :, EOS_ID if it.final else WAIT_ID] for i, it in enumerate(its)])
        tok_lp[s0: s0 + B] = tl.double().cpu().numpy()
        end_lp[s0: s0 + B] = el.double().cpu().numpy()
        del h, lg, lsm
    return tok_lp, end_lp


def run_scoring(args, recs: list[dict], out: Path) -> None:
    from experiments.mist_w3po_v1.core.engine import Engine

    eng = Engine.from_merged(args.thinker, "zh2en", args.max_gib)
    bad = [r["id"] for r in recs[:50] if eng.tok(r["S"], add_special_tokens=False)["input_ids"] != r["S_ids"]]
    if bad:
        raise SystemExit(f"tokenizer mismatch between sched_v1 and the engine: {bad[:5]}")
    lp = out / "logprobs"
    lp.mkdir()
    t0 = time.time()
    for k, rec in enumerate(recs):
        tok_lp, end_lp = score_window(eng, rec, args.batch, not args.no_final_bar_wait)
        np.savez(lp / f"{rec['id']}.npz", tok_lp=tok_lp, end_lp=end_lp)
        if (k + 1) % 25 == 0:
            print(f"scored {k + 1}/{len(recs)} {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ analysis (CPU)
def commit_reads(states: np.ndarray, n: int) -> np.ndarray:
    """[R] committed counts -> read index of every token."""
    r_of = np.zeros(n, np.int64)
    prev = 0
    for r, b in enumerate(states.tolist()):
        r_of[prev:b] = r
        prev = b
    return r_of


def block_words(rec: dict, states: np.ndarray) -> list[int]:
    """Words of every non-empty block (text tokens that start a word)."""
    ws = np.asarray(rec["wstart"])
    prev, out = 0, []
    for b in states.tolist():
        if b > prev:
            out.append(int(ws[prev:b].sum()))
        prev = b
    return out


def sched_stats(rec: dict, states: np.ndarray) -> dict:
    n = rec["n_tok"]
    t = np.asarray(rec["t_ms"]) / 1000.0
    r_of = commit_reads(states, n)
    ch_last = [c["tok1"] - 1 for c in rec["chunks"]]
    unit_last = {}
    for c in rec["chunks"]:
        unit_last[c["utt"]] = max(unit_last.get(c["utt"], -1), c["tok1"] - 1)
    prev = np.concatenate([[0], states[:-1]])
    blens = states - prev
    seg = set(rec["seg_bounds"]) | {0}
    return {"chunk_read": [int(r_of[j]) for j in ch_last], "chunk_t": [float(t[r_of[j]]) for j in ch_last],
            "unit_read": {u: int(r_of[j]) for u, j in unit_last.items()},
            "words": block_words(rec, states), "blens": [int(x) for x in blens if x > 0],
            "empty_reads": int((blens == 0).sum()), "reads": len(blens),
            "mid_chunk_ends": int(sum(1 for b, x in zip(states.tolist(), blens.tolist()) if x > 0 and b not in seg))}


def analyse(recs: list[dict], lps: dict, args, mode: str, dev: str) -> dict:
    import torch

    from experiments.mist_w3po_v1.crf import schedule_lattice as sl

    wins = []
    for rec in recs:
        tok_lp, end_lp = lps[rec["id"]]
        end_lp = end_lp.copy()
        if mode == "chunk":
            allowed = np.zeros(rec["n_tok"] + 1, bool)
            allowed[[0] + rec["seg_bounds"]] = True
            end_lp[:, ~allowed] = -np.inf
        wins.append({"tok_lp": tok_lp, "end_lp": end_lp, "lo": rec["lo"], "hi": rec["hi"], "lag": lag_matrix(rec)})
    bs = args.lattice_batch
    lats = [sl.pack(wins[i: i + bs], W=args.W, device=dev) for i in range(0, len(wins), bs)]
    lat_heur = np.array([r["lat_heur"] for r in recs])
    l_ref = float(lat_heur.mean()) if args.lref == "sample" else json.loads(
        (Path(args.sched) / "summary.json").read_text())["L_ref_mean_lat_heur_windows"]
    feas = np.concatenate([sl.marginals(l, 0.0)["feasible"].cpu().numpy() for l in lats])
    sol = sl.solve_lambda(lats, l_ref)
    lam = sol["lam"]
    e_lat0 = sl.expected_lat(lats, 0.0)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    per, agg = [], {"heur": [], "map": [], "ffbs": []}
    shift_map, shift_q, shift_map_all = [], [], []
    unit_diff, unit_n, chunk_diff, chunk_n, short_diff, short_n = 0, 0, 0, 0, 0, 0
    k = 0
    for lat in lats:
        mg = sl.marginals(lat, lam)
        st, _ = sl.viterbi(lat, lam)
        smp = sl.sample(lat, lam, n=args.samples, generator=gen)
        for i in range(lat.B):
            rec = recs[k]
            k += 1
            if not bool(mg["feasible"][i]):
                continue
            R = rec["n_read"]
            heur = np.asarray(rec["beta_heur"])
            mp = st[i, :R].cpu().numpy()
            sh, sm = sched_stats(rec, heur), sched_stats(rec, mp)
            agg["heur"].append(sh)
            agg["map"].append(sm)
            for s in smp[i, :, :R].cpu().numpy():
                agg["ffbs"].append(sched_stats(rec, s))
            t = np.asarray(rec["t_ms"]) / 1000.0
            w_tok = mg["w_tok"][i, :R, : rec["n_tok"]].cpu().numpy()
            units = {u["utt"]: u for u in rec["units"]}
            for ci, c in enumerate(rec["chunks"]):
                j = c["tok1"] - 1
                d_map = sm["chunk_t"][ci] - sh["chunk_t"][ci]
                shift_map_all.append(d_map)
                chunk_n += 1
                chunk_diff += sm["chunk_read"][ci] != sh["chunk_read"][ci]
                if units[c["utt"]]["ov_s"] > 0:
                    shift_map.append(d_map)
                    shift_q.append(float((w_tok[:, j] * t).sum()) - sh["chunk_t"][ci])
            for u, r_h in sh["unit_read"].items():
                unit_n += 1
                d = sm["unit_read"][u] != r_h
                unit_diff += d
                if units[u]["b_ms"] - units[u]["a_ms"] <= 1000:
                    short_n += 1
                    short_diff += d
            per.append({"id": rec["id"], "heur": heur.tolist(), "map": mp.tolist(), "lat_heur": rec["lat_heur"],
                        "lat_map": sl_lat(rec, mp), "exp_lat": float(mg["exp_lat"][i]), "logZ": float(mg["logZ"][i])})

    def blk(name):
        ws = [x for s in agg[name] for x in s["words"]]
        bl = [x for s in agg[name] for x in s["blens"]]
        return {"blocks": len(ws), "share_le2_words": float(np.mean([x <= 2 for x in ws])) if ws else None,
                "mean_block_tokens": float(np.mean(bl)) if bl else None,
                "share_empty_reads": sum(s["empty_reads"] for s in agg[name]) / max(1, sum(s["reads"] for s in agg[name])),
                "mid_chunk_ends": sum(s["mid_chunk_ends"] for s in agg[name])}

    b_h, b_m, b_f = blk("heur"), blk("map"), blk("ffbs")
    c1 = unit_diff / max(unit_n, 1)
    c2 = float(np.mean(shift_map)) if shift_map else None
    c2q = float(np.mean(shift_q)) if shift_q else None
    rise = max(b_m["share_le2_words"] - b_h["share_le2_words"], b_f["share_le2_words"] - b_h["share_le2_words"])
    lat_map = float(np.mean([p["lat_map"] for p in per])) if per else None
    return {
        "mode": mode, "W": args.W, "windows": len(recs), "feasible_lam0": int(feas.sum()),
        "L_ref": l_ref, "lref_source": args.lref, "lam_star": lam, "bisection": sol,
        "E_lat_at_lam0": e_lat0, "mean_lat_map": lat_map,
        "C1_unit_commit_read_differs": c1, "C1_chunk_commit_read_differs": chunk_diff / max(chunk_n, 1),
        "units_le1s_commit_read_differs": short_diff / max(short_n, 1), "units": unit_n, "units_le1s": short_n,
        "C2_ov_chunk_shift_s_map": c2, "C2_ov_chunk_shift_s_q": c2q, "ov_chunks": len(shift_map),
        "all_chunk_shift_s_map": float(np.mean(shift_map_all)) if shift_map_all else None,
        "C3_blocks": {"heur": b_h, "map": b_m, "ffbs": b_f}, "C3_rise_le2_words": rise,
        "decision": {"C1_stop": c1 < 0.10, "C2_stop": c2 is not None and c2 < -0.3,
                     "C3_add_g": rise > 0.05,
                     "go": not (c1 < 0.10) and not (c2 is not None and c2 < -0.3)},
        "per_window": per,
    }


def sl_lat(rec: dict, states: list) -> float:
    r_of = commit_reads(np.asarray(states), rec["n_tok"])
    return float(np.mean((np.asarray(rec["t_ms"])[r_of] - np.asarray(rec["tau_ms"])) / 1000.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sched", default=SCHED)
    ap.add_argument("--thinker", default=V8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--batch", type=int, default=4, help="reads per forward (Engine.read-style left padding)")
    ap.add_argument("--max-gib", type=int, default=40)
    ap.add_argument("--no-final-bar-wait", action="store_true")
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--lattice-batch", type=int, default=64)
    ap.add_argument("--lref", choices=["sample", "train"], default="sample")
    ap.add_argument("--analysis-only", default="", help="a scored dir (logprobs/ + windows.json): analysis only, CPU")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run the analysis on random log-probs "
                                                           "for 20 windows; no model, nothing written")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    if args.analysis_only:
        src = Path(args.analysis_only)
        ids = json.loads((src / "windows.json").read_text())["ids"]
        args.sched = json.loads((src / "windows.json").read_text())["sched"]
        allr, want = {}, set(ids)
        for f in sorted(Path(args.sched).glob("shard_*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r["id"] in want:
                    allr[r["id"]] = r
        recs = [allr[i] for i in ids]
    else:
        recs = load_records(args.sched, args.n, args.seed)
    n_reads = sum(r["n_read"] for r in recs)
    n_tok = sum(r["n_tok"] for r in recs)
    plan = {"windows": len(recs), "forwards (reads)": n_reads, "forward batches": sum(-(-r["n_read"] // args.batch)
                                                                                         for r in recs),
            "S tokens": n_tok, "scored positions": sum(r["n_read"] * (r["n_tok"] + 1) for r in recs),
            "max N": max(r["n_tok"] for r in recs), "max R": max(r["n_read"] for r in recs),
            "logits per batch (MB, fp32)": round(args.batch * (max(r["n_tok"] for r in recs) + 1) * 151936 * 4 / 2**20),
            "thinker": args.thinker, "thinker exists": Path(args.thinker, "config.json").exists(),
            "out": str(out), "W": args.W, "final_bar_wait": not args.no_final_bar_wait, "lref": args.lref}
    print(json.dumps(plan, indent=1), flush=True)
    if args.dry_run:
        rng = np.random.default_rng(0)
        sub = recs[:20]
        lps = {r["id"]: (-rng.exponential(1.0, (r["n_read"], r["n_tok"])),
                         np.log(rng.uniform(1e-3, 1, (r["n_read"], r["n_tok"] + 1)))) for r in sub}
        for mode in ("any", "chunk"):
            rep = analyse(sub, lps, args, mode, "cpu")
            rep.pop("per_window")
            print(json.dumps({k: rep[k] for k in ("mode", "L_ref", "lam_star", "E_lat_at_lam0", "mean_lat_map",
                                                  "C1_unit_commit_read_differs", "C2_ov_chunk_shift_s_map",
                                                  "C3_rise_le2_words", "decision")}), flush=True)
        print("DRY_RUN_OK (random log-probs; nothing written)")
        return
    out.mkdir(parents=True)
    (out / "windows.json").write_text(json.dumps({"sched": args.sched, "ids": [r["id"] for r in recs],
                                                  "plan": plan, "args": vars(args)}, indent=1))
    if not args.analysis_only:
        run_scoring(args, recs, out)
        lp_dir = out / "logprobs"
    else:
        lp_dir = Path(args.analysis_only) / "logprobs"
    lps = {}
    for r in recs:
        z = np.load(lp_dir / f"{r['id']}.npz")
        lps[r["id"]] = (z["tok_lp"], z["end_lp"])
    dev = "cuda" if not args.analysis_only else "cpu"
    reports = {}
    for mode in ("any", "chunk"):
        rep = analyse(recs, lps, args, mode, dev)
        with open(out / f"per_window_{mode}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(p) + "\n" for p in rep.pop("per_window"))
        reports[mode] = rep
    (out / "report.json").write_text(json.dumps(reports, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(reports, indent=1, ensure_ascii=False))
    print(f"SCHED_PRECHECK_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
