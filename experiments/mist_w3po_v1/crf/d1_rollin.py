"""D1 roll-in and manifest (reports/mist_w3po_v1/thinker_dagger_d1_prereg_v1.zh-CN.md §1.2-1.3, §4).

The learner (T4 or T5 step_03000, the base picked by crf/j1_decide.py -> logs/mist_w3po_v1/d1/base_decision.json)
decides only WHEN to write; the content is the reference pieces in kappa order.
  state     H (written pieces), a prefix of the kappa order (V.kap(rec, e, all pieces)); empty at the start
  read r    (r < R - 1) ONE teacher-forced forward of committed = kappa(H) ++ kappa(rest) (the training row's input:
            audio <= t_r, the engine's posterior cut, ctx ""), the full-vocabulary argmax at the position of the first
            token of every rest piece j = 0, 1, ...: the first j whose argmax is <wait> ends the read with j pieces
            written (greedy decoding's stop rule); no <wait> -> everything left is written. The final read writes all
            (its <wait> is barred).
  output    rho~_i = the read at which piece i is written (monotone along every speaker's kappa order; may be < e_i)
Causal attention makes the one-forward argmax at position k equal to the argmax after the prefix committed[:k]:
--selfcheck N re-runs the first N windows of a shard with one prefix forward per boundary and asserts equal decisions.

D1 rows (dart_aug.aug_requests_rho): per read with an overdue piece (e_i <= r - 1, not written), history kappa(H) and
target = every unwritten piece due by r (e_i <= r), kappa order, then e_r. Manifest: per (step, window) of the base's
3000-step plan, the window's rows; a step whose D1 rows exceed 25 % of its read rows keeps a uniform random 25 %
(numpy default_rng([20261112, step])).

    python -m experiments.mist_w3po_v1.crf.d1_rollin --score --sched <base sched> --ckpt <learner step_03000> \
        --root <new root> --shard k/n [--selfcheck 20]                          (2 visible cards)
    python -m experiments.mist_w3po_v1.crf.d1_rollin --manifest --sched <base sched> --root <root> --lw-mode t2|t5 \
        --out <new manifest dir>
    tests: PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_d1

D1v (reports/mist_w3po_v1/thinker_dagger_d1v_prereg_v1.zh-CN.md; additive, the defaults are D1's): --scorer v2 =
gpu_batch_scorer_v2 (right padding + position ids from 0 + an explicit 4-D mask + audio embeddings shared per read).
The D1 selfcheck failed 16/20 and 15/20 with v1 for two reasons (crf/test_d1v.py R0, CPU decomposition): v1's left
padding without position ids shifts every padded row's RoPE positions, and the padded batch and the unpadded per-boundary
forward take different attention paths in transformers (explicit mask + repeat_kv vs is_causal + GQA inside SDPA); the
bf16 differences of either are amplified by the top-k router into decision flips at near-ties. v2 removes both.
--selfcheck-min-reads, --selfcheck-tol (0 = registered), --manifest --lw-mode v2p (the VE1 learner).
    python -m experiments.mist_w3po_v1.crf.d1_rollin --score --scorer v2 --sched crf/sched_t5_merge3_w10_v1 \
        --ckpt <VE1 step_03000> --root crf/d1_rollin_ve1_v1 --shard k/n --selfcheck 20 --selfcheck-min-reads 3
    tests: PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_d1v
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V

WAIT_ID = 151649
D1_SEED = 20261112
CAP = 0.25
PLAN_SEED, PLAN_STEPS, PLAN_ROWS = 20261109, 3000, 64
THINKER = "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged"   # = B.THINKER (v8)


# ------------------------------------------------------------------------------------------------ pure parts
def kap_all(rec: dict) -> list[int]:
    e = S.chunk_reads(rec)
    return V.kap(rec, e, range(len(rec["chunks"])))


def read_input(rec: dict, order: list[int], n_hist: int) -> tuple[list[int], list[int]]:
    """committed = ids of order[:n_hist] ++ ids of order[n_hist:]; positions (indices into committed) of the first token
    of every rest piece."""
    ids, pos = [], []
    for k, c in enumerate(order):
        if k >= n_hist:
            pos.append(len(ids))
        ids += S.ids_of(rec, [c])
    return ids, pos


def decide(arg_at: list[int]) -> int:
    """pieces written at this read: the first boundary whose argmax is <wait>, else all."""
    for j, a in enumerate(arg_at):
        if a == WAIT_ID:
            return j
    return len(arg_at)


def rho_from_counts(rec: dict, order: list[int], written: list[int]) -> list[int]:
    """written[r] = pieces written at read r (in kappa order) -> rho~ per chunk index."""
    rho = [None] * len(rec["chunks"])
    k = 0
    for r, w in enumerate(written):
        for c in order[k: k + w]:
            rho[c] = r
        k += w
    assert k == len(order) and None not in rho, rec["id"]
    return rho


def simulate(rec: dict, scorer) -> dict:
    """the literal per-window loop: scorer(rec, r, committed, positions) -> argmax ids at those positions."""
    order = kap_all(rec)
    R, n = rec["n_read"], len(order)
    h, written = 0, []
    for r in range(R):
        if r == R - 1:
            j = n - h
        elif h == n:
            j = 0
        else:
            ids, pos = read_input(rec, order, h)
            j = decide(scorer(rec, r, ids, pos))
        written.append(j)
        h += j
    return {"id": rec["id"], "rho": rho_from_counts(rec, order, written), "written": written}


def simulate_rounds(recs: list[dict], batch_scorer) -> list[dict]:
    """the same trajectories, read by read over all windows: batch_scorer([(rec, r, committed, positions)]) -> list."""
    order = {rec["id"]: kap_all(rec) for rec in recs}
    h = {rec["id"]: 0 for rec in recs}
    written = {rec["id"]: [] for rec in recs}
    for r in range(max(rec["n_read"] for rec in recs)):
        todo = []
        for rec in recs:
            i, n, R = rec["id"], len(order[rec["id"]]), rec["n_read"]
            if r >= R:
                continue
            if r == R - 1:
                written[i].append(n - h[i])
                h[i] = n
            elif h[i] == n:
                written[i].append(0)
            else:
                ids, pos = read_input(rec, order[i], h[i])
                todo.append((rec, r, ids, pos))
        if todo:
            for (rec, r_, ids, pos), am in zip(todo, batch_scorer(todo)):
                j = decide(list(am))
                written[rec["id"]].append(j)
                h[rec["id"]] += j
    return [{"id": rec["id"], "rho": rho_from_counts(rec, order[rec["id"]], written[rec["id"]]),
             "written": written[rec["id"]]} for rec in recs]


# ------------------------------------------------------------------------------------------------ GPU scoring
def gpu_batch_scorer(eng, wavs: dict, budget: int, batch: int):
    """batch_scorer for simulate_rounds: the training row's input (the T0 / B-2 forward: Engine.audio_embeds, Prompter
    ids with ctx "", left padding), full-vocabulary argmax at the requested positions of committed."""
    import torch

    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD, ReadItem
    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
    from experiments.mist_w3po_v1.rl.rollout import load_window

    def one(part):
        its = []
        for rec, r, ids, _ in part:
            if rec["id"] not in wavs:
                wavs[rec["id"]] = load_window({"audio": rec["audio"], "post": {"lowlat": rec["post"]}})
            wav, post = wavs[rec["id"]]
            t = rec["t_ms"][r]
            its.append(ReadItem(wav=wav, post=post, t_ms=t, final=False, committed=list(ids), ctx="",
                                akey=f"{rec['id']}|{t}"))
        out = []
        with torch.no_grad():
            ae = eng.audio_embeds(its, eng.emb_layer.weight.dtype)
            prompts = [eng.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(its, ae)]
            Bn, L = len(its), max(len(p) for p in prompts)
            tid = torch.full((Bn, L), eng.tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((Bn, L), dtype=torch.long)
            for i, p in enumerate(prompts):
                tid[i, L - len(p):] = torch.as_tensor(p)
                mask[i, L - len(p):] = 1
            tid, mask = tid.to(eng.dev), mask.to(eng.dev)
            emb = eng.emb_layer(tid)
            emb = emb.masked_scatter((tid == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
            hs = eng.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                                   return_dict=True).last_hidden_state
            for i, (rec, r, ids, pos) in enumerate(part):
                M = len(ids)
                # position p of committed is predicted by the hidden state at L - M - 1 + p (p = M: after all)
                idx = torch.as_tensor([L - M - 1 + p for p in pos], device=hs.device)
                lg = eng.logits(hs[i, idx]).to(eng.dev)
                out.append(lg.argmax(-1).cpu().tolist())
                del lg
            del hs, emb
        return out

    def scorer(todo):
        res = [None] * len(todo)
        at = {(rec["id"], r): k for k, (rec, r, _, _) in enumerate(todo)}
        assert len(at) == len(todo)
        for part in B.batches([(rec, r) for rec, r, _, _ in todo], batch, budget):
            ks = [at[(rec["id"], r)] for rec, r in part]
            got = one([todo[k] for k in ks])
            for k, g in zip(ks, got):
                res[k] = g
        return res
    return scorer


def right_padded(prompts: list[list[int]], pad_id: int):
    """D1v scorer v2's layout: right padding (train_crf_sft.collate's), explicit position ids 0..L-1 for every row (the
    real tokens sit where an unpadded forward puts them), and an explicit 4-D boolean mask (B, 1, L, L): causal and
    key-valid. A 4-D mask is passed through by transformers' mask builder as is, so a batch with and a batch without
    padding run the same SDPA path (attn_mask set, no is_causal, no GQA-in-SDPA); pad query rows attend to their
    row's real keys (never a fully masked row)."""
    import torch

    B_, L = len(prompts), max(len(p) for p in prompts)
    ids = torch.full((B_, L), pad_id, dtype=torch.long)
    valid = torch.zeros((B_, L), dtype=torch.bool)
    for i, p in enumerate(prompts):
        ids[i, : len(p)] = torch.as_tensor(p)
        valid[i, : len(p)] = True
    causal = torch.ones((L, L), dtype=torch.bool).tril()
    mask4 = causal[None, None] & valid[:, None, None, :]
    pos = torch.arange(L, dtype=torch.long)[None].expand(B_, L).contiguous()
    return ids, mask4, pos


def gpu_batch_scorer_v2(eng, wavs: dict, budget: int, batch: int, acache: dict, diag: dict | None = None,
                        tag: str = "b"):
    """D1v (reports/mist_w3po_v1/thinker_dagger_d1v_prereg_v1.zh-CN.md §1.2, R0): gpu_batch_scorer with
      - right padding + explicit position ids from 0 (v1 left-pads without position ids, so transformers numbers a
        row's real tokens from its pad count: the RoPE angles, rounded to bf16 cos / sin, differ from the unpadded
        forward's; the per-boundary prefix check runs unpadded),
      - an explicit 4-D mask (the same SDPA path whether or not a batch has padding),
      - audio embeddings cached per akey (acache) and shared by every forward of the same read (the one-forward batch
        and the prefix check see identical audio inputs).
    The read-out index of committed position p in row i is len(prompt_i) - M + p - 1. diag (selfcheck): per
    (tag, id, r, p) the argmax and the margin logit[<wait>] - max(other logits)."""
    import torch

    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD, ReadItem
    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
    from experiments.mist_w3po_v1.rl.rollout import load_window

    def one(part):
        its = []
        for rec, r, ids, _ in part:
            if rec["id"] not in wavs:
                wavs[rec["id"]] = load_window({"audio": rec["audio"], "post": {"lowlat": rec["post"]}})
            wav, post = wavs[rec["id"]]
            t = rec["t_ms"][r]
            its.append(ReadItem(wav=wav, post=post, t_ms=t, final=False, committed=list(ids), ctx="",
                                akey=f"{rec['id']}|{t}"))
        out = []
        with torch.no_grad():
            ae = eng.audio_embeds(its, eng.emb_layer.weight.dtype, cache=acache)
            prompts = [eng.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(its, ae)]
            tid, mask4, pos = right_padded(prompts, eng.tok.pad_token_id or 0)
            tid, mask4, pos = tid.to(eng.dev), mask4.to(eng.dev), pos.to(eng.dev)
            emb = eng.emb_layer(tid)
            emb = emb.masked_scatter((tid == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
            hs = eng.thinker.model(inputs_embeds=emb, attention_mask=mask4, position_ids=pos, use_cache=False,
                                   return_dict=True).last_hidden_state
            for i, (rec, r, ids, ps) in enumerate(part):
                M, Lp = len(ids), len(prompts[i])
                idx = torch.as_tensor([Lp - M - 1 + p for p in ps], device=hs.device)
                lg = eng.logits(hs[i, idx]).to(eng.dev)
                am = lg.argmax(-1)
                out.append(am.cpu().tolist())
                if diag is not None:
                    other = lg.clone()
                    other[:, WAIT_ID] = float("-inf")
                    marg = (lg[:, WAIT_ID] - other.max(-1).values).cpu().tolist()
                    for p, a, m in zip(ps, out[-1], marg):
                        diag[(tag, rec["id"], r, p)] = (int(a), float(m))
                del lg
            del hs, emb
        return out

    def scorer(todo):
        res = [None] * len(todo)
        at = {(rec["id"], r): k for k, (rec, r, _, _) in enumerate(todo)}
        assert len(at) == len(todo)
        for part in B.batches([(rec, r) for rec, r, _, _ in todo], batch, budget):
            ks = [at[(rec["id"], r)] for rec, r in part]
            got = one([todo[k] for k in ks])
            for k, g in zip(ks, got):
                res[k] = g
        return res
    return scorer


def selfcheck_report(lit: list[dict], bat: list[dict], diag: dict | None) -> dict:
    """selfcheck summary: windows with equal rho; for every differing window the first differing read and, with diag,
    the decisions and <wait> margins of both paths at that read's boundaries (p, argmax lit / bat, margin lit / bat)."""
    sc = {"windows": len(lit), "equal": sum(a["rho"] == b["rho"] for a, b in zip(lit, bat)), "differ": []}
    for a, b in zip(lit, bat):
        if a["rho"] == b["rho"]:
            continue
        r0 = next((r for r, (x, y) in enumerate(zip(a["written"], b["written"])) if x != y), None)
        d = {"id": a["id"], "read": r0, "written_lit": a["written"], "written_bat": b["written"], "at_read": []}
        if diag is not None and r0 is not None:
            ps = sorted({k[3] for k in diag if k[1] == a["id"] and k[2] == r0})
            d["at_read"] = [(p, diag.get(("lit", a["id"], r0, p)), diag.get(("bat", a["id"], r0, p))) for p in ps]
        sc["differ"].append(d)
    if diag is not None:
        both = [(diag[k][1], diag[("bat",) + k[1:]][1]) for k in diag if k[0] == "lit" and ("bat",) + k[1:] in diag]
        if both:
            dm = np.abs(np.asarray([x - y for x, y in both]))
            sc["margin_abs_diff"] = {"n": len(both), "max": float(dm.max()), "mean": float(dm.mean())}
            sc["min_abs_margin_lit"] = float(min(abs(x) for x, _ in both))
    return sc


def selfcheck_recs(recs: list[dict], n: int, min_reads: int) -> list[dict]:
    """the selfcheck windows: the first n of the shard (min_reads 0: v1's choice), or the first n with at least
    min_reads reads and 2 pieces."""
    if min_reads <= 0:
        return recs[:n]
    return [r for r in recs if r["n_read"] >= min_reads and len(r["chunks"]) >= 2][:n]


def run_score(args) -> None:
    if getattr(args, "scorer", "v1") == "v2":
        return run_score_v2(args)
    import torch

    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
    from experiments.mist_w3po_v1.crf.decode_crf_ckpt import apply_ckpt

    root = Path(args.root)
    k, n = B.parse_shard(args.shard)
    out = root / f"rollin_{k}of{n}"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    recs, plan = load_plan(args.sched)
    used = sorted({i for s in plan for i in s})
    recs = B.shard_of([recs[i] for i in used], k, n)
    if args.limit:
        recs = recs[: args.limit]
    out.mkdir(parents=True)
    t0 = time.time()
    if args.mem_cap_gib > 0:
        for i in range(torch.cuda.device_count()):
            tot = torch.cuda.get_device_properties(i).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gib * 2**30 / tot), i)
    eng = B.load_engine(args.thinker, "zh2en")
    info = apply_ckpt(eng, args.ckpt)
    print("learner ckpt applied " + json.dumps(info), flush=True)
    bad = [r["id"] for r in recs if eng.tok(r["S"], add_special_tokens=False)["input_ids"] != r["S_ids"]]
    if bad:
        print(f"note: {len(bad)} windows whose joint tokenisation differs from S_ids (S_ids used as is)", flush=True)
    load_s = time.time() - t0
    wavs = {}
    scorer = gpu_batch_scorer(eng, wavs, args.tok_budget, args.batch)
    sc = {}
    if args.selfcheck:   # one prefix forward per boundary (the literal loop) vs the one-forward batch
        def prefix_scorer(rec, r, ids, pos):
            return [scorer([(rec, r, ids[:p], [p])])[0][0] for p in pos]
        lit = [simulate(rec, prefix_scorer) for rec in recs[: args.selfcheck]]
        bat = simulate_rounds(recs[: args.selfcheck], scorer)
        sc = {"windows": len(lit), "equal": sum(a["rho"] == b["rho"] for a, b in zip(lit, bat))}
        print("selfcheck " + json.dumps(sc), flush=True)
        if sc["equal"] != sc["windows"]:
            raise SystemExit("selfcheck failed: one-forward decisions differ from the per-boundary prefix forwards")
    t1 = time.time()
    res = []
    CH = 512
    with open(out / "rho.jsonl", "x", encoding="utf-8") as fh:
        for a in range(0, len(recs), CH):
            part = simulate_rounds(recs[a: a + CH], scorer)
            for x in part:
                fh.write(json.dumps(x) + "\n")
            res += part
            wavs.clear()
            el = time.time() - t1
            print(f"{min(a + CH, len(recs))}/{len(recs)} windows, {el:.0f}s", flush=True)
    peak = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]
    st = {"windows": len(recs), "forwards": int(sum(sum(1 for r in range(len(x["written"]) - 1)) for x in res)),
          "load_s": round(load_s, 1), "rollin_s": round(time.time() - t1, 1), "selfcheck": sc, "peak_alloc_gib": peak,
          "ckpt": args.ckpt, "ckpt_info": info, "args": vars(args)}
    (out / "done.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(f"D1_ROLLIN_DONE {out} {json.dumps(st)}", flush=True)


def run_score_v2(args) -> None:
    """--scorer v2 (D1v): run_score with gpu_batch_scorer_v2 (shared audio cache, cleared with the wav cache), the
    selfcheck on selfcheck_recs(.., --selfcheck-min-reads) with decision margins (selfcheck.json in the shard dir), and
    split-K bf16 reductions in fp32 (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False).
    --selfcheck-tol 0 (registered): every selfcheck window must agree; tol > 0 (only on the coordinator's decision)
    accepts a differing window when every differing decision has |margin| <= tol in both paths."""
    import torch

    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
    from experiments.mist_w3po_v1.crf.decode_crf_ckpt import apply_ckpt

    root = Path(args.root)
    k, n = B.parse_shard(args.shard)
    out = root / f"rollin_{k}of{n}"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    recs, plan = load_plan(args.sched)
    used = sorted({i for s in plan for i in s})
    recs = B.shard_of([recs[i] for i in used], k, n)
    if args.limit:
        recs = recs[: args.limit]
    out.mkdir(parents=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.mem_cap_gib > 0:
        for i in range(torch.cuda.device_count()):
            tot = torch.cuda.get_device_properties(i).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gib * 2**30 / tot), i)
    eng = B.load_engine(args.thinker, "zh2en")
    info = apply_ckpt(eng, args.ckpt)
    print("learner ckpt applied " + json.dumps(info), flush=True)
    bad = [r["id"] for r in recs if eng.tok(r["S"], add_special_tokens=False)["input_ids"] != r["S_ids"]]
    if bad:
        print(f"note: {len(bad)} windows whose joint tokenisation differs from S_ids (S_ids used as is)", flush=True)
    load_s = time.time() - t0
    wavs, acache = {}, {}
    scorer = gpu_batch_scorer_v2(eng, wavs, args.tok_budget, args.batch, acache)
    sc = {}
    if args.selfcheck:
        diag = {}
        lit_sc = gpu_batch_scorer_v2(eng, wavs, args.tok_budget, args.batch, acache, diag, "lit")
        bat_sc = gpu_batch_scorer_v2(eng, wavs, args.tok_budget, args.batch, acache, diag, "bat")
        chk = selfcheck_recs(recs, args.selfcheck, args.selfcheck_min_reads)

        def prefix_scorer(rec, r, ids, pos):
            return [lit_sc([(rec, r, ids[:p], [p])])[0][0] for p in pos]
        lit = [simulate(rec, prefix_scorer) for rec in chk]
        bat = simulate_rounds(chk, bat_sc)
        sc = selfcheck_report(lit, bat, diag)
        sc["tol"] = args.selfcheck_tol
        ok = sc["equal"] == sc["windows"]
        if not ok and args.selfcheck_tol > 0:
            ok = all(all(abs(x[1][1]) <= args.selfcheck_tol and abs(y[1][1]) <= args.selfcheck_tol
                         for _, x, y in d["at_read"] if x and y and x[0] != y[0]) for d in sc["differ"])
            sc["accepted_by_tol"] = ok
        (out / "selfcheck.json").write_text(json.dumps(sc, indent=1, default=str), encoding="utf-8")
        print("selfcheck " + json.dumps({k_: sc[k_] for k_ in ("windows", "equal")} |
                                        {"margin_abs_diff": sc.get("margin_abs_diff")}), flush=True)
        if not ok:
            raise SystemExit("selfcheck failed: one-forward decisions differ from the per-boundary prefix forwards "
                             f"(details {out / 'selfcheck.json'})")
        wavs.clear()
        acache.clear()
    t1 = time.time()
    res = []
    CH = 512
    with open(out / "rho.jsonl", "x", encoding="utf-8") as fh:
        for a in range(0, len(recs), CH):
            part = simulate_rounds(recs[a: a + CH], scorer)
            for x in part:
                fh.write(json.dumps(x) + "\n")
            res += part
            wavs.clear()
            acache.clear()
            el = time.time() - t1
            print(f"{min(a + CH, len(recs))}/{len(recs)} windows, {el:.0f}s", flush=True)
    peak = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]
    st = {"windows": len(recs), "forwards": int(sum(sum(1 for r in range(len(x["written"]) - 1)) for x in res)),
          "load_s": round(load_s, 1), "rollin_s": round(time.time() - t1, 1),
          "selfcheck": {k_: sc[k_] for k_ in ("windows", "equal", "tol") if k_ in sc}, "peak_alloc_gib": peak,
          "scorer": "v2", "ckpt": args.ckpt, "ckpt_info": info, "args": vars(args)}
    (out / "done.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(f"D1_ROLLIN_DONE {out} {json.dumps(st)}", flush=True)


# ------------------------------------------------------------------------------------------------ manifest (CPU)
def load_plan(sched: str) -> tuple[list[dict], list[list[int]]]:
    """the trainer's windows (load_windows order: n_tok > 0, sorted by id) and plan_steps(20261109, 3000, 64)."""
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    recs = []
    for f in sorted(Path(sched).glob("shard_*.jsonl")):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    recs = [r for r in recs if r["n_tok"] > 0]
    recs.sort(key=lambda r: r["id"])
    return recs, T.plan_steps([r["n_read"] for r in recs], PLAN_SEED, PLAN_STEPS, PLAN_ROWS)


def cap_rows(step: int, rows: list[tuple[int, int]], read_rows: int, cap: float = CAP) -> list[tuple[int, int]]:
    """rows = [(window idx, r)] of one step; more than cap x read_rows -> a uniform random subset of that size."""
    m = int(np.floor(cap * read_rows))
    if len(rows) <= m:
        return list(rows)
    rng = np.random.default_rng([D1_SEED, step])
    keep = sorted(rng.choice(len(rows), size=m, replace=False).tolist())
    return [rows[i] for i in keep]


def run_manifest(args) -> None:
    from experiments.mist_w3po_v1.crf import dart_aug as DA

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    recs, plan = load_plan(args.sched)
    rho = {}
    for f in sorted(Path(args.root).glob("rollin_*of*/rho.jsonl")):
        for l in open(f, encoding="utf-8"):
            x = json.loads(l)
            rho[x["id"]] = x["rho"]
    used = sorted({i for s in plan for i in s})
    miss = [recs[i]["id"] for i in used if recs[i]["id"] not in rho]
    if miss:
        raise SystemExit(f"roll-in missing for {len(miss)} plan windows (e.g. {miss[:3]})")
    reqs = {i: DA.aug_requests_rho(recs[i], rho[recs[i]["id"]], args.lw_mode) for i in used}
    st = Counter()
    lines, per_step, kept_frac = [], [], []
    for s, ids in enumerate(plan):
        rows = [(i, q["r"]) for i in ids for q in reqs[i]]
        read_rows = sum(recs[i]["n_read"] for i in ids)
        kept = cap_rows(s, rows, read_rows)
        per_step.append((len(rows), len(kept), read_rows))
        kept_frac.append(len(kept) / max(1, len(rows)) if rows else 1.0)
        ks = {}
        for i, r in kept:
            ks.setdefault(i, []).append(r)
        for i in ids:
            if i in ks:
                lines.append({"step": s, "idx": i, "id": recs[i]["id"], "rho": rho[recs[i]["id"]], "keep": ks[i]})
    # R4 and descriptive statistics over the plan windows
    over_w = ahead = n_items = 0
    lag, overdue_reads, reads = [], 0, 0
    for i in used:
        rec, rh = recs[i], rho[recs[i]["id"]]
        e = S.chunk_reads(rec)
        q = reqs[i]
        over_w += bool(q)
        overdue_reads += len(q)
        reads += rec["n_read"]
        n_items += len(e)
        ahead += sum(1 for a, b in zip(rh, e) if a < b)
        lag += [a - b for a, b in zip(rh, e)]
    lag = np.asarray(lag)
    a = np.asarray(per_step, np.float64)
    summ = {"sched": args.sched, "root": args.root, "lw_mode": args.lw_mode, "plan": {"seed": PLAN_SEED, "steps": PLAN_STEPS,
            "rows_per_step": PLAN_ROWS}, "cap": CAP, "seed": D1_SEED, "windows_in_plan": len(used),
            "windows_with_overdue": over_w, "share_windows_with_overdue": round(over_w / len(used), 4),
            "R4_pass": bool(over_w / len(used) >= 0.05), "overdue_read_share": round(overdue_reads / max(1, reads), 4),
            "ahead_item_share": round(ahead / max(1, n_items), 4),
            "rho_minus_e": {"mean": round(float(lag.mean()), 3), "p50": float(np.median(lag)),
                            "p90": float(np.percentile(lag, 90)), "share_gt0": round(float((lag > 0).mean()), 4)},
            "d1_rows_per_step_before_cap": {"mean": round(float(a[:, 0].mean()), 2), "p95": float(np.percentile(a[:, 0], 95))},
            "d1_rows_per_step_after_cap": {"mean": round(float(a[:, 1].mean()), 2), "p95": float(np.percentile(a[:, 1], 95))},
            "d1_over_read_rows": round(float(a[:, 1].sum() / a[:, 2].sum()), 4),
            "steps_capped": int((a[:, 0] > a[:, 1]).sum()), "rows_dropped_share": round(float(1 - a[:, 1].sum() /
                                                                                              max(1, a[:, 0].sum())), 4)}
    out.mkdir(parents=True)
    with open(out / "d1.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in lines)
    with open(out / "summary.json", "x", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))


def load_manifest(path: str, wins: list, plan: list[list[int]], sched: str) -> dict:
    """{step: {window idx: (rho, keep reads)}}, checked against the trainer's plan and windows."""
    sm = json.loads((Path(path) / "summary.json").read_text())
    if sm["sched"] != sched or sm["plan"] != {"seed": PLAN_SEED, "steps": len(plan), "rows_per_step": PLAN_ROWS}:
        raise SystemExit(f"--d1 manifest made for {sm['sched']} / {sm['plan']}, not this run")
    out: dict = {}
    for l in open(Path(path) / "d1.jsonl", encoding="utf-8"):
        x = json.loads(l)
        s, i = x["step"], x["idx"]
        if i not in plan[s] or wins[i].rec["id"] != x["id"]:
            raise SystemExit(f"--d1 manifest entry ({s}, {i}, {x['id']}) is not in this plan")
        out.setdefault(s, {})[i] = (x["rho"], x["keep"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--sched", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--thinker", default=THINKER)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--selfcheck", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tok-budget", type=int, default=24000)
    ap.add_argument("--mem-cap-gib", type=float, default=60.0)
    ap.add_argument("--lw-mode", default="t2", choices=["t2", "t5", "v2p"],
                    help="--manifest: lattice mode of the in-lattice statistic (v2p: D1v, the VE1 learner)")
    ap.add_argument("--out", default="")
    ap.add_argument("--scorer", default="v1", choices=["v1", "v2"],
                    help="--score: v1 = gpu_batch_scorer (left padding, D1); v2 = gpu_batch_scorer_v2 (D1v: right "
                         "padding, position ids from 0, 4-D mask, shared audio cache)")
    ap.add_argument("--selfcheck-min-reads", type=int, default=0,
                    help="--scorer v2: selfcheck windows = the first --selfcheck with >= this many reads and >= 2 "
                         "pieces (0: the shard's first windows, as v1)")
    ap.add_argument("--selfcheck-tol", type=float, default=0.0,
                    help="--scorer v2: 0 = every selfcheck window must agree (registered); > 0 only on the "
                         "coordinator's decision")
    a = ap.parse_args()
    if a.score:
        if not a.ckpt:
            raise SystemExit("--score needs --ckpt")
        run_score(a)
    if a.manifest:
        if not a.out:
            raise SystemExit("--manifest needs --out")
        run_manifest(a)


if __name__ == "__main__":
    main()
