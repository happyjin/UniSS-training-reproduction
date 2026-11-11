"""T1 / T2 training data (reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md §T1, §T2): the sched_v1 windows in the
sched record format of crf/build_sched_windows.py, with
  items  = the <= 4-word pieces of B-2 / T0 (t0_agree_labels.train_target: the same utterances, slots, character-
           proportional source spans; the punctuation chunks asserted equal to sched_v1's),
  reads  = every 0.5 s plus the window end (B.read_times(dur, 500)),
  labels = e_read per piece: T0's write read (running max per speaker; labels.jsonl of the T0 train root) for the
           windows of the 3000-step plan, S_prop's read u for the other windows (never trained: asserted below that the
           trainer's plan_steps on THIS data touches only T0-labelled windows); --labels prop: u everywhere.
So `train_crf_sft.py --sched <out> --objective serial_v2p` is T1 (v2' on these items / reads / labels) and
`--objective latent_write` is T2 (crf/latent_write_crf.py, which also reads u_read).

Per window (sched_v1's fields, same meaning):
  chunks   the pieces in S order, S sorted by (e_read, src_end_ms, src_start_ms, utt, k); per chunk the sched_v1 keys
           (piece, text, slot, utt, src_start_ms, src_end_ms, commit_ms, q_start, q_end, tok0, tok1) plus k (piece index
           in its utterance), n_words, e_read, u_read, t0_e_raw, t0_zero, label_src.
           commit_ms = src_end_ms + 1000 UNCAPPED: it is only the secondary key of precheck_serial_v2.kap, which then
           orders like precheck_serial.kappa (rho, source end, source start) on every set (asserted).
  beta_heur  committed tokens after each read on the label path (pieces with e_read <= r; N at the final read), so
           precheck_serial.chunk_reads(rec) == e_read (asserted).
  lo / hi / req_ms / tau_ms / kind / chunk / unit / wstart / units / lat_heur: build_sched_windows' rules on the 0.5 s
           reads (D_max 4 s); lo / hi only feed the trainer's feasibility filter (set_reach), checked feasible below.
Checks written to summary.json: label path rows == C0 rows on every plan window (serial_v2p_train._check_one), the
trainer's plan == the T0-labelled windows, feasibility of every plan window, S tokenisation == per-piece ids.

    python -m experiments.mist_w3po_v1.crf.build_piece_sched --t0-root eval_outputs/mist_w3po_v1/crf/t0_agree_train \\
        --out eval_outputs/mist_w3po_v1/crf/sched_piece_t0_v1 [--labels t0|prop] [--workers 32]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
from experiments.mist_w3po_v1.crf import t0_agree_labels as T0

K_SPK, K_TA, K_TEXT, K_TB = 0, 1, 2, 3
LAG_MS = 1000
DMAX_MS = 4000


def sched_record(w: dict, tg: dict, e: list[int], u: list[int], extra: dict, tok) -> dict:
    """The sched-format record of window w from its T0 target tg (B-2 pieces in kappa order) and labels e (per piece)."""
    from experiments.mist_w3po_v1.crf.build_sched_windows import first_read_at
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import q

    P = tg["pieces"]
    J = len(P)
    assert len(e) == len(u) == J
    t_ms = tg["t_ms"]
    R = len(t_ms)
    assert all(0 <= x < R for x in e)
    order = sorted(range(J), key=lambda j: (e[j], P[j]["src_end_ms"], P[j]["src_start_ms"], P[j]["utt"], P[j]["k"]))
    S_old = tg["S_ids"]
    ids, kind, chunk_of, unit_of, wstart, chunks = [], [], [], [], [], []
    for c, j in enumerate(order):
        p = P[j]
        pi = S_old[p["tok0"]: p["tok1"]]
        strs = tok.convert_ids_to_tokens(pi[2:-1])
        ch = {"piece": p["piece"], "text": p["text"], "slot": p["slot"], "utt": p["utt"], "k": p["k"],
              "src_start_ms": p["src_start_ms"], "src_end_ms": p["src_end_ms"], "commit_ms": p["src_end_ms"] + LAG_MS,
              "q_start": q(p["src_start_ms"] / 1000), "q_end": q(p["src_end_ms"] / 1000),
              "tok0": len(ids), "tok1": len(ids) + len(pi), "n_words": p["n_words"], "e_read": int(e[j]),
              "u_read": int(u[j])}
        for key, v in extra.items():
            ch[key] = v[j]
        chunks.append(ch)
        ids += pi
        kind += [K_SPK, K_TA] + [K_TEXT] * (len(pi) - 3) + [K_TB]
        chunk_of += [c] * len(pi)
        unit_of += [p["utt"]] * len(pi)
        wstart += [0, 0] + [int(k == 0 or s.startswith("Ġ")) for k, s in enumerate(strs)] + [0]
    S = "".join(c["piece"] for c in chunks)
    joint = tok(S, add_special_tokens=False)["input_ids"]
    assert joint == ids, f"joint tokenisation differs from per-piece: {w['id']}"
    N = len(ids)
    beta = [N if r == R - 1 else sum(c["tok1"] - c["tok0"] for c in chunks if c["e_read"] <= r) for r in range(R)]
    assert all(a <= b for a, b in zip(beta, beta[1:]))
    utts = w["utts"]
    units = {}
    for c in chunks:
        ui = c["utt"]
        if ui not in units:
            uu = utts[ui]
            a, b = uu["begin_s"], uu["end_s"]
            ov = sum(max(0.0, min(b, v["end_s"]) - max(a, v["begin_s"])) for v in utts
                     if v["speaker_id"] != uu["speaker_id"])
            units[ui] = {"utt": ui, "a_ms": round(a * 1000), "b_ms": round(b * 1000), "slot": c["slot"],
                         "speaker_id": uu["speaker_id"], "ov_s": round(ov, 3), "words": len(uu["ref"].split())}
    lo, hi, req, tau = [], [], [], []
    for jt in range(N):
        c = chunks[chunk_of[jt]]
        un = units[c["utt"]]
        kd = kind[jt]
        rq = c["q_start"] * 80 if kd in (K_SPK, K_TA) else un["a_ms"] if kd == K_TEXT else c["q_end"] * 80
        req.append(rq)
        lo.append(first_read_at(t_ms, rq))
        hi.append(first_read_at(t_ms, un["b_ms"] + DMAX_MS))
        tau.append(c["src_end_ms"])
    r_of = np.zeros(N, np.int64)
    prev = 0
    for r, b in enumerate(beta):
        r_of[prev:b] = r
        prev = b
    lat_heur = float(np.mean((np.array(t_ms)[r_of] - np.array(tau)) / 1000.0))
    return {"id": w["id"], "meeting": w["meeting"], "start_s": w["start_s"], "dur_ms": w["dur_ms"], "audio": w["audio"],
            "post": w["post"]["lowlat"], "t_ms": t_ms, "S": S, "S_ids": ids, "kind": kind, "chunk": chunk_of,
            "unit": unit_of, "wstart": wstart, "lo": lo, "hi": hi, "req_ms": req, "tau_ms": tau, "beta_heur": beta,
            "lat_heur": lat_heur, "n_tok": N, "n_read": R, "seg_bounds": [c["tok1"] for c in chunks], "chunks": chunks,
            "units": [units[k] for k in sorted(units)], "label_src": extra["label_src"][0] if J else None}


def check_record(rec: dict) -> dict:
    """chunk_reads == e_read; kap(all) == S order == precheck_serial.kappa; per-speaker monotone e / u; the label
    path's v2' rows == C0's rows (serial_v2p_train._check_one)."""
    from experiments.mist_w3po_v1.crf import precheck_serial as S
    from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
    from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T

    rho = S.chunk_reads(rec)
    n = len(rec["chunks"])
    ok = rho == [c["e_read"] for c in rec["chunks"]]
    ok &= V.kap(rec, rho, range(n)) == list(range(n)) == S.kappa(rec, rho, range(n))
    last = {}
    for c in rec["chunks"]:
        pe, pu = last.get(c["slot"], (-1, -1))
        ok &= c["e_read"] >= pe and c["u_read"] >= pu
        last[c["slot"]] = (c["e_read"], c["u_read"])
    v = V2T._check_one(rec)
    return {"ok": bool(ok and v["ok"]), "p1": v["p1"], "p2": v["p2"], "tok1": v["tok1"], "tok2": v["tok2"], "R": v["R"]}


def _job(args):
    k, ws, tgs, labs, smap, mode, out = args
    from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer

    tok = tokenizer()
    st, chk = Counter(), {}
    with open(Path(out) / f"shard_{k:03d}.jsonl", "x", encoding="utf-8") as fh:
        for w in ws:
            tg = tgs.get(w["id"]) or T0.train_target(w, smap[w["meeting"]], tok, None)
            u = T0.prop_reads(tg)
            J = len(tg["pieces"])
            if mode == "t0" and w["id"] in labs:
                lb = labs[w["id"]]
                assert lb["t_ms"] == tg["t_ms"] and lb["u"] == u
                e, src = lb["e"], "t0"
                extra = {"t0_e_raw": lb["e_raw"], "t0_zero": lb["zero"]}
            else:
                e, src = u, ("prop" if mode == "prop" else "prop_unused")
                extra = {"t0_e_raw": [None] * J, "t0_zero": [None] * J}
            extra["label_src"] = [src] * J
            rec = sched_record(w, tg, e, u, extra, tok)
            st[src] += 1
            st["pieces"] += J
            st["tokens"] += rec["n_tok"]
            st["reads"] += rec["n_read"]
            st["final_pieces"] += sum(c["e_read"] == rec["n_read"] - 1 for c in rec["chunks"])
            if w["id"] in tgs:
                chk[w["id"]] = check_record(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return dict(st), chk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t0-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", choices=["t0", "prop"], default="t0")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="smoke: the first n windows (the plan check is skipped)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    src = Path(T0.SRC_TRAIN)
    smap = json.loads((src / "slot_map.json").read_text(encoding="utf-8"))["lowlat"]
    W = [json.loads(l) for l in open(src / "windows_train_ref.jsonl", encoding="utf-8")]
    W.sort(key=lambda w: w["id"])
    if args.limit:
        W = W[: args.limit]
    root = Path(args.t0_root)
    tgs = {r["id"]: r for r in map(json.loads, open(root / "cpu" / "targets.jsonl", encoding="utf-8"))}
    labs = {}
    if args.labels == "t0":
        labs = {r["id"]: r for r in map(json.loads, open(root / "labels.jsonl", encoding="utf-8"))}
        assert set(labs) == set(tgs), "labels and targets differ"
    out.mkdir(parents=True)
    shards = [W[i: i + args.shard_size] for i in range(0, len(W), args.shard_size)]
    jobs = [(k, sh, {w["id"]: tgs[w["id"]] for w in sh if w["id"] in tgs}, {w["id"]: labs[w["id"]] for w in sh
            if w["id"] in labs}, {m: smap[m] for m in {w["meeting"] for w in sh}}, args.labels, str(out))
            for k, sh in enumerate(shards)]
    import multiprocessing as mp

    st, chk = Counter(), {}
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for i, (s, c) in enumerate(pool.imap_unordered(_job, jobs)):
            st.update(s)
            chk.update(c)
            print(f"shard {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    # the trainer's own view: load_windows + plan_steps + set_reach on this directory
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    wins = T.load_windows(str(out), True)
    plan = T.plan_steps([w.R for w in wins], T0.PLAN_SEED, T0.PLAN_STEPS, T0.PLAN_ROWS)
    used = sorted({i for s in plan for i in s})
    used_ids = {wins[i].rec["id"] for i in used}
    plan_ok = bool(args.limit) or (used_ids == set(tgs) and all(wins[i].rec["label_src"] == args.labels
                                                                  or args.labels == "prop" for i in used))
    uw = [wins[i] for i in used]
    for k in range(0, len(uw), 256):
        T.set_reach(uw[k: k + 256], 128)
    infeasible = [w.rec["id"] for w in uw if not w.feasible]
    bad = sorted(i for i, c in chk.items() if not c["ok"])
    per_step = {}
    for key in ("p1", "p2", "tok1", "tok2"):
        v = [sum(chk[wins[i].rec["id"]][key] for i in s if wins[i].rec["id"] in chk) for s in plan]
        per_step[key] = {"mean": float(np.mean(v)), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                         "max": int(np.max(v))}
    lat = np.array([w.lat_heur for w in wins])
    summ = {"windows": len(wins), "stats": dict(st), "plan_windows": len(used), "plan_equals_t0_windows": plan_ok,
            "plan_windows_infeasible": infeasible[:20], "n_infeasible": len(infeasible),
            "label_path_check_windows": len(chk), "label_path_check_bad": bad[:20], "n_bad": len(bad),
            "v2p_plan_per_step": per_step, "L_ref_mean_lat_heur_windows": float(lat.mean()),
            "lat_heur_plan_windows": float(np.mean([w.lat_heur for w in uw])), "args": vars(args),
            "elapsed_s": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1), flush=True)
    if not plan_ok or infeasible or bad:
        raise SystemExit("PIECE_SCHED_CHECK_FAILED")
    print(f"PIECE_SCHED_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
