"""Scheme 1' precheck follow-up: why does the gold OPENING token of a short unit get low probability under v8?
(reports/mist_w3po_v1/sched_precheck_v8.zh-CN.md §4.3; GPU scoring with the engine path of crf/precheck_sched.py.)

Two explanations:
  ATTRIBUTION  v8 would say the interjection under the wrong speaker (or merge it into the previous speaker)
  ORDER        v8 would first emit another speaker's pending segment (the serialization order differs)

Units: the reference utterances with duration <= --max-dur (1 s) of the windows of a precheck scored dir (the same
seeded sample: train 500 / dev 300). Per unit u (gold slot k, opening token j0 = its first chunk's speaker token)
and per read r in {final read, the heuristic commit read of j0} (one item when they coincide), prefix = S[:j0]
(gold), audio <= t_r, <wait> barred at the final read (as Engine.read / precheck_sched):

  pass 1  G = prefix + the gold first chunk (<spk_k> time text time): the full next-token distribution at the opening
          position, collapsed into classes (<spk_1..8> separately, <wait>, <eos>, time tokens, text = the rest),
          its argmax, and lpG = mean log-prob of the chunk's tokens after <spk_k>
  pass 2  with k' = the most probable speaker token other than k (= the argmax when the argmax is a wrong speaker):
          A = prefix + <spk_k'> + the gold chunk after its speaker token     lpA = mean log-prob of those tokens
          B = prefix + the first chunk of k' in S after j0 (pending at j0)   lpB = same, after <spk_k'> (if any)
          prev_same = k' is the slot of the segment right before j0 in S (collapse into the previous speaker)

DECISION RULE (pre-stated in this file before any result was looked at):
  per item, by the argmax class at the opening position:
    gold speaker token                 -> GOLD
    a speaker token k' != k            -> if k' has no pending chunk after j0: ATTRIBUTION_NOPENDING
                                          elif lpA - lpB >= 0.5 nats: ATTRIBUTION
                                          elif lpB - lpA >= 0.5 nats: ORDER
                                          else: AMBIGUOUS
    a time or text token               -> OTHER_CONT (continuation without a new speaker header)
    <eos> or <wait>                    -> END_WAIT
  summarised per read (final / heuristic), for <= 1 s and <= 0.5 s units, isolated vs ov50 (other speakers cover
  >= 50 % of the unit); dev (never trained on) is the primary evidence. The same A/B comparison is also reported for
  the runner-up speaker of GOLD items (secondary, not part of the rule).

    python -m experiments.mist_w3po_v1.crf.rescore_openers --scored <precheck scored dir(s)> --out <new dir> \
        [--shard K/N] [--limit n]
    python -m experiments.mist_w3po_v1.crf.rescore_openers --summarise <dir>[,<dir>...] --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import diag_short_units as D
from experiments.mist_w3po_v1.crf import precheck_sched as P

WAIT_ID, EOS_ID = 151649, 151645
NEW0, N_TIME, N_SLOT = 151676, 251, 8
SPK0 = NEW0 + N_TIME
CLASSES = [f"spk{k}" for k in range(1, N_SLOT + 1)] + ["wait", "eos", "time", "text"]
MARGIN = 0.5


# ------------------------------------------------------------------------------------------------ items (CPU)
def spk_id(slot: int) -> int:
    return SPK0 + slot - 1


def unit_items(recs: list[dict], max_dur: float) -> list[dict]:
    """One item per (short unit, read) in a deterministic order; the heuristic read is merged into the final one when
    they coincide (reads = ["final", "heur"])."""
    out = []
    for rec in recs:
        R = rec["n_read"]
        rh = D.heur_reads(rec)
        chunks = sorted(rec["chunks"], key=lambda c: c["tok0"])
        for u in rec["units"]:
            dur = (u["b_ms"] - u["a_ms"]) / 1000.0
            if dur > max_dur:
                continue
            first = min((c for c in chunks if c["utt"] == u["utt"]), key=lambda c: c["tok0"])
            j0 = first["tok0"]
            assert rec["S_ids"][j0] == spk_id(first["slot"])
            prev = next((c for c in chunks if c["tok1"] == j0), None)
            base = {"id": rec["id"], "utt": u["utt"], "dur_s": dur, "words": u["words"], "ov50": D.ov50(rec["units"], u),
                    "slot": first["slot"], "j0": j0, "tok1": first["tok1"], "R": R,
                    "prev_slot": prev["slot"] if prev else None}
            r_final, r_heur = R - 1, int(rh[j0])
            if r_heur == r_final:
                out.append({**base, "r": r_final, "reads": ["final", "heur"]})
            else:
                out.append({**base, "r": r_final, "reads": ["final"]})
                out.append({**base, "r": r_heur, "reads": ["heur"]})
    return out


def pending_chunk(rec: dict, j0: int, slot: int) -> dict | None:
    """The first chunk of speaker `slot` that starts after j0 in S (not committed at the opening position)."""
    cs = [c for c in rec["chunks"] if c["slot"] == slot and c["tok0"] > j0]
    return min(cs, key=lambda c: c["tok0"]) if cs else None


def forced_G(rec: dict, it: dict) -> list[int]:
    return list(rec["S_ids"][it["j0"]: it["tok1"]])


def forced_A(rec: dict, it: dict, k2: int) -> list[int]:
    return [spk_id(k2)] + list(rec["S_ids"][it["j0"] + 1: it["tok1"]])


def forced_B(rec: dict, c: dict) -> list[int]:
    return list(rec["S_ids"][c["tok0"]: c["tok1"]])


def classify(it: dict) -> str:
    """The pre-stated decision rule (module docstring)."""
    cls = it["argmax_class"]
    if cls == f"spk{it['slot']}":
        return "GOLD"
    if cls.startswith("spk"):
        if it.get("lpB") is None:
            return "ATTRIBUTION_NOPENDING"
        d = it["lpA"] - it["lpB"]
        return "ATTRIBUTION" if d >= MARGIN else "ORDER" if d <= -MARGIN else "AMBIGUOUS"
    if cls in ("time", "text"):
        return "OTHER_CONT"
    return "END_WAIT"


def collapse(lp_row: np.ndarray) -> dict:
    """[V] log-probs -> class masses."""
    p = np.exp(lp_row.astype(np.float64))
    m = {f"spk{k}": float(p[spk_id(k)]) for k in range(1, N_SLOT + 1)}
    m["wait"], m["eos"] = float(p[WAIT_ID]), float(p[EOS_ID])
    m["time"] = float(p[NEW0:SPK0].sum())
    m["text"] = float(max(0.0, 1.0 - sum(m.values())))
    return m


def argmax_class(tok: int) -> str:
    if SPK0 <= tok < SPK0 + N_SLOT:
        return f"spk{tok - SPK0 + 1}"
    if tok == WAIT_ID:
        return "wait"
    if tok == EOS_ID:
        return "eos"
    if NEW0 <= tok < SPK0:
        return "time"
    return "text"


# ------------------------------------------------------------------------------------------------ scoring (GPU)
def score(eng, reqs: list[tuple], batch: int, final_bar_wait: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """reqs: (rec, r, prefix ids, forced ids). Per req: (log-probs [V] at the position after the prefix, float32 on
    CPU; log-probs of the forced tokens [m]). Engine.read-style left padding, <wait> barred at the final read."""
    import torch

    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD, ReadItem
    from experiments.mist_w3po_v1.rl.rollout import load_window

    wavs = {}
    out = []
    emb_dtype = eng.emb_layer.weight.dtype
    for s0 in range(0, len(reqs), batch):
        part = reqs[s0: s0 + batch]
        its, ms = [], []
        for rec, r, prefix, forced in part:
            if rec["id"] not in wavs:
                wavs[rec["id"]] = load_window({"audio": rec["audio"], "post": {"lowlat": rec["post"]}})
            wav, post = wavs[rec["id"]]
            its.append(ReadItem(wav=wav, post=post, t_ms=rec["t_ms"][r], final=r == rec["n_read"] - 1,
                                committed=list(prefix) + list(forced), ctx="", akey=f"{rec['id']}|{rec['t_ms'][r]}"))
            ms.append(len(forced))
        with torch.no_grad():
            ae = eng.audio_embeds(its, emb_dtype)
            prompts = [eng.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(its, ae)]
            B, L, M = len(its), max(len(p) for p in prompts), max(ms)
            ids = torch.full((B, L), eng.tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((B, L), dtype=torch.long)
            for i, p in enumerate(prompts):
                ids[i, L - len(p):] = torch.as_tensor(p)
                mask[i, L - len(p):] = 1
            ids, mask = ids.to(eng.dev), mask.to(eng.dev)
            emb = eng.emb_layer(ids)
            emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
            h = eng.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                                  return_dict=True).last_hidden_state[:, L - M - 1:]
            lg = eng.logits(h).to(eng.dev)                                    # [B, M+1, V] float32
            for i, it in enumerate(its):
                if it.final and final_bar_wait:
                    lg[i, :, WAIT_ID] = -1e9
            lsm = torch.log_softmax(lg, -1)
            for i, (rec, r, prefix, forced) in enumerate(part):
                rows = lsm[i, M - ms[i]:]                                     # row t predicts forced[t]
                first = rows[0].float().cpu().numpy()
                f_lp = (rows[torch.arange(ms[i], device=eng.dev), torch.as_tensor(forced, device=eng.dev)].double().cpu().numpy()
                        if ms[i] else np.zeros(0))
                out.append((first, f_lp))
        del h, lg, lsm
    return out


def run(args) -> None:
    from experiments.mist_w3po_v1.core.engine import Engine

    sched, ids, _ = P.load_scored([d for d in args.scored.split(",") if d])
    want, allr = set(ids), {}
    for f in sorted(Path(sched).glob("shard_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in want:
                allr[r["id"]] = r
    recs = [allr[i] for i in ids]
    items = unit_items(recs, args.max_dur)
    n_all = len(items)
    sh = P.parse_shard(args.shard)
    if sh:
        items = P.shard_select(items, *sh)
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True)
    (out / "meta.json").write_text(json.dumps({"scored": args.scored, "sched": sched, "items_all": n_all,
                                               "items": len(items), "args": vars(args)}, indent=1))
    print(f"{len(items)} items (of {n_all}) from {len(recs)} windows", flush=True)
    t0 = time.time()
    eng = Engine.from_merged(args.thinker, "zh2en", args.max_gib)
    byid = {r["id"]: r for r in recs}
    bar = not args.no_final_bar_wait
    # pass 1: G
    res1 = score(eng, [(byid[it["id"]], it["r"], byid[it["id"]]["S_ids"][: it["j0"]], forced_G(byid[it["id"]], it))
                       for it in items], args.batch, bar)
    for it, (first, f_lp) in zip(items, res1):
        rec = byid[it["id"]]
        it["masses"] = collapse(first)
        top = int(np.argmax(first))
        it["argmax_tok"], it["argmax_class"] = top, argmax_class(top)
        it["lp_open"] = float(first[spk_id(it["slot"])])
        it["lpG"] = float(f_lp[1:].mean())
        others = [(it["masses"][f"spk{k}"], k) for k in range(1, N_SLOT + 1) if k != it["slot"]]
        it["k2"] = max(others)[1]
        it["k2_is_argmax"] = it["argmax_class"] == f"spk{it['k2']}"
        it["prev_same"] = it["prev_slot"] is not None and it["k2"] == it["prev_slot"]
        c = pending_chunk(rec, it["j0"], it["k2"])
        it["pending"] = None if c is None else {"tok0": c["tok0"], "tok1": c["tok1"], "utt": c["utt"],
                                                 "src_start_ms": c["src_start_ms"], "src_end_ms": c["src_end_ms"],
                                                 "started": c["src_start_ms"] <= rec["t_ms"][it["r"]],
                                                 "heard": c["src_end_ms"] <= rec["t_ms"][it["r"]]}
    print(f"pass 1 done {time.time() - t0:.0f}s", flush=True)
    # pass 2: A and B for the runner-up speaker k2 (the argmax when the argmax is a wrong speaker)
    reqs, where = [], []
    for n, it in enumerate(items):
        rec = byid[it["id"]]
        pre = rec["S_ids"][: it["j0"]]
        reqs.append((rec, it["r"], pre, forced_A(rec, it, it["k2"])))
        where.append((n, "A"))
        if it["pending"] is not None:
            reqs.append((rec, it["r"], pre, forced_B(rec, it["pending"])))
            where.append((n, "B"))
    res2 = score(eng, reqs, args.batch, bar)
    for (n, kind), (first, f_lp) in zip(where, res2):
        items[n]["lp" + kind] = float(f_lp[1:].mean())
        items[n]["lp" + kind + "_spk"] = float(f_lp[0])
    for it in items:
        it.setdefault("lpB", None)
        it["label"] = classify(it)
    import torch

    peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
    with open(out / "items.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(it) + "\n" for it in items)
    print(f"RESCORE_OPENERS_DONE {out} items {len(items)} forwards {len(items) + len(reqs)} "
          f"s {time.time() - t0:.0f} peak_alloc_gib {peak:.1f}", flush=True)


# ------------------------------------------------------------------------------------------------ summary (CPU)
def qs(x) -> dict | None:
    x = np.asarray([v for v in x if v is not None], np.float64)
    if not len(x):
        return None
    return {"n": int(len(x)), "median": float(np.median(x)), "q1": float(np.percentile(x, 25)),
            "q3": float(np.percentile(x, 75)), "mean": float(x.mean())}


def summarise(items: list[dict]) -> dict:
    groups = {"all": lambda it: True, "le0.5s": lambda it: it["dur_s"] <= 0.5,
              "iso": lambda it: not it["ov50"], "ov50": lambda it: it["ov50"],
              "le0.5s_iso": lambda it: it["dur_s"] <= 0.5 and not it["ov50"],
              "le0.5s_ov50": lambda it: it["dur_s"] <= 0.5 and it["ov50"]}
    labels = ["GOLD", "ATTRIBUTION", "ATTRIBUTION_NOPENDING", "ORDER", "AMBIGUOUS", "OTHER_CONT", "END_WAIT"]
    out = {}
    for read in ("final", "heur"):
        for g, f in groups.items():
            sub = [it for it in items if read in it["reads"] and f(it)]
            if not sub:
                continue
            lab = Counter(it["label"] for it in sub)
            wrong = [it for it in sub if it["argmax_class"].startswith("spk") and it["label"] != "GOLD"]
            gold = [it for it in sub if it["label"] == "GOLD"]
            am = Counter("gold_spk" if it["argmax_class"] == f"spk{it['slot']}" else
                         ("prev_spk" if it["prev_slot"] is not None and it["argmax_class"] == f"spk{it['prev_slot']}"
                          else "other_spk") if it["argmax_class"].startswith("spk") else it["argmax_class"]
                         for it in sub)

            def mass(it, key):
                m = it["masses"]
                if key == "gold_spk":
                    return m[f"spk{it['slot']}"]
                if key == "prev_spk":
                    return m[f"spk{it['prev_slot']}"] if it["prev_slot"] not in (None, it["slot"]) else 0.0
                if key == "other_spk":
                    return sum(m[f"spk{k}"] for k in range(1, N_SLOT + 1)
                               if k != it["slot"] and k != it["prev_slot"])
                return m[key]

            d = {"items": len(sub),
                 "labels": {k: lab.get(k, 0) / len(sub) for k in labels},
                 "label_counts": {k: lab.get(k, 0) for k in labels},
                 "argmax_class": {k: v / len(sub) for k, v in am.most_common()},
                 "mass": {k: qs([mass(it, k) for it in sub]) for k in
                          ("gold_spk", "prev_spk", "other_spk", "wait", "eos", "time", "text")},
                 "prev_differs_from_gold": float(np.mean([it["prev_slot"] not in (None, it["slot"]) for it in sub])),
                 "wrong_spk_argmax": {
                     "n": len(wrong),
                     "collapse_into_prev": float(np.mean([it["prev_same"] for it in wrong])) if wrong else None,
                     "has_pending": float(np.mean([it["pending"] is not None for it in wrong])) if wrong else None,
                     "pending_started": float(np.mean([it["pending"]["started"] for it in wrong
                                                       if it["pending"]])) if wrong else None,
                     "lpA": qs([it["lpA"] for it in wrong]), "lpB": qs([it["lpB"] for it in wrong]),
                     "lpG": qs([it["lpG"] for it in wrong]),
                     "lpA_minus_lpB": qs([it["lpA"] - it["lpB"] for it in wrong if it["lpB"] is not None]),
                     "lpA_minus_lpG": qs([it["lpA"] - it["lpG"] for it in wrong])},
                 "gold_argmax_runner_up": {
                     "n": len(gold),
                     "lpA_minus_lpB": qs([it["lpA"] - it["lpB"] for it in gold if it["lpB"] is not None]),
                     "rule_on_runner_up": dict(Counter(
                         "NOPENDING" if it["lpB"] is None else "ATTRIBUTION" if it["lpA"] - it["lpB"] >= MARGIN
                         else "ORDER" if it["lpB"] - it["lpA"] >= MARGIN else "AMBIGUOUS" for it in gold))}}
            out[f"{read}|{g}"] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", default="", help="precheck scored dir(s) (the window sample)")
    ap.add_argument("--summarise", default="", help="rescore output dir(s) to merge and summarise (CPU)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinker", default=P.V8)
    ap.add_argument("--max-gib", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-dur", type=float, default=1.0)
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-final-bar-wait", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    if args.summarise:
        items = []
        for d in args.summarise.split(","):
            if d:
                items += [json.loads(l) for l in open(Path(d) / "items.jsonl", encoding="utf-8")]
        keys = [(it["id"], it["utt"], it["r"]) for it in items]
        assert len(keys) == len(set(keys)), "an item is in more than one dir"
        res = {"items": len(items), "units": len({(it["id"], it["utt"]) for it in items}),
               "rule": f"argmax wrong speaker k': lpA - lpB >= {MARGIN} ATTRIBUTION, <= -{MARGIN} ORDER, else "
                       f"AMBIGUOUS; no pending chunk of k': ATTRIBUTION_NOPENDING", "groups": summarise(items)}
        out.mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1))
        print(f"RESCORE_SUMMARY_DONE {out}", flush=True)
        return
    run(args)


if __name__ == "__main__":
    main()
