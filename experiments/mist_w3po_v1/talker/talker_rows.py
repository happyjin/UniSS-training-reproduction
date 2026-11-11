"""Training rows of the two talker schemes from si_infer --dump-cands runs (reports/mist_w3po_v1/
talker_crf_prereg_v1.zh-CN.md §1.3, §2.3). CPU only.

Streams: one per dumped item (window x speaker); block availability a_n = speech_latency_lit's online protocol
(max(write_ms, open_detect_ms); a block written at/after the window end also >= close_detect_ms).

plem    the PL-EM E step: lattice per stream (playback_lattice.Stream: tau 0.5, beta 5, f = -min(NLL, 10)), lambda by the
        dual bisection to budget = (the A' rule path's own corpus latency) - 0.3 s; per block, q(c | b_n = the A' path's
        state) over its 8 candidates; candidate rows with q >= 0.01. Own sequence = the backlog format (S2T speed =
        speed_base, backlog token; TTS speed = the candidate's), history = the A' rule path; reference sequence (T1c's
        KL) = the native format with both speed tokens = the candidate's speed. Speed soft label q(v | b_n) on the
        35 speed tokens at the TTS speed position, once per block.
align   scheme 2: native format (A''s talker input); the offset-0 candidates of a block; chosen = cover 1 first, then the
        lowest NLL (no full cover: highest cover, then NLL) -> CE + KL; every offset-0 candidate with a cover value is
        an alignment row, label 1 iff cover == 1.

bon     (2026-11-11, talker_crf_prereg_v2 experiment 1, the "no lattice" ablation) PL-EM's rows with the weights
        replaced: per block, q(c) ~ exp(tau*lpbar + beta*f) over its candidates (same tau 0.5, beta 5, f) -- no latency
        term, no lambda, no backward messages, blocks independent; same Q_MIN, sequences, history, speed soft label
        (from this q). Saved with mode "plem" (pretrain_talker_crf --crf-mode plem reads them unchanged), variant "bon".
--scope meeting  (plem only; experiment 2) one shared output channel per meeting (shared_channel.py): one lattice
        chain per meeting, blocks FIFO by meeting-absolute availability, state = the shared backlog (Delta 80 ms,
        cap MEETING_CAP_MS, rounding "nearest"), l_n the same word-delay proxy on the shared channel, lambda by the
        dual bisection to budget = (the A' rule path's own shared-channel corpus latency) - 1.0 s; q(c | b_n = the A'
        path's exact shared backlog, nearest grid state); every block's backlog token (history included) = the A'
        path's shared-backlog bin. Saved with mode "plem", variant "meeting". Default --scope stream: unchanged.

Writes <out> (torch.save of a dict; refuses to overwrite) and <out>.json (stats).

    python -m experiments.mist_w3po_v1.talker.talker_rows --mode plem --cands '<dir>/cands_*.jsonl' --win <windows> --out <file>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.talker import playback_lattice as PL
from experiments.mist_w3po_v1.talker import shared_channel as SC

SEMANTIC_0, END_SEMANTIC, WAIT_READ, STREAMING_MODE = 155761, 180392, 180395, 180406
TASK_S2S, BALANCE, START_GLOBAL, END_GLOBAL = 180379, 180400, 180387, 180391
GLOBAL_0, GLM_0 = 151665, 163953
WRITE, TASK_S2T, TASK_TTS = 180396, 180380, 180375
START_CONTENT, END_CONTENT, START_SEMANTIC = 180386, 180390, 180388
LANG_ENG = 180373
SPEED_0, N_SPEED = 180337, 35
BACKLOG_IDS = (180381, 180382, 180383, 180384, 180385, 180397, 180398, 180405)
Q_MIN = 0.01
BUDGET_DROP_S = 0.3
BUDGET_DROP_1CH_S = 1.0      # --scope meeting (talker_crf_prereg_v2 experiment 2)
MEETING_CAP_MS = 32000       # A' path's shared backlog on the train rows: max 31.4 s (10% of blocks > 8 s)
MEETING_ROUNDING = "nearest"


def closing(c: dict, final: bool) -> int:
    return c["closing"] if c.get("closing") in (WAIT_READ, END_SEMANTIC) else (END_SEMANTIC if final else WAIT_READ)


def prefix(it: dict, r: int, backlog: bool, lang: int = LANG_ENG) -> tuple[list[int], list[int]]:
    """Block r's input up to and including <write> <task_tts> <lang> (history = the chosen candidates), split as
    (head up to and including the S2T <lang>, rest from START_CONTENT ... <lang>) so the S2T speed token (native
    format: the block's speed) can be put between. backlog: head ends with nothing; the caller adds speed_base and the
    backlog token. Mirrors si_infer.build_seq."""
    bl = it["blocks"]
    glm = bl[r]["glm"]
    seq = [TASK_S2S, BALANCE, lang, START_GLOBAL] + [GLOBAL_0 + g for g in it["voice"]] + [END_GLOBAL]
    g_prev = 0
    for k, b in enumerate(bl[: r + 1]):
        seq += [GLM_0 + t for t in glm[g_prev: b["glm_end"]]]
        if k == r:
            head = seq + [WRITE, TASK_S2T, lang]
            rest = [START_CONTENT] + b["ids"] + [END_CONTENT, WRITE, TASK_TTS, lang]
            return head, rest
        if backlog:
            seq += [WRITE, TASK_S2T, lang, b["speed_base"], BACKLOG_IDS[b["backlog_bin"]], START_CONTENT] + b["ids"] + [END_CONTENT]
        else:
            seq += [WRITE, TASK_S2T, lang, b["speed"], START_CONTENT] + b["ids"] + [END_CONTENT]
        c = b["cands"][b["chosen"]]
        seq += [WRITE, TASK_TTS, lang, b["speed"]] + ([] if b["final"] else [STREAMING_MODE]) + [START_SEMANTIC]
        seq += [SEMANTIC_0 + x for x in c["codes"]] + [closing(c, b["final"])]
        g_prev = b["glm_end"]
    raise AssertionError("unreachable")


def full_seq(it: dict, r: int, v: int, backlog: bool) -> list[int]:
    """the whole input of block r with TTS speed v (native: S2T speed v as well), ending at START_SEMANTIC."""
    head, rest = prefix(it, r, backlog)
    b = it["blocks"][r]
    mid = [b["speed_base"], BACKLOG_IDS[b["backlog_bin"]]] if backlog else [v]
    return head + mid + rest + [v] + ([] if b["final"] else [STREAMING_MODE]) + [START_SEMANTIC]


def availability(w: dict, write_ms: int) -> float:
    c = max(write_ms, w["open_detect_ms"])
    if write_ms >= w["dur_ms"]:
        c = max(c, w["close_detect_ms"])
    return float(c)


def feat(c: dict) -> float:
    lpbar = (c["lp"] or 0.0) / max(1, c["ntok"])
    nll = PL.NLL_CAP if c["nll"] is None else min(float(c["nll"]), PL.NLL_CAP)
    return PL.TAU * lpbar + PL.BETA * (-nll)


def stream_of(it: dict, w: dict) -> PL.Stream:
    bl = it["blocks"]
    return PL.Stream([availability(w, b["write_ms"]) for b in bl],
                     [[len(c["codes"]) * 20 for c in b["cands"]] for b in bl],
                     [[feat(c) for c in b["cands"]] for b in bl],
                     [len(b["text"].split()) for b in bl])


def load_items(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["blocks"] and all(b["cands"] for b in r["blocks"]):
                out.append(r)
    return out


def windows(path: str) -> dict:
    return {w["id"]: w for w in map(json.loads, open(path, encoding="utf-8"))}


def lattice(items: list[dict], W: dict, drop: float = BUDGET_DROP_S):
    """-> streams, lambda, stats (the A' rule path's corpus latency, the budget, E_q at lambda)."""
    streams = [stream_of(it, W[it["id"].split("__S")[0]]) for it in items]
    words = sum(sum(st.words) for st in streams)
    a_lat = sum(st.path_latency([b["chosen"] for b in it["blocks"]]) for st, it in zip(streams, items)) / words
    lam, e = PL.solve_lambda(streams, a_lat - drop)
    return streams, lam, {"aprule_latency_s": round(a_lat, 4), "budget_s": round(a_lat - drop, 4), "lambda": lam,
                          "eq_latency_s": round(e, 4), "eq_latency_lam0_s": round(PL.corpus_latency(streams, 0.0), 4),
                          "streams": len(streams), "words": words}


def plem_rows(items: list[dict], W: dict) -> tuple[list[dict], dict]:
    streams, lam, stats = lattice(items, W)
    blocks, kl_u, ent = [], [], []
    for it, st in zip(items, streams):
        _, marg, _, beta = st.forward_backward(lam)
        states = st.path_states([b["chosen"] for b in it["blocks"]])
        for r, b in enumerate(it["blocks"]):
            C = len(b["cands"])
            qm = marg[r].sum(0)
            qm = qm / qm.sum()
            kl_u.append(float(np.sum(qm * (np.log(np.maximum(qm, 1e-300)) + np.log(C)))))
            ent.append(float(-np.sum(qm * np.log(np.maximum(qm, 1e-300)))))
            q = st.conditional(r, states[r], lam, beta)
            qv = np.zeros(N_SPEED)
            for c, x in zip(b["cands"], q):
                qv[c["v"] - SPEED_0] += x
            head, rest = prefix(it, r, True)
            head_ref, rest_ref = prefix(it, r, False)
            cands = []
            for c, x in zip(b["cands"], q):
                if x < Q_MIN:
                    continue
                tail = [c["v"]] + ([] if b["final"] else [STREAMING_MODE]) + [START_SEMANTIC]
                y = [SEMANTIC_0 + t for t in c["codes"]] + [closing(c, b["final"])]
                cands.append({"own": head + [b["speed_base"], BACKLOG_IDS[b["backlog_bin"]]] + rest + tail,
                              "ref": head_ref + [c["v"]] + rest_ref + tail, "y": y, "w": float(x)})
            if not cands:
                continue
            s = sum(c["w"] for c in cands)
            for c in cands:
                c["w"] /= s
            spd_pos = len(head) + 2 + len(rest) - 1   # the TTS <lang>: its output predicts the speed token
            blocks.append({"id": it["id"], "r": r, "cands": cands, "spd_pos": spd_pos, "qv": qv.tolist()})
    stats.update({"blocks": len(blocks), "rows": sum(len(x["cands"]) for x in blocks),
                  "p3_kl_uniform_mean": round(float(np.mean(kl_u)), 4) if kl_u else None,
                  "posterior_entropy_mean": round(float(np.mean(ent)), 4) if ent else None})
    return blocks, stats


def block_rows(it: dict, r: int, q) -> dict | None:
    """block r's candidate rows with weights q over its candidates -- plem_rows' construction verbatim (rows with
    q >= Q_MIN, renormalised; the speed soft label from the full q; own = the backlog format with the item's
    backlog_bin tokens, ref = the native format). None when no candidate passes Q_MIN."""
    b = it["blocks"][r]
    qv = np.zeros(N_SPEED)
    for c, x in zip(b["cands"], q):
        qv[c["v"] - SPEED_0] += x
    head, rest = prefix(it, r, True)
    head_ref, rest_ref = prefix(it, r, False)
    cands = []
    for c, x in zip(b["cands"], q):
        if x < Q_MIN:
            continue
        tail = [c["v"]] + ([] if b["final"] else [STREAMING_MODE]) + [START_SEMANTIC]
        y = [SEMANTIC_0 + t for t in c["codes"]] + [closing(c, b["final"])]
        cands.append({"own": head + [b["speed_base"], BACKLOG_IDS[b["backlog_bin"]]] + rest + tail,
                      "ref": head_ref + [c["v"]] + rest_ref + tail, "y": y, "w": float(x)})
    if not cands:
        return None
    s = sum(c["w"] for c in cands)
    for c in cands:
        c["w"] /= s
    spd_pos = len(head) + 2 + len(rest) - 1   # the TTS <lang>: its output predicts the speed token
    return {"id": it["id"], "r": r, "cands": cands, "spd_pos": spd_pos, "qv": qv.tolist()}


def _kl_ent(qm: np.ndarray) -> tuple[float, float]:
    qm = qm / qm.sum()
    return (float(np.sum(qm * (np.log(np.maximum(qm, 1e-300)) + np.log(len(qm))))),
            float(-np.sum(qm * np.log(np.maximum(qm, 1e-300)))))


def bon_weights(b: dict) -> np.ndarray:
    """experiment 1: q(c) ~ exp(tau*lpbar + beta*f) over the block's candidates (feat), no latency term."""
    x = np.asarray([feat(c) for c in b["cands"]], dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def bon_rows(items: list[dict]) -> tuple[list[dict], dict]:
    blocks, kl_u, ent = [], [], []
    for it in items:
        for r, b in enumerate(it["blocks"]):
            q = bon_weights(b)
            k, h = _kl_ent(q)
            kl_u.append(k)
            ent.append(h)
            bk = block_rows(it, r, q)
            if bk is not None:
                blocks.append(bk)
    return blocks, {"weights": "bon", "tau": PL.TAU, "beta": PL.BETA, "items": len(items),
                    "blocks_in": len(kl_u), "blocks": len(blocks), "rows": sum(len(x["cands"]) for x in blocks),
                    "kl_uniform_mean": round(float(np.mean(kl_u)), 4) if kl_u else None,
                    "entropy_mean": round(float(np.mean(ent)), 4) if ent else None}


def chosen_durs(items: list[dict]) -> dict:
    """(item index, block index) -> the A' rule path's duration (ms, 20 ms codes)."""
    return {(n, r): len(b["cands"][b["chosen"]]["codes"]) * 20 for n, it in enumerate(items)
            for r, b in enumerate(it["blocks"])}


def meeting_lattice(items: list[dict], W: dict, drop: float = BUDGET_DROP_1CH_S, cap: int = MEETING_CAP_MS,
                    rounding: str = MEETING_ROUNDING):
    """--scope meeting: one PL.Stream per meeting over its blocks in FIFO order (shared_channel.meeting_queues).
    -> chains (meeting -> [(item index, block index)]), streams (same order), lambda, stats."""
    Q = SC.meeting_queues(items, W)
    chains, streams = {}, []
    for m, L in Q.items():
        chains[m] = [(n, r) for _, n, r in L]
        bl = [items[n]["blocks"][r] for _, n, r in L]
        streams.append(PL.Stream([a for a, _, _ in L], [[len(c["codes"]) * 20 for c in b["cands"]] for b in bl],
                                 [[feat(c) for c in b["cands"]] for b in bl], [len(b["text"].split()) for b in bl],
                                 cap=cap, rounding=rounding))
    words = sum(sum(st.words) for st in streams)
    a_lat = sum(st.path_latency([items[n]["blocks"][r]["chosen"] for n, r in ch])
                for ch, st in zip(chains.values(), streams)) / words
    lam, e = PL.solve_lambda(streams, a_lat - drop)
    return chains, streams, lam, {
        "scope": "meeting", "cap_ms": cap, "rounding": rounding, "aprule_latency_1ch_s": round(a_lat, 4),
        "budget_s": round(a_lat - drop, 4), "lambda": lam, "eq_latency_s": round(e, 4),
        "eq_latency_lam0_s": round(PL.corpus_latency(streams, 0.0), 4), "meetings": len(streams),
        "chain_len_max": max(len(c) for c in chains.values()), "words": words}


def plem_meeting_rows(items: list[dict], W: dict) -> tuple[list[dict], dict]:
    chains, streams, lam, stats = meeting_lattice(items, W)
    bl_ms = SC.shared_backlogs(items, W, chosen_durs(items))
    items2 = [{**it, "blocks": [{**b, "backlog_bin": SC.backlog_bin(bl_ms[(n, r)])} for r, b in enumerate(it["blocks"])]}
              for n, it in enumerate(items)]
    qs, kl_u, ent = {}, [], []
    for ch, st in zip(chains.values(), streams):
        _, marg, _, beta = st.forward_backward(lam)
        for pos, (n, r) in enumerate(ch):
            k, h = _kl_ent(marg[pos].sum(0))
            kl_u.append(k)
            ent.append(h)
            qs[(n, r)] = st.conditional(pos, PL.quant_nearest(bl_ms[(n, r)], st.delta, st.cap), lam, beta)
    blocks = []
    for n, it in enumerate(items2):
        for r in range(len(it["blocks"])):
            bk = block_rows(it, r, qs[(n, r)])
            if bk is not None:
                blocks.append(bk)
    bins = np.bincount([SC.backlog_bin(v) for v in bl_ms.values()], minlength=8)
    bl = np.asarray(list(bl_ms.values())) / 1000.0
    stats.update({"blocks": len(blocks), "rows": sum(len(x["cands"]) for x in blocks),
                  "p3_kl_uniform_mean": round(float(np.mean(kl_u)), 4) if kl_u else None,
                  "posterior_entropy_mean": round(float(np.mean(ent)), 4) if ent else None,
                  "aprule_shared_backlog_s": {"mean": round(float(bl.mean()), 3), "p50": round(float(np.median(bl)), 3),
                                              "p90": round(float(np.percentile(bl, 90)), 3), "max": round(float(bl.max()), 3)},
                  "backlog_bin_hist": bins.tolist()})
    return blocks, stats


def cover_pick(b: dict) -> int | None:
    """§2.2: among the offset-0 candidates: cover 1 first, the lowest NLL; else the highest cover, then NLL."""
    js = [j for j, c in enumerate(b["cands"]) if c["o"] == 0]
    if not js:
        return None
    nll = lambda j: PL.NLL_CAP if b["cands"][j]["nll"] is None else min(float(b["cands"][j]["nll"]), PL.NLL_CAP)
    cov = lambda j: -1.0 if b["cands"][j]["cover"] is None else float(b["cands"][j]["cover"])
    full = [j for j in js if cov(j) >= 1.0]
    if full:
        return min(full, key=nll)
    return min(js, key=lambda j: (-cov(j), nll(j)))


def align_rows(items: list[dict], tok=None) -> tuple[list[dict], dict]:
    """tok: the text tokenizer, for the units' character lengths (duration prior); None -> all 1."""
    blocks, n_pos, n_neg = [], 0, 0
    for it in items:
        for r, b in enumerate(it["blocks"]):
            pick = cover_pick(b)
            if pick is None:
                continue
            head, rest = prefix(it, r, False)
            rows = []
            for j, c in enumerate(b["cands"]):
                if c["o"] != 0:
                    continue
                v = c["v"]
                seq = head + [v] + rest + [v] + ([] if b["final"] else [STREAMING_MODE]) + [START_SEMANTIC]
                y = [SEMANTIC_0 + t for t in c["codes"]] + [closing(c, b["final"])]
                lab = -1 if c["cover"] is None else int(float(c["cover"]) >= 1.0)
                if j != pick and lab < 0:
                    continue
                t0 = len(head) + 2   # START_CONTENT of block r is head + [v] -> index len(head)+1; text from +2
                lens = ([max(1, len(tok.decode([t]).strip())) for t in b["ids"]] if tok is not None
                        else [1] * len(b["ids"]))
                rows.append({"seq": seq, "y": y, "ce": int(j == pick), "lab": lab, "text_start": t0,
                             "M": len(b["ids"]), "text_ids": b["ids"], "lens": lens})
                n_pos += lab == 1
                n_neg += lab == 0
            blocks.append({"id": it["id"], "r": r, "rows": rows})
    fr = sum(len(x["y"]) - 1 for bk in blocks for x in bk["rows"] if x["lab"] == 1)
    ch = sum(sum(x["lens"]) for bk in blocks for x in bk["rows"] if x["lab"] == 1)
    return blocks, {"blocks": len(blocks), "rows": sum(len(x["rows"]) for x in blocks), "align_pos": n_pos,
                    "align_neg": n_neg, "kappa": round(fr / ch, 4) if ch else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("plem", "align", "lattice", "bon"), required=True)
    ap.add_argument("--scope", choices=("stream", "meeting"), default="stream",
                    help="plem: the lattice per stream (default, unchanged) or one shared-channel chain per meeting")
    ap.add_argument("--cands", required=True, help="glob of cands_*.jsonl")
    ap.add_argument("--win", required=True, help="the online windows the run's text was decoded on")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() or Path(str(out) + ".json").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    if a.scope == "meeting" and a.mode != "plem":
        raise SystemExit("--scope meeting: plem only")
    items = load_items(a.cands)
    if a.mode in ("bon", "plem") and (a.mode == "bon" or a.scope == "meeting"):   # talker_crf_prereg_v2
        rows, stats = bon_rows(items) if a.mode == "bon" else plem_meeting_rows(items, windows(a.win))
        variant = "bon" if a.mode == "bon" else "meeting"
        import torch

        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mode": "plem", "variant": variant, "blocks": rows, "stats": stats, "cands": a.cands}, out)
        Path(str(out) + ".json").write_text(json.dumps({"mode": a.mode, "variant": variant, "cands": a.cands, **stats},
                                                       indent=1) + "\n")
        print(json.dumps(stats))
        return
    if a.mode == "plem":
        rows, stats = plem_rows(items, windows(a.win))
    elif a.mode == "align":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
        rows, stats = align_rows(items, tok)
    else:
        raise SystemExit("--mode lattice: use precheck_plcrf.py")
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mode": a.mode, "blocks": rows, "stats": stats, "cands": a.cands}, out)
    Path(str(out) + ".json").write_text(json.dumps({"mode": a.mode, "cands": a.cands, **stats}, indent=1) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
