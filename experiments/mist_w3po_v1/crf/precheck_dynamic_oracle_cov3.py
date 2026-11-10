"""CRF dynamic oracle precheck, coverage judge v3 (re-registration 2026-11-10 ~21:15 UTC, user OK, before any v3 result:
reports/mist_w3po_v1/crf_dynamic_oracle_precheck.zh-CN.md §7). v2 stopped at O1 (iv): its lexical content test called
paraphrase translations "not covered" (§6). v3 changes only the content test of rules b and c; rule a' (no explaining
away, nearest short response), O1 (ii) / (iii), the population, budget, scoring and thresholds are v2's / §1's.

  M3(x, u)  v2's content_match (edge match, same short-response class, light-stemmed content words) OR the cosine of
            all-MiniLM-L12-v2 sentence embeddings (mean pooling, L2-normalised; the tcpSemER model) of x's text and u's
            reference >= SIM_T = 0.3 (fixed here, before any v3 result)
  (a'')     = v2's a'
  (b'')     the scorer credits x to u's unit within 1 s (v1's rule b) AND (M3(x, u) OR (x's mapped speaker is u's speaker
            AND x's own source span overlaps u by >= 0.5 of their union)): a credited segment of the same speaker over
            the same span is u's translation even when it is a wrong one; a credited segment that is neither similar
            nor u's own span (garbage, another sentence repeated) does not cover u
  (c'')     only for u longer than 1 s: own-span IoU >= 0.3 AND M3(x, u)
  review    a fresh sample, seed SEED + 3; agreement < 0.95 -> STOP (unchanged)

    python -m experiments.mist_w3po_v1.crf.precheck_dynamic_oracle_cov3 --o1 --root <new root> --procs 64
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_dynamic_oracle as DO
from experiments.mist_w3po_v1.crf import precheck_dynamic_oracle_cov2 as C2

SIM_T = 0.3
SAME_SPAN_IOU = 0.5
MINILM = "/opt/dlami/nvme/jasonleeeli/evaluation_models/all-MiniLM-L12-v2"
_EMB: dict = {"pool": None, "cache": {}}


def _embed_worker(texts: list[str]) -> list:
    """runs in a spawned process (the O1 parent must not touch CUDA before its fork pool): the MiniLM recipe of
    speech_gap_decomp.sem_err, on CPU."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    g = globals()
    if "_TOKM" not in g:
        g["_TOKM"] = (AutoTokenizer.from_pretrained(MINILM), AutoModel.from_pretrained(MINILM).eval())
    tok, mod = g["_TOKM"]
    out = []
    for i in range(0, len(texts), 256):
        b = texts[i:i + 256]
        x = tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            o = mod(**x).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).float()
        out += list(torch.nn.functional.normalize((o * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=-1).numpy())
    return out


def embed(texts: list[str]) -> dict:
    """text -> L2-normalised MiniLM mean-pooled embedding, cached; computed in one spawned worker process."""
    from multiprocessing import get_context
    cache = _EMB["cache"]
    need = [t for t in dict.fromkeys(texts) if t not in cache]
    if need:
        if _EMB["pool"] is None:
            _EMB["pool"] = get_context("spawn").Pool(1)
        for t, v in zip(need, _EMB["pool"].apply(_embed_worker, (need,))):
            cache[t] = v
    return cache


def iou(x: dict, u: dict) -> float:
    inter = DO.ov(x["a"], x["b"], u["a"], u["b"])
    union = max(x["b"], u["b"]) - min(x["a"], u["a"])
    return inter / union if union > 0 else 0.0


def meeting_tables_v3(st: dict, seglist: list[dict], sim=None) -> dict:
    """v2's tables with the v3 content test in rules b and c (keys a / b / c / b_raw). sim(text, ref) -> cosine
    (default: MiniLM); injectable for tests."""
    from experiments.mist_w3po_v1.crf import diag_order_attr as D

    sc, got, unit_of, mp = st["sc"], st["got"], st["unit_of"], st.get("mp", {})
    if sim is None:
        cache = embed([x["text"] for x in seglist if x["text"]] + [v["ref"] for v in sc.values()])
        sim = lambda a, b: float(np.dot(cache[a], cache[b]))  # noqa: E731
    base = C2.meeting_tables_v2(st, seglist)                 # a' and b_raw (v1 credit) are v2's
    nr = {rid: D.norm(v["ref"]) for rid, v in sc.items()}
    idx = {id(x): k for k, x in enumerate(seglist)}

    def m3(x, u, rid):
        return C2.content_match(D.norm(x["text"]), nr[rid]) or sim(x["text"], u["ref"]) >= SIM_T

    out = {}
    for rid, u in sc.items():
        rb, rc = [], []
        for x, t in got.get(unit_of[rid], []):
            if t.strip() and DO.ov(x["a"] - DO.RECALL_COLLAR, x["b"] + DO.RECALL_COLLAR, u["a"], u["b"]) > 0:
                same = mp.get(x["spk"]) == u["spk"] and iou(x, u) >= SAME_SPAN_IOU
                if same or m3(x, u, rid):
                    rb.append(idx[id(x)])
        if u["b"] - u["a"] > DO.LONG_S:
            for k, x in enumerate(seglist):
                if x["text"] and iou(x, u) >= C2.IOU_C and m3(x, u, rid):
                    rc.append(k)
        out[rid] = {"a": base[rid]["a"], "b": sorted(set(rb)), "c": sorted(set(rc)), "b_raw": base[rid]["b_raw"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--o1", action="store_true")
    ap.add_argument("--root", required=True)
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    C2.TABLES = meeting_tables_v3
    C2.REVIEW_SEED = DO.SEED + 3
    C2.COVERAGE_TAG = "v3"
    if a.o1:
        C2.run_o1_v2(a)


if __name__ == "__main__":
    main()
