"""Robust wave 5, side line V (reports/mist_w3po_v1/robust_wave5_prereg.zh-CN.md): LocalAgreement-2 text replay with
view-marginalised drafts (K = 8 fixed views) against the same code at K = 1 (the original audio only).

Model: the base Thinker (pretrained_models/Qwen3-Omni-30B-A3B-Instruct, no adapter), HF bf16, loaded and prompted as
wave3_T_thinker.BaseThinker (si_thinker_stream's base prompt). The read loop is si_thinker_stream.run_window's plain
la2 branch, verbatim in its rules: read r (r = 0, 1, ...) hears t = min((r + 1) * 1000, dur_ms) ms; the prompt is
head + join(committed); the continuation is greedy (max 160 new tokens, stop at <|im_end|> 151645); draft =
units(join(committed) + sep + continuation) with si_thinker_stream's separator rule; at t >= dur_ms everything after
the committed units is committed and the sentence is done; otherwise the longest common prefix with the previous draft
beyond the committed units is committed; the previous draft is then this draft. Output rows as si_thinker_stream's
blocks file ({id, translation, blocks[{write_ms, text, final}], drafts}), with its last-empty-block rule.

K = 8: the 8 views of wave 4 #4 (wave4_views.views: +-20 ms shift, speed 0.95 / 1.05, +-3 dB gain, low shelf +3 dB @
300 Hz, high shelf -3 dB @ 3 kHz) of the heard prefix are 8 rows of ONE generate batch; a logits processor replaces
every row's scores by log((1/8) sum_k softmax(scores_k)) of its group, so greedy picks the same token in all 8 rows and
their KV caches advance together (one draft per sentence). K = 1: the original prefix, one row, the same processor
(identity for a group of one). No threshold, no flip gating: the commit rule is LA2's.

Data: S1's items (eval_outputs/mist_w3po_v1/wave2_S1/data/items_<dir>.jsonl): set (a) a_{clean,noisy}__*, and set (b)
b_{clean,noisy}__* of the pairs whose twins both have ovl = false (meta).
Workers claim (K, dir, shard) units (O_EXCL claim files); outputs .partial -> renamed, never rewritten.
    CUDA_VISIBLE_DEVICES=<card> python -I experiments/mist_w3po_v1/voicecond/wave5_viewmarg.py --worker k [--max-gib 66]
    CUDA_VISIBLE_DEVICES=<card> python -I .../wave5_viewmarg.py --check 5    (K = 1 on 5 zh2en a_noisy sentences vs S1)
    CUDA_VISIBLE_DEVICES=<card> python -I .../wave5_viewmarg.py --latency 200 (exclusive card: per-read wall clock, K = 1
        and K = 8, one sentence per batch as in deployment, >= 200 reads each after 5 warm-up reads)
    python -I .../wave5_viewmarg.py --summary                                 (CPU: BLEU, D1 text LAAL, gates V1-V4)
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
OUT = REPO / "eval_outputs/mist_w3po_v1/wave5/V"
RUNS, CLAIMS = OUT / "runs", OUT / "claims"
DIRS = ("zh2en", "en2zh")
KS = (8, 1)
N_SHARD = 8
SR = 16000
WINDOW = 1000
THINKER_EOS = 151645
MAX_NEW = 160
GROUP = {8: 4, 1: 32}          # sentences per generate batch (rows = sentences x K)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} {msg}", flush=True)


def units(text: str, zh: bool) -> list[str]:          # si_thinker_stream.units
    return [c for c in text if not c.isspace()] if zh else text.split()


def join(us: list[str], zh: bool) -> str:             # si_thinker_stream.join
    return "".join(us) if zh else " ".join(us)


def items_of(d: str) -> list[dict]:
    meta = {r["id"]: r for r in map(json.loads, open(S1 / f"data/meta_{d}.jsonl", encoding="utf-8"))}
    pairs: dict = {}
    for m in meta.values():
        pairs.setdefault((m["set"], m["pair"]), []).append(m)
    keep = set()
    for (s, _), ms in pairs.items():
        if s == "a" or (s == "b" and len(ms) == 2 and not any(m["ovl"] for m in ms)):
            keep.update(m["id"] for m in ms)
    return [r for r in map(json.loads, open(S1 / f"data/items_{d}.jsonl", encoding="utf-8")) if r["id"] in keep]


def wav_cache():
    import soundfile as sf
    cache = {}

    def get(p):
        if p not in cache:
            x, sr = sf.read(str(REPO / p), dtype="float32", always_2d=True)
            assert sr == SR, p
            cache[p] = x.mean(axis=1)
        return cache[p]
    return get


# ------------------------------------------------------------------------------------------------------ GPU
class Engine:
    def __init__(self, max_gib: int):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
        from experiments.mist_w3po_v1.voicecond.wave3_T_thinker import BaseThinker
        from experiments.mist_w3po_v1.voicecond.wave4_views import views
        self.torch, self.views = torch, views
        self.bt = BaseThinker(max_gib)
        eng = self

        class Marg(LogitsProcessor):
            def __call__(self, input_ids, scores):
                K = eng.K
                s = scores.float()
                p = torch.softmax(s, -1).view(-1, K, s.shape[-1]).mean(1)
                lp = torch.log(p.clamp_min(1e-30))
                return lp.repeat_interleave(K, 0).to(scores.dtype)
        self.lp = LogitsProcessorList([Marg()])
        self.K = 1

    def drafts(self, d: str, K: int, reqs: list[tuple[str, np.ndarray]]) -> list[str]:
        """one greedy continuation per (prompt prefix text, heard audio) request, view-marginalised over K rows"""
        torch, bt = self.torch, self.bt
        rows, texts = [], []
        for pref, x in reqs:
            vs = [x] if K == 1 else self.views(x)
            assert len(vs) == K
            for v in vs:
                rows.append(v if len(v) >= 1600 else np.pad(v, (0, 1600 - len(v))))
                texts.append(bt.head[d] + pref)
        inp = bt.proc(text=texts, audio=rows, return_tensors="pt", padding=True)
        inputs = {k: v.to(bt.dev) for k, v in inp.items()}
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
        self.K = K
        with torch.no_grad():
            out = bt.thinker.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=THINKER_EOS,
                                      pad_token_id=bt.tok.pad_token_id, logits_processor=self.lp)
        gen = out[:, inputs["input_ids"].shape[1]:].tolist()
        res = []
        for j in range(len(reqs)):
            g = [r[: r.index(THINKER_EOS)] if THINKER_EOS in r else r for r in gen[j * K: (j + 1) * K]]
            g = [[i for i in r if i != bt.tok.pad_token_id] for r in g]
            assert all(r == g[0] for r in g), "view rows diverged"
            res.append(bt.tok.decode(g[0], skip_special_tokens=True))
        return res


def la2(eng: Engine, d: str, K: int, its: list[dict], get, timer=None) -> list[dict]:
    """si_thinker_stream.run_window's la2 branch for a group of sentences (round-synchronous)"""
    zh = d.endswith("zh")
    st = []
    for it in its:
        a = get(it["audio"])
        st.append({"id": it["id"], "audio": a, "dur_ms": len(a) // 16, "committed": [], "prev": None, "blocks": [],
                   "drafts": [], "done": False})
    r = 0
    while any(not s["done"] for s in st):
        live = [s for s in st if not s["done"]]
        for s in live:
            s["t"] = min((r + 1) * WINDOW, s["dur_ms"])
        reqs = [(join(s["committed"], zh), s["audio"][: s["t"] * 16]) for s in live]
        if timer is not None:
            eng.torch.cuda.synchronize()
            t0 = time.perf_counter()
        conts = eng.drafts(d, K, reqs)
        if timer is not None:
            eng.torch.cuda.synchronize()
            timer.append(time.perf_counter() - t0)
        for s, cont in zip(live, conts):
            sep = "" if zh or not s["committed"] or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
            draft = units(join(s["committed"], zh) + sep + cont, zh)
            s["drafts"].append({"t": s["t"], "draft": join(draft, zh)})
            n = len(s["committed"])
            if s["t"] >= s["dur_ms"]:
                new, s["done"] = draft[n:], True
            else:
                agreed = 0
                if s["prev"] is not None:
                    while agreed < min(len(draft), len(s["prev"])) and draft[agreed] == s["prev"][agreed]:
                        agreed += 1
                new = draft[n: agreed] if agreed > n else []
                s["prev"] = draft
            if new or s["done"]:
                s["blocks"].append({"write_ms": s["t"], "text": ("" if zh or not n else " ") + join(new, zh),
                                    "final": s["done"]})
                s["committed"] += new
        r += 1
    out = []
    for s in st:
        blocks = [b for b in s["blocks"] if b["text"].strip() or b["final"]]
        if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:
            blocks = blocks[:-1]
            blocks[-1]["final"] = True
        out.append({"id": s["id"], "translation": join(s["committed"], zh), "blocks": blocks, "drafts": s["drafts"]})
    return out


def worker(a) -> None:
    eng = Engine(a.max_gib)
    get = wav_cache()
    log(f"worker {a.worker} ready on {socket.gethostname()} cuda {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    RUNS.mkdir(parents=True, exist_ok=True)
    CLAIMS.mkdir(parents=True, exist_ok=True)
    order = [(K, d, k) for k in range(N_SHARD) for K in KS for d in DIRS]
    for K, d, k in order:
        name = f"K{K}_{d}_s{k}"
        if (RUNS / f"{name}.jsonl").exists():
            continue
        try:
            fd = os.open(CLAIMS / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{a.worker} {os.getpid()} {socket.gethostname()}\n".encode())
            os.close(fd)
        except FileExistsError:
            continue
        its = [x for i, x in enumerate(items_of(d)) if i % N_SHARD == k]
        t0 = time.time()
        part = RUNS / f"{name}.jsonl.partial"
        with open(part, "w", encoding="utf-8") as fh:
            for g in range(0, len(its), GROUP[K]):
                for row in la2(eng, d, K, its[g: g + GROUP[K]], get):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
        os.rename(part, RUNS / f"{name}.jsonl")
        log(f"{name}: {len(its)} sentences in {time.time() - t0:.0f}s")


def check(a) -> None:
    eng = Engine(a.max_gib)
    get = wav_cache()
    d = "zh2en"
    s1 = {r["id"]: r for r in map(json.loads, open(S1 / f"thinker_{d}/blocks_{d}_w1.jsonl", encoding="utf-8"))}
    its = [x for x in items_of(d) if x["id"].startswith("a_noisy__")][: a.check]
    res = la2(eng, d, 1, its, get)
    zh = False
    rows, tot, agree, same_tr, same_bl = [], 0, 0, 0, 0
    for r in res:
        ref = s1[r["id"]]
        u, v = units(r["translation"], zh), units(ref["translation"], zh)
        L = 0
        while L < min(len(u), len(v)) and u[L] == v[L]:
            L += 1
        tot += max(len(u), len(v))
        agree += L
        same_tr += int(r["translation"] == ref["translation"])
        same_bl += int([(b["write_ms"], b["text"]) for b in r["blocks"]] == [(b["write_ms"], b["text"]) for b in ref["blocks"]])
        rows.append({"id": r["id"], "hf": r, "s1_translation": ref["translation"], "s1_blocks": ref["blocks"]})
    summ = {"n": len(res), "translation_identical": same_tr, "blocks_identical": same_bl,
            "unit_prefix_agreement": agree / max(1, tot)}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "check5.json"
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    json.dump({"summary": summ, "rows": rows}, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"check: {summ}")


def latency(a) -> None:
    import torch
    eng = Engine(a.max_gib)
    get = wav_cache()
    d = "zh2en"
    its = [x for x in items_of(d) if x["id"].startswith("a_noisy__")]
    res = {}
    for K in (1, 8):
        la2(eng, d, K, its[:2], get)                   # warm-up (>= 5 reads), not timed
        times, j = [], 2
        while len(times) < a.latency:
            la2(eng, d, K, its[j: j + 1], get, timer=times)
            j += 1
        t = np.array(times) * 1000
        res[f"K{K}"] = {"n_reads": len(t), "n_sentences": j - 2, "median_ms": float(np.median(t)),
                        "p95_ms": float(np.percentile(t, 95)), "mean_ms": float(t.mean()), "max_ms": float(t.max()),
                        "times_ms": t.round(2).tolist()}
        log(f"K={K}: {res[f'K{K}']['n_reads']} reads, median {res[f'K{K}']['median_ms']:.0f} ms, p95 {res[f'K{K}']['p95_ms']:.0f} ms")
    res["ratio_median"] = res["K8"]["median_ms"] / res["K1"]["median_ms"]
    res["ratio_p95"] = res["K8"]["p95_ms"] / res["K1"]["p95_ms"]
    res["device"] = torch.cuda.get_device_name(0)
    res["note"] = "HF transformers bf16 generate, one sentence per batch (K rows), feature extraction included, exclusive card"
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "latency.json"
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    json.dump(res, open(p, "w", encoding="utf-8"), indent=1)
    log(f"latency ratio median {res['ratio_median']:.2f}, p95 {res['ratio_p95']:.2f}")


# ------------------------------------------------------------------------------------------------------ CPU
def summary(a) -> None:
    from sacrebleu.metrics import BLEU
    from evaluation.text_metrics import normalize_for_bleu
    spec = importlib.util.spec_from_file_location(
        "laal_defs", REPO / "eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112/laal_defs.py")
    LD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LD)
    p = OUT / "summary.json"
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    rng = np.random.default_rng(0)
    res = {}
    for d in DIRS:
        NL = "cmn" if d == "en2zh" else "eng"
        bl = BLEU(tokenize="zh" if d == "en2zh" else "13a")
        its = {x["id"]: x for x in items_of(d)}
        meta = {r["id"]: r for r in map(json.loads, open(S1 / f"data/meta_{d}.jsonl", encoding="utf-8"))}
        hyp = {}
        for K in KS:
            hyp[K] = {}
            files = sorted(glob.glob(str(RUNS / f"K{K}_{d}_s*.jsonl")))
            assert len(files) == N_SHARD, (K, d, len(files))
            for f in files:
                for r in map(json.loads, open(f, encoding="utf-8")):
                    hyp[K][r["id"]] = r
            assert set(hyp[K]) == set(its), (K, d)

        def stat(h, ref):
            s = bl.corpus_score([normalize_for_bleu(h, NL)], [[normalize_for_bleu(ref, NL)]])
            return [s.sys_len, s.ref_len] + s.counts + s.totals

        def cb(S):
            sl, rl, c, t = S[0], S[1], S[2:6], S[6:10]
            if np.any(c == 0):
                return 0.0
            return 100 * (1.0 if sl > rl else np.exp(1 - rl / max(sl, 1))) * np.exp(np.mean(np.log(c / t)))

        def d1(r, it):   # text-level D1: one delay per committed unit = its block's write time (capped at |X|)
            X = it["dur_ms"]
            T = len(LD.si_units(it["translation"], NL))
            dw = [min(b["write_ms"], X) for b in r["blocks"] for _ in LD.si_units(b["text"], NL)]
            return LD.si_laal(dw, X, T) / 1000 if dw else np.nan
        for set_ in ("a", "b"):
            for cond in ("noisy", "clean"):
                ids = sorted(i for i in its if i.startswith(f"{set_}_{cond}__"))
                units_of = [meta[i]["stream"] if set_ == "b" else i for i in ids]
                U = sorted(set(units_of))
                ui = np.array([U.index(u) for u in units_of])
                S = {K: np.array([stat(hyp[K][i]["translation"], its[i]["translation"]) for i in ids], float) for K in KS}
                agg = {K: np.zeros((len(U), 10)) for K in KS}
                for K in KS:
                    np.add.at(agg[K], ui, S[K])
                lat = {K: np.array([d1(hyp[K][i], its[i]) for i in ids]) for K in KS}
                lu = {K: np.array([np.nanmean(lat[K][ui == j]) for j in range(len(U))]) for K in KS}
                b8, b1 = cb(agg[8].sum(0)), cb(agg[1].sum(0))
                draws = rng.integers(0, len(U), (2000, len(U)))
                bd = np.array([cb(agg[8][j].sum(0)) - cb(agg[1][j].sum(0)) for j in draws])
                ld = np.array([np.nanmean(lu[8][j]) - np.nanmean(lu[1][j]) for j in draws])
                p_le0 = float((bd <= 0).mean())
                res[f"{d}_{set_}_{cond}"] = {
                    "n_sentences": len(ids), "n_units": len(U), "bleu_K1": b1, "bleu_K8": b8, "delta": b8 - b1,
                    "delta_ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                    "p_one_sided": p_le0, "d1_K1": float(np.nanmean(lu[1])), "d1_K8": float(np.nanmean(lu[8])),
                    "d1_delta": float(np.nanmean(lu[8]) - np.nanmean(lu[1])),
                    "d1_delta_ci": [float(np.percentile(ld, 2.5)), float(np.percentile(ld, 97.5))]}
                log(f"{d} {set_} {cond}: K1 {b1:.2f} K8 {b8:.2f} delta {b8 - b1:+.2f} "
                    f"[{res[f'{d}_{set_}_{cond}']['delta_ci'][0]:+.2f}, {res[f'{d}_{set_}_{cond}']['delta_ci'][1]:+.2f}] "
                    f"D1 {np.nanmean(lu[1]):.3f} -> {np.nanmean(lu[8]):.3f}")
    lat = json.load(open(OUT / "latency.json")) if (OUT / "latency.json").exists() else None
    chk = json.load(open(OUT / "check5.json"))["summary"] if (OUT / "check5.json").exists() else None
    g = {}
    z = res["zh2en_a_noisy"]
    g["V1"] = {"pass": z["delta"] >= 1.0 and z["p_one_sided"] < 0.05, "delta": z["delta"], "p": z["p_one_sided"]}
    g["V2"] = {"pass": all(res[f"{d}_a_clean"]["delta"] >= -0.3 for d in DIRS),
               **{d: res[f"{d}_a_clean"]["delta"] for d in DIRS}}
    g["V3"] = {"pass": res["en2zh_a_noisy"]["delta"] >= 0 and res["en2zh_b_noisy"]["delta"] >= 0,
               "a": res["en2zh_a_noisy"]["delta"], "b": res["en2zh_b_noisy"]["delta"]}
    lat_ok = lat is not None and lat["K8"]["p95_ms"] <= 1000
    d1_ok = all(res[k]["d1_delta"] <= 0.2 for k in res)
    g["V4"] = {"pass": bool(lat_ok and d1_ok), "p95_K8_ms": lat and lat["K8"]["p95_ms"], "p95_K1_ms": lat and lat["K1"]["p95_ms"],
               "ratio_median": lat and lat["ratio_median"], "ratio_p95": lat and lat["ratio_p95"],
               "d1_delta_max": max(res[k]["d1_delta"] for k in res)}
    g["pass"] = all(g[k]["pass"] for k in ("V1", "V2", "V3", "V4"))
    json.dump({"cells": res, "gates": g, "latency": {k: v for k, v in (lat or {}).items() if k not in ("K1", "K8")} |
               ({f"K{K}": {x: y for x, y in lat[f"K{K}"].items() if x != "times_ms"} for K in KS} if lat else {}),
               "backend_check": chk}, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"gates: {json.dumps(g)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--max-gib", type=int, default=66)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--latency", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(a)
    elif a.check:
        check(a)
    elif a.latency:
        latency(a)
    elif a.worker >= 0:
        worker(a)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
