"""Robust wave 2, diagnosis S1 scoring (reports/mist_w3po_v1/robust_wave2_diag_prereg.zh-CN.md).

System: the paper's LocalAgreement-2 baseline at a 1 s read window (wave2_s1_run.sh). Outputs under
eval_outputs/mist_w3po_v1/wave2_S1/ (<root>):
  thinker_<dir>/blocks_<dir>_w1.jsonl          the Thinker's commits (both sets, both conditions)
  talker_<dir>_s<seed>/{part_*.jsonl, results.jsonl, asr_speech.jsonl, squim.jsonl}   one TTS seed
  repro/...                                    the reproduction check on a RealSI subset

pesq    SQUIM objective (torchaudio, first 20 s, game_muffle_analysis.squim) of each sentence's speech-only output
        (results.jsonl speech_path). CPU pool; uniss-eval python. Refuses an existing output.
          python experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run <talker dir> --out <jsonl> [--procs 24]
score   per direction x set (a: CoVoST 2 pairs; b: OpenBroadcast-SI dev single-speaker cuts) x condition:
        text BLEU of the committed translation, ASR-BLEU per TTS seed (speech-only ASR, paper protocol), LAAL D1 (paper's
        segment LAAL) and D2 (speech_laal_online) from laal_defs.sentence_defs, SQUIM PESQ; clean - noisy with a
        paired bootstrap 95 % CI (2000 resamples, seed 0) over sentences (a) or single-speaker streams (b); (b) also
        on the turns without crosstalk. qwen3omni python.
          python experiments/mist_w3po_v1/voicecond/wave2_s1_score.py score --root <root> --seeds 20261021,1 --out <json>
repro   reproduction check on the RealSI subset: our Thinker commits vs the paper's stored LA2 commits (exact), and
        text BLEU / ASR-BLEU / D1 on the same sentences vs the paper's three talker seeds.
          python experiments/mist_w3po_v1/voicecond/wave2_s1_score.py repro --root <root> --out <json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
PAPER_E = REPO / "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500"
PAPER_BLOCKS = REPO / "eval_outputs/uniss_qwen3_omni_v1/realsi_thinker_stream_v1/blocks_{d}_w1.jsonl"
PAPER_RUNS = ("system_la2_m1", "system_la2-seed1_m1", "system_la2-seed2_m1")
PAPER_POINT = {"en2zh": (29.75, 2.51), "zh2en": (23.19, 2.70)}   # results.json realsi LA2 m=1: ASR-BLEU, D1 (s)
LANG = {"en2zh": "cmn", "zh2en": "eng"}
B = 2000


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def laal_defs():
    spec = importlib.util.spec_from_file_location(
        "laal_defs", REPO / "eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112/laal_defs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ PESQ
def _squim(a):
    sid, path = a
    try:
        from experiments.mist_w3po_v1.evaluation.game_muffle_analysis import read16, squim
        q = squim(read16(path)) if path and Path(path).exists() else None
        return {"id": sid, **(q or {"pesq": None})}
    except Exception as e:  # noqa: BLE001
        return {"id": sid, "pesq": None, "error": repr(e)[:200]}


def pesq(a) -> None:
    import multiprocessing as mp
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    recs = jl(Path(a.run) / "results.jsonl")
    jobs = [(r["id"], r.get("speech_path")) for r in recs]
    with mp.get_context("spawn").Pool(a.procs) as pool, open(out, "x", encoding="utf-8") as fh:
        for r in pool.imap(_squim, jobs, chunksize=4):
            fh.write(json.dumps(r) + "\n")
    print(out, len(jobs))


# ------------------------------------------------------------------ BLEU with sufficient statistics
class Bleu:
    def __init__(self, lang):
        from sacrebleu.metrics import BLEU
        from evaluation.text_metrics import normalize_for_bleu
        self.lang = lang
        self.norm = normalize_for_bleu
        self.m = BLEU(tokenize="zh" if lang == "cmn" else "13a")

    def stats(self, hyps, refs):
        h = [self.norm(x, self.lang) for x in hyps]
        r = [self.norm(x, self.lang) for x in refs]
        return np.array(self.m._extract_corpus_statistics(h, [r]), dtype=np.int64)

    def score(self, st):
        return float(self.m._compute_score_from_stats(list(st.sum(0) if st.ndim == 2 else st)).score)


def corpus_bleu(hyps, refs, lang):
    from evaluation.text_metrics import corpus_bleu as cb
    return cb(hyps, refs, language=lang)["score"]


# ------------------------------------------------------------------ score
def load_dir(root: Path, d: str, seeds: list, LD):
    meta = {m["id"]: m for m in jl(root / "data" / f"meta_{d}.jsonl")}
    items = {r["id"]: r for r in jl(root / "data" / f"items_{d}.jsonl")}
    blocks = {r["id"]: r for r in jl(root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
    per = {i: {"ref": items[i]["translation"], "hyp": "".join(b["text"] for b in blocks[i]["blocks"]) if i in blocks else ""}
           for i in meta}
    for s in seeds:
        run = root / f"talker_{d}_s{s}"
        recs = {r["id"]: r for r in jl(run / "results.jsonl")}
        heard = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
        sq = {r["id"]: r.get("pesq") for r in jl(run / "squim.jsonl")} if (run / "squim.jsonl").exists() else {}
        for i in meta:
            r = recs.get(i)
            sd = LD.sentence_defs(r) if r else None
            per[i][f"asr{s}"] = heard.get(i, "")
            per[i][f"d1_{s}"] = sd["D1"] / 1000 if sd else None
            per[i][f"d2_{s}"] = sd["D2"] / 1000 if sd else None
            per[i][f"pesq{s}"] = sq.get(i)
    return meta, per


def mean_or_nan(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("nan")


def summarize(meta, per, lang, seeds, set_, units_key, keep=lambda m: True):
    bl = Bleu(lang)
    pairs = sorted({m["pair"] for m in meta.values() if m["set"] == set_ and keep(m)})
    idx = {(m["pair"], m["cond"]): i for i, m in meta.items() if m["set"] == set_}
    pairs = [p for p in pairs if (p, "clean") in idx and (p, "noisy") in idx]
    unit_of = {p: meta[idx[(p, "clean")]][units_key] for p in pairs}
    units = sorted(set(unit_of.values()))
    upos = {u: k for k, u in enumerate(units)}
    pu = np.array([upos[unit_of[p]] for p in pairs])
    out = {"n_sentences": len(pairs), "n_units": len(units)}
    ids = {c: [idx[(p, c)] for p in pairs] for c in ("clean", "noisy")}
    metrics = {}
    # sufficient statistics per sentence -> per unit
    def unit_sum(st):
        U = np.zeros((len(units), st.shape[1]), dtype=np.int64)
        np.add.at(U, pu, st)
        return U
    def unit_mean(vals):   # (sum, count) per unit over defined values
        s, n = np.zeros(len(units)), np.zeros(len(units))
        for k, v in enumerate(vals):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                s[pu[k]] += v
                n[pu[k]] += 1
        return s, n
    for c in ("clean", "noisy"):
        metrics[("text", c)] = ("bleu", [unit_sum(bl.stats([per[i]["hyp"] for i in ids[c]], [per[i]["ref"] for i in ids[c]]))])
        metrics[("asr", c)] = ("bleu", [unit_sum(bl.stats([per[i][f"asr{s}"] for i in ids[c]], [per[i]["ref"] for i in ids[c]]))
                                        for s in seeds])
        for k in ("d1", "d2", "pesq"):
            metrics[(k, c)] = ("mean", [unit_mean([per[i][f"{k}_{s}" if k != "pesq" else f"pesq{s}"] for i in ids[c]])
                                        for s in seeds])

    def value(kind, parts, w):
        vs = []
        for p in parts:
            if kind == "bleu":
                vs.append(bl.score((p * w[:, None]).sum(0)))
            else:
                s, n = p
                vs.append(float((s * w).sum() / max((n * w).sum(), 1e-9)))
        return vs

    rng = np.random.default_rng(0)
    W = [np.ones(len(units))] + [np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)).astype(float)
                                 for _ in range(B)]
    for name in ("text", "asr", "d1", "d2", "pesq"):
        kind = metrics[(name, "clean")][0]
        res = {}
        boot = []
        for b, w in enumerate(W):
            vc = value(kind, metrics[(name, "clean")][1], w)
            vn = value(kind, metrics[(name, "noisy")][1], w)
            if b == 0:
                res = {"clean": round(float(np.mean(vc)), 3), "noisy": round(float(np.mean(vn)), 3),
                       "clean_per_seed": [round(x, 3) for x in vc] if len(vc) > 1 else None,
                       "noisy_per_seed": [round(x, 3) for x in vn] if len(vn) > 1 else None,
                       "loss": round(float(np.mean(vc) - np.mean(vn)), 3)}
            else:
                boot.append(float(np.mean(vc) - np.mean(vn)))
        res["loss_ci95"] = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
        out[name] = res
    return out


def score(a) -> None:
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    root = Path(a.root)
    seeds = a.seeds.split(",")
    LD = laal_defs()
    res = {"system": "LA2 (LocalAgreement-2), read window 1 s, base Qwen3-Omni Thinker + T1c Talker (paper baseline)",
           "seeds": seeds, "bootstrap": f"{B} paired resamples, seed 0; units = sentences (a) / single-speaker streams (b)",
           "loss": "clean - noisy (BLEU, PESQ: positive = noise hurts; LAAL: negative = noise is slower)"}
    for d in a.dirs.split(","):
        meta, per = load_dir(root, d, seeds, LD)
        lang = LANG[d]
        res[d] = {"a": summarize(meta, per, lang, seeds, "a", "stream"),
                  "b": summarize(meta, per, lang, seeds, "b", "stream")}
        # b_no_crosstalk: pairs whose noisy and clean turns both have no other speaker overlapping
        ovl = {m["pair"] for m in meta.values() if m["set"] == "b" and m["ovl"]}
        res[d]["b_no_crosstalk"] = summarize(meta, per, lang, seeds, "b", "stream", keep=lambda m, o=ovl: m["pair"] not in o)
        # how often the Talker / ASR outputs are missing
        res[d]["coverage"] = {c: {"asr_nonempty": float(np.mean([bool(per[i][f"asr{s}"].strip()) for i, m in meta.items()
                                                                  if m["cond"] == c for s in seeds])),
                                  "hyp_nonempty": float(np.mean([bool(per[i]["hyp"].strip()) for i, m in meta.items()
                                                                 if m["cond"] == c]))} for c in ("clean", "noisy")}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ reproduction check
def repro(a) -> None:
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    root = Path(a.root) / "repro"
    LD = laal_defs()
    res = {"paper_point": {d: {"asr_bleu": p[0], "d1_s": p[1]} for d, p in PAPER_POINT.items()}}
    for d in a.dirs.split(","):
        lang = LANG[d]
        ours = {r["id"]: r for r in jl(root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
        paper = {r["id"]: r for r in jl(str(PAPER_BLOCKS).format(d=d))}
        ids = sorted(ours)
        key = lambda r: [(b["text"], b["write_ms"]) for b in r["blocks"]]
        same = sum(key(ours[i]) == key(paper[i]) for i in ids)
        same_text = sum("".join(b["text"] for b in ours[i]["blocks"]) == "".join(b["text"] for b in paper[i]["blocks"]) for i in ids)
        prec = {r["id"]: r for r in jl(PAPER_E / PAPER_RUNS[0] / f"realsi_{d}" / "results.jsonl")}
        refs = [prec[i]["translation_ref"] for i in ids]
        tb_ours = corpus_bleu(["".join(b["text"] for b in ours[i]["blocks"]) for i in ids], refs, lang)
        tb_paper = corpus_bleu(["".join(b["text"] for b in paper[i]["blocks"]) for i in ids], refs, lang)
        paper_asr, paper_d1 = [], []
        for run in PAPER_RUNS:
            p = PAPER_E / run / f"realsi_{d}"
            heard = {r["id"]: r.get("asr_text", "") for r in jl(p / "asr_speech.jsonl")}
            recs = {r["id"]: r for r in jl(p / "results.jsonl")}
            paper_asr.append(corpus_bleu([heard.get(i, "") for i in ids], refs, lang))
            paper_d1.append(mean_or_nan([(LD.sentence_defs(recs[i]) or {}).get("D1") for i in ids]) / 1000)
        t = root / f"talker_{d}"
        mine = {}
        if (t / "asr_speech.jsonl").exists():
            heard = {r["id"]: r.get("asr_text", "") for r in jl(t / "asr_speech.jsonl")}
            recs = {r["id"]: r for r in jl(t / "results.jsonl")}
            mine = {"asr_bleu": round(corpus_bleu([heard.get(i, "") for i in ids], refs, lang), 2),
                    "d1_s": round(mean_or_nan([(LD.sentence_defs(recs[i]) or {}).get("D1") if i in recs else None
                                               for i in ids]) / 1000, 3)}
        res[d] = {"n": len(ids), "blocks_identical": same, "translation_identical": same_text,
                  "text_bleu_ours": round(tb_ours, 2), "text_bleu_paper": round(tb_paper, 2),
                  "paper_subset_asr_bleu_per_seed": [round(x, 2) for x in paper_asr],
                  "paper_subset_d1_per_seed": [round(x, 3) for x in paper_d1],
                  "ours_talker_default_seed": mine}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pesq"); p.add_argument("--run", required=True); p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=24); p.set_defaults(f=pesq)
    p = sub.add_parser("score"); p.add_argument("--root", required=True); p.add_argument("--seeds", default="20261021,1")
    p.add_argument("--dirs", default="en2zh,zh2en"); p.add_argument("--out", required=True); p.set_defaults(f=score)
    p = sub.add_parser("repro"); p.add_argument("--root", required=True); p.add_argument("--dirs", default="en2zh,zh2en")
    p.add_argument("--out", required=True); p.set_defaults(f=repro)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
