"""Scores of Hy-SimulMT + Qwen3-TTS on RealSI under the paper's protocol (registration
paper/baselines_run_plan_hysimulmt_qwen3tts_v1.md §3) = R2's experiments/baselines_hysimulmt_v1/score_hy.py with the
speech runs swapped; one json per point and direction, tiers.json and summary.md in --out (refuses existing files),
with R2's "Hy-SimulMT + paper talker" numbers (paper/baselines_results/hysimulmt/) side by side.

Per TTS seed run (assemble.py results.jsonl + speech-only ASR):
  asr_bleu   evaluation.text_metrics.corpus_bleu over ALL RealSI sentences (missing ASR row = "")
  laal_d1    collect_results.realsi_points verbatim (every unit at its block's playback start, queue included)
  laal_d0w   laal_defs.sentence_defs()["D0w"] (units at their block's write_ms)
text_bleu: the committed text, as R2 (checked equal to R2's value). A point = mean over the three TTS seeds.

    PYTHONPATH=. conda_envs/qwen3omni/bin/python experiments/baselines_hysimulmt_qwen3tts_v1/score.py \
        --out paper/baselines_results/hysimulmt_qwen3tts
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/baselines_hysimulmt_v1"))
from evaluation.text_metrics import corpus_bleu  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units  # noqa: E402
from hy_realsi import GRID, realsi_items  # noqa: E402

sys.path.insert(0, str(REPO / "eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112"))
from laal_defs import sentence_defs  # noqa: E402

E = REPO / "eval_outputs/baselines_hysimulmt_qwen3tts_v1/a1"   # amendment 1 runs
THINKER = REPO / "eval_outputs/baselines_hysimulmt_v1/thinker"
R2 = REPO / "paper/baselines_results/hysimulmt"
LANG = {"en2zh": "cmn", "zh2en": "eng"}
BUDGETS = {"en2zh": [("m1", 1.18), ("m2", 1.98), ("m3", 2.46), ("m4", 2.98)],
           "zh2en": [("m1", 1.18), ("m2", 1.92), ("m3", 2.40), ("m4", 2.98)]}
OMNI = {"en2zh": [19.80, 24.21, 25.01, 25.54], "zh2en": [14.89, 19.55, 21.07, 22.47]}   # make_tables.OMNI (reported)
SEEDS = (0, 1, 2)


def run_dir(point: str, d: str, seed: int) -> Path:
    w = GRID[point][0] / 1000
    return E / "runs" / f"system_{point}{f'-seed{seed}' if seed else ''}_m{w:g}" / f"realsi_{d}"


def seed_run(point: str, d: str, seed: int, items: list[dict]) -> dict:
    run = run_dir(point, d, seed)
    recs = [json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")]
    heard = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(run / "asr_speech.jsonl", encoding="utf-8"))}
    lang = LANG[d]
    ref = {it["id"]: it["trg_text"] for it in items}
    for r in recs:
        assert r["translation_ref"] == ref[r["id"]], r["id"]
        assert r["tgt_lang"] == lang
    cover = len([r for r in recs if r["id"] in heard]) / max(len(recs), 1)
    assert cover >= 0.99, (run, cover)
    bleu = corpus_bleu([heard.get(it["id"], "") for it in items], [it["trg_text"] for it in items], language=lang)
    seg, d0w = [], []
    for r in recs:   # collect_results.realsi_points, verbatim
        ds = [bl["start_ms"] for bl in r["blocks"] for _ in units(bl["text"], lang)]
        if ds:
            seg.append(laal(ds, r["source_duration_ms"], len(units(r["translation_ref"], lang))))
        x = sentence_defs(r)
        if x is not None:
            d0w.append(x["D0w"])
    bl = [b for r in recs for b in r["blocks"]]
    tts = [b for b in bl if b["tts"]]
    return {"run": str(run.relative_to(REPO)), "asr_bleu": round(bleu["score"], 2), "laal_d1": st.mean(seg) / 1000,
            "laal_d0w": st.mean(d0w) / 1000, "n_talker_rows": len(recs), "n_asr_rows": len(heard),
            "asr_cover": round(cover, 4), "n_d1": len(seg), "n_d0w": len(d0w), "errors": sum(bool(r.get("error")) for r in recs),
            "n_blocks": len(bl), "n_tts_blocks": len(tts), "n_hit_cap": sum(b["tts_hit_cap"] for b in tts),
            "speech_s_per_sentence": round(st.mean(sum(b["dur_ms"] for b in r["blocks"]) for r in recs) / 1000, 3),
            "end_offset_s_mean": round(st.mean(r["end_offset_ms"] for r in recs) / 1000, 3),
            "bleu_detail": {k: bleu[k] for k in ("bp", "sys_len", "ref_len", "sample_count")}}


def point_dir(point: str, d: str) -> dict:
    items = realsi_items(d)
    bl = {r["id"]: r for r in map(json.loads, open(THINKER / point / f"blocks_{d}.jsonl", encoding="utf-8"))}
    assert set(bl) == {it["id"] for it in items}
    lang = LANG[d]
    tb = round(corpus_bleu([bl[it["id"]]["translation"] for it in items], [it["trg_text"] for it in items],
                           language=lang)["score"], 2)
    r2 = json.load(open(R2 / f"{point}_{d}.json", encoding="utf-8"))
    assert tb == r2["text_bleu"], (point, d, tb, r2["text_bleu"])
    runs = {s: seed_run(point, d, s, items) for s in SEEDS}
    return {"system": "Hy-SimulMT-30B-A3B + Qwen3-TTS-12Hz-0.6B-Base", "point": point, "direction": d,
            "config": {"chunk_ms_first": 1600, "chunk_ms": GRID[point][0], "st_wait_chunks": GRID[point][1],
                       "native_mode": "standard" if GRID[point][0] == 1600 else "short_chunk",
                       "blocks": str((THINKER / point / f"blocks_{d}.jsonl").relative_to(REPO)) + " (R2, unchanged)",
                       "tts": "Qwen/Qwen3-TTS-12Hz-0.6B-Base @ 5d83992436eae1d760afd27aff78a71d676296fc, qwen-tts 0.1.1, "
                              "bf16, sdpa",
                       "voice": "x-vector only, speaker embedding of the source up to the first commit (>= 1 s; the paper "
                                "talker's clip), SimulInfer peak normalisation",
                       "text_input": "block text stripped, one request per committed block, non_streaming_mode=True, "
                                     "punctuation-only blocks not spoken",
                       "decoding": "generation_config.json sampling defaults (do_sample, T 0.9, top_k 50, top_p 1.0, "
                                   "rep 1.05, sub-talker 0.9/50/1.0); amendment 1: per-block cap min(1800, max(180, "
                                   "18 * chars)) codec frames (SimulInfer tts_backends/qwen.py), capped audio kept",
                       "seeds": "3 (batch-level torch.manual_seed 20261112 / 1000000 / 2000000 + batch index, batch 32)",
                       "timeline": "start = max(write_ms, previous block end), no compute time; 24 kHz -> 16 kHz",
                       "registration": "paper/baselines_run_plan_hysimulmt_qwen3tts_v1.md (+ amendment 1)"},
            "n": len(items),
            "asr_bleu": round(st.mean(r["asr_bleu"] for r in runs.values()), 2),
            "asr_bleu_sd": round(st.stdev(r["asr_bleu"] for r in runs.values()), 2),
            "text_bleu": tb,
            "laal_d1": round(st.mean(r["laal_d1"] for r in runs.values()), 3),
            "laal_d0w": round(st.mean(r["laal_d0w"] for r in runs.values()), 3),
            "speech_s_per_sentence": round(st.mean(r["speech_s_per_sentence"] for r in runs.values()), 3),
            "end_offset_s_mean": round(st.mean(r["end_offset_s_mean"] for r in runs.values()), 3),
            "n_hit_cap_3seeds": sum(r["n_hit_cap"] for r in runs.values()),
            "paper_talker": {k: r2[k] for k in ("asr_bleu", "asr_bleu_sd", "text_bleu", "laal_d1", "laal_d0w")},
            "seeds": runs}


def tiers(res: dict, key: str) -> dict:
    t = {}
    for d, bud in BUDGETS.items():
        for k, (name, x) in enumerate(bud):
            ok = [r for (p, dd), r in res.items() if dd == d and r[key] <= x + 1e-9]
            best = max(ok, key=lambda r: r["asr_bleu"]) if ok else None
            t[f"{d}|{name}"] = {"budget": x, "omni_reported": OMNI[d][k],
                                "point": best and best["point"], "asr_bleu": best and best["asr_bleu"],
                                "laal_d1": best and best["laal_d1"], "laal_d0w": best and best["laal_d0w"]}
    return t


def write_summary(res: dict, t1: dict, t0: dict, r2t: dict, path: Path) -> None:
    dname = {"en2zh": "En→Zh", "zh2en": "Zh→En"}
    L = ["# Hy-SimulMT-30B-A3B + Qwen3-TTS-12Hz-0.6B-Base，RealSI 句级（自动生成，"
         "`experiments/baselines_hysimulmt_qwen3tts_v1/score.py`）", "",
         "登记：`paper/baselines_run_plan_hysimulmt_qwen3tts_v1.md`。只替换语音生成器：Hy-SimulMT 的提交块（R2，未重跑）、排队模型、"
         "speech-only ASR-BLEU、D1（段 LAAL，块开始播放时刻，含排队）、D0w（块写出时刻）、分档规则都与 R2 的“+ 论文 talker”相同。"
         "Qwen3-TTS 0.6B Base，x-vector 克隆源说话人（参考片段同论文 talker），官方默认采样解码，三个种子取均值。"
         "BLEU 分母为全部句子（En→Zh 346，Zh→En 431）。", "",
         "## 全部点（并列：+ 论文 talker，R2）", "",
         "| 方向 | 点 | 原生模式 | ASR-BLEU Qwen3-TTS（±sd） | ASR-BLEU 论文 talker（±sd） | 文本 BLEU | D1 Qwen3-TTS (s) | D1 论文 talker (s) | "
         "D0w (s) | 每句语音 (s) | 句尾后延续 (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (p, d), r in sorted(res.items(), key=lambda x: (x[0][1], x[1]["laal_d1"])):
        q = r["paper_talker"]
        L.append(f"| {dname[d]} | `{p}` | {r['config']['native_mode']} | {r['asr_bleu']:.2f} (±{r['asr_bleu_sd']:.2f}) | "
                 f"{q['asr_bleu']:.2f} (±{q['asr_bleu_sd']:.2f}) | {r['text_bleu']:.2f} | {r['laal_d1']:.2f} | "
                 f"{q['laal_d1']:.2f} | {r['laal_d0w']:.2f} | {r['speech_s_per_sentence']:.2f} | {r['end_offset_s_mean']:.2f} |")
    for title, t, rt, key in (("## 分档（按 D1，论文规则）", t1, r2t["by_D1"], "laal_d1"),
                              ("## 分档（按 D0w，仅作参考）", t0, r2t["by_D0w_reference_only"], "laal_d0w")):
        L += ["", title, "", "| 方向 | 档 | 预算 (s) | SimulS2ST-Omni（报告值） | Hy-SimulMT + Qwen3-TTS | Hy-SimulMT + 论文 talker |",
              "|---|---|---|---|---|---|"]
        for k, v in t.items():
            d, m = k.split("|")
            cell = "--" if v["point"] is None else f"{v['asr_bleu']:.2f} ({v[key]:.2f})，`{v['point']}`"
            w = rt[k]
            cell2 = "--" if w["point"] is None else f"{w['asr_bleu']:.2f} ({w[key]:.2f})，`{w['point']}`"
            L.append(f"| {dname[d]} | {m} | {v['budget']:.2f} | {v['omni_reported']:.2f} | {cell} | {cell2} |")
    with path.open("x", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {(p, d): point_dir(p, d) for p in GRID for d in ("en2zh", "zh2en")}
    for (p, d), r in res.items():
        with open(out / f"{p}_{d}.json", "x", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=1)
    t1, t0 = tiers(res, "laal_d1"), tiers(res, "laal_d0w")
    with open(out / "tiers.json", "x", encoding="utf-8") as fh:
        json.dump({"by_D1": t1, "by_D0w_reference_only": t0}, fh, ensure_ascii=False, indent=1)
    r2t = json.load(open(R2 / "tiers.json", encoding="utf-8"))
    write_summary(res, t1, t0, r2t, out / "summary.md")
    print(json.dumps({f"{p}|{d}": {k: r[k] for k in ("asr_bleu", "asr_bleu_sd", "text_bleu", "laal_d1", "laal_d0w",
                                                       "n_hit_cap_3seeds", "paper_talker")}
                      for (p, d), r in res.items()}, indent=1, ensure_ascii=False))
    print(json.dumps(t1, indent=1))


if __name__ == "__main__":
    main()
