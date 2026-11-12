"""Scores of the Hy-SimulMT RealSI baseline under the paper's protocol (plan §4) -> one json per point and direction
plus summary.md in --out (refuses existing files).

Per talker seed run (results.jsonl + asr_speech.jsonl of si_infer + speech-only ASR):
  asr_bleu   evaluation.text_metrics.corpus_bleu over ALL RealSI sentences (a sentence the talker never spoke, because
             Hy-SimulMT committed nothing, or without an ASR row counts as ""), as collect_results.realsi_points
  laal_d1    collect_results.realsi_points verbatim: every unit at its block's playback start (start_ms), si_metrics.laal,
             sentence mean (sentences with no unit skipped)
  laal_d0w   laal_defs.sentence_defs()["D0w"] (units at their block's write_ms), same sentence set
text_bleu = the committed text (blocks file "translation" = Hy-SimulMT's full_text) against trg_text, same function.
A point = mean over the three talker seeds (paper rule). Tiers: per direction and budget, the point with the highest
ASR-BLEU whose D1 fits (also listed by D0w, reference only).

    PYTHONPATH=. conda_envs/qwen3omni/bin/python experiments/baselines_hysimulmt_v1/score_hy.py \
        --out paper/baselines_results/hysimulmt
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.text_metrics import corpus_bleu  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units  # noqa: E402
from hy_realsi import GRID, realsi_items  # noqa: E402

sys.path.insert(0, str(REPO / "eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112"))
from laal_defs import sentence_defs  # noqa: E402

E = REPO / "eval_outputs/baselines_hysimulmt_v1"
LANG = {"en2zh": "cmn", "zh2en": "eng"}
BUDGETS = {"en2zh": [("m1", 1.18), ("m2", 1.98), ("m3", 2.46), ("m4", 2.98)],
           "zh2en": [("m1", 1.18), ("m2", 1.92), ("m3", 2.40), ("m4", 2.98)]}
OMNI = {"en2zh": [19.80, 24.21, 25.01, 25.54], "zh2en": [14.89, 19.55, 21.07, 22.47]}   # make_tables.OMNI (reported)
SEEDS = (0, 1, 2)


def seed_run(point: str, d: str, seed: int, items: list[dict]) -> dict:
    w = GRID[point][0] / 1000
    base = E / "talker" / f"system_{point}{f'-seed{seed}' if seed else ''}_m{w:g}" / f"realsi_{d}"
    # talker_worker_hy.variants: a failed attempt keeps its folder, the next one writes <folder>-r<n>; use the complete one
    done = [o for o in [base] + [Path(f"{base}-r{n}") for n in range(1, 10)] if (o / "asr_speech.jsonl").exists()]
    assert len(done) == 1, (base, done)
    run = done[0]
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
    return {"run": str(run.relative_to(REPO)), "asr_bleu": round(bleu["score"], 2), "laal_d1": st.mean(seg) / 1000,
            "laal_d0w": st.mean(d0w) / 1000, "n_talker_rows": len(recs), "n_asr_rows": len(heard),
            "asr_cover": round(cover, 4), "n_d1": len(seg), "n_d0w": len(d0w), "errors": sum(bool(r.get("error")) for r in recs),
            "bleu_detail": {k: bleu[k] for k in ("bp", "sys_len", "ref_len", "sample_count")}}


def point_dir(point: str, d: str) -> dict:
    items = realsi_items(d)
    th = E / "thinker" / point
    bl = {r["id"]: r for r in map(json.loads, open(th / f"blocks_{d}.jsonl", encoding="utf-8"))}
    meta = json.load(open(th / f"meta_{d}.json", encoding="utf-8"))
    assert set(bl) == {it["id"] for it in items}, "blocks file must hold every RealSI id"
    lang = LANG[d]
    tb = corpus_bleu([bl[it["id"]]["translation"] for it in items], [it["trg_text"] for it in items], language=lang)
    no_out = [i for i, r in bl.items() if not any(units(b["text"], lang) for b in r["blocks"])]
    early = [i for i, r in bl.items() if r["blocks"] and r["blocks"][-1]["write_ms"] < r["dur_ms"]]
    nb = [len(r["blocks"]) for r in bl.values() if r["blocks"]]
    runs = {s: seed_run(point, d, s, items) for s in SEEDS}
    out = {"system": "Hy-SimulMT-30B-A3B + streaming UniSS talker (T1c@1500)", "point": point, "direction": d,
           "config": {"chunk_ms_first": 1600, "chunk_ms": GRID[point][0], "st_wait_chunks": GRID[point][1],
                      "native_mode": "standard" if GRID[point][0] == 1600 else "short_chunk",
                      "simulinfer_settings": meta["settings"], "versions": meta["versions"],
                      "talker": "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss",
                      "talker_args": "--text-policy file --min-words 2 --speed-policy block --history-ban 8 --gpu-mem 0.15 "
                                     "--num-shards 2, seeds 20261021 / 1 / 2",
                      "registration": "paper/baselines_run_plan_hysimulmt_v1.md (+ amendment 1)"},
           "n": len(items),
           "asr_bleu": round(st.mean(r["asr_bleu"] for r in runs.values()), 2),
           "asr_bleu_sd": round(st.stdev(r["asr_bleu"] for r in runs.values()), 2),
           "text_bleu": round(tb["score"], 2),
           "laal_d1": round(st.mean(r["laal_d1"] for r in runs.values()), 3),
           "laal_d0w": round(st.mean(r["laal_d0w"] for r in runs.values()), 3),
           "n_no_output": len(no_out), "n_last_commit_before_end": len(early),
           "blocks_per_sentence": round(st.mean(nb), 2) if nb else None,
           "thinker_errors": meta["n_errors"], "seeds": runs}
    return out


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


def write_summary(res: dict, t1: dict, t0: dict, path: Path) -> None:
    dname = {"en2zh": "En→Zh", "zh2en": "Zh→En"}
    L = ["# Hy-SimulMT-30B-A3B + 论文 talker，RealSI 句级（自动生成，`experiments/baselines_hysimulmt_v1/score_hy.py`）", "",
         "登记：`paper/baselines_run_plan_hysimulmt_v1.md`（含修订 1：只跑两档原生设置）。协议与论文 QABC 点相同："
         "streaming UniSS talker T1c@1500，三个 talker 种子取均值；speech-only ASR-BLEU；D1 = 段 LAAL（块开始播放时刻，含排队）；"
         "D0w = SimulEval 逐词 LAAL（块写出时刻）。BLEU 分母为全部句子（En→Zh 346，Zh→En 431）。", "",
         "## 全部点", "",
         "| 方向 | 点 | 原生模式 | ASR-BLEU（±sd） | 文本 BLEU | D1 (s) | D0w (s) | 无输出句 | 末次提交早于句尾 | 块/句 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for (p, d), r in sorted(res.items(), key=lambda x: (x[0][1], x[1]["laal_d1"])):
        L.append(f"| {dname[d]} | `{p}` | {r['config']['native_mode']} | {r['asr_bleu']:.2f} (±{r['asr_bleu_sd']:.2f}) | "
                 f"{r['text_bleu']:.2f} | {r['laal_d1']:.2f} | {r['laal_d0w']:.2f} | {r['n_no_output']} | "
                 f"{r['n_last_commit_before_end']} | {r['blocks_per_sentence']} |")
    for title, t, key in (("## 分档（按 D1，论文规则）", t1, "laal_d1"), ("## 分档（按 D0w，仅作参考）", t0, "laal_d0w")):
        L += ["", title, "", "| 方向 | 档 | 预算 (s) | SimulS2ST-Omni（报告值） | Hy-SimulMT + talker |", "|---|---|---|---|---|"]
        for k, v in t.items():
            d, m = k.split("|")
            cell = "--" if v["point"] is None else f"{v['asr_bleu']:.2f} ({v[key]:.2f})，`{v['point']}`"
            L.append(f"| {dname[d]} | {m} | {v['budget']:.2f} | {v['omni_reported']:.2f} | {cell} |")
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
    write_summary(res, t1, t0, out / "summary.md")
    print(json.dumps({f"{p}|{d}": {k: r[k] for k in ("asr_bleu", "asr_bleu_sd", "text_bleu", "laal_d1", "laal_d0w",
                                                       "n_no_output", "n_last_commit_before_end")}
                      for (p, d), r in res.items()}, indent=1))
    print(json.dumps(t1, indent=1))


if __name__ == "__main__":
    main()
