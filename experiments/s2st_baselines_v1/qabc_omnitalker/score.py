"""Score the QABC tier points spoken by the paper's talker (stored runs, recomputed) and by Qwen3-Omni's native talker
(registration paper/baselines_run_plan_qabc_omnitalker_v1.md §3), with the paper's code:

  ASR-BLEU   asr_speech.jsonl (speech-only wav, Whisper-large-v3 / Paraformer) -> evaluation.text_metrics.corpus_bleu
             over all sentences (a missing ASR row counts as ""), as collect_results.realsi_points
  D1         collect_results.realsi_points' segment LAAL, verbatim (unit = start_ms of its block, queue included)
  D0w        laal_defs.sentence_defs()["D0w"] (unit = its block's write time; talker-independent)
  talker WER sentence level: the speech-only ASR against the committed text the talker was given (the concatenated
             block texts), both normalised as BLEU (text_metrics.normalize_for_bleu: English words, Chinese characters),
             corpus rate = edits / reference tokens, with substitutions / deletions / insertions
  text BLEU  the committed text against the reference (identical for both talkers: a check)
  queue      every block starts at max(write_ms, previous block's end) (checked on every run)
A point = the mean over three talker seeds (ASR-BLEU, D1, D0w, talker WER). Tiers: per direction, the best ASR-BLEU
among the 4 points whose D1 fits the paper's budget (and the same by D0w, for reference).

    PYTHONPATH=. uniss-eval/bin/python experiments/s2st_baselines_v1/qabc_omnitalker/score.py paper    # check only
    ... score.py all --out paper/baselines_results/qabc_omnitalker                                       # refuses existing files
    ... score.py blocks --out paper/baselines_results/qabc_omnitalker    # block-level WER (Omni, seed 0) once its ASR is done
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evaluation.text_metrics import corpus_bleu, normalize_for_bleu  # noqa: E402
from experiments.s2st_baselines_v1.qabc_omnitalker.common import (  # noqa: E402
    OUT, PAPER_SEEDS, POINTS, REPO, SEEDS, omni_run, paper_run, point_name)
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "laal_defs", REPO / "eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112/laal_defs.py")
LD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LD)


def edit_ops(ref: list, hyp: list) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of a minimum edit alignment."""
    n, m = len(ref), len(hyp)
    D = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]   # (cost, S, D, I)
    for i in range(1, n + 1):
        D[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        D[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c, s, d, ins = D[i - 1][j - 1]
            best = (c, s, d, ins) if ref[i - 1] == hyp[j - 1] else (c + 1, s + 1, d, ins)
            c, s, d, ins = D[i - 1][j]
            if c + 1 < best[0]:
                best = (c + 1, s, d + 1, ins)
            c, s, d, ins = D[i][j - 1]
            if c + 1 < best[0]:
                best = (c + 1, s, d, ins + 1)
            D[i][j] = best
    return D[n][m][1:]


def wer_parts(pairs: list, lang: str) -> dict:
    S = Dl = I = N = 0
    for ref, hyp in pairs:
        r, h = normalize_for_bleu(ref, lang).split(), normalize_for_bleu(hyp, lang).split()
        s, d, i = edit_ops(r, h)
        S, Dl, I, N = S + s, Dl + d, I + i, N + len(r)
    return {"wer": round(100 * (S + Dl + I) / max(N, 1), 2), "sub": round(100 * S / max(N, 1), 2),
            "del": round(100 * Dl / max(N, 1), 2), "ins": round(100 * I / max(N, 1), 2), "ref_tokens": N,
            "unit": "word" if lang == "eng" else "char"}


def score_run(run: Path) -> dict:
    recs = [json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")]
    heard = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(run / "asr_speech.jsonl", encoding="utf-8"))}
    lang = recs[0]["tgt_lang"]
    asr = corpus_bleu([heard.get(r["id"], "") for r in recs], [r["translation_ref"] for r in recs], language=lang)
    committed = ["".join(b["text"] for b in r["blocks"]) for r in recs]
    text = corpus_bleu(committed, [r["translation_ref"] for r in recs], language=lang)
    seg, d0w, queue_bad = [], [], 0
    for r in recs:   # collect_results.realsi_points, verbatim
        ds = [bl["start_ms"] for bl in r["blocks"] for _ in units(bl["text"], lang)]
        if ds:
            seg.append(laal(ds, r["source_duration_ms"], len(units(r["translation_ref"], lang))))
        sd = LD.sentence_defs(r)
        if sd is not None:
            d0w.append(sd["D0w"])
        end = 0
        for b in r["blocks"]:
            queue_bad += abs(b["start_ms"] - max(b["write_ms"], end)) > 1
            end = b["start_ms"] + b["dur_ms"]
    spoken = [sum(b["dur_ms"] for b in r["blocks"]) for r in recs]
    return {"run": str(run.relative_to(REPO)), "n": len(recs), "n_asr": len(heard), "asr_cover_ok": len(heard) >= 0.99 * len(recs),
            "asr_bleu": asr["score"], "text_bleu": text["score"], "laal_d1": st.mean(seg) / 1000, "n_d1": len(seg),
            "laal_d0w": st.mean(d0w) / 1000, "n_d0w": len(d0w),
            "talker_wer": wer_parts([(c, heard.get(r["id"], "")) for c, r in zip(committed, recs)], lang),
            "queue_violations": queue_bad, "speech_s_per_sentence": st.mean(spoken) / 1000,
            "blocks": sum(len(r["blocks"]) for r in recs), "blocks_no_audio": sum(b["dur_ms"] <= 0 for r in recs for b in r["blocks"])}


def mean_point(runs: list) -> dict:
    out = {"n_seeds": len(runs), "n": runs[0]["n"]}
    for k in ("asr_bleu", "text_bleu", "laal_d1", "laal_d0w", "speech_s_per_sentence"):
        out[k] = round(st.mean(r[k] for r in runs), 4)
    out["asr_bleu_sd"] = round(st.stdev(r["asr_bleu"] for r in runs), 3) if len(runs) > 1 else None
    out["talker_wer"] = round(st.mean(r["talker_wer"]["wer"] for r in runs), 2)
    for k in ("sub", "del", "ins"):
        out[f"talker_wer_{k}"] = round(st.mean(r["talker_wer"][k] for r in runs), 2)
    out["talker_wer_unit"] = runs[0]["talker_wer"]["unit"]
    out["queue_violations"] = sum(r["queue_violations"] for r in runs)
    out["asr_cover_ok"] = all(r["asr_cover_ok"] for r in runs)
    return out


def tiers(points: dict, key: str) -> dict:
    out = {}
    for d in ("en2zh", "zh2en"):
        for p in POINTS:
            if p[0] != d:
                continue
            budget = p[2]
            ok = [(n, v) for n, v in points.items() if n.startswith(d) and v[key] <= budget + 1e-9]
            best = max(ok, key=lambda x: x[1]["asr_bleu"]) if ok else None
            out[f"{d}|{p[1]}"] = {"budget": budget, "point": best[0] if best else None,
                                  "asr_bleu": round(best[1]["asr_bleu"], 2) if best else None,
                                  key: round(best[1][key], 2) if best else None}
    return out


def paper_points() -> dict:
    res = {}
    for p in POINTS:
        runs = [score_run(paper_run(p, s)) for s in PAPER_SEEDS]
        m = mean_point(runs)
        m.update(config={"family": p[3], "window_s": p[4], "b0": p[5]},
                 paper_printed={"asr_bleu": p[6], "laal_d1": p[7]},
                 reproduces_paper=round(m["asr_bleu"], 2) == p[6] and round(m["laal_d1"], 2) == p[7], seeds=runs)
        res[point_name(p[0], p[1])] = m
    return res


def block_asr(name: str) -> dict | None:
    """secondary (Omni talker, seed 0): each block's own wav transcribed, against the text sent for it."""
    d = name.split("_")[0]
    f = OUT / "block_asr" / f"asr_{d}_s0.jsonl"
    if not f.exists():
        return None
    lang = "cmn" if d == "en2zh" else "eng"
    rows = [r for r in map(json.loads, open(f, encoding="utf-8")) if r["id"].startswith(f"{name}_s0__")]
    want = sum(1 for l in open(OUT / "block_asr" / f"blocks_{d}_s0.jsonl", encoding="utf-8")
               if json.loads(l)["id"].startswith(f"{name}_s0__"))
    if len(rows) < want:   # still being transcribed: reported later (block_wer_s0.json), never on a partial set
        return None
    out = wer_parts([(r["text_sent"], r.get("asr_text", "")) for r in rows], lang)
    out["blocks"] = len(rows)
    return out


def omni_points() -> dict:
    res = {}
    for p in POINTS:
        runs = [score_run(omni_run(p, s)) for s in SEEDS]
        m = mean_point(runs)
        n = point_name(p[0], p[1])
        fid = json.load(open(OUT / "fidelity" / f"{n}.json", encoding="utf-8"))
        m.update(config={"family": p[3], "window_s": p[4], "b0": p[5]}, thinker_fidelity=fid,
                 block_talker_wer_s0=block_asr(n), seeds=runs)
        res[n] = m
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("paper", "all", "blocks"))
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.what == "blocks":   # the secondary block-level WER once the block ASR is complete
        res = {point_name(p[0], p[1]): block_asr(point_name(p[0], p[1])) for p in POINTS}
        assert all(v is not None for v in res.values()), "block ASR incomplete"
        with open(Path(a.out) / "block_wer_s0.json", "x", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print(json.dumps(res, indent=1))
        return
    paper = paper_points()
    if a.what == "paper":
        for n, v in paper.items():
            print(n, {k: v[k] for k in ("asr_bleu", "laal_d1", "laal_d0w", "talker_wer", "talker_wer_sub", "talker_wer_del",
                                        "talker_wer_ins", "text_bleu", "queue_violations", "reproduces_paper",
                                        "speech_s_per_sentence")})
        return
    omni = omni_points()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in paper:
        with open(out / f"{n}.json", "x", encoding="utf-8") as fh:
            json.dump({"point": n, "omni_native_talker": omni[n], "paper_talker": paper[n]}, fh, ensure_ascii=False, indent=1)
    summ = {"points": {n: {"paper": {k: paper[n][k] for k in ("asr_bleu", "asr_bleu_sd", "laal_d1", "laal_d0w", "talker_wer",
                                                               "text_bleu", "n", "reproduces_paper", "speech_s_per_sentence")},
                           "omni": {**{k: omni[n][k] for k in ("asr_bleu", "asr_bleu_sd", "laal_d1", "laal_d0w", "talker_wer",
                                                               "text_bleu", "n", "speech_s_per_sentence")},
                                    "thinker_exact_share_s0": omni[n]["thinker_fidelity"]["per_seed"]["0"]["exact_share"],
                                    "thinker_wer_s0": omni[n]["thinker_fidelity"]["per_seed"]["0"]["thinker_wer"],
                                    "block_talker_wer_s0": (omni[n]["block_talker_wer_s0"] or {}).get("wer")}}
                for n in paper},
            "tiers_d1": {"paper": tiers(paper, "laal_d1"), "omni": tiers(omni, "laal_d1")},
            "tiers_d0w": {"paper": tiers(paper, "laal_d0w"), "omni": tiers(omni, "laal_d0w")}}
    with open(out / "summary.json", "x", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=1)
    with open(out / "summary_tables.md", "x", encoding="utf-8") as fh:
        fh.write(tables_md(paper, omni, summ))
    print(json.dumps(summ, ensure_ascii=False, indent=1))


def tables_md(paper: dict, omni: dict, summ: dict) -> str:
    f2 = lambda x: "--" if x is None else f"{x:.2f}"   # noqa: E731
    L = ["| 方向 | 档 | ASR-BLEU 论文 talker | ASR-BLEU Omni talker | Δ (Omni − 论文) | D1 论文 | D1 Omni | D0w（两边） "
         "| talker WER 论文 | talker WER Omni | 语音时长/句 论文 | 语音时长/句 Omni |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in POINTS:
        n = point_name(p[0], p[1])
        a, b = paper[n], omni[n]
        u = "CER" if p[0] == "en2zh" else "WER"
        L.append(f"| {'En→Zh' if p[0] == 'en2zh' else 'Zh→En'} | {p[1]} | {a['asr_bleu']:.2f} ± {a['asr_bleu_sd']:.2f} "
                 f"| {b['asr_bleu']:.2f} ± {b['asr_bleu_sd']:.2f} | {b['asr_bleu'] - a['asr_bleu']:+.2f} | {a['laal_d1']:.2f} "
                 f"| {b['laal_d1']:.2f} | {a['laal_d0w']:.2f} / {b['laal_d0w']:.2f} | {a['talker_wer']:.2f} ({u}) "
                 f"| {b['talker_wer']:.2f} ({u}) | {a['speech_s_per_sentence']:.2f} s | {b['speech_s_per_sentence']:.2f} s |")
    L += ["", "| 方向 | 档 | Omni talker WER 替换/删除/插入 | 论文 talker 替换/删除/插入 | Omni Thinker 与送入文本完全一致（种子 0） "
          "| Omni Thinker WER（种子 0） | Omni 块级 talker WER（种子 0） |", "|---|---|---|---|---|---|---|"]
    for p in POINTS:
        n = point_name(p[0], p[1])
        a, b = paper[n], omni[n]
        fs = b["thinker_fidelity"]["per_seed"]["0"]
        bw = b["block_talker_wer_s0"]
        L.append(f"| {'En→Zh' if p[0] == 'en2zh' else 'Zh→En'} | {p[1]} | {b['talker_wer_sub']:.2f} / {b['talker_wer_del']:.2f} / "
                 f"{b['talker_wer_ins']:.2f} | {a['talker_wer_sub']:.2f} / {a['talker_wer_del']:.2f} / {a['talker_wer_ins']:.2f} "
                 f"| {100 * fs['exact_share']:.1f}% | {fs['thinker_wer']:.2f} | {f2(bw and bw['wer'])} |")
    L += ["", "| 方向 | 档（预算 s） | 论文 talker：点 / ASR-BLEU / D1 | Omni talker：点 / ASR-BLEU / D1 |", "|---|---|---|---|"]
    for k, t in summ["tiers_d1"]["paper"].items():
        o = summ["tiers_d1"]["omni"][k]
        L.append(f"| {k.split('|')[0]} | {k.split('|')[1]} ({t['budget']}) | {t['point'] or '--'} / {f2(t['asr_bleu'])} / "
                 f"{f2(t['laal_d1'])} | {o['point'] or '--'} / {f2(o['asr_bleu'])} / {f2(o['laal_d1'])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
