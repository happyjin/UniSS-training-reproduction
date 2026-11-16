"""E9 cascade summary (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §1 S3, §2): per test set, the
cascade's numbers read from the outputs of e9_test.sh, with B's from the same files / functions for reference.

  text     decode/<name>_rel/score_text.json: tcpBLEU all / ov / 1spk, CR, collapse (the relabelled text; score_run);
           the raw speaker-blind decode's score_text.json (the model's own speaker tokens) as raw_*;
           chunk-protocol text LAAL (crf/t_eval/<rel>/textlat_chunk_<rel>.json, unit.text_laal_online all)
  speech   tcpASR-BLEU = score_run --asr tcpBLEU_all per TTS seed (decode/<rel>__t1c[_tts2]/score_asr.json), mean of the
           two seeds; LAAL_block / LAAL_spread / StartOffset / EndOffset = table1_analysis.lat_stats on the Table 1
           latency units (joint/table1_v14/lat/{blockstart,perword}/<run>.json.units.jsonl; mean over ov + 1spk units,
           mean over seeds); per-seed chunk-protocol speech LAAL (lat_chunk_<run>.json unit.speech_laal_online all)
  counts   windows, segments, unk segments, relabel changes, Talker streams / dropped streams, ASR rows
B (rows R0 of v11f_analysis: jtest_b<set>_..._s06000__t1c[_tts2]) is read the same way where its files exist.
Refuses to overwrite.

    python -m experiments.mist_w3po_v1.evaluation.e9_summary --out eval_outputs/mist_w3po_v1/e9_cascade/e9_summary_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import table1_analysis as TA

REPO = Path(__file__).resolve().parents[3]
DEC = REPO / "eval_outputs/mist_w3po_v1/decode"
SPD = REPO / "eval_outputs/mist_w3po_v1/speech"
TE = REPO / "eval_outputs/mist_w3po_v1/crf/t_eval"
STEP = "s03000"
BT = "joint_v8h6k_20261113T073658Z_s06000"
SETS = {"ami": "AMI Zh->En", "ali": "AliMeeting Zh->En", "en": "AMI En->Zh"}


def jl(p: Path):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def nlines(p: Path) -> int | None:
    return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else None


def text(d: Path) -> dict:
    r = jl(d / "score_text.json") or {}
    return {k: r.get(k) for k in ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk", "CR", "collapse", "duplicate_rate")}


def unit_mean(p: Path, key: str):
    r = jl(p)
    try:
        return r["unit"][key]["all"]["mean"]
    except (TypeError, KeyError):
        return None


def speech(runs: list[str], lat_files: list[Path], te_dir: Path) -> dict:
    asr = [(jl(DEC / n / "score_asr.json") or {}).get("tcpBLEU_all") for n in runs]
    out = {"runs": runs, "tcpASR_BLEU_per_seed": asr,
           "tcpASR_BLEU": round(float(np.mean(asr)), 2) if all(a is not None for a in asr) else None,
           "speech_laal_online_per_seed": [unit_mean(te_dir / f"lat_chunk_{n}.json", "speech_laal_online") for n in runs]}
    if all(f.exists() for f in lat_files):
        for name, (mu, _) in TA.lat_stats(lat_files).items():
            out[{"laal_block": "LAAL_block", "laal_spread": "LAAL_spread", "startoff": "StartOffset",
                 "endoff": "EndOffset"}[name]] = round(mu, 3)
    else:
        out["latency_units_missing"] = [str(f.relative_to(REPO)) for f in lat_files if not f.exists()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {"registration": "reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §1 S3 (E9)",
           "models": {"zh2en": "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_zh2en_v8/thinker_merged",
                      "en2zh": "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_en2zh_v8/thinker_merged"},
           "sets": {}}
    for s, label in SETS.items():
        n = f"jtest_{s}_cascade_blind_{STEP}"
        r, sp = f"{n}_rel", f"{n}_rel__t1c"
        runs = [sp, f"{sp}_tts2"]
        lf = [TA.TL / mode / f"{x}.json.units.jsonl" for x in runs for mode in ("perword", "blockstart")]
        rel = jl(DEC / r / "relabel_stats.json") or {}
        dropped = [sum(1 for _ in open(SPD / x / "talker_in/DROPPED.txt")) if (SPD / x / "talker_in/DROPPED.txt").exists()
                   else None for x in runs]
        row = {"label": label, "decode": n, "relabelled": r,
               "text": text(DEC / r), "raw_blind_text": text(DEC / n),
               "text_laal_chunk": unit_mean(TE / r / f"textlat_chunk_{r}.json", "text_laal_online"),
               "speech": speech(runs, lf, TE / sp),
               "counts": {"windows": nlines(DEC / r / "hyp.jsonl"), "relabel": rel.get("stats"),
                          "spk_segments": rel.get("spk_segments"),
                          "talker_streams": [nlines(SPD / x / "talker_in/items.jsonl") for x in runs],
                          "talker_results": [nlines(SPD / x / "talker/results.jsonl") for x in runs],
                          "dropped_streams": dropped}}
        b = f"jtest_b{s}_{BT}"
        bl = [TA.FL / mode / f"{s}_R0_s{k}.json.units.jsonl" for k in (1, 2) for mode in ("perword", "blockstart")]
        row["B_reference"] = {"decode": b, "text": text(DEC / b),
                              "text_laal_chunk": unit_mean(TE / b / f"textlat_chunk_{b}.json", "text_laal_online"),
                              "speech": speech([f"{b}__t1c", f"{b}__t1c_tts2"], bl, TE / f"{b}__t1c")}
        res["sets"][s] = row
        print(s, json.dumps({"text": row["text"], "tcpASR": row["speech"].get("tcpASR_BLEU"),
                             "LAAL_block": row["speech"].get("LAAL_block"), "B_tcpASR": row["B_reference"]["speech"].get("tcpASR_BLEU"),
                             "B_LAAL_block": row["B_reference"]["speech"].get("LAAL_block")}, ensure_ascii=False), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
