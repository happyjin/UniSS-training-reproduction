"""E8 summary: summarize_e7.py (copied, not changed) for the X-Translator runs -> eval_outputs/mist_w3po_v1/e8_xtranslator/
e8_summary_v1.json (refuses to overwrite) + markdown tables.

  e8          the original runs jtest_{ami,en,ali}_xtranslator_default (official default, audio as read)
  e8_level    the level-calibrated runs jtest_{ami,en}_xtranslator_level (prereg v2 amendment 2: x' = clip(g x, -1, 1))
  table1      amendment 2 item 4 (fixed before any score): AMI sets -> level, Ali -> default
Per run the e7_summary_v1.json fields (text / text_1spk_lower_bound / asr tcpBLEU, latency_chunk, build,
spk_segments), plus
  cp           cpASR-BLEU (hyp_asr.jsonl) and text cpBLEU (hyp.jsonl), cp_asr_bleu.py outputs in e8_xtranslator/cp/
  latency_table1  table1_lat.sh units (joint/table1_v14/lat/{blockstart,perword}/<run>.json): speech LAAL_block
               (primary), LAAL_spread, StartOffset, EndOffset
  counts       windows, segments, unspoken (TTS-failed) segments, TTS / MT failures, empty windows, error windows
  input        gain, clipped samples (from the raw rows' counters), energy-gate statistics (input only)
The E7 rows and our strict baseline / method test rows (joint/test_summary_v1.json) are copied in for reference.

    python experiments/s2st_baselines_v1/xtranslator_multispk/summarize_e8.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
DEC = REPO / "eval_outputs/mist_w3po_v1/decode"
TE = REPO / "eval_outputs/mist_w3po_v1/crf/t_eval"
T1 = REPO / "eval_outputs/mist_w3po_v1/joint/table1_v14/lat"
E8 = REPO / "eval_outputs/mist_w3po_v1/e8_xtranslator"
OUT = E8 / "e8_summary_v1.json"
SETS = {"ami": "AMI zh->en", "en": "AMI En->Zh", "ali": "AliMeeting zh->en"}
VARIANTS = {"default": ("raw", ("ami", "en", "ali")), "level": ("raw_level", ("ami", "en"))}
TABLE1 = {"ami": "level", "en": "level", "ali": "default"}
HEAD = ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk", "CR", "collapse", "duplicate_rate", "omitted_ov", "omitted_1spk",
        "inserted_ov", "inserted_1spk")


def j(p: Path):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def m(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def raw_input(raw: str, s: str) -> dict:
    g, clipped, samples, rows = set(), 0, 0, 0
    for f in sorted(glob.glob(str(E8 / raw / s / "part_*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            rows += 1
            c = r.get("counters") or {}
            g.add(c.get("gain", r.get("config", {}).get("gain", 1.0)))
            clipped += c.get("clipped_samples", 0)
            samples += r.get("src_samples", 0)
    return {"gain": sorted(g), "clipped_samples": clipped, "samples": samples,
            "clipped_sample_share": round(clipped / samples, 7) if samples else None, "rows": rows}


def run_summary(s: str, n: str, raw: str) -> dict:
    t, t1, a = j(DEC / n / "score_text.json"), j(DEC / f"{n}_1spk" / "score_text.json"), j(DEC / n / "score_asr.json")
    tl, sl = j(TE / n / f"textlat_chunk_{n}.json"), j(TE / n / f"lat_chunk_{n}.json")
    bs, pw = j(T1 / "blockstart" / f"{n}.json"), j(T1 / "perword" / f"{n}.json")
    b = j(DEC / n / "build_e8.json")
    ca, ct = j(E8 / "cp" / f"{n}_asr.json"), j(E8 / "cp" / f"{n}_text.json")
    u = lambda d, k, r="all", q="mean": m(d, "unit", k, r, q)  # noqa: E731
    st = (b or {}).get("stats", {})
    return {
        "set": SETS[s], "run": n,
        "text": {k: m(t, k) for k in HEAD},
        "text_1spk_lower_bound": {k: m(t1, k) for k in ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk")},
        "asr": {k: m(a, k) for k in HEAD},
        "cp": {"cpASR_BLEU": m(ca, "cpBLEU"), "cpBLEU_text": m(ct, "cpBLEU")},
        "latency_chunk": {
            "text_laal": {r: u(tl, "text_laal_online", r) for r in ("all", "ov", "1spk")},
            "speech_laal": {r: u(sl, "speech_laal_online", r) for r in ("all", "ov", "1spk")},
            "speech_laal_1ch": {r: u(sl, "speech_laal_online_1ch", r) for r in ("all", "ov", "1spk")},
            "speech_start_offset": u(sl, "speech_start_online"), "speech_end_offset": u(sl, "speech_end_online"),
            "speech_end_offset_p50": u(sl, "speech_end_online", q="p50"),
            "speech_end_offset_1ch": u(sl, "speech_end_online_1ch"),
            "segment_block_mismatch": m(sl, "segment_block_mismatch"), "check": m(sl, "check"),
            "units_text": u(tl, "text_laal_online", q="n"), "units_speech": u(sl, "speech_laal_online", q="n"),
            "load_speech_over_src_union": m(sl, "load", "speech_over_src_union"),
        },
        "latency_table1": {
            "laal_block": {r: u(bs, "speech_laal_online", r) for r in ("all", "ov", "1spk")},
            "laal_spread": {r: u(pw, "speech_laal_online", r) for r in ("all", "ov", "1spk")},
            "start_offset": u(bs, "speech_start_online"), "end_offset": u(bs, "speech_end_online"),
            "segment_block_mismatch": m(bs, "segment_block_mismatch"), "units": u(bs, "speech_laal_online", q="n"),
        },
        "counts": {"windows": st.get("windows"), "segments": st.get("segments"),
                   "unspoken_segments": st.get("unspoken_segments", 0),
                   "tts_failures": st.get("tts_fail_items", 0), "tts_timeouts": st.get("tts_timeout_items", 0),
                   "mt_failures": st.get("mt_fail_items", 0),
                   "empty_translation_items": st.get("empty_translation_items", 0),
                   "empty_windows": st.get("empty_windows"), "windows_without_speech": st.get("windows_without_speech"),
                   "error_windows": st.get("error_windows"), "unk_segments": st.get("unk_segments"),
                   "boundary_flushes": st.get("boundary_flushes"), "streams": st.get("streams"),
                   "speech_s": st.get("speech_s")},
        "input": raw_input(raw, s),
        "build": b and b["stats"], "spk_segments": b and b["spk_segments"],
    }


def main() -> None:
    res = {v: {s: run_summary(s, f"jtest_{s}_xtranslator_{v}", raw) for s in sets} for v, (raw, sets) in VARIANTS.items()}
    gate, gain = j(E8 / "input_gate_stats_v1.json"), j(E8 / "level_gain_v1.json")
    out = {"e8": res["default"], "e8_level": res["level"],
           "table1": {s: {"variant": v, "run": f"jtest_{s}_xtranslator_{v}"} for s, v in TABLE1.items()},
           "input_gate_stats_raw": gate, "level_gain": gain,
           "reference_e7": m(j(REPO / "eval_outputs/mist_w3po_v1/e7_seamless/e7_summary_v1.json"), "e7"),
           "reference_ours_test_summary_v1": j(REPO / "eval_outputs/mist_w3po_v1/joint/test_summary_v1.json"),
           "prereg": "reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md (+ amendment 2, c18ee367a)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    f = lambda x: "--" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))  # noqa: E731
    print("| variant | set | tcpBLEU all / ov | cpASR-BLEU | tcpASR-BLEU | text LAAL | LAAL_block | LAAL_spread | "
          "StartOffset | EndOffset | windows | segments | unspoken | empty win | err win | clipped |")
    print("|" + "---|" * 16)
    for v, R in res.items():
        for s, r in R.items():
            L, T, C, I = r["latency_chunk"], r["latency_table1"], r["counts"], r["input"]
            print(f"| {v} | {r['set']} | {f(r['text']['tcpBLEU_all'])} / {f(r['text']['tcpBLEU_ov'])} "
                  f"| {f(r['cp']['cpASR_BLEU'])} | {f(r['asr']['tcpBLEU_all'])} | {f(L['text_laal']['all'])} "
                  f"| {f(T['laal_block']['all'])} | {f(T['laal_spread']['all'])} | {f(L['speech_start_offset'])} "
                  f"| {f(L['speech_end_offset'])} | {f(C['windows'])} | {f(C['segments'])} | {f(C['unspoken_segments'])} "
                  f"| {f(C['empty_windows'])} | {f(C['error_windows'])} | {f(I['clipped_sample_share'])} |")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
