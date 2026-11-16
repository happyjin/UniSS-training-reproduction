"""E8 summary: summarize_e7.py (copied, not changed) for the X-Translator runs -> eval_outputs/mist_w3po_v1/e8_xtranslator/
e8_summary_v1.json (refuses to overwrite) + a markdown table. Same fields as e7_summary_v1.json, plus
  latency_table1   the Table 1 latency units (table1_lat.sh -> joint/table1_v14/lat/{blockstart,perword}/<run>.json):
                   speech LAAL_block (block-start convention, the paper's primary), LAAL_spread, StartOffset, EndOffset
  counts           windows, segments, unspoken (TTS-failed) segments, TTS / MT failures, empty windows, error windows
Our strict baseline / method test rows (joint/test_summary_v1.json) and the E7 rows are copied in for reference.

    python experiments/s2st_baselines_v1/xtranslator_multispk/summarize_e8.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
DEC = REPO / "eval_outputs/mist_w3po_v1/decode"
TE = REPO / "eval_outputs/mist_w3po_v1/crf/t_eval"
T1 = REPO / "eval_outputs/mist_w3po_v1/joint/table1_v14/lat"
OUT = REPO / "eval_outputs/mist_w3po_v1/e8_xtranslator/e8_summary_v1.json"
SETS = {"ami": "AMI zh->en", "ali": "AliMeeting zh->en", "en": "AMI En->Zh"}
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


def main() -> None:
    res = {}
    for s, label in SETS.items():
        n = f"jtest_{s}_xtranslator_default"
        t, t1, a = j(DEC / n / "score_text.json"), j(DEC / f"{n}_1spk" / "score_text.json"), j(DEC / n / "score_asr.json")
        tl, sl = j(TE / n / f"textlat_chunk_{n}.json"), j(TE / n / f"lat_chunk_{n}.json")
        bs, pw = j(T1 / "blockstart" / f"{n}.json"), j(T1 / "perword" / f"{n}.json")
        b = j(DEC / n / "build_e8.json")
        u = lambda d, k, r="all", q="mean": m(d, "unit", k, r, q)  # noqa: E731
        st = (b or {}).get("stats", {})
        res[s] = {
            "set": label, "run": n,
            "text": {k: m(t, k) for k in HEAD},
            "text_1spk_lower_bound": {k: m(t1, k) for k in ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk")},
            "asr": {k: m(a, k) for k in HEAD},
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
            "build": b and b["stats"], "spk_segments": b and b["spk_segments"],
        }
    ours = j(REPO / "eval_outputs/mist_w3po_v1/joint/test_summary_v1.json")
    e7 = j(REPO / "eval_outputs/mist_w3po_v1/e7_seamless/e7_summary_v1.json")
    out = {"e8": res, "reference_ours_test_summary_v1": ours, "reference_e7": e7 and e7.get("e7"),
           "prereg": "reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    f = lambda x: "--" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))  # noqa: E731
    print("| set | tcpBLEU all | ov | 1spk | 1-spk LB all | ASR-BLEU all | ov | 1spk | text LAAL | speech LAAL (spread) | "
          "LAAL_block | StartOffset | EndOffset | windows | segments | unspoken | empty win |")
    print("|" + "---|" * 17)
    for s, r in res.items():
        L, T, C = r["latency_chunk"], r["latency_table1"], r["counts"]
        print(f"| {r['set']} | {f(r['text']['tcpBLEU_all'])} | {f(r['text']['tcpBLEU_ov'])} | {f(r['text']['tcpBLEU_1spk'])} "
              f"| {f(r['text_1spk_lower_bound']['tcpBLEU_all'])} | {f(r['asr']['tcpBLEU_all'])} | {f(r['asr']['tcpBLEU_ov'])} "
              f"| {f(r['asr']['tcpBLEU_1spk'])} | {f(L['text_laal']['all'])} | {f(L['speech_laal']['all'])} "
              f"| {f(T['laal_block']['all'])} | {f(L['speech_start_offset'])} | {f(L['speech_end_offset'])} "
              f"| {f(C['windows'])} | {f(C['segments'])} | {f(C['unspoken_segments'])} | {f(C['empty_windows'])} |")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
