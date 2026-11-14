"""E7 summary (reports/mist_w3po_v1/e7_seamless_multispk_prereg.zh-CN.md §5-§6): collects the score / latency / build
files of the three runs into eval_outputs/mist_w3po_v1/e7_seamless/e7_summary_v1.json (refuses to overwrite) and prints
a markdown table. Our strict baseline / method test rows (joint/test_summary_v1.json) are copied in for reference.

    python experiments/s2st_baselines_v1/seamless_multispk/summarize_e7.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
DEC = REPO / "eval_outputs/mist_w3po_v1/decode"
TE = REPO / "eval_outputs/mist_w3po_v1/crf/t_eval"
OUT = REPO / "eval_outputs/mist_w3po_v1/e7_seamless/e7_summary_v1.json"
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
        n = f"jtest_{s}_seamless_default"
        t, t1, a = j(DEC / n / "score_text.json"), j(DEC / f"{n}_1spk" / "score_text.json"), j(DEC / n / "score_asr.json")
        tl, sl = j(TE / n / f"textlat_chunk_{n}.json"), j(TE / n / f"lat_chunk_{n}.json")
        b = j(DEC / n / "build_e7.json")
        u = lambda d, k, r="all", q="mean": m(d, "unit", k, r, q)  # noqa: E731
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
            "build": b and b["stats"], "spk_segments": b and b["spk_segments"],
        }
    ours = j(REPO / "eval_outputs/mist_w3po_v1/joint/test_summary_v1.json")
    out = {"e7": res, "reference_ours_test_summary_v1": ours,
           "prereg": "reports/mist_w3po_v1/e7_seamless_multispk_prereg.zh-CN.md"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    f = lambda x: "--" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))  # noqa: E731
    print("| set | tcpBLEU all | ov | 1spk | 1-spk LB all | ASR-BLEU all | ov | 1spk | text LAAL | speech LAAL | "
          "speech LAAL 1ch | StartOffset | EndOffset | windows | segments |")
    print("|" + "---|" * 15)
    for s, r in res.items():
        L = r["latency_chunk"]
        print(f"| {r['set']} | {f(r['text']['tcpBLEU_all'])} | {f(r['text']['tcpBLEU_ov'])} | {f(r['text']['tcpBLEU_1spk'])} "
              f"| {f(r['text_1spk_lower_bound']['tcpBLEU_all'])} | {f(r['asr']['tcpBLEU_all'])} | {f(r['asr']['tcpBLEU_ov'])} "
              f"| {f(r['asr']['tcpBLEU_1spk'])} | {f(L['text_laal']['all'])} | {f(L['speech_laal']['all'])} "
              f"| {f(L['speech_laal_1ch']['all'])} | {f(L['speech_start_offset'])} | {f(L['speech_end_offset'])} "
              f"| {f(m(r, 'build', 'windows'))} | {f(m(r, 'build', 'segments'))} |")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
