"""OpenBroadcast-SI v1: markdown table of bench_eval metrics (eval_outputs/mist_w3po_v1/voicecond_bench/<name>/metrics.json),
one row per run x TTS seed. Read-only.

    python -m experiments.mist_w3po_v1.voicecond.bench_table <name> [<name> ...]    (e.g. obsi_b_{id2,ood1,clean}_zh2en__t1c)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "eval_outputs/mist_w3po_v1/voicecond_bench"
COLS = ["条件", "方向", "种子", "tcpASR-BLEU", "不分说话人 ASR-BLEU", "文本 tcpBLEU", "文本不分说话人 BLEU", "语音 LAAL (s)",
        "EndOffset (s)", "SQUIM-PESQ", "SQUIM SI-SDR", "频谱对比度", "频谱平坦度", "说话人相似度"]


def f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> None:
    print("| " + " | ".join(COLS) + " |")
    print("|" + "---|" * len(COLS))
    for name in sys.argv[1:]:
        M = json.load(open(OUT / name / "metrics.json"))
        for i, (n, R) in enumerate(M["seeds"].items(), 1):
            g = lambda k, s="median": (R.get(k) or {}).get(s)  # noqa: E731
            lat = R.get("latency_chunk") or {}
            print("| " + " | ".join([M["cond"], M["dir"], str(i), f((R.get("tcp_asr") or {}).get("tcpBLEU_all")),
                                     f((R.get("blind_asr") or {}).get("corpus")), f((R.get("tcp_text") or {}).get("tcpBLEU_all")),
                                     f((R.get("blind_text") or {}).get("corpus")), f(lat.get("speech_laal"), 3),
                                     f(lat.get("end_offset"), 3), f(g("pesq")), f(g("sisdr")), f(g("contrast")),
                                     f(g("flatness"), 4), f(g("sim", "mean"), 3)]) + " |")


if __name__ == "__main__":
    main()
