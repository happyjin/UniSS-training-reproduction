"""OpenBroadcast-SI v1: markdown table of bench_eval metrics (eval_outputs/mist_w3po_v1/voicecond_bench/<name>/metrics.json),
one row per run x TTS seed. Read-only.

    python -m experiments.mist_w3po_v1.voicecond.bench_table <name> [<name> ...]    (e.g. obsi_b_{id2,ood1,clean}_zh2en__t1c)
    python -m experiments.mist_w3po_v1.voicecond.bench_table --paired <A> <B>   paired by stream key (B - A; per-stream
        means averaged over the TTS seeds): mean difference, bootstrap 95 % CI (10k resamples, seed 0), Wilcoxon p
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


def paired(a: str, b: str) -> None:
    import numpy as np
    from scipy.stats import wilcoxon

    def load(n):
        M = json.load(open(OUT / n / "metrics.json"))
        out = {}
        for seed, ps in M["per_stream"].items():
            for m, v in ps.items():
                for k, x in v.items():
                    if k != "stream" and x is not None:
                        out.setdefault((v["stream"], k), []).append(x)
        return {key: float(np.mean(v)) for key, v in out.items()}
    A, B = load(a), load(b)
    print(f"| 指标 | {a} | {b} | 差（后 − 前） | 95% CI | Wilcoxon p | 流数 |")
    print("|---|---|---|---|---|---|---|")
    for k in ("blind_asr", "pesq", "sisdr", "contrast", "flatness", "sim"):
        keys = sorted(s for (s, kk) in A if kk == k and (s, k) in B)
        x = np.array([A[(s, k)] for s in keys])
        y = np.array([B[(s, k)] for s in keys])
        d = y - x
        rng = np.random.default_rng(0)
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"| {k} | {x.mean():.4f} | {y.mean():.4f} | {d.mean():+.4f} | [{np.percentile(bs, 2.5):+.4f}, "
              f"{np.percentile(bs, 97.5):+.4f}] | {p:.2g} | {len(d)} |")


def main() -> None:
    if sys.argv[1:2] == ["--paired"]:
        paired(sys.argv[2], sys.argv[3])
        return
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
