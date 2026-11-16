"""input-only statistics for the E8 report: per set, share of online test windows in which X-Translator's client
energy gate (Qwen3StreamingRecognizer: any 2560-sample chunk with RMS >= 0.01) never opens, their share of the
window duration and of diarizer-active speech (post.lowlat > 0.5), and the median max-chunk RMS. No outputs / scores."""
import json, sys
import numpy as np, soundfile as sf
SETS = {"ami": "eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl",
        "en": "eval_outputs/mist_w3po_v1/data/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl",
        "ali": "eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_ali.jsonl"}
out = {}
for s, f in SETS.items():
    W = [json.loads(l) for l in open(f)]
    mx, dur, act = [], [], []
    for w in W:
        x, _ = sf.read(w["audio"], dtype="float32", always_2d=True)
        x = x.mean(1)
        n = len(x) // 2560
        mx.append(float(np.sqrt((x[:n * 2560].reshape(n, 2560) ** 2).mean(1)).max()) if n else 0.0)
        dur.append(w["dur_ms"] / 1000)
        p = np.load(w["post"]["lowlat"])
        act.append(float((p.max(1) > 0.5).sum() * 0.08))
    mx, dur, act = map(np.array, (mx, dur, act))
    c = mx < 0.01
    out[s] = {"windows": int(len(W)), "gate_closed_windows": int(c.sum()), "gate_closed_share": round(float(c.mean()), 4),
              "gate_closed_dur_share": round(float(dur[c].sum() / dur.sum()), 4),
              "gate_closed_active_speech_share": round(float(act[c].sum() / act.sum()), 4),
              "median_max_chunk_rms": round(float(np.median(mx)), 4)}
    print(s, out[s], flush=True)
json.dump(out, open(sys.argv[1], "x"), indent=1)
