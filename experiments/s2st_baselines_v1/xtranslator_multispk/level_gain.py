"""prereg v2 amendment 2: per AMI set, g = clip(0.1 / m_dev, 1, 31.6), m_dev = median over the set's DEV online windows
of the window's loudest 0.16 s (2560-sample) chunk RMS (input_gate_stats' statistic). Reports, input only: the dev
gate-closed share after g, and for the TEST windows the gate-closed share before / after g and the clipped-sample share
of x' = clip(g x, -1, 1). No outputs / scores are read."""
import json, sys
import numpy as np, soundfile as sf
DD = "eval_outputs/mist_w3po_v1/data"
SETS = {"ami": (f"{DD}/windows_zh2en_c2g06_v1/windows_dev_online.jsonl", f"{DD}/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl"),
        "en": (f"{DD}/windows_en2zh_c2g06_v1/windows_dev_online.jsonl", f"{DD}/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl")}


def maxrms(x):
    n = len(x) // 2560
    return float(np.sqrt((x[:n * 2560].reshape(n, 2560).astype(np.float64) ** 2).mean(1)).max()) if n else 0.0


def load(w):
    x, _ = sf.read(w["audio"], dtype="float32", always_2d=True)
    return x.mean(1)


out = {"rule": "g = clip(0.1 / m_dev, 1, 31.6); x' = clip(g x, -1, 1); gate = any 2560-sample chunk RMS >= 0.01",
       "prereg": "reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md amendment 2 (c18ee367a)", "sets": {}}
for s, (fd, ft) in SETS.items():
    D = [json.loads(l) for l in open(fd)]
    X = [load(w) for w in D]
    m = float(np.median([maxrms(x) for x in X]))
    g = float(np.clip(0.1 / m, 1.0, 31.6))
    dev_closed_raw = float(np.mean([maxrms(x) < 0.01 for x in X]))
    dev_closed = float(np.mean([maxrms(np.clip(g * x, -1, 1)) < 0.01 for x in X]))
    T = [json.loads(l) for l in open(ft)]
    tc_raw = tc = 0
    clip_n = tot = clip_w = 0
    for w in T:
        x = load(w)
        y = np.clip(g * x, -1, 1)
        tc_raw += maxrms(x) < 0.01
        tc += maxrms(y) < 0.01
        k = int((np.abs(g * x) > 1.0).sum())
        clip_n += k
        clip_w += k > 0
        tot += len(x)
    out["sets"][s] = {"dev_windows_file": fd, "dev_windows": len(D), "m_dev": round(m, 5), "g": round(g, 4),
                      "g_db": round(20 * np.log10(g), 2), "dev_gate_closed_share_raw": round(dev_closed_raw, 4),
                      "dev_gate_closed_share_after_g": round(dev_closed, 4), "test_windows_file": ft,
                      "test_windows": len(T), "test_gate_closed_share_raw": round(tc_raw / len(T), 4),
                      "test_gate_closed_share_after_g": round(tc / len(T), 4),
                      "test_clipped_sample_share": round(clip_n / tot, 7), "test_windows_with_clipping": int(clip_w),
                      "test_windows_with_clipping_share": round(clip_w / len(T), 4)}
    print(s, out["sets"][s], flush=True)
out["sets"]["ali"] = {"g": 1.0, "note": "AliMeeting: no tuning data allowed, no dev; g = 1 (amendment 2 item 2)"}
json.dump(out, open(sys.argv[1], "x"), indent=1)
