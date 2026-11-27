"""portfolio 2026-11-27 (prereg change 55): test verdict of one method from a collect.py JSON that also holds p3hk and p3hk2.
Per direction: (i) ASR >= p3hk ASR - 0.2 and D2 <= p3hk D2 - 0.10 s, or (ii) ASR >= the p3hk-p3hk2 line at the method's D2 + 0.2
(linear through the two points, extrapolated). Both directions -> effective; one -> single-direction effective; none -> ineffective.
    python3 experiments/sst_hsmm_v1/pf_verdict.py <collect.json> <system> [<dev choice.json>]
"""
import json
import sys

d = json.load(open(sys.argv[1]))
s = sys.argv[2]
ch = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
ok = {}
for dr in ("en2zh", "zh2en"):
    g = lambda n, k: d[n]["NEW_vs_B_b"][dr][k]["diff"]
    a, x = g(s, "asr"), g(s, "d2")
    ha, hx, ka, kx = g("p3hk", "asr"), g("p3hk", "d2"), g("p3hk2", "asr"), g("p3hk2", "d2")
    line = ha + (ka - ha) / (kx - hx) * (x - hx)
    c1, c2 = a >= ha - 0.2 and x <= hx - 0.10, a - line >= 0.2
    ok[dr] = c1 or c2
    sp = ch.get(dr, {})
    scr = "" if not sp else f" dev={sp['spec']} screen={'有效' if sp['screen_effective'] else '无效'}"
    print(f"{s} {dr}: vs p3hk ASR {a - ha:+.3f} D2 {x - hx:+.3f} s | vs line {a - line:+.3f} | (i) {c1} (ii) {c2}{scr}")
n = sum(ok.values())
print(s, "verdict:", "有效" if n == 2 else ("单方向有效 (" + [k for k, v in ok.items() if v][0] + ")" if n == 1 else "无效"))
