"""Paper v4 under the 2026-10-30 compute deadline (cards 4-7 until 10:15 UTC): the Zh->En S2-dl is retrained for 1,000
steps (thinker_gstar_s2dl_zh2en_1k_20261030; the 3,000-step run would end after the deadline and was stopped by PID),
then decoded at 1 s on AMI (8 shards, cards 4-5) and AliMeeting (12 shards, cards 6-7), one Thinker copy per card.

    python3 reports/gstar_si_paper_v1/make_v4c_runs.py
"""
import json
from pathlib import Path

EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v4_20261030"
S2DLZ = f"{EV}/thinker_gstar_s2dl_zh2en_1k_20261030/thinker_merged"
J = Path("reports/gstar_si_paper_v1/jobs/v4_runs")


def runs(name, win, n):
    return [{"module": "gstar_si_stream_online",
             "argv": ["--thinker", S2DLZ, "--windows", win, "--out", f"{OUT}/{name}/hyp_s{k}.jsonl", "--shard", str(k),
                      "--num-shards", str(n), "--policy", "wait", "--window", "1", "--post-key", "lowlat", "--dir", "zh2en"],
             "log": f"{OUT}/{name}/infer_s{k}.log"} for k in range(n)]


z = runs("zh2en_S2dl_d1", "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl", 8)
a = runs("ali_S2dl_d1", "data/alimeeting_si_v1/windows_test_ref.jsonl", 12)
for card, rs in ((4, z[:4]), (5, z[4:]), (6, a[:6]), (7, a[6:])):
    with open(J / f"p3k_card{card}.json", "x") as fh:
        json.dump(rs, fh, indent=1)
print("ok")
