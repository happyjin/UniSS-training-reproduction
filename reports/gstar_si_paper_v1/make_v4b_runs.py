"""Paper v4, second chain: gstar_multi run files (cards 4-7, six runs per card) for the En->Zh deadline-label model
trained with mixed read intervals (S2-dl-m, build_ami_gstar_stream_rows_multid rows), decoded at 0.5, 1 and 2 s.

    python3 reports/gstar_si_paper_v1/make_v4b_runs.py
"""
import json
from pathlib import Path

EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v4_20261030"
S2DLM = f"{EV}/thinker_gstar_s2dlm_20261030/thinker_merged"
WIN = "data/ami_multispk_si_v1/e2e_gstar_stream_v1/windows_test_ref.jsonl"
J = Path("reports/gstar_si_paper_v1/jobs/v4_runs")
N = 8
rs = [{"module": "gstar_si_stream_online",
       "argv": ["--thinker", S2DLM, "--windows", WIN, "--out", f"{OUT}/en2zh_S2dlm_d{w}/hyp_s{k}.jsonl", "--shard", str(k),
                "--num-shards", str(N), "--policy", "wait", "--window", w, "--post-key", "lowlat", "--dir", "en2zh"],
       "log": f"{OUT}/en2zh_S2dlm_d{w}/infer_s{k}.log"} for w in ("0.5", "1", "2") for k in range(N)]
for c in range(4):
    with open(J / f"p5_card{4 + c}.json", "x") as fh:
        json.dump(rs[6 * c: 6 * c + 6], fh, indent=1)
print(len(rs))
