"""Paper v4 (no Bayesian commit rule; the simultaneous system is the deadline-label model S2-dl): gstar_multi run files
for cards 4-7, one file per card and phase (four runs per card, one Thinker copy each).

  p1  En->Zh S2-dl at read intervals 0.5 and 2 s (1 s exists in gstar_v3_20261029)
  p3  Zh->En S2-dl (trained by run_v4.sh on build_ami_gstar_stream_rows_zh2en rows): AMI at 1 s and AliMeeting at 1 s,
  p4  then AMI at 0.5 and 2 s
Outputs go to eval_outputs/uniss_qwen3_omni_v1/gstar_v4_20261030/<system>/hyp_s<k>.jsonl (8 shards per system).

    python3 reports/gstar_si_paper_v1/make_v4_runs.py
"""
import json
from pathlib import Path

EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v4_20261030"
D = "data/ami_multispk_si_v1"
S2DL = f"{EV}/thinker_gstar_s2dl_20261029b/thinker_merged"
S2DLZ = f"{EV}/thinker_gstar_s2dl_zh2en_20261030/thinker_merged"
WIN = {"en2zh": f"{D}/e2e_gstar_stream_v1/windows_test_ref.jsonl", "zh2en": f"{D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
       "ali": "data/alimeeting_si_v1/windows_test_ref.jsonl"}
J = Path("reports/gstar_si_paper_v1/jobs/v4_runs")
N = 8


def runs(name, thinker, win, d, w):
    return [{"module": "gstar_si_stream_online",
             "argv": ["--thinker", thinker, "--windows", win, "--out", f"{OUT}/{name}/hyp_s{k}.jsonl", "--shard", str(k),
                      "--num-shards", str(N), "--policy", "wait", "--window", w, "--post-key", "lowlat", "--dir", d],
             "log": f"{OUT}/{name}/infer_s{k}.log"} for k in range(N)]


def write(phase, systems):
    rs = [r for s in systems for r in runs(*s)]
    for c in range(4):   # cards 4..7, four runs each
        with open(J / f"{phase}_card{4 + c}.json", "x") as fh:
            json.dump(rs[4 * c: 4 * c + 4], fh, indent=1)


write("p1", [("en2zh_S2dl_d0.5", S2DL, WIN["en2zh"], "en2zh", "0.5"), ("en2zh_S2dl_d2", S2DL, WIN["en2zh"], "en2zh", "2")])
write("p3", [("zh2en_S2dl_d1", S2DLZ, WIN["zh2en"], "zh2en", "1"), ("ali_S2dl_d1", S2DLZ, WIN["ali"], "zh2en", "1")])
write("p4", [("zh2en_S2dl_d0.5", S2DLZ, WIN["zh2en"], "zh2en", "0.5"), ("zh2en_S2dl_d2", S2DLZ, WIN["zh2en"], "zh2en", "2")])
print(sorted(p.name for p in J.iterdir()))
