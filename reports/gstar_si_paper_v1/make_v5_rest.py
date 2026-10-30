"""Paper v5: finish a gstar_v5 system whose decode workers were stopped by PID (e.g. Zh->En S2-dl 3k at 0.5 s, whose
runaway generations grew the batch-16 workers past the memory limit on cards shared with a colleague). For every shard k of
the original N, the windows it has not written yet (by id) go to <system>/rest_s<k>.windows.jsonl, decoded into
<system>/hyp_s<k>r.jsonl with a smaller batch; the existing hyp_s<k>.jsonl files are kept as they are, and the scorer's
hyp_s*.jsonl glob picks up both. gstar_multi run files spread over the given cards, REST_WORKER_MIB (25 GB) per worker.

    python3 reports/gstar_si_paper_v1/make_v5_rest.py zh2en_S2dl3k_d0.5 7 0.5 0,3 [batch]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_v5_runs import CAP, OTHERS_MARGIN, OUT, S2DLZ, THINKER, WIN, foreign, spread, write  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

name, n, w, cards = sys.argv[1], int(sys.argv[2]), sys.argv[3], [int(x) for x in sys.argv[4].split(",")]
batch = sys.argv[5] if len(sys.argv) > 5 else "8"
d = Path(OUT) / name
allw = [json.loads(l) for l in open(WIN["zh2en"], encoding="utf-8")]
runs = []
for k in range(n):
    done = {json.loads(l)["id"] for l in open(d / f"hyp_s{k}.jsonl", encoding="utf-8")} if (d / f"hyp_s{k}.jsonl").exists() else set()
    rest = [x for x in allw[k::n] if x["id"] not in done]
    print(f"shard {k}: {len(done)} done, {len(rest)} left", file=sys.stderr)
    if not rest:
        continue
    wf = d / f"rest_s{k}.windows.jsonl"
    with open(wf, "x", encoding="utf-8") as fh:
        for x in rest:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    runs.append({"module": "gstar_si_stream_online",
                 "argv": ["--thinker", S2DLZ, "--windows", str(wf), "--out", str(d / f"hyp_s{k}r.jsonl"), "--shard", "0",
                          "--num-shards", "1", "--policy", "wait", "--window", w, "--post-key", "lowlat", "--dir", "zh2en",
                          "--batch", batch],
                 "log": str(d / f"infer_s{k}r.log")})
per = int(os.environ.get("REST_WORKER_MIB", "25000"))   # these workers grow far past the usual ~6 GB


def cap_of(c):
    used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                        "-i", str(c)]).decode().strip())
    return max(0, (CAP - (OTHERS_MARGIN if foreign(c) else 0) - used - THINKER) // per)


caps = {c: cap_of(c) for c in cards}
print("capacity", caps, file=sys.stderr)
write(f"rest_{name}", spread(runs, cards, caps))
