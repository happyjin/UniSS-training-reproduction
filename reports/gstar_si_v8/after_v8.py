"""Paper v8: merge the shards of finished whole-meeting runs (eval_outputs/.../gstar_v8_20261103/runs/on_<set>_<sys>/
hyp_s<k>.jsonl) into hyp.jsonl once every window of the set is present (an existing hyp.jsonl is kept), then re-score
everything (rescore_existing.py) and rebuild the paper's v8 tables (paper/gstar_si_v8/scripts/make_tables_v8.py).

    /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python reports/gstar_si_v8/after_v8.py
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs"
ON = REPO / "data/online_v6_20261031"
PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"


def main():
    for d in sorted(RUNS.glob("on_*")):
        if (d / "hyp.jsonl").exists():
            continue
        st = d.name.split("_")[1]
        Wst = [json.loads(ln) for ln in open(ON / f"windows_test_online_{st}.jsonl")]
        if "lag2" in d.name:   # prereg_v8 amendment 2: fixed lag on the v7 meeting subset only
            sub = set(json.load(open(ON / "subset_meetings.json"))["ali" if st == "ali" else "ami"])
            Wst = [w for w in Wst if w["meeting"] in sub]
        want = {w["id"] for w in Wst}
        got = {}
        for f in sorted(d.glob("hyp_*.jsonl")):   # shards (hyp_s*), resumes (hyp_r*, hyp_s1r), helpers (hyp_h*)
            if f.name == "hyp.jsonl":
                continue
            for ln in open(f, encoding="utf-8"):
                try:
                    h = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                got.setdefault(h["id"], ln)
        if set(got) >= want:
            with open(d / "hyp.jsonl", "x", encoding="utf-8") as fh:
                fh.writelines(got[i] if got[i].endswith("\n") else got[i] + "\n" for i in sorted(want))
            print(f"{d.name}: merged {len(want)} windows", flush=True)
        else:
            print(f"{d.name}: {len(set(got) & want)}/{len(want)} windows", flush=True)
    subprocess.run([PY, str(REPO / "reports/gstar_si_v8/rescore_existing.py"), "--jobs", "24"], check=False)
    subprocess.run([PY, "scripts/make_tables_v8.py"], cwd=REPO / "paper/gstar_si_v8", check=False,
                   stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    sys.exit(main())
