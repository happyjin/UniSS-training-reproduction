"""Paper v7 (prereg_v7 §5): does the ranking on AliMeeting change without its worst machine-translated references?

The GEMBA-DA scores of reports/gstar_si_v7/qe_gemba.py (one per reference utterance) give a threshold at the lowest
10 %; every reference window that holds an utterance below it is left out, and the four AliMeeting systems are scored by
gstar_score on the remaining windows (new files under eval_outputs/.../gstar_v7_20261101/qe/rank/). Writes
generated/v7_qe_rank.json: the threshold, the share of windows kept, and the differences \\sys{} - same-data cascade and
\\sys{}-Simul - \\sys{} + LA2 on all windows and on the kept ones.

    python scripts/qe_rank_v7.py      (from the paper folder)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systems_v7 import ROOT, V7, hyp_dir, windows  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated"
PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"


def main():
    rows = []
    for f in sorted((V7 / "qe").glob("ali_ref_gemba*.jsonl")):
        if "smoke" not in f.name:
            rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    sc = {r["record_id"]: r["score"] for r in rows if r.get("score") is not None}
    if not sc:
        raise SystemExit("no QE scores yet")
    thr = float(np.percentile(list(sc.values()), 10))
    out = V7 / "qe" / "rank"
    out.mkdir(parents=True, exist_ok=True)
    W = [json.loads(l) for l in open(windows("ali", "O2"), encoding="utf-8")]
    keep = [w for w in W if all(sc.get(u["record_id"], 100.0) > thr for u in w["utts"] if not u["hesitation"])]
    wf = out / "windows_kept.jsonl"
    if not wf.exists():
        with open(wf, "x", encoding="utf-8") as fh:
            for w in keep:
                fh.write(json.dumps(w, ensure_ascii=False) + "\n")
    res = {"threshold_p10": round(thr, 1), "utterances_scored": len(sc), "windows": len(W), "windows_kept": len(keep)}
    cp = {}
    for n in ("O2", "C-ft2", "S2dl_d1", "O2_LA2_d1"):
        o = out / f"{n}.json"
        pk = "lowlat" if n in ("S2dl_d1", "O2_LA2_d1") else "offline"
        if not o.exists():
            subprocess.run(f"PYTHONPATH=. {PY} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_score --windows {wf} "
                           f"--hyp {hyp_dir('ali', n)}/hyp.jsonl --post-key {pk} --out {o} > {o}.log 2>&1",
                           shell=True, cwd=ROOT, check=True)
        cp[n] = (json.load(open(hyp_dir("ali", n) / "score.json"))["all"]["cpBLEU_session"],
                 json.load(open(o))["all"]["cpBLEU_session"])
    res["cas_all"] = round(cp["O2"][0] - cp["C-ft2"][0], 2)
    res["cas"] = round(cp["O2"][1] - cp["C-ft2"][1], 2)
    res["sim_all"] = round(cp["S2dl_d1"][0] - cp["O2_LA2_d1"][0], 2)
    res["sim"] = round(cp["S2dl_d1"][1] - cp["O2_LA2_d1"][1], 2)
    res["cp_kept"] = {k: v[1] for k, v in cp.items()}
    (GEN / "v7_qe_rank.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res)


if __name__ == "__main__":
    main()
