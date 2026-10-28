"""RealSI post-processing of the bayesrisk v2 systems (after jobs/realsi_jobs.json; --round r2: the second zh2en test
after jobs/realsi_r2_jobs.json, written to final_r2_zh2en.json / matched_r2.md next to the first round's systems): per system the continuous-timeline
abrupt analysis (reports/uniss_m2_fluency_results_v2/abrupt/abrupt.py, mode s1b1.2q) and UTMOS; then the comparison
with the whole v3 curve (final_<d>.json), a paired bootstrap of every system against the nearest-LAAL v3 point
(pair_<system>.json), and the matched-LAAL table (matched.md). v3's numbers come from the same runs and scripts as the
v1 report (its UTMOS file and abrupt analyses). Outputs that exist are kept.
    python3 post.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
HERE = ROOT / "reports/uniss_m2_bayesrisk_v2"
V1 = ROOT / "reports/uniss_m2_bayesrisk_v1"
E = "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500"
TB = "eval_outputs/uniss_qwen3_omni_v1/bayesrisk_v2/realsi"
EV = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
CMP = f"env PYTHONPATH=.:.tmp/pylib_jieba {EV} -m experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_compare"

V3 = {"en2zh": {"v3b0.5_en2zh": "flu7-sq_m0.5", "v3_en2zh": "flu7-sq-b1_m0.5", "v3b1.5_en2zh": "bay-v3-b1.5_m0.5",
                "v3b2_en2zh": "bay-v3-b2_m0.5", "v3b2.5_en2zh": "bay-v3-b2.5_m0.5", "v3b3_en2zh": "bay-v3-b3_m0.5",
                "v3b3.5_en2zh": "bay-v3-b3.5_m0.5"},
      "zh2en": {"v3_zh2en": "flu7-sq-zh_m1", "v3bm0.5_zh2en": "flu7-sq-bm0p5-zh_m1", "v3b0_zh2en": "bay-v3-zh-b0_m1",
                "v3b0.5_zh2en": "bay-v3-zh-b0.5_m1", "v3b1_zh2en": "bay-v3-zh-b1_m1", "v3b1.5_zh2en": "bay-v3-zh-b1.5_m1"}}
# name, system dir, direction, window, blocks suffix (jobs/realsi_jobs.json); "secondary" = online posterior
SYS = [("br2P-a0.75-bm0.5_zh2en", "br2P-a0.75-zh-bm0.5_m1", "zh2en", "1", "_brop1b-0.5mposterior_dev_v3pua0.75nnext_it1_zh2en_cln"),
       ("br2P-a0.75-b0_zh2en", "br2P-a0.75-zh-b0_m1", "zh2en", "1", "_brop1b0mposterior_dev_v3pua0.75nnext_it1_zh2en_cln"),
       ("br2Ph4t-b0_zh2en", "br2Ph4t-zh-b0_m1", "zh2en", "1", "_brop1b0mposterior_dev_v3puh4t_cln"),
       ("br2PO-a0.75-bm0.5_zh2en", "br2PO-a0.75-zh-bm0.5_m1", "zh2en", "1", "_brop1b-0.5mposterior_dev_v3po0ua0.75nnext_it1_zh2en_cln"),
       ("br2PO-a0.75-b0_zh2en", "br2PO-a0.75-zh-b0_m1", "zh2en", "1", "_brop1b0mposterior_dev_v3po0ua0.75nnext_it1_zh2en_cln"),
       ("br2Ah3t-b1_en2zh", "br2Ah3t-b1_m0.5", "en2zh", "0.5", "_brop1b1puh3t_cln"),
       ("br2Ah3t-b1.5_en2zh", "br2Ah3t-b1.5_m0.5", "en2zh", "0.5", "_brop1b1.5puh3t_cln"),
       ("br2Ah3t-b2.5_en2zh", "br2Ah3t-b2.5_m0.5", "en2zh", "0.5", "_brop1b2.5puh3t_cln")]


def sh(cmd):
    print("+", cmd[:200], flush=True)
    subprocess.run(["bash", "-c", cmd], cwd=ROOT, check=True)


SYS_R2 = [("br3Pd-a1-b0_zh2en", "br3Pd-a1-zh-b0_m1", "zh2en", "1", "_brop1b0mposterior_dev_v3puda1nnext_it3_zh2en_cln"),
          ("br3Pd-a1-b0.5_zh2en", "br3Pd-a1-zh-b0.5_m1", "zh2en", "1", "_brop1b0.5mposterior_dev_v3puda1nnext_it3_zh2en_cln")]


def main():
    global SYS
    r2 = sys.argv[1:] == ["--round", "r2"]
    tag = "_r2" if r2 else ""
    if r2:
        SYS = [s for s in SYS if s[2] == "zh2en"] + SYS_R2
    (HERE / "abrupt").mkdir(exist_ok=True)
    (HERE / "utmos").mkdir(exist_ok=True)
    procs = []
    for name, sd, d, w, suf in SYS:
        run = f"{E}/system_{sd}/realsi_{d}"
        ab = HERE / f"abrupt/{name}_s1b1.2q.txt"
        if not ab.exists():
            procs.append(subprocess.Popen(["bash", "-c", f"PYTHONPATH=. {EV} reports/uniss_m2_fluency_results_v2/abrupt/abrupt.py "
                                           f"'{name}={run}=realsi:{d}={TB}/blocks_{d}_w{w}{suf}.jsonl=s1b1.2q' > {ab}.part 2>&1 "
                                           f"&& command mv -f {ab}.part {ab}"], cwd=ROOT))
        ut = HERE / f"utmos/{name}.jsonl"
        if not ut.exists():
            procs.append(subprocess.Popen(["bash", "-c", f"UTMOS_THREADS=6 {EV} reports/uniss_m2_fluency_analysis_v1/scripts/utmos_runs.py "
                                           f"{ut} '{name}={run}' > {HERE}/utmos/{name}.log 2>&1"], cwd=ROOT))
    for p in procs:
        assert p.wait() == 0, p.args
    ut_args = f"--utmos {V1}/utmos_all.jsonl " + " ".join(f"--utmos {HERE}/utmos/{n}.jsonl" for n, *_ in SYS)
    for d in (("zh2en",) if r2 else ("en2zh", "zh2en")):
        out = HERE / f"final{tag}_{d}.json"
        runs = [f"--run {k}={E}/system_{v}/realsi_{d}" for k, v in V3[d].items()]
        runs += [f"--run {n}={E}/system_{sd}/realsi_{dd}" for n, sd, dd, *_ in SYS if dd == d]
        if not out.exists():
            sh(f"{CMP} --base {next(iter(V3[d])) if d == 'zh2en' else 'v3_en2zh'} {ut_args} {' '.join(runs)} --out {out} > {out.with_suffix('.log')} 2>&1")
        res = json.load(open(out))
        for n, sd, dd, *_ in SYS:
            if dd != d:
                continue
            near = min(V3[d], key=lambda k: abs(res[k]["laal_s"] - res[n]["laal_s"]))
            po = HERE / f"pair_{n}.json"
            if not po.exists():
                sh(f"{CMP} --base {near} {ut_args} --run {near}={E}/system_{V3[d][near]}/realsi_{d} "
                   f"--run {n}={E}/system_{sd}/realsi_{d} --out {po} > {po.with_suffix('.log')} 2>&1")
    if r2:
        sh(f"{EV} {HERE}/matched.py {HERE}/final_r2_zh2en.json > {HERE}/matched_r2.md")
    else:
        sh(f"{EV} {HERE}/matched.py {HERE}/final_en2zh.json {HERE}/final_zh2en.json > {HERE}/matched.md")
    print("POST_DONE", flush=True)


if __name__ == "__main__":
    main()
