"""AliMeeting chain of v3_jobs.json again: ali_mt failed at engine start (FileNotFoundError: 'ninja' -- its command lacks
the vLLM venv on PATH, so the FP8 MoE kernels could not build) and the 66 jobs depending on it were skipped. Same commands,
ali_mt with the venv's bin first on PATH; the jobs are taken from the queue log, so nothing that ran is repeated.

    python3 reports/gstar_si_paper_v1/make_v3_ali_fix_jobs.py logs/v3_eval5_run.log reports/gstar_si_paper_v1/jobs/v3_ali_fix_jobs.json
"""
import json, re, sys

VENV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin"
log = open(sys.argv[1]).read()
redo = {"ali_mt"} | set(re.findall(r"skip (\S+) \(dependency failed\)", log))
assert re.search(r" fail ali_mt$", log, re.M), "ali_mt did not fail in this log"
jobs = [j for j in json.load(open("reports/gstar_si_paper_v1/jobs/v3_jobs.json")) if j["id"] in redo]
assert len(jobs) == len(redo), sorted(redo - {j["id"] for j in jobs})
for j in jobs:
    j["deps"] = [d for d in j["deps"] if d in redo]   # ali_nemo_offline / ali_nemo_lowlat are done
    if j["id"] == "ali_mt":
        assert "PATH=" + VENV not in j["cmd"]
        j["cmd"] = f"PATH={VENV}:$PATH " + j["cmd"]
with open(sys.argv[2], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")
