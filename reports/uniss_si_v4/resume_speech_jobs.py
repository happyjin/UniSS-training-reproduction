"""Resume RealSI speech specs (make_speech_jobs.py) from what is on disk, after the S1 move bug and the ASR memory
overrun: for each planned run, nothing if transcribed; ASR only if the talker finished (its results.jsonl or two part
files exist); move back and ASR if the run sits where S1 moved it; the whole job otherwise (a failed talker's empty
folder is moved aside to <run>_failed<k>, never deleted).

    python3 reports/uniss_si_v4/resume_speech_jobs.py <out_jobs.json> <spec.json> ...
"""
import json, os, subprocess, sys

SP = "eval_outputs/uniss_qwen3_omni_v1/si_v4/speech"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"


def asr(folder):
    return (f"{PYE} reports/uniss_si_v4/speech_only.py {folder} > /dev/null && CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} "
            f"-m evaluation.asr_transcribe --input {folder}/results_speechonly.jsonl --output {folder}/asr_speech.jsonl "
            f"--device cuda:0 --batch-size 4 && [ -f {folder}/asr_speech.jsonl ]")


def n_lines(f):
    return sum(1 for _ in open(f, encoding="utf-8")) if os.path.exists(f) else 0


jobs, todo, counts = [], [], {"done": 0, "asr": 0, "move+asr": 0, "full": 0}
for spec_file in sys.argv[2:]:
    for s in json.load(open(spec_file)):
        d = s["dir"]
        for seed in s["seeds"]:
            name = s["name"] + (f"-seed{seed}" if seed else "")
            sysd = f"{SP}/system_{name}_m{s['window']:g}"
            home, moved = f"{sysd}/_{d}/realsi_{d}", f"{sysd}/realsi_{d}"
            jid = f"{name}_m{s['window']:g}_{d}"
            talked = os.path.exists(f"{home}/results.jsonl") or len([f for f in os.listdir(home) if f.startswith("part_")]) == 2 \
                if os.path.isdir(home) else False
            if os.path.isdir(moved):
                counts["move+asr"] += 1
                partial = f"{moved}/asr_speech.jsonl"
                aside = f"&& command mv {partial} {partial}.partial " if os.path.exists(partial) else ""
                jobs.append({"id": f"fix_{jid}", "gpus": 1, "vllm": False, "mem": 16000, "deps": [],
                             "cmd": f"command mv {moved} {home} {aside.replace(moved, home)}&& " + asr(home)})
            elif talked and n_lines(f"{home}/asr_speech.jsonl") >= max(n_lines(f"{home}/results.jsonl"), 1):
                counts["done"] += 1
            elif talked:
                counts["asr"] += 1
                partial = f"{home}/asr_speech.jsonl"
                aside = f"command mv {partial} {partial}.partial && " if os.path.exists(partial) else ""
                jobs.append({"id": f"asr_{jid}", "gpus": 1, "vllm": False, "mem": 16000, "deps": [], "cmd": aside + asr(home)})
            else:
                counts["full"] += 1
                if os.path.exists(f"{sysd}/_{d}"):
                    k = 1
                    while os.path.exists(f"{sysd}/_{d}_failed{k}"):
                        k += 1
                    subprocess.run(["mv", "-n", f"{sysd}/_{d}", f"{sysd}/_{d}_failed{k}"], check=True)
                todo.append({**s, "seeds": [seed]})
if todo:
    spec_out, jobs_out = sys.argv[1].replace(".json", "_todo_spec.json"), sys.argv[1].replace(".json", "_todo_jobs.json")
    with open(spec_out, "x") as fh:
        json.dump(todo, fh, indent=1)
    subprocess.run(["python3", "reports/uniss_si_v4/make_speech_jobs.py", jobs_out, spec_out], check=True)
    jobs += json.load(open(jobs_out))
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(counts, len(jobs), "jobs")
