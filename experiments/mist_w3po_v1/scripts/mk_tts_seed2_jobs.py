"""Seed-2 speech jobs (LOG 2026-11-09 03:09 UTC, 2-seed speech gate) for evaluations that are still queued: the
36 pipelined speech jobs of EA 50 (076_*) re-targeted at <name>_tts2 with talker --seed 20261109; the sprep job waits
for <name>__score (text score done -> hyp.jsonl / score_text.json exist) and copies those two files (cp -n) into
decode/<name>_tts2 first. Refuses existing job files / output dirs. A gate job runs gate_2seed when both draws of the
candidate and of the refs exist.

    python mk_tts_seed2_jobs.py <prefix e.g. 088> <name> [<name> ...] [--dry]
"""
import json
import os
import sys

Q = "logs/mist_w3po_v1/queue_v2/jobs"
D = "eval_outputs/mist_w3po_v1/decode"
SRC = "dev_pilot50_eagrpoctrl631sc_ctx0_guard_s8"
PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
REFS = ("dev_v8_ctx0_guard_s8", "dev_v8_ctx2_guard_s8")


def main():
    args = [a for a in sys.argv[1:] if a != "--dry"]
    dry = "--dry" in sys.argv
    prefix, names = args[0], args[1:]
    tmpl = [json.load(open(f"{Q}/076_{SRC}_{i:02d}.json")) for i in range(45)]
    tmpl = [t for t in tmpl if t["id"].split("__")[-1] not in [f"dec{k}" for k in range(8)] + ["score"]]
    assert len(tmpl) == 36 and tmpl[0]["id"].endswith("__sprep"), [t["id"] for t in tmpl[:2]]
    for name in names:
        new = name + "_tts2"
        assert not os.path.exists(f"{D}/{new}") and not os.path.exists(f"eval_outputs/mist_w3po_v1/speech/{new}"), new
        jobs = []
        for t in tmpl:
            j = json.loads(json.dumps(t).replace(SRC, new))
            if j["id"].endswith("__sprep"):
                j["deps"] = [f"{name}__score"]
                k = j["cmd"].index("mkdir -p ")
                j["cmd"] = (j["cmd"][:k] + f"mkdir -p {D}/{new} && cp -n {D}/{name}/hyp.jsonl {D}/{name}/score_text.json "
                            f"{D}/{new}/ && " + j["cmd"][k:])
            if "__talk" in j["id"]:
                assert "--num-shards 8 >" in j["cmd"]
                j["cmd"] = j["cmd"].replace("--num-shards 8 >", "--num-shards 8 --seed 20261109 >")
            jobs.append(j)
        refs = " ".join(f"--ref {D}/{r} {D}/{r}_tts2" for r in REFS)
        jobs.append({"id": f"gate2_{name}", "mem_gb": 0, "deps": [f"{new}__sscore", f"{name}__sscore"] + [f"{r}_tts2__sscore" for r in REFS], "retries": 2,
                     "cmd": f"export PYTHONPATH=.:third_party/Megatron-LM; {PY} -m experiments.mist_w3po_v1.evaluation."
                            f"gate_2seed --cand {D}/{name} {D}/{new} {refs} --out {D}/{name}/gate_2seed.json "
                            f"> {D}/{name}/gate_2seed.log 2>&1"})
        for i, j in enumerate(jobs):
            out = f"{Q}/{prefix}_tts2_{name}_{i:02d}.json"
            assert not os.path.exists(out), out
            if dry:
                if i in (0, 1, 36):
                    print(j["id"], j["deps"], j["cmd"][-260:])
            else:
                with open(out, "x") as fh:
                    json.dump(j, fh)
        print(name, len(jobs), "jobs" + (" (dry)" if dry else ""))


if __name__ == "__main__":
    main()
