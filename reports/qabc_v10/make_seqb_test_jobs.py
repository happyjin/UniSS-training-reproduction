"""gpuq9 jobs for the test-set run of the decision rule that passed the dev gate (prereg_v5 section 4): seq-B on the s0
posterior, RealSI (15 windows, both directions, text; p5 exists in realsi_rules) and ACL60/60 (seq-B and the p5 replay,
then StreamLAAL), into eval_outputs/uniss_qwen3_omni_v1/qabc_v10/{realsi_rules,acl_rules}.

    python3 reports/qabc_v10/make_seqb_test_jobs.py <out.json>
"""
import json
import sys

sys.path.insert(0, "reports/qabc_v10")
from make_v10_jobs import M, O10, QR, SCORE, W15  # noqa: E402

jobs = [{"id": f"rules_realsi_seqb_{d}", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
         "cmd": f"{QR} --direction {d} --windows {W15} --rules seqb --thinker {M['s0']} --thinker-tag s0 --b0 1 "
                f"--rate {r} --out {O10}/realsi_rules"} for d, r in (("en2zh", "0.191"), ("zh2en", "0.339"))]
jobs.append({"id": "rules_acl_seqb_p5", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
             "cmd": f"{QR} --direction en2zh --items-file data/acl6060_dev_v1/items.jsonl --windows {W15} --rules p5,seqb "
                    f"--thinker {M['s0']} --thinker-tag s0 --b0 1 --rate 0.191 --out {O10}/acl_rules && {SCORE} --blocks "
                    f"'{O10}/acl_rules/blocks_en2zh_*.jsonl' --dump {O10}/acl_rules/seg > {O10}/acl_rules/streamlaal.jsonl"})
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")
