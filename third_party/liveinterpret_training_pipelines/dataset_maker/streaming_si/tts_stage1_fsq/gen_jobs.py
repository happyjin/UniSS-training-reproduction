#!/usr/bin/env python3
"""按数据源生成多份 Indexed Job yaml（改 name / SOURCE / LANG / 并行度）。

用法：python3 gen_jobs.py   # 产物 k8s_job_<src>_<lang>.yaml
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "k8s_stage1_fsq_job.yaml")

# (source, lang, parallelism, extra_env)
JOBS = [
    ("emilia", "zh", 16, {}),
    ("yodas", "jp", 8, {}),
    ("wenet", "zh", 8, {}),
    ("iscslp", "zh", 8, {"ISCSLP_MOVIES": "movie-zh1,movie-zh2,movie-zh3,movie-zh4"}),
    ("iscslp", "en", 8, {"ISCSLP_MOVIES": "movie-en1,movie-en2"}),
    ("ko", "ko", 8, {}),
    ("lance", "ja,ko", 8, {}),
    ("hifitts", "en", 8, {}),
]
NPROC_PER_GPU = 3

base = open(TEMPLATE).read()
for src, lang, par, extra in JOBS:
    s = base
    tag = f"{src}-{lang.replace(',', '-')}"
    s = s.replace("name: tts-stage1-fsq-emilia-zh", f"name: tts-stage1-fsq-{tag}")
    s = s.replace("    source: emilia", f"    source: {src}")
    s = re.sub(r"  parallelism: \d+", f"  parallelism: {par}", s)
    s = re.sub(r"  completions: \d+", f"  completions: {par}", s)
    s = re.sub(r'(value: ")emilia(" *#?.*)', rf'\g<1>{src}\g<2>', s, count=1)
    s = s.replace('value: "zh"              # 输出 language 字段（iscslp 必填；lance 可 "ja,ko"）',
                  f'value: "{lang}"')
    s = s.replace('value: "48"', f'value: "{par * NPROC_PER_GPU}"')
    for k, v in extra.items():
        s = s.replace('            # - name: TEXT_TOKENIZER',
                      f'            - name: {k}\n              value: "{v}"\n            # - name: TEXT_TOKENIZER')
    out = os.path.join(HERE, f"k8s_job_{tag}.yaml")
    open(out, "w").write(s)
    print("生成", out, f"parallelism={par} nshard={par * NPROC_PER_GPU}")
