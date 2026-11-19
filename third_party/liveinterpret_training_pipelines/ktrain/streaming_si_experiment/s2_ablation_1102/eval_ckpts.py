#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 消融的中间 checkpoint 评测编排（在本机跑，可重复执行，只处理新出现的 checkpoint）。

流程（每个 checkpoint）：
  1. 列 S3 s2_ablation_1102/ckpt/<组>/checkpoint-N/（有 _UPLOAD_DONE 才算传完）
  2. 生成预签名下载脚本 -> 雅加达 si-baseline-api（FSx，无 AWS 凭证，故用预签名）
  3. 选显存最空的 GPU，用与 realeval / KIC 完全相同的命令推理：
       realeval_en_v1/input.jsonl（40 条 × zh/en = 80）+ testreal3/stage.jsonl（KIC 分段 10 条）
  4. 结果拉回 inference_outputs/s2_ablation_1102/<组>/<ckpt>/{realeval,kic}.jsonl，删 FSx 上的模型
  5. 汇总：python eval_ckpts.py --report  -> 轨迹表（每组 × 每步的 fail 数、KIC 覆盖）

用法：
  python eval_ckpts.py              # 评测所有新 checkpoint
  python eval_ckpts.py --report     # 只出轨迹表
"""
import json
import os
import re
import subprocess
import sys

import boto3
from botocore.config import Config

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
OUT = os.path.join(REPO, "inference_outputs/s2_ablation_1102")
B = "stevehguo-ap-southeast-3"
PFX = "speech_model_backup/stevehguo/s2_ablation_1102/ckpt/"
POD = ["kubectl", "-n", "speech-h200", "exec", "si-baseline-api", "--"]
POD_I = ["kubectl", "-n", "speech-h200", "exec", "-i", "si-baseline-api", "--"]   # 需要传 stdin 时用
E = "/fsx/stevehguo/eval"
GROUPS = ("G0", "G1", "G2", "PC")

s3 = boto3.client("s3", region_name="ap-southeast-3", endpoint_url="https://s3.ap-southeast-3.amazonaws.com",
                  config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))


def sh(cmd: str, timeout=3600) -> str:
    r = subprocess.run(POD + ["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return (r.stdout + r.stderr).replace("Defaulted container", "")


def list_ckpts():
    out = []
    for g in GROUPS:
        pg = s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=f"{PFX}{g}/", Delimiter="/")
        for p in pg:
            for cp in p.get("CommonPrefixes", []):
                name = cp["Prefix"].rstrip("/").rsplit("/", 1)[1]
                if not name.startswith("checkpoint-"):
                    continue
                try:
                    s3.head_object(Bucket=B, Key=f"{cp['Prefix']}_UPLOAD_DONE")
                except s3.exceptions.ClientError:
                    continue
                out.append((g, name, cp["Prefix"]))
    return sorted(out, key=lambda x: (x[0], int(x[1].split("-")[1])))


def eval_one(g, name, prefix):
    tag = f"{g}_{name.replace('checkpoint-', 'ck')}"
    dst = os.path.join(OUT, g, name)
    if os.path.exists(os.path.join(dst, "kic.jsonl")) and os.path.exists(os.path.join(dst, "realeval.jsonl")):
        return False
    os.makedirs(dst, exist_ok=True)
    keys = [o["Key"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=prefix)
            for o in p.get("Contents", []) if not o["Key"].endswith("_UPLOAD_DONE")]
    md = f"{E}/models/abl/{tag}"
    lines = ["set -u", f"rm -rf {md}"]
    for k in keys:
        rel = k[len(prefix):]
        u = s3.generate_presigned_url("get_object", Params={"Bucket": B, "Key": k}, ExpiresIn=7200)
        lines.append(f"mkdir -p '{os.path.dirname(f'{md}/{rel}')}' && curl -sS --fail --retry 3 -o '{md}/{rel}' '{u}' &")
    lines.append("wait")
    script = "\n".join(lines) + "\n"
    print(f"[eval] {tag}: 下载 {len(keys)} 个文件", flush=True)
    subprocess.run(POD_I + ["bash", "-c", f"mkdir -p {E}/models/abl {E}/out/abl && cat > /tmp/dl_{tag}.sh"],
                   input=script, text=True, check=True, capture_output=True)
    out = sh(f"bash /tmp/dl_{tag}.sh; du -sh {md}; ls {md}/*.safetensors | wc -l", timeout=1800).strip().splitlines()
    print("   ", out[-2:], flush=True)
    if out[-1].strip() != "3":
        print(f"[eval] {tag}: 模型分片不全，跳过", flush=True)
        return False
    run = f"""
E={E}; md={md}; o=$E/out/abl/{tag}
gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
export CUDA_VISIBLE_DEVICES=$gpu HF_HOME=$E/hf PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd /tmp
for job in realeval:$E/realeval_en_v1/input.jsonl kic:$E/testreal3/stage.jsonl; do
  n=${{job%%:*}}; inp=${{job#*:}}; rm -rf $o/$n
  PYTHONPATH=$E/code/custom_1031_langfix timeout 1800 /opt/venv/bin/python -m streaming_si.inference.infer_s2st \\
     --model_dir $md --input_jsonl $inp --out_dir $o/$n --text_only --device cuda:0 > $o.$n.log 2>&1
  echo "$n gpu$gpu rows=$(wc -l < $o/$n/results.jsonl 2>/dev/null)"
done
"""
    print(sh(f"mkdir -p {E}/out/abl; " + run, timeout=3600).strip(), flush=True)
    for n in ("realeval", "kic"):
        r = subprocess.run(POD + ["cat", f"{E}/out/abl/{tag}/{n}/results.jsonl"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            open(os.path.join(dst, f"{n}.jsonl"), "w").write(r.stdout)
    sh(f"rm -rf {md}")
    return True


# ---------------- 打分：与 realeval_en_v1/analyze.py 同口径 ----------------
def _wer_cov_en(ref, hyp, n_words):
    r = re.findall(r"[a-z0-9']+", re.sub(r"<[^>]+>|\[[^\]]+\]", " ", ref.lower()))
    h = re.findall(r"[a-z0-9']+", re.sub(r"<[^>]+>|\[[^\]]+\]", " ", hyp.lower()))
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, dp[j] = dp[j], cur
    w = dp[-1] / max(1, len(r))
    cov = len(re.findall(r"[A-Za-z0-9']+", hyp)) / max(1, n_words)
    return w, cov


def score_realeval(path, man):
    fz = fe = 0
    noisy_fail = 0
    for r in map(json.loads, open(path)):
        base, lang = r["id"].rsplit("__", 1)
        m, text = man[base], r.get("text") or ""
        if lang == "en":
            # 输出以中文为主 = 任务混淆（旧 S2 / PC 没见过同语言任务，会给出中文翻译），不算退化
            if len(re.findall(r"[\u4e00-\u9fff]", text)) > len(re.findall(r"[A-Za-z]", text)) / 3:
                continue
            w, cov = _wer_cov_en(m["ref_en"], text, m["n_words"])
            f = cov < 0.4 or (w >= 0.8 and cov < 0.7)
            fe += f
        else:
            draft = m.get("ref_zh_draft") or ""
            cov = len(re.sub(r"\s+", "", text)) / max(1, len(re.sub(r"\s+", "", draft)))
            f = cov < 0.35
            fz += f
        if f and m["source"] in ("kic", "urgent2024", "peoples"):
            noisy_fail += 1
    return fz, fe, noisy_fail


KIC_DRAFT_ZH_CHARS = None


def score_kic(path):
    d = {r["id"]: (r.get("text") or "") for r in map(json.loads, open(path))}
    full = d.get("full__zh", "")
    seg02 = d.get("seg0to2__en", "")
    w = re.findall(r"[A-Za-z']+", seg02.lower())
    uniq = len(set(w)) / max(1, len(w))
    # 词数很多但去重词比 < 0.4 = 重复循环（正常译文约 0.8），记 0 词以免被“词数多=正常”误读
    n_eff = 0 if (len(w) >= 20 and uniq < 0.4) else len(w)
    return len(re.sub(r"\s+", "", full)), n_eff, full[:40], ("[循环] " if n_eff == 0 and len(w) >= 20 else "") + seg02[:50]


def report():
    man = {r["id"]: r for r in map(json.loads, open(os.path.join(REPO, "inference_outputs/realeval_en_v1/manifest.jsonl")))}
    rows = []
    # 参照：旧 S2 / 新 S2 的既有结果
    ref = {"旧S2@25046": "out_s2_ck_old.jsonl", "新S2@56924": "out_s2_ck_v2.jsonl"}
    print(f"{'组':4s} {'step':>6s} | {'zh fail':>7s} {'en fail':>7s} {'带噪fail':>8s} | {'KIC full_zh 字数':>15s} {'seg0to2_en 词数':>15s} | KIC full__zh / seg0to2__en")
    for lab, f in ref.items():
        p = os.path.join(REPO, "inference_outputs/realeval_en_v1", f)
        fz, fe, nf = score_realeval(p, man)
        print(f"{lab:11s}| {fz:7d} {fe:7d} {nf:8d} |  （KIC 见主报告：旧 S2 全段正常 / 新 S2「是的，我知道。」）")
    for g in GROUPS:
        gd = os.path.join(OUT, g)
        if not os.path.isdir(gd):
            continue
        for name in sorted(os.listdir(gd), key=lambda x: int(x.split("-")[1])):
            p1, p2 = os.path.join(gd, name, "realeval.jsonl"), os.path.join(gd, name, "kic.jsonl")
            if not (os.path.exists(p1) and os.path.exists(p2)):
                continue
            fz, fe, nf = score_realeval(p1, man)
            nzh, nen, tzh, ten = score_kic(p2)
            step = int(name.split("-")[1])
            rows.append(dict(group=g, step=step, zh_fail=fz, en_fail=fe, noisy_fail=nf, kic_full_zh_chars=nzh,
                             kic_seg02_en_words=nen, kic_full_zh=tzh, kic_seg02_en=ten))
            print(f"{g:4s} {step:6d} | {fz:7d} {fe:7d} {nf:8d} | {nzh:15d} {nen:15d} | {tzh} / {ten}")
    json.dump(rows, open(os.path.join(OUT, "trajectory.json"), "w"), ensure_ascii=False, indent=1)
    print("\n判读：主指标是 zh fail（/40）与 KIC full__zh 字数（旧 S2 = 144 字，新 S2 = 7 字「是的，我知道。」）。"
          "\n      seg0to2__en 旧 S2 = 55 词，新 S2 = 7 词；重复循环（去重词比 < 0.4，如 'uh, uh, uh…'）记 0 词并标 [循环]。"
          "\n      en fail 已排除任务混淆（en→en 输出中文）。")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "--report" not in sys.argv:
        # 单实例：两个进程同时处理同一 checkpoint 时，一方下载前的 rm -rf 会删掉另一方正在加载的模型
        import fcntl
        _lock = open(os.path.join(OUT, ".eval.lock"), "w")
        try:
            fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("[eval] 已有一个 eval_ckpts.py 在跑，退出")
        for g, name, prefix in list_ckpts():
            eval_one(g, name, prefix)
    report()
