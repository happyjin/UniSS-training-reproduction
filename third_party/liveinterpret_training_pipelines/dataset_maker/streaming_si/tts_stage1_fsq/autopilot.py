#!/usr/bin/env python3
"""stage1 全链路自动驾驶：造数 -> QA -> 过滤 -> 腾卡 -> 交棒本地提交训练。

在集群里以 Job 方式长跑（无人值守）：
  1. 等 8 个造数 Job 全部 Complete；Failed/Missing 自动删掉重建（模板在 ConfigMap），最多 3 次
  2. 等 QA Job（UTMOS 抽样 + 全量统计 + 报告）Complete，同上自动重建
  3. 建 filter Job（8 路并行）把异常样本过滤到 TTS_FSQ_FILTERED
  4. 腾卡：删掉 4 台目标节点上所有占 GPU 的 Pod 及其 owner（PyTorchJob/Job）
  5. 写 S3 marker（_QA/FILTERED_READY.json），本地 autopilot_local.sh 收到后提交训练
  6. 继续巡检 12h

所有 k8s 操作走 kubectl（python kubernetes client 对 `nvidia.com/gpu` 这类
扩展资源名的 key 有正则校验，会拒绝提交 Job，故直接用 kubectl）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import boto3

NS = "speech-h200"
CODE_DIR = "/opt/tts_stage1_fsq"
TOKEN = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

BUCKET = "stevehguo-ap-southeast-3"
REGION = "ap-southeast-3"
BASE = "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version"
TFS = f"{BASE}/TTS_FSQ"
FILTERED = f"{BASE}/TTS_FSQ_FILTERED"
QA_PREFIX = f"{TFS}/_QA"

PRODUCE_JOBS = ["emilia-zh", "yodas-jp", "wenet-zh", "iscslp-zh", "iscslp-en", "ko-ko",
                "lance-ja-ko", "hifitts-en"]
TARGET_NODES = ["hyperpod-i-03e7d1f6d95864e1a", "hyperpod-i-08e3257c716579618",
                "hyperpod-i-09a68c454800140d6", "hyperpod-i-0a3ebd369de798192"]
MAX_RECREATE = 3
MAX_RUNTIME_H = 12
# 可选任务：允许"没跑完也放行"，不阻塞后续 QA/过滤/训练
OPTIONAL_JOBS = {"tts-stage1-fsq-hifitts-en"}
OPTIONAL_WAIT_H = 0.5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def k(args, check=False):
    server = f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:{os.environ['KUBERNETES_SERVICE_PORT']}"
    cmd = ["kubectl", "--server", server, "--certificate-authority", CA,
           "--token", open(TOKEN).read().strip(), "-n", NS] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr[:300]}")
    return r


def job_state(name):
    r = k(["get", "job", name, "-o", "json"], check=False)
    if r.returncode != 0:
        return "Missing", 0, 0
    d = json.loads(r.stdout)
    st = d.get("status", {})
    comp = d.get("spec", {}).get("completions") or 1
    succ, fail = st.get("succeeded", 0) or 0, st.get("failed", 0) or 0
    if succ >= comp:
        return "Complete", succ, fail
    backoff = d.get("spec", {}).get("backoffLimit", 0) or 0
    if fail > backoff:
        return "Failed", succ, fail
    return "Running", succ, fail


def template_for(name):
    short = name
    for p in ("tts-stage1-fsq-", "tts-stage1-"):
        if short.startswith(p):
            short = short[len(p):]
            break
    for cand in (f"k8s_job_{short}.yaml", f"k8s_job_{name}.yaml", f"k8s_{short}_job.yaml"):
        p = os.path.join(CODE_DIR, cand)
        if os.path.exists(p):
            return p
    return None


def recreate(name):
    path = template_for(name)
    if not path:
        log(f"  找不到模板 {name}，跳过重建")
        return False
    k(["delete", "job", name, "--wait=false"], check=False)
    time.sleep(10)
    r = k(["apply", "-f", path], check=False)
    if r.returncode != 0:
        log(f"  重建失败 {name}: {r.stderr[:200]}")
        return False
    log(f"  已重建 Job {name}（模板 {os.path.basename(path)}）")
    return True


def wait_jobs(names, tag, soft_deadline_h=0, min_ratio=1.0):
    recreated = {n: 0 for n in names}
    t0 = time.time()
    while True:
        states = {n: job_state(n) for n in names}
        done = [n for n in states if states[n][0] == "Complete"]
        bad = [n for n in states if states[n][0] in ("Failed", "Missing")]
        log(f"[{tag}] 完成 {len(done)}/{len(names)} 异常 {len(bad)} | "
            + " ".join(f"{n.replace('tts-stage1-fsq-','')}:{states[n][1]}" for n in names))
        if not bad and len(done) == len(names):
            return True
        pending = [n for n in names if states[n][0] != "Complete"]
        # 只差可选任务（hifitts 章节下载极慢）时不必死等：等够 OPTIONAL_WAIT_H 就放行
        if pending and all(n in OPTIONAL_JOBS for n in pending) \
                and (time.time() - t0) / 3600 > OPTIONAL_WAIT_H:
            log(f"[{tag}] 只剩可选任务 {pending}，已等 {OPTIONAL_WAIT_H}h，放行（让它继续在后台跑）")
            return True
        if soft_deadline_h and (time.time() - t0) / 3600 > soft_deadline_h \
                and len(done) / len(names) >= min_ratio:
            log(f"[{tag}] 超过 {soft_deadline_h}h 软 deadline，放行")
            return True
        for n in bad:
            if recreated[n] < MAX_RECREATE:
                recreated[n] += 1
                log(f"[{tag}] {n} 异常，第 {recreated[n]} 次重建")
                recreate(n)
            else:
                log(f"[{tag}] {n} 放弃（超过 {MAX_RECREATE} 次）")
                return False
        time.sleep(120)


def free_gpus():
    r = k(["get", "pods", "-A", "-o", "json"], check=True)
    pods = json.loads(r.stdout)["items"]
    victims = []
    for p in pods:
        if p["spec"].get("nodeName") not in TARGET_NODES:
            continue
        if p["status"].get("phase") not in ("Running", "Pending"):
            continue
        g = 0
        for c in p["spec"]["containers"]:
            req = (c.get("resources") or {}).get("requests") or {}
            if req.get("nvidia.com/gpu"):
                g += int(req["nvidia.com/gpu"])
        if g:
            victims.append((p["metadata"]["namespace"], p["metadata"]["name"],
                            p["spec"]["nodeName"], g, p["metadata"].get("ownerReferences") or []))
    if not victims:
        log("[free] 目标节点上没有占卡任务")
        return
    for ns, name, node, g, owners in victims:
        log(f"[free] {node} {ns}/{name} 占 {g} 卡 -> 删除")
        for o in owners:
            kind, oname = o.get("kind"), o.get("name")
            try:
                if kind == "PyTorchJob":
                    k(["-n", ns, "delete", "pytorchjob", oname, "--wait=false"], check=False)
                elif kind == "Job":
                    k(["-n", ns, "delete", "job", oname, "--wait=false"], check=False)
            except Exception as e:
                log(f"      删 owner 失败 {kind}/{oname}: {str(e)[:100]}")
        k(["-n", ns, "delete", "pod", name, "--wait=false"], check=False)
    time.sleep(90)


def create_job(path, name):
    k(["delete", "job", name, "--wait=false"], check=False)
    time.sleep(10)
    r = k(["apply", "-f", path], check=False)
    if r.returncode != 0:
        log(f"创建失败 {name}: {r.stderr[:200]}")
        return False
    log(f"已创建 Job {name}")
    return True


def main() -> int:
    s3 = boto3.client("s3", region_name=REGION)
    t0 = time.time()

    log("=== 阶段1：等待造数 ===")
    ok = wait_jobs([f"tts-stage1-fsq-{s}" for s in PRODUCE_JOBS], "produce",
                   soft_deadline_h=6.0, min_ratio=0.75)

    log("=== 阶段2：QA 报告 ===")
    ok_qa = wait_jobs(["tts-stage1-fsq-qa"], "qa")

    log("=== 阶段3：过滤异常样本 ===")
    if create_job(os.path.join(CODE_DIR, "k8s_job_filter.yaml"), "tts-stage1-filter"):
        pass
    # 已有的 emilia_fsq_v3 英文（46,828h）一并并入训练集：同样过一遍过滤
    create_job(os.path.join(CODE_DIR, "k8s_job_filter_v3en.yaml"), "tts-stage1-filter-v3en")
    ok_filter = wait_jobs(["tts-stage1-filter", "tts-stage1-filter-v3en"], "filter",
                          soft_deadline_h=4.0, min_ratio=0.5)

    log("=== 阶段4：腾出 4 台目标节点的 GPU ===")
    free_gpus()

    stat = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "produce_ok": ok, "qa_ok": ok_qa, "filter_ok": ok_filter,
            "target_nodes": TARGET_NODES}
    try:
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{FILTERED}/_FILTER_STAT/")
        rows_in = rows_out = 0
        for o in r.get("Contents", []):
            d = json.loads(s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read())
            rows_in += d.get("rows_in", 0)
            rows_out += d.get("rows_out", 0)
        stat.update({"rows_in": rows_in, "rows_out": rows_out,
                     "drop_ratio": round(1 - rows_out / max(rows_in, 1), 4)})
    except Exception as e:
        log(f"统计失败: {str(e)[:120]}")
    s3.put_object(Bucket=BUCKET, Key=f"{QA_PREFIX}/FILTERED_READY.json",
                  Body=json.dumps(stat, indent=2).encode(), ContentType="application/json")
    log(f"[marker] 已写 FILTERED_READY.json: {stat}")

    log("=== 阶段5：巡检 ===")
    while (time.time() - t0) / 3600 < MAX_RUNTIME_H:
        try:
            r = k(["get", "jobs", "-l", "app=tts-stage1-fsq", "--no-headers"], check=False)
            log("巡检: " + " | ".join(l.split()[:3] for l in r.stdout.strip().splitlines()[:9]))
        except Exception as e:
            log(f"巡检异常: {str(e)[:120]}")
        time.sleep(600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
