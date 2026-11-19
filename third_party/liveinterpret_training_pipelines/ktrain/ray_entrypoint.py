#!/usr/bin/env python3
# =============================================================================
# ray_entrypoint.py — ktrain `framework: ray`（KubeRay RayJob）的 job 入口。
#
# 为什么需要它（两个 KubeRay 固有行为，都有实证）：
#   1. RayJob 的 entrypoint 经 submitter `ray job submit` 提交后，job driver
#      固定落在 0-GPU 的 head pod（headGroupSpec.rayStartParams num-gpus='0'，
#      ktrainjob schema 不可改）。ms-swift 的 driver 进程需要 GPU：
#      TrainingArguments(bf16=True) 在 __post_init__ 调
#      torch.cuda.is_bf16_supported()，无 GPU 直接 raise（与 anyscale
#      train/job.yaml:124-132 注释记录的问题相同）。
#   2. 训练所需的节点侧文件副作用（/curated_audio、/unist 音频软链、EFA
#      NCCL 插件软链）必须在**每个** GPU worker 上都做一遍——launch_train.sh
#      只会在 driver 所在的 worker 上执行。
#
# 因此本脚本只做三件事（参考 anyscale/train/dispatch_train.py 的钉 driver
# 设计，按 KubeRay 裁剪，不做插件 staging / S3 manifest 拉取）：
#   a. 等集群 GPU 凑齐（KubeRay worker 加入是异步的）；
#   b. 在每个 GPU worker 上跑一次 launch_train.sh 的 NODE_PREP_ONLY 模式
#      （只落文件副作用，不启动训练）；
#   c. 按活集群派生 RAY_DEVICE_GROUPS，把真正的训练 driver（launch_train.sh）
#      用 NodeAffinity 钉到第一个 GPU worker 上执行，透传退出码。
# =============================================================================
import json
import os
import subprocess
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

LAUNCHER = os.environ.get("LAUNCHER_PATH", "/opt/launcher/launch_train.sh")
EXPECTED_GPUS = int(os.environ.get("RAY_EXPECTED_GPUS", "16"))
WAIT_TIMEOUT = int(os.environ.get("RAY_CLUSTER_WAIT_TIMEOUT", "1800"))


def log(msg: str) -> None:
    print(f"[ray-entrypoint] {msg}", flush=True)


@ray.remote(num_cpus=1, num_gpus=0)
def _node_prep(env: dict) -> int:
    """在指定节点跑 launch_train.sh 的 NODE_PREP_ONLY 模式（软链/EFA 插件软链）。"""
    prep_env = dict(env)
    prep_env["NODE_PREP_ONLY"] = "true"
    return subprocess.call(["bash", LAUNCHER], env=prep_env)


@ray.remote(num_cpus=2, num_gpus=0)
def _driver(env: dict) -> int:
    """训练 driver：完整跑 launch_train.sh -> train script -> swift sft --use_ray。"""
    return subprocess.call(["bash", LAUNCHER], env=env)


def main() -> None:
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

    # ---- a. 等 GPU worker 加入 ----
    deadline = time.time() + WAIT_TIMEOUT
    gpu_nodes = []
    total = 0
    while True:
        gpu_nodes = [n for n in ray.nodes()
                     if n.get("Alive") and int(n.get("Resources", {}).get("GPU", 0)) > 0]
        total = sum(int(n["Resources"]["GPU"]) for n in gpu_nodes)
        if total >= EXPECTED_GPUS:
            break
        if time.time() > deadline:
            log(f"FATAL: only {total}/{EXPECTED_GPUS} GPUs after {WAIT_TIMEOUT}s")
            sys.exit(1)
        log(f"waiting GPUs {total}/{EXPECTED_GPUS} (gpu_nodes={len(gpu_nodes)})")
        time.sleep(15)
    per_node = total // len(gpu_nodes)
    device_groups = {
        "nproc_per_node": per_node,
        "default": {"device": "GPU", "ranks": list(range(total)), "workers": ["default"]},
    }
    log(f"cluster ready: {total} GPUs on {len(gpu_nodes)} workers, "
        f"{per_node}/node; RAY_DEVICE_GROUPS={json.dumps(device_groups, separators=(',', ':'))}")

    env = dict(os.environ)
    env["RAY_DEVICE_GROUPS"] = json.dumps(device_groups, separators=(",", ":"))
    env["RAY_WORLD_SIZE"] = str(total)
    env["RAY_NPROC_PER_NODE"] = str(per_node)
    env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = \
        env.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    env["RAY_DEDUP_LOGS"] = env.get("RAY_DEDUP_LOGS", "0")

    def _on_node(node):
        return NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)

    # ---- b. 每个 GPU worker 做节点侧文件副作用（幂等）----
    prep_refs = [
        _node_prep.options(scheduling_strategy=_on_node(n),
                           runtime_env={"env_vars": env}).remote(env)
        for n in gpu_nodes
    ]
    prep_rcs = ray.get(prep_refs)
    if any(rc != 0 for rc in prep_rcs):
        log(f"FATAL: node prep failed, rcs={prep_rcs}")
        sys.exit(1)
    log(f"node prep done on {len(gpu_nodes)} workers")

    # ---- c. driver 钉到第一个 GPU worker ----
    target = gpu_nodes[0]
    log(f"pinning driver to node {target.get('NodeManagerAddress')} "
        f"(node_id={target['NodeID'][:12]}...)")
    handle = _driver.options(scheduling_strategy=_on_node(target),
                             runtime_env={"env_vars": env}).remote(env)
    rc = ray.get(handle)
    log(f"driver exited rc={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
