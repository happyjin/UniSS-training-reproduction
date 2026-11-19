#!/usr/bin/env python3
"""从训练 PyTorchJob yaml 派生「工件同步」Job（batch/v1 Job，非 PyTorchJob）。

训练 yaml（pytorch_job/pytorchjob.yaml）是任务配置的唯一真源，本脚本用
**结构化 YAML 解析**（不是正则）读取它，派生出 4 个串行执行的同步任务
（model / tokenizer / manifest / audio）。

=============================================================================
关键设计 —— 每一条都对应一个真实踩过的坑，改动前请先读懂原因
=============================================================================

1) **为什么用 batch/v1 Job + completions=N，而不是 PyTorchJob**
   NVMe 是 hostPath，每个节点各存一份，训练在全部 N 个白名单节点上跑，
   所以工件必须在**每个节点**就绪。
   PyTorchJob 的致命问题：Master 成功时 operator 即判定整体 Succeeded，
   并按 cleanPodPolicy 清掉仍在同步的 Worker Pod —— 会出现「报成功但部分
   节点没同步完」，训练一起步就挂。这正是本次要根除的 bug。
   batch/v1 Job 的 `completions=N` 语义是「N 个 Pod 全部成功才 Complete」，
   "全部完成"是硬条件，无法被提前判成功。

2) **为什么靠 podAntiAffinity 而不是逐节点生成 Job**
   nodeAffinity 沿用训练 Master 的白名单（N 个 hostname）+
   podAntiAffinity(required, topologyKey=kubernetes.io/hostname, 匹配本 Job
   自身 Pod) => N 个 Pod 必须落在 N 个不同节点上；白名单恰好 N 个节点，
   于是「全覆盖、一节点一个」由调度器强制保证，且只需 1 个 Job 对象。
   前提是断言 N == len(白名单)（见 A3），不等就 FATAL。
   注：曾怀疑「Job 一次性创建 N 个 Pod 时 podAntiAffinity 在并发调度下会漏
   节点」，经集群实测（含 32C/32Gi 资源量的受控 Job）证实 podAntiAffinity
   正常生效——节点不足时是 Pending 而非违反反亲和落到他处。漏同步的真凶是
   「目标节点当时被训练 Job 占用（Insufficient cpu）导致 Pod Pending」，而非
   反亲和失效。故保留 podAntiAffinity 方案。

3) **Pod template = 深拷贝训练 Master.template 后做减法**
   旧实现用「白名单摘取字段手工重建」，结果漏掉了
   `sidecar.istio.io/inject: "false"` 注解，Pod 被注入 istio sidecar 后
   Job 永远不 Complete（sidecar 不退出）。深拷贝 + 显式减法可以保证
   「训练 yaml 里新增的调度相关字段自动被继承」，不会再漏。
   派生完成后还会**断言**该注解仍在（见 assert_pod_sanity）。

4) **绝不能在容器里用 nproc 决定并发**
   容器 CPU 限额由 cgroup cfs quota 实现，不改 /proc/cpuinfo，
   nproc 在 P5E 上读到 192，按它并发会 CPU throttling + 进程堆积。
   并发度由本脚本按 Job 的 limits.cpu 显式注入 SYNC_PAR。

5) **同步与就绪判定完全交给 `aws s3 sync` 自身**
   `aws s3 sync` 原生按「相对 key 名 + 文件大小」增量比对（--size-only），
   本地缺失或大小不符即重新下载覆盖，已一致则跳过。同一组文件只需执行
   一次 `aws s3 sync --size-only`，其返回码即代表成败。
   不做任何额外的复核 / 检查机制，不写就绪标记文件，不遍历本地磁盘计数，
   不做 dry-run —— 相对 sync 自带的比对，这些机制都是多余的。

6) **工件 URI 的三种 scheme（唯一约定，与 launch_train.sh 严格一一对应）**

   | 形式                 | 含义                       | 容器内最终路径        | 同步行为                          |
   |----------------------|----------------------------|-----------------------|-----------------------------------|
   | s3://<bucket>/<key>  | S3 直链                    | ${NVME_MOUNT}/<key>   | aws s3 sync --size-only -> NVMe   |
   | s3mnt://<key>        | 已由 S3 CSI 挂进容器的对象 | ${NVME_MOUNT}/<key>   | 从 <s3all_mount>/<key> 本地同步   |
   | fsx://<path>         | 已由 FSX CSI 挂进容器      | <fsx_mount>/<path>    | **完全跳过同步**                  |

   fsx:// 条目不产生任何 sync Job，但会写进 plan.env 的 FSX_ARTIFACTS 并由
   submit.sh 打印 —— 「跳过」必须可见，静默跳过会让人误以为同步过了。
   若某个 task 的全部条目都是 fsx://，该 task 整体不生成 Job（也不算 skipped，
   而是记为 fsx-direct）。

用法：
  derive_sync_jobs.py --yaml <pytorchjob.yaml> --out <dir> [--run-id <id>]
写出：
  <out>/plan.env          供 submit.sh source 的调度计划（任务顺序、超时、节点等）
  <out>/sync_<task>.yaml  每个任务一个 Job（model/tokenizer/manifest/audio）
"""
import argparse
import copy
import os
import re
import sys

try:
    import yaml
except ImportError:
    print("[derive][FATAL] PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# 常量区 —— 超时与 deadline 成对定义（B3）
# =============================================================================
# 用户原话："同步任务所用的 CPU 资源一般和训练任务一致，但最多 32 核。"
# 即：跟随训练容器的 requests/limits，再对 32 核封顶。
# 32 这个上限的依据：aws CLI v2 每个 cp 子进程是一个完整 Python 解释器，
# 并发再高只增加进程调度与内存开销，不提升 S3 吞吐。
SYNC_CPU_HARD_CAP = 32

# 内存：必须显式声明 requests/limits，否则是 BestEffort QoS，节点内存吃紧时
# 最先被 kubelet 驱逐（同步任务动辄跑几十分钟，被驱逐等于白干）。
# 取值依据：同步是 IO 密集型，内存只用于 aws CLI 进程（每进程 ~150MB）与页缓存，
# 32 并发 * 1GB 余量已非常宽裕 -> 32Gi limit；requests 取 limit 的 1/4 保证可调度。
# 同时**不得超过训练容器对应值**（训练 requests 1350Gi / limits 1500Gi），
# 32Gi 远低于此，因此恒满足；下方 _clamp_mem 仍做一次硬性夹取以防训练侧改小。
SYNC_MEM_REQUEST_GI = 8
SYNC_MEM_LIMIT_GI = 32

# ---- deadline 与 kubectl wait timeout 必须成对定义（B3）----
# 规则：wait_timeout 严格大于 activeDeadlineSeconds，留 120s 缓冲，
# 否则会出现「Job 还没被 k8s 判 DeadlineExceeded，本地 wait 就先超时返回」
# 的竞态：脚本报 ERR 但 Job 其实还在跑，后续 delete 会中断正在同步的节点。
WAIT_BUFFER_SEC = 120
DEADLINE_SEC = {
    "model": 3600,      # 模型 ~几十 GB
    "tokenizer": 1800,  # tokenizer 很小
    "manifest": 3600,   # manifest 文件数多但单个小
    "audio": 21600,     # WDS 原音频 shards，最大头，6h
}
WAIT_TIMEOUT_SEC = {k: v + WAIT_BUFFER_SEC for k, v in DEADLINE_SEC.items()}

# 任务定义：(task 名, 来源 env 列表)。列表顺序即串行执行顺序。
# tokenizer 的 BASE 变量可选（不是所有模型都有），缺失时只同步主 tokenizer。
TASK_DEFS = [
    ("model", ["MODEL_PATH"]),
    ("tokenizer", ["QWEN3_TTS_TOKENIZER", "QWEN3_TTS_TOKENIZER_BASE"]),
    ("manifest", ["BRIDGE_ALIGNMENT_DATA"]),
    ("audio", ["ARTIFACT_AUDIO_SHARDS"]),
]

# A4：Master / Worker 之间必须逐字一致的工件相关 env。
# 不一致意味着「同步的东西」和「训练读的东西」对不上，属于配置事故，必须 FATAL。
CRITICAL_ENVS = [
    "MODEL_PATH",
    "QWEN3_TTS_TOKENIZER",
    "QWEN3_TTS_TOKENIZER_BASE",
    "BRIDGE_ALIGNMENT_DATA",
    "ARTIFACT_AUDIO_SHARDS",
    "NVME_MOUNT",
    # FSX_MOUNT 参与 fsx:// 的路径拼接（launch_train.sh），Master/Worker 不一致
    # 会让两边读到不同目录 —— 与 NVME_MOUNT 同等级别的配置事故。
    "FSX_MOUNT",
]

# 1.3 减法清单：这些 env 前缀/名字与同步无关，必须从深拷贝的容器里剔除。
# 尤其 NCCL_*/FI_* 会让容器尝试初始化 EFA（我们已剥离 EFA 资源），
# WANDB_* 会带上密钥且触发无谓的网络连接。
STRIP_ENV_PREFIXES = ("NCCL_", "FI_", "WANDB_", "TORCH_NCCL_")
STRIP_ENV_NAMES = {
    "TRAIN_SCRIPT", "MODEL_TYPE", "OUTPUT_ROOT", "CODE_PREFIX",
    "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK",
}

# 1.3 减法清单：训练专用的卷。dshm(64Gi tmpfs)/workspace/launcher 同步用不到，
# 留着 dshm 会白占 64Gi 节点内存。
STRIP_VOLUMES = {"launcher", "workspace", "dshm"}

# 必须从 resources 中剥离的扩展资源：同步不需要 GPU / EFA，
# 留着会导致同步 Job 与训练 Job 抢同一批 GPU（节点上 8 卡已被训练独占，
# 同步 Pod 会永久 Pending 直到 deadline）。
STRIP_RESOURCES = ("nvidia.com/gpu", "vpc.amazonaws.com/efa")

ISTIO_ANNOTATION = ("sidecar.istio.io/inject", "false")

SCRIPT_MOUNT = "/opt/sync"
SCRIPT_CONFIGMAP = "artifact-sync-script"
SCRIPT_VOLUME = "syncbin"


def fatal(msg):
    print(f"[derive][FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# CPU / 内存数量解析
# =============================================================================
def cpu_to_cores(val):
    """k8s CPU 数量 -> float 核数。

    值可能是 "135"、"135000m"、"1.5"、135(int)。旧实现直接当整数用，
    遇到 "135000m" 会解析成 135000 核，min(135000,32) 侥幸对了，
    但 "500m" 会被当成 500 核 —— 必须正确解析。
    """
    s = str(val).strip()
    if not s:
        raise ValueError("empty cpu quantity")
    if s.endswith("m"):
        return float(s[:-1]) / 1000.0
    return float(s)


def cores_to_cpu_str(cores):
    """核数 -> k8s CPU 字符串。整数用整数形式，小数用毫核，避免浮点串。"""
    milli = int(round(cores * 1000))
    if milli % 1000 == 0:
        return str(milli // 1000)
    return f"{milli}m"


_MEM_UNITS = {
    "": 1, "k": 10 ** 3, "M": 10 ** 6, "G": 10 ** 9, "T": 10 ** 12,
    "Ki": 2 ** 10, "Mi": 2 ** 20, "Gi": 2 ** 30, "Ti": 2 ** 40,
}


def mem_to_bytes(val):
    """k8s 内存数量 -> 字节。支持 Ki/Mi/Gi/Ti 与 k/M/G/T 与裸字节。"""
    s = str(val).strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([EPTGMk]i?|)", s)
    if not m:
        raise ValueError(f"unparseable memory quantity: {val!r}")
    num, unit = m.group(1), m.group(2)
    if unit not in _MEM_UNITS:
        raise ValueError(f"unsupported memory unit in {val!r}")
    return float(num) * _MEM_UNITS[unit]


def _clamp_cpu(train_val, cap_cores, what):
    """requests/limits.cpu = min(训练值, 32)。训练侧缺失时用 cap 兜底。"""
    if train_val is None:
        return cores_to_cpu_str(cap_cores)
    try:
        cores = cpu_to_cores(train_val)
    except ValueError as e:
        fatal(f"cannot parse training {what}: {e}")
    return cores_to_cpu_str(min(cores, cap_cores))


def _clamp_mem(train_val, want_gi, what):
    """同步内存 = min(想要的值, 训练容器对应值)。

    规格要求"不超过训练容器对应值"：训练已经把节点内存吃掉大半，
    同步 Job 申请超过训练的量只会导致不可调度。
    """
    want = want_gi * (2 ** 30)
    if train_val is None:
        return f"{want_gi}Gi"
    try:
        train_bytes = mem_to_bytes(train_val)
    except ValueError as e:
        fatal(f"cannot parse training {what}: {e}")
    if train_bytes < want:
        # 训练侧比我们想要的还小，直接跟随训练值（原样字符串，避免单位转换失真）
        return str(train_val)
    return f"{want_gi}Gi"


# =============================================================================
# 训练 yaml 解析
# =============================================================================
def _container(pod_spec):
    """取名为 pytorch 的训练容器，找不到则取第一个。"""
    containers = pod_spec.get("containers") or []
    if not containers:
        raise ValueError("no containers in pod spec")
    for c in containers:
        if c.get("name") == "pytorch":
            return c
    return containers[0]


def _env_map(container):
    """env 列表 -> dict。仅取带字面 value 的项；valueFrom（Secret/ConfigMap/
    fieldRef）无法静态求值，跳过而不是误读成空串。"""
    out = {}
    for e in container.get("env") or []:
        name = e.get("name")
        if name and "value" in e:
            out[name] = str(e["value"])
    return out


def _node_whitelist(pod_spec):
    """从 nodeAffinity 取 kubernetes.io/hostname 白名单（去重保序）。"""
    terms = (
        (pod_spec.get("affinity") or {})
        .get("nodeAffinity", {})
        .get("requiredDuringSchedulingIgnoredDuringExecution", {})
        .get("nodeSelectorTerms")
        or []
    )
    nodes = []
    for t in terms:
        for expr in t.get("matchExpressions") or []:
            if expr.get("key") == "kubernetes.io/hostname" and expr.get("operator") == "In":
                nodes.extend(expr.get("values") or [])
    return list(dict.fromkeys(nodes))


def _mount_map(container):
    return {m.get("name"): m.get("mountPath") for m in container.get("volumeMounts") or []}


def _replicas(spec, role):
    """读 replicas。缺省按 k8s 语义视为 1（PyTorchJob 同）。"""
    if spec is None:
        return 0
    n = spec.get("replicas")
    if n is None:
        return 1
    if not isinstance(n, int) or n < 0:
        fatal(f"{role}.replicas must be a non-negative integer, got {n!r}")
    return n


def _assert_master_worker_consistent(master_pod, master_ctr, worker_pod, worker_ctr):
    """A4：Master / Worker 的关键字段必须一致，不一致 FATAL（旧实现只 WARN）。

    为什么必须 FATAL 而不是 WARN：我们只按 Master 派生同步 Job。
    若 Worker 的 MODEL_PATH 不同，Worker 节点训练时读的是另一个路径，
    同步再成功也没用 —— 这类配置事故必须在提交前挡住，WARN 会被日志淹没。
    """
    diffs = []

    m_env, w_env = _env_map(master_ctr), _env_map(worker_ctr)
    for k in CRITICAL_ENVS:
        mv, wv = m_env.get(k), w_env.get(k)
        if mv != wv:
            diffs.append(f"env[{k}]: Master={mv!r} Worker={wv!r}")

    if master_ctr.get("image") != worker_ctr.get("image"):
        diffs.append(
            f"image: Master={master_ctr.get('image')!r} Worker={worker_ctr.get('image')!r}")

    # 结构化比较（dict/list 直接 ==），不做字符串比较，避免键顺序造成假阳性
    for field in ("nodeSelector", "affinity", "volumes", "serviceAccountName"):
        mv, wv = master_pod.get(field), worker_pod.get(field)
        if mv != wv:
            diffs.append(f"pod.{field} differs: Master={mv!r} Worker={wv!r}")

    if diffs:
        fatal(
            "Master/Worker mismatch (sync is derived from Master only, "
            "so any divergence means workers would read artifacts we never synced):\n  "
            + "\n  ".join(diffs)
        )


def parse(path):
    doc = yaml.safe_load(open(path))
    if not doc or doc.get("kind") != "PyTorchJob":
        raise ValueError(f"{path} is not a PyTorchJob manifest")

    specs = doc["spec"]["pytorchReplicaSpecs"]
    if "Master" not in specs:
        raise ValueError("pytorchReplicaSpecs.Master not found")

    master_tmpl = specs["Master"]["template"]
    master_pod = master_tmpl["spec"]
    master_ctr = _container(master_pod)

    worker_spec = specs.get("Worker")
    n_master = _replicas(specs.get("Master"), "Master")
    n_worker = _replicas(worker_spec, "Worker") if worker_spec else 0

    if worker_spec and n_worker > 0:
        worker_pod = worker_spec["template"]["spec"]
        _assert_master_worker_consistent(
            master_pod, master_ctr, worker_pod, _container(worker_pod))

    nodes = _node_whitelist(master_pod)
    if not nodes:
        raise ValueError("node whitelist (kubernetes.io/hostname) not found in Master affinity")

    # ---- A3：训练节点数必须等于白名单长度 ----
    # 不等意味着「同步覆盖的节点集合」与「训练实际使用的节点集合」不同：
    #  - N < 白名单：同步了训练不用的节点（浪费，但更危险的是掩盖了配置漂移）
    #  - N > 白名单：不可能调度成功，训练本身就会卡 Pending
    # 无论哪种都必须在提交前挡住。
    n_train = n_master + n_worker
    if n_train != len(nodes):
        fatal(
            f"replica/whitelist mismatch: Master.replicas({n_master}) + "
            f"Worker.replicas({n_worker}) = {n_train}, but node whitelist has "
            f"{len(nodes)} entries {nodes}. "
            "Sync coverage is derived from the whitelist and enforced by "
            "podAntiAffinity(one pod per host); these two MUST be equal."
        )

    mounts = _mount_map(master_ctr)
    env = _env_map(master_ctr)

    return {
        "doc": doc,
        "ns": doc["metadata"].get("namespace", "speech-h200"),
        "env": env,
        "nodes": nodes,
        "n": n_train,
        "master_template": master_tmpl,
        "master_container_name": master_ctr.get("name", "pytorch"),
        "nvme_mount": mounts.get("nvme", env.get("NVME_MOUNT", "/nvme")),
        "data_mount": mounts.get("s3all", "/s3all"),
        "models_mount": mounts.get("s3models", "/s3models"),
        # fsx 挂载点：**只从 volumeMounts 取，取不到就是 None**（不给默认值）。
        # 这样 build_tasks 见到 fsx:// 却拿到 None 时能 FATAL —— 若这里兜底成
        # "/fsx"，yaml 漏挂 fsx volume 的配置事故会被掩盖，训练启动后才报
        # "文件不存在"，那时已经浪费了整轮同步与调度。
        "fsx_mount": mounts.get("fsx"),
        "resources": master_ctr.get("resources") or {},
    }


# =============================================================================
# 工件推导
# =============================================================================
def _norm_key(key):
    """归一化路径段：折叠重复斜杠、去掉首尾空段。

    yaml 里写成 `s3://bucket//data/...` 时不能产生空段（会拼出 /nvme/data//x，
    与 launch_train.sh 归一化后的训练侧路径对不上，训练会静默读不到文件）。
    """
    return "/".join(p for p in key.split("/") if p)


def s3key_of(val, nvme_mount="/nvme"):
    """从 env 值提取「NVMe 下的相对 key」（仅对落 NVMe 的两种 scheme 有意义）。

    s3://<bucket>/<key> -> <key>；s3mnt://<key> -> <key>；<nvme_mount>/<key> -> <key>。
    其余（含 fsx://）返回 ""，由调用方按 scheme 分支处理。

    nvme_mount 由调用方从 yaml 的 volumeMounts 解析后传入 —— 早期版本把 "/nvme"
    硬编码在这里，NVME_MOUNT 改成别的挂载点时会静默失配（返回 "" -> 报"不是合法
    URI"，错误信息还指向错误的方向）。
    """
    if not val:
        return ""
    prefix = nvme_mount.rstrip("/") + "/"
    if val.startswith("s3://"):
        body = val[len("s3://"):]
        key = body.split("/", 1)[1] if "/" in body else ""
    elif val.startswith("s3mnt://"):
        key = val[len("s3mnt://"):]
    elif val.startswith(prefix):
        key = val[len(prefix):]
    else:
        return ""
    return _norm_key(key)


def _assert_no_traversal(en, tok, key):
    """拒绝含 `..` 的路径段。

    key 会被直接拼进 `<nvme>/<key>` 或 `<fsx>/<key>` 并交给 rsync/cp/aws sync，
    `..` 能让同步目标逃出 NVMe 挂载点写到宿主机任意位置（sync Pod 挂的是
    hostPath，权限足够），属于必须在派生期挡住的高危输入。
    """
    if any(p == ".." for p in key.split("/")):
        fatal(f"env {en} value {tok!r} contains a '..' path segment; "
              f"path traversal is not allowed in artifact URIs")


def _normalize_s3(val):
    """归一化 s3:// URL 的重复斜杠（bucket 与 key 段都不留空段）。"""
    return "s3://" + _norm_key(val[len("s3://"):])


# 工件条目的三种 kind（与 sync_artifact.sh 的 KINDS 取值、launch_train.sh 的
# scheme 分支严格一一对应，改一处必须同时改另两处）：
KIND_S3 = "s3"      # aws s3 sync --size-only
KIND_MNT = "mnt"    # 从 CSI 挂载点做本地文件系统增量同步
KIND_FSX = "fsx"    # 不同步，训练直读


def build_tasks(meta):
    """返回 [(task, [(kind, src, dst, key), ...]), ...]，顺序即串行顺序。

    * kind=s3  : src=s3://bucket/key            dst=<nvme>/<key>
    * kind=mnt : src=<s3all_mount>/<key>        dst=<nvme>/<key>
    * kind=fsx : src=<fsx_mount>/<path>         dst=src（不搬运，仅用于打印）

    env 缺失/为空的任务会以空 items 出现（submit.sh 据此打印 skipped），
    而不是被静默丢掉 —— 静默丢掉会让人以为"同步过了"。
    """
    env = meta["env"]
    nvme = meta["nvme_mount"]
    s3all = meta["data_mount"]
    fsx = meta["fsx_mount"]
    tasks = []
    for task, env_names in TASK_DEFS:
        items = []
        for en in env_names:
            raw = env.get(en, "")
            # ARTIFACT_AUDIO_SHARDS 是空格分隔的多前缀，同属一个任务；
            # 其余变量 split() 后也就一个元素，统一处理即可。
            for tok in raw.split():
                if tok.startswith("fsx://"):
                    # ---- fsx://：训练直读，不进任何 sync task ----
                    path = _norm_key(tok[len("fsx://"):])
                    if not path:
                        fatal(f"env {en} value {tok!r} has an empty fsx path")
                    if not fsx:
                        # yaml 用了 fsx:// 却没给容器挂 fsx volume：训练一定
                        # 读不到。这是配置事故，必须在提交前挡住。
                        fatal(
                            f"env {en} value {tok!r} uses the fsx:// scheme, but the "
                            f"training container has no volumeMount named 'fsx'. "
                            f"Add an fsx volume+volumeMount (and set FSX_MOUNT env to "
                            f"its mountPath) before submitting."
                        )
                    if re.search(r"\s", path):
                        fatal(f"env {en}: fsx path contains whitespace: {path!r}")
                    _assert_no_traversal(en, tok, path)
                    full = f"{fsx.rstrip('/')}/{path}"
                    items.append((KIND_FSX, full, full, path))
                    continue

                key = s3key_of(tok, nvme)
                if not key:
                    fatal(
                        f"env {en} value {tok!r} is not a supported artifact URI. "
                        f"Expected one of: s3://<bucket>/<key>, s3mnt://<key>, "
                        f"fsx://<path>, or an absolute {nvme}/ path."
                    )
                # 1.5：SRCS/DSTS 用空格分隔传给脚本，key 里带空白会把一个路径
                # 拆成两个，导致同步到错误位置且不报错。必须在派生期就挡住。
                if re.search(r"\s", key):
                    fatal(f"env {en}: artifact key contains whitespace, cannot be "
                          f"passed space-separated: {key!r}")
                _assert_no_traversal(en, tok, key)

                if tok.startswith("s3://"):
                    items.append((KIND_S3, _normalize_s3(tok), f"{nvme}/{key}", key))
                elif tok.startswith("s3mnt://"):
                    # 源是容器内的 CSI 挂载点路径，不是 s3:// URL —— sync Job
                    # 必须挂上 s3all PVC（need_csi），否则源目录不存在。
                    if not s3all:
                        fatal(
                            f"env {en} value {tok!r} uses the s3mnt:// scheme, but the "
                            f"training container has no volumeMount named 's3all'."
                        )
                    items.append(
                        (KIND_MNT, f"{s3all.rstrip('/')}/{key}", f"{nvme}/{key}", key))
                else:
                    # 走到这里说明是裸 <nvme>/ 路径：无法判断数据从哪来，
                    # 也就无从同步。旧实现同样拒绝，保持不变。
                    fatal(
                        f"env {en} value {tok!r} is a bare NVMe path with no source; "
                        f"use s3://, s3mnt:// or fsx:// so the sync source is explicit."
                    )
        tasks.append((task, items))
    return tasks


# =============================================================================
# Pod template 派生：深拷贝训练 Master.template 后做减法
# =============================================================================
def _strip_container(ctr, cpu_req, cpu_lim, mem_req, mem_lim, command, env, keep_volumes):
    """对深拷贝出来的训练容器做减法。原地修改。"""
    ctr["command"] = command
    ctr.pop("args", None)          # 训练的 args 对同步无意义，残留会被 command 之后误传
    ctr.pop("livenessProbe", None)
    ctr.pop("readinessProbe", None)
    ctr.pop("startupProbe", None)
    ctr.pop("lifecycle", None)
    ctr.pop("ports", None)
    ctr["env"] = env               # 只留同步所需（见 build_sync_env）

    # resources：显式重建，确保 GPU/EFA 被彻底剥离（不能只 pop，
    # 训练 yaml 未来可能新增别的扩展资源，白名单式重建更安全）。
    ctr["resources"] = {
        "requests": {"cpu": cpu_req, "memory": mem_req},
        "limits": {"cpu": cpu_lim, "memory": mem_lim},
    }
    # ephemeral-storage 有意不声明：同步写的是 hostPath(NVMe)，不占容器
    # ephemeral 层；训练那 200Gi 的申请会白白抬高调度门槛。

    ctr["volumeMounts"] = [
        m for m in (ctr.get("volumeMounts") or []) if m.get("name") in keep_volumes
    ]
    ctr["volumeMounts"].append(
        {"name": SCRIPT_VOLUME, "mountPath": SCRIPT_MOUNT, "readOnly": True})
    # imagePullPolicy / envFrom / securityContext 原样保留 —— 同步与训练读的是同
    # 一批 S3 对象，凭证来源必须与训练容器完全一致。
    return ctr


def build_sync_env(meta, kinds, srcs, dsts, sync_par):
    """同步脚本的全部入参。NODE_NAME 用 fieldRef 取真实节点名，
    仅用于日志，便于失败时精确定位到节点。

    KINDS 与 SRCS/DSTS 同序、同长度、空格分隔，取值 s3 / mnt，
    sync_artifact.sh 据此选择 `aws s3 sync` 还是本地文件系统同步。
    （fsx 条目不会进到这里 —— 它们根本不产生 Job。）
    """
    return [
        {"name": "NODE_NAME",
         "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        {"name": "NVME_MOUNT", "value": meta["nvme_mount"]},
        {"name": "KINDS", "value": " ".join(kinds)},
        {"name": "SRCS", "value": " ".join(srcs)},
        {"name": "DSTS", "value": " ".join(dsts)},
        {"name": "SYNC_PAR", "value": str(sync_par)},
    ]


def derive_job(meta, *, name, task, command, env, cpu_req, cpu_lim, mem_req, mem_lim, deadline, need_csi):
    """把训练 Master.template 深拷贝成一个 batch/v1 Job。

    深拷贝的意义（B1 的根治手段）：annotations（含 istio 禁注入）、labels、
    serviceAccountName、nodeSelector、affinity.nodeAffinity、tolerations、
    imagePullSecrets、securityContext、priorityClassName、schedulerName、
    dnsPolicy/hostNetwork 等**全部自动继承**，不需要维护白名单，
    训练侧新增调度字段也不会漏。
    """
    tmpl = copy.deepcopy(meta["master_template"])
    pod = tmpl["spec"]

    # ---- 卷：只留 nvme（+ s3mnt:// 源需要的 s3all/s3models），追加脚本 ConfigMap ----
    # 注意 fsx 卷**永远不保留**：fsx:// 工件根本不产生 sync Job，给同步 Pod 挂
    # FSX 只会增加挂载失败面（FSX CSI 挂载慢，且 Pod 起不来时很难定位）。
    keep_volumes = {"nvme"}
    if need_csi:
        keep_volumes |= {"s3all", "s3models"}
    pod["volumes"] = [
        v for v in (pod.get("volumes") or [])
        if v.get("name") in keep_volumes and v.get("name") not in STRIP_VOLUMES
    ]
    pod["volumes"].append({
        "name": SCRIPT_VOLUME,
        "configMap": {"name": SCRIPT_CONFIGMAP, "defaultMode": 0o755},
    })
    present_volumes = {v["name"] for v in pod["volumes"]}

    # ---- 容器：只保留训练容器本体，剥离可能存在的 sidecar ----
    ctrs = pod.get("containers") or []
    main = None
    for c in ctrs:
        if c.get("name") == meta["master_container_name"]:
            main = c
            break
    if main is None:
        main = ctrs[0]
    _strip_container(main, cpu_req, cpu_lim, mem_req, mem_lim, command, env,
                     present_volumes)
    # sync Pod 显式注入 AWS 凭证 secret（aws-local-creds），使 `aws s3 sync`
    # 能访问模型/数据桶。训练容器出于安全考量不携带凭证，这里为同步任务补上
    # 集群内默认身份（见 sync_artifact.sh：aws CLI 读取注入的
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION 环境变量）。
    _AWS_CREDS_SECRET = "aws-local-creds"
    main.setdefault("envFrom", [])
    if not any(e.get("secretRef", {}).get("name") == _AWS_CREDS_SECRET
               for e in main["envFrom"]):
        main["envFrom"].append({"secretRef": {"name": _AWS_CREDS_SECRET}})
    pod["containers"] = [main]
    pod.pop("initContainers", None)

    # ---- restartPolicy 必须是 Never ----
    # backoffLimit=0 只在 restartPolicy=Never 下生效为"不重试"；
    # OnFailure 会让 kubelet 就地无限重启容器，Job 永远不判失败。
    pod["restartPolicy"] = "Never"

    # ---- podAntiAffinity：N 个 Pod 必须落 N 个不同节点 ----
    # nodeAffinity 直接沿用深拷贝下来的训练值（白名单），此处只追加互斥。
    affinity = pod.setdefault("affinity", {})
    affinity["podAntiAffinity"] = {
        "requiredDuringSchedulingIgnoredDuringExecution": [{
            "labelSelector": {"matchLabels": {"sync-job": name}},
            "topologyKey": "kubernetes.io/hostname",
        }]
    }
    # 自身 Pod 打上该 label，作为上面 labelSelector 的匹配对象
    tmpl.setdefault("metadata", {}).setdefault("labels", {})["sync-job"] = name
    tmpl["metadata"]["labels"]["sync-task"] = task

    n = meta["n"]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": meta["ns"],
            "labels": {"sync-job": name, "sync-task": task},
        },
        "spec": {
            # completions=parallelism=N + Indexed：N 个 Pod 全成功才 Complete。
            # 这是"全节点覆盖"的硬保证，替代了 PyTorchJob 的假成功语义。
            "completions": n,
            "parallelism": n,
            "completionMode": "Indexed",
            # backoffLimit=0：脚本内部已对单对象重试 3 次；Job 层再重试会让
            # 失败节点上重新起一个 Pod，可能落到已成功的节点（podAntiAffinity
            # 只约束同时存在的 Pod），破坏一节点一个的不变式。
            "backoffLimit": 0,
            "activeDeadlineSeconds": deadline,
            # 不设 ttlSecondsAfterFinished：submit.sh 需要在 Job 结束后逐 Pod
            # 抓日志（B4），TTL 到期会连 Pod 一起删掉，日志就没了。由 submit.sh
            # 在采集完日志后显式删除。
            "template": tmpl,
        },
    }


# =============================================================================
# 派生结果自检（B1）
# =============================================================================
def assert_pod_sanity(job, name):
    """对派生结果做硬断言。任何一条不满足都是"静默错配"，必须当场炸掉，
    绝不能让它被提交到集群后才在 Pending/不 Complete 里慢慢暴露。"""
    tmpl = job["spec"]["template"]
    pod = tmpl["spec"]
    ann = (tmpl.get("metadata") or {}).get("annotations") or {}
    k, v = ISTIO_ANNOTATION
    if ann.get(k) != v:
        fatal(f"{name}: missing pod annotation {k}={v}; istio sidecar would be "
              f"injected and the Job could never reach Complete "
              f"(sidecar never exits). Got annotations={ann!r}")

    ctr = pod["containers"][0]
    res = ctr.get("resources") or {}
    for scope in ("requests", "limits"):
        block = res.get(scope) or {}
        for bad in STRIP_RESOURCES:
            if bad in block:
                fatal(f"{name}: resources.{scope} still contains {bad}")
        if "cpu" not in block or "memory" not in block:
            fatal(f"{name}: resources.{scope} must declare both cpu and memory "
                  f"(otherwise QoS is BestEffort and the pod gets evicted first)")
    lim_cores = cpu_to_cores(res["limits"]["cpu"])
    req_cores = cpu_to_cores(res["requests"]["cpu"])
    if lim_cores > SYNC_CPU_HARD_CAP or req_cores > SYNC_CPU_HARD_CAP:
        fatal(f"{name}: cpu exceeds hard cap {SYNC_CPU_HARD_CAP} "
              f"(requests={req_cores} limits={lim_cores})")
    if req_cores > lim_cores:
        fatal(f"{name}: requests.cpu({req_cores}) > limits.cpu({lim_cores})")

    if pod.get("restartPolicy") != "Never":
        fatal(f"{name}: restartPolicy must be Never, got {pod.get('restartPolicy')!r}")

    spec = job["spec"]
    if not (spec["completions"] == spec["parallelism"] >= 1):
        fatal(f"{name}: completions/parallelism invalid: {spec}")
    if spec.get("backoffLimit") != 0:
        fatal(f"{name}: backoffLimit must be 0")
    if spec.get("completionMode") != "Indexed":
        fatal(f"{name}: completionMode must be Indexed")

    aff = pod.get("affinity") or {}
    if not (aff.get("nodeAffinity") or {}).get(
            "requiredDuringSchedulingIgnoredDuringExecution"):
        fatal(f"{name}: nodeAffinity(required) lost during derivation")
    anti = (aff.get("podAntiAffinity") or {}).get(
        "requiredDuringSchedulingIgnoredDuringExecution")
    if not anti or anti[0].get("topologyKey") != "kubernetes.io/hostname":
        fatal(f"{name}: podAntiAffinity(hostname) missing; pods could pile up on "
              f"one node and leave other nodes unsynced")

    mount_names = {m["name"] for m in ctr.get("volumeMounts") or []}
    vol_names = {v["name"] for v in pod.get("volumes") or []}
    if not mount_names <= vol_names:
        fatal(f"{name}: volumeMounts reference undefined volumes: "
              f"{sorted(mount_names - vol_names)}")
    for bad in STRIP_VOLUMES:
        if bad in vol_names:
            fatal(f"{name}: volume {bad} should have been stripped")

    env_names = {e["name"] for e in ctr.get("env") or []}
    for e in env_names:
        if e in STRIP_ENV_NAMES or e.startswith(STRIP_ENV_PREFIXES):
            fatal(f"{name}: training-only env {e} leaked into sync job")


# =============================================================================
# main
# =============================================================================
def sh_quote(v):
    return "'" + str(v).replace("'", "'\\''") + "'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="run")
    args = ap.parse_args()

    # ConfigMap 名带 run-id（D4）：避免两次 submit 并发时互相覆盖脚本。
    # submit.sh 用同样的约定 `artifact-sync-script-${RUN_ID}` 建 CM，必须对齐。
    global SCRIPT_CONFIGMAP
    SCRIPT_CONFIGMAP = f"artifact-sync-script-{args.run_id}"

    try:
        meta = parse(args.yaml)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - 顶层需给出可读诊断
        fatal(f"cannot parse {args.yaml}: {e}")

    tasks = build_tasks(meta)
    # 「一个工件都没解析出来」才是错。只要有条目（哪怕全是 fsx:// 直读、
    # 一个 sync Job 都不生成）就是合法配置 —— 不能因为没有同步任务就 FATAL。
    if not any(items for _, items in tasks):
        fatal("no artifacts derived from yaml env (all task envs empty)")

    # ---- CPU/内存：跟随训练容器，对 32 核封顶（1.4）----
    train_req = (meta["resources"].get("requests") or {})
    train_lim = (meta["resources"].get("limits") or {})
    cpu_req = _clamp_cpu(train_req.get("cpu"), SYNC_CPU_HARD_CAP, "requests.cpu")
    cpu_lim = _clamp_cpu(train_lim.get("cpu"), SYNC_CPU_HARD_CAP, "limits.cpu")
    # requests 不得大于 limits（训练 requests<limits，封顶后可能同时撞到 32，
    # 也可能 requests 更小；这里只需保证不倒挂）
    if cpu_to_cores(cpu_req) > cpu_to_cores(cpu_lim):
        cpu_req = cpu_lim
    mem_req = _clamp_mem(train_req.get("memory"), SYNC_MEM_REQUEST_GI, "requests.memory")
    mem_lim = _clamp_mem(train_lim.get("memory"), SYNC_MEM_LIMIT_GI, "limits.memory")
    # SYNC_PAR = limits.cpu（核数取整，至少 1）。绝不用容器内 nproc。
    sync_par = max(1, int(cpu_to_cores(cpu_lim)))

    os.makedirs(args.out, exist_ok=True)

    active_tasks, skipped_tasks, fsx_tasks = [], [], []
    fsx_artifacts = []      # 全局：所有 fsx:// 直读工件的容器内路径
    n_artifacts = 0         # 已解析的工件总数（含 fsx），用于"是否什么都没解析出来"判断
    for task, items in tasks:
        n_artifacts += len(items)
        if not items:
            skipped_tasks.append(task)
            continue

        # fsx:// 条目不参与同步，从 sync 列表中剔除，但必须记录下来供打印。
        fsx_items = [it for it in items if it[0] == KIND_FSX]
        sync_items = [it for it in items if it[0] != KIND_FSX]
        fsx_artifacts.extend(src for _, src, _, _ in fsx_items)

        if not sync_items:
            # 该 task 全部走 FSX 直读 -> 整体不生成 Job。这不是 "skipped"
            # （env 没配），而是 "fsx-direct"（配了但无需同步），两者必须区分，
            # 否则日志会误导成"忘了配数据"。
            fsx_tasks.append(task)
            continue

        active_tasks.append(task)
        kinds = [k for k, _, _, _ in sync_items]
        srcs = [s for _, s, _, _ in sync_items]
        dsts = [d for _, _, d, _ in sync_items]
        job = derive_job(
            meta,
            name=f"sync-{task}-{args.run_id}",
            task=task,
            command=["bash", "-c", f"{SCRIPT_MOUNT}/sync_artifact.sh"],
            env=build_sync_env(meta, kinds, srcs, dsts, sync_par),
            cpu_req=cpu_req, cpu_lim=cpu_lim, mem_req=mem_req, mem_lim=mem_lim,
            deadline=DEADLINE_SEC[task],
            # 只有 CSI 挂载源（s3mnt://，kind=mnt）才需要给 sync Pod 挂 PVC；
            # s3:// 直链源不需要（模型桶 CSI 未授权，挂了反而会让 Pod 起不来）。
            need_csi=any(k == KIND_MNT for k in kinds),
        )
        assert_pod_sanity(job, task)
        with open(os.path.join(args.out, f"sync_{task}.yaml"), "w") as f:
            yaml.safe_dump(job, f, sort_keys=False, default_flow_style=False,
                           allow_unicode=True)

    if n_artifacts == 0:
        fatal("no artifacts derived from yaml env (all task envs empty)")

    # ---- plan.env：submit.sh 的调度计划 ----
    with open(os.path.join(args.out, "plan.env"), "w") as f:
        f.write(f"NAMESPACE={sh_quote(meta['ns'])}\n")
        f.write(f"NODES={sh_quote(' '.join(meta['nodes']))}\n")
        f.write(f"NODE_COUNT={meta['n']}\n")
        f.write(f"SYNC_TASKS={sh_quote(' '.join(active_tasks))}\n")
        f.write(f"SKIPPED_TASKS={sh_quote(' '.join(skipped_tasks))}\n")
        # FSX 直读：task 粒度与工件粒度都写出，submit.sh 全部打印（不静默跳过）
        f.write(f"FSX_TASKS={sh_quote(' '.join(fsx_tasks))}\n")
        f.write(f"FSX_ARTIFACTS={sh_quote(' '.join(fsx_artifacts))}\n")
        f.write(f"FSX_MOUNT={sh_quote(meta['fsx_mount'] or '')}\n")
        f.write(f"ARTIFACT_COUNT={n_artifacts}\n")
        f.write(f"SYNC_PAR={sync_par}\n")
        f.write(f"SYNC_CPU_REQUEST={sh_quote(cpu_req)}\n")
        f.write(f"SYNC_CPU_LIMIT={sh_quote(cpu_lim)}\n")
        for task in active_tasks:
            f.write(f"WAIT_TIMEOUT_{task}={WAIT_TIMEOUT_SEC[task]}\n")
            f.write(f"JOB_NAME_{task}={sh_quote(f'sync-{task}-{args.run_id}')}\n")

    print(f"[derive] N={meta['n']} nodes={len(meta['nodes'])} "
          f"cpu={cpu_req}/{cpu_lim} par={sync_par}")
    print(f"[derive] sync tasks: {' '.join(active_tasks) or '(none)'}")
    if fsx_tasks:
        print(f"[derive] fsx-direct tasks (no sync job): {' '.join(fsx_tasks)}")
    for p in fsx_artifacts:
        print(f"[derive]   fsx-direct artifact: {p}")
    if skipped_tasks:
        print(f"[derive] skipped (env not set): {' '.join(skipped_tasks)}")


if __name__ == "__main__":
    main()
