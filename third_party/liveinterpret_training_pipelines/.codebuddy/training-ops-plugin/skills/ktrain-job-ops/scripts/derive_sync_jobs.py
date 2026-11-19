#!/usr/bin/env python3
"""从 KtrainJob 任务 yaml 派生「工件同步」Job（batch/v1 Job，非 KtrainJob）。

输入契约（全部来自任务 yaml 或命令行参数，脚本无任何环境硬编码）：
  metadata.namespace                命名空间（必填）
  spec.image                        同步 Pod 镜像（与训练一致）
  spec.numNodes                     节点数 N，必须等于白名单长度
  spec.resources                    训练资源（平铺 cpu/memory/gpu 或 requests/limits），
                                    同步 Pod 资源由此 clamp 派生
  spec.env                          工件声明扫描源：值中空格分隔的 token，
      s3://<bucket>/<key>  -> 工件，同步到 <NVME_MOUNT>/<key>（aws s3 sync --size-only）
      s3mnt://<key>        -> 工件，源已在 S3 CSI 挂载点上，从 <S3MNT_MOUNT>/<key>
                              本地增量同步（rsync/cp/tar）到 <NVME_MOUNT>/<key>
      fsx://<path>         -> 工件，不同步（训练经 FSX CSI 直读 <FSX_MOUNT>/<path>）
      其余 token           -> 非工件，忽略
  spec.env.NVME_MOUNT               NVMe 容器挂载点（存在 s3:// 或 s3mnt:// 工件时必填）；
                                    按 mountPath == NVME_MOUNT 在 render-extra
                                    extra_volume_mounts 里定位 NVMe 卷
  spec.env.FSX_MOUNT                FSX 容器挂载点（存在 fsx:// 工件时必填）
  spec.env.S3ALL_MOUNT              s3mnt 源的 CSI 挂载点（存在 s3mnt:// 工件时：
                                    未设置且 render-extra 中只有一个 PVC 卷时自动
                                    取该卷挂载点，多个 PVC 卷时必须显式设置）
  metadata.annotations['ktrain.oit-svc.com/render-extra'] JSON：
      affinity.nodeAffinity...kubernetes.io/hostname In   节点白名单（必填）
      extra_volumes / extra_volume_mounts                 NVMe 卷来源
      node_selector / tolerations / service_account_name  原样复制到同步 Pod

同步 Job 结构（派生规则，不可改）：
  completions=N / parallelism=N / completionMode=Indexed + podAntiAffinity(hostname)
  -> N 个 Pod 一一落到 N 个白名单节点，全部成功才 Complete。

用法：
  derive_sync_jobs.py --yaml <ktrainjob.yaml> --out <dir> [--run-id <id>]
                      [--deadline-sec N] [--task-deadline <任务名|env名>=<秒>]...
                      [--aws-creds-secret <secret名>]
                      [--cpu-cap 32] [--mem-request-gi 8] [--mem-limit-gi 32]
写出：
  <out>/plan.env          供 submit.sh source 的调度计划
  <out>/sync_<task>.yaml  每个同步任务一个 Job
"""
import argparse
import copy
import json
import os
import re
import sys

try:
    import yaml
except ImportError:
    print("[derive][FATAL] PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

WAIT_BUFFER_SEC = 120
SCRIPT_MOUNT = "/opt/sync"
SCRIPT_CONFIGMAP = "artifact-sync-script"
SCRIPT_VOLUME = "syncbin"

_KIND_S3 = "s3"
_KIND_S3MNT = "s3mnt"
_KIND_FSX = "fsx"


def fatal(msg):
    print(f"[derive][FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


# ---- CPU / 内存解析 ----
def cpu_to_cores(val):
    s = str(val).strip()
    if not s:
        raise ValueError("empty cpu quantity")
    if s.endswith("m"):
        return float(s[:-1]) / 1000.0
    return float(s)


def cores_to_cpu_str(cores):
    milli = int(round(cores * 1000))
    if milli % 1000 == 0:
        return str(milli // 1000)
    return f"{milli}m"


_MEM_UNITS = {
    "": 1, "k": 10 ** 3, "M": 10 ** 6, "G": 10 ** 9, "T": 10 ** 12,
    "Ki": 2 ** 10, "Mi": 2 ** 20, "Gi": 2 ** 30, "Ti": 2 ** 40,
}


def mem_to_bytes(val):
    s = str(val).strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([EPTGMk]i?|)", s)
    if not m:
        raise ValueError(f"unparseable memory quantity: {val!r}")
    num, unit = m.group(1), m.group(2)
    if unit not in _MEM_UNITS:
        raise ValueError(f"unsupported memory unit in {val!r}")
    return float(num) * _MEM_UNITS[unit]


def _clamp_cpu(train_val, cap_cores, what):
    if train_val is None:
        return cores_to_cpu_str(cap_cores)
    try:
        cores = cpu_to_cores(train_val)
    except ValueError as e:
        fatal(f"cannot parse training {what}: {e}")
    return cores_to_cpu_str(min(cores, cap_cores))


def _clamp_mem(train_val, want_gi, what):
    want = want_gi * (2 ** 30)
    if train_val is None:
        return f"{want_gi}Gi"
    try:
        train_bytes = mem_to_bytes(train_val)
    except ValueError as e:
        fatal(f"cannot parse training {what}: {e}")
    if train_bytes < want:
        return str(train_val)
    return f"{want_gi}Gi"


# ---- KtrainJob yaml 解析 ----
def _render_extra(doc):
    ann = (doc.get("metadata") or {}).get("annotations") or {}
    raw = ann.get("ktrain.oit-svc.com/render-extra")
    if not raw:
        fatal("metadata.annotations['ktrain.oit-svc.com/render-extra'] not found")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        fatal(f"render-extra annotation is not valid JSON: {e}")


def _env_map(spec):
    env = spec.get("env") or {}
    return {k: str(v) for k, v in env.items() if v is not None}


def _node_whitelist(re):
    terms = (
        (re.get("affinity") or {})
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


def parse(path):
    doc = yaml.safe_load(open(path))
    if not doc or doc.get("kind") != "KtrainJob":
        fatal(f"{path} is not a KtrainJob manifest "
              f"(kind={doc.get('kind') if doc else 'None'})")

    spec = doc.get("spec") or {}
    re_extra = _render_extra(doc)

    ns = (doc.get("metadata") or {}).get("namespace")
    if not ns:
        fatal("metadata.namespace missing in task yaml")

    nodes = _node_whitelist(re_extra)
    if not nodes:
        fatal("node whitelist (kubernetes.io/hostname In) not found in "
              "render-extra.affinity")

    n = int(spec.get("numNodes") or 1)
    # framework=ray 时 numNodes = head(0GPU) + workers（probe 渲染实测：
    # numNodes=3 -> workerGroupSpecs.replicas=2），真正吃 GPU 物理节点的是
    # workers = numNodes-1，白名单长度应与之一致；pytorch 框架则 numNodes
    # 本身就是 worker 数，与白名单 1:1。
    framework = str(spec.get("framework") or "pytorch").lower()
    worker_nodes = n - 1 if framework == "ray" else n
    if worker_nodes != len(nodes):
        fatal(f"framework={framework} numNodes({n}) -> worker nodes "
              f"({worker_nodes}) != len(node whitelist)({len(nodes)} {nodes})")

    env = _env_map(spec)
    mounts = {m.get("name"): m.get("mountPath")
              for m in re_extra.get("extra_volume_mounts") or [] if m.get("name")}
    volumes = re_extra.get("extra_volumes") or []

    def _volume_by_mount_path(mp):
        name = next((n for n, p in mounts.items()
                     if p and p.rstrip("/") == mp.rstrip("/")), None)
        if not name:
            return None
        return next((v for v in volumes if v.get("name") == name), None)

    nvme_mount = env.get("NVME_MOUNT")
    nvme_volume = None
    if nvme_mount:
        nvme_volume = _volume_by_mount_path(nvme_mount)
        if nvme_volume is None:
            fatal(f"no extra_volume_mounts entry with mountPath == NVME_MOUNT "
                  f"({nvme_mount}); cannot locate the NVMe volume")

    # s3mnt 源挂载：env.S3ALL_MOUNT 显式指定；否则取唯一 PVC 卷的挂载点
    s3mnt_mount = env.get("S3ALL_MOUNT")
    s3mnt_volume = None
    pvc_volumes = [v for v in volumes
                   if (v.get("persistentVolumeClaim") or {}).get("claimName")]
    if s3mnt_mount:
        s3mnt_volume = _volume_by_mount_path(s3mnt_mount)
        if s3mnt_volume is None:
            fatal(f"no extra_volume_mounts entry with mountPath == S3ALL_MOUNT "
                  f"({s3mnt_mount}); cannot locate the s3mnt source volume")
    elif len(pvc_volumes) == 1:
        s3mnt_volume = pvc_volumes[0]
        s3mnt_mount = mounts.get(s3mnt_volume["name"])

    return {
        "spec": spec,
        "render_extra": re_extra,
        "ns": ns,
        "env": env,
        "nodes": nodes,
        # completions / NODE_COUNT = 物理 GPU 节点数。ray 的 spec.numNodes
        # 含 0-GPU head，若原样写入会让第 3 个 sync Pod 因 hostname
        # anti-affinity 永远 Pending（白名单只有 workers 台）。
        "n": worker_nodes,
        "image": spec.get("image", ""),
        "nvme_mount": nvme_mount,
        "nvme_volume": nvme_volume,
        "s3mnt_mount": s3mnt_mount,
        "s3mnt_volume": s3mnt_volume,
        "fsx_mount": env.get("FSX_MOUNT"),
        "resources": spec.get("resources") or {},
        "service_account": (re_extra.get("service_account_name")
                            or re_extra.get("serviceAccountName")),
        "node_selector": re_extra.get("node_selector") or re_extra.get("nodeSelector") or {},
        "affinity": re_extra.get("affinity") or {},
        "tolerations": re_extra.get("tolerations") or [],
    }


# ---- 工件推导：扫描全部 env，按 URI scheme 分类 ----
def _norm_key(key):
    return "/".join(p for p in key.split("/") if p)


def _task_name(env_name):
    return re.sub(r"[^a-z0-9]+", "-", env_name.lower()).strip("-")


def build_tasks(meta):
    env = meta["env"]
    nvme = meta["nvme_mount"]
    s3mnt = meta["s3mnt_mount"]
    fsx = meta["fsx_mount"]
    tasks = []  # [(task_name, env_name, items)]  item=(kind, src, dst)
    seen = set()
    for en in sorted(env):
        items = []
        for tok in env[en].split():
            if tok.startswith("s3://"):
                body = tok[len("s3://"):]
                key = _norm_key(body.split("/", 1)[1] if "/" in body else "")
                if not key:
                    fatal(f"env {en} value {tok!r}: empty s3 key")
                if not nvme:
                    fatal(f"env {en} uses s3:// but spec.env.NVME_MOUNT is not set")
                if any(p == ".." for p in key.split("/")):
                    fatal(f"env {en} value {tok!r}: '..' path segment not allowed")
                src = "s3://" + _norm_key(body)
                items.append((_KIND_S3, src, f"{nvme.rstrip('/')}/{key}"))
            elif tok.startswith("s3mnt://"):
                key = _norm_key(tok[len("s3mnt://"):])
                if not key:
                    fatal(f"env {en} value {tok!r}: empty s3mnt key")
                if not nvme:
                    fatal(f"env {en} uses s3mnt:// but spec.env.NVME_MOUNT is not set")
                if not s3mnt:
                    fatal(f"env {en} uses s3mnt:// but no s3mnt source mount can be "
                          f"located: set spec.env.S3ALL_MOUNT, or keep exactly one "
                          f"PVC-backed volume in render-extra extra_volumes")
                if any(p == ".." for p in key.split("/")):
                    fatal(f"env {en} value {tok!r}: '..' path segment not allowed")
                items.append((_KIND_S3MNT, f"{s3mnt.rstrip('/')}/{key}",
                              f"{nvme.rstrip('/')}/{key}"))
            elif tok.startswith("fsx://"):
                path = _norm_key(tok[len("fsx://"):])
                if not path:
                    fatal(f"env {en} value {tok!r}: empty fsx path")
                if not fsx:
                    fatal(f"env {en} uses fsx:// but spec.env.FSX_MOUNT is not set")
                if any(p == ".." for p in path.split("/")):
                    fatal(f"env {en} value {tok!r}: '..' path segment not allowed")
                full = f"{fsx.rstrip('/')}/{path}"
                items.append((_KIND_FSX, full, full))
            # 非 URI token 不是工件，忽略
        if not items:
            continue
        name = _task_name(en)
        if name in seen:
            fatal(f"duplicate derived task name {name!r} (env {en})")
        seen.add(name)
        tasks.append((name, en, items))
    return tasks


# ---- 同步 Job 构造 ----
def derive_job(meta, *, name, task, kinds, srcs, dsts, sync_par, cpu_req, cpu_lim,
               mem_req, mem_lim, deadline, aws_creds_secret, run_id):
    need_s3mnt = any(k == _KIND_S3MNT for k in kinds)
    ctr = {
        "name": "sync",
        "image": meta["image"],
        "imagePullPolicy": "IfNotPresent",
        "command": ["bash", "-c", f"{SCRIPT_MOUNT}/sync_artifact.sh"],
        "env": [
            {"name": "NODE_NAME",
             "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
            {"name": "NVME_MOUNT", "value": meta["nvme_mount"]},
            {"name": "KINDS", "value": " ".join(kinds)},
            {"name": "SRCS", "value": " ".join(srcs)},
            {"name": "DSTS", "value": " ".join(dsts)},
            {"name": "SYNC_PAR", "value": str(sync_par)},
        ],
        "resources": {
            "requests": {"cpu": cpu_req, "memory": mem_req},
            "limits": {"cpu": cpu_lim, "memory": mem_lim},
        },
        "volumeMounts": [
            {"name": meta["nvme_volume"]["name"], "mountPath": meta["nvme_mount"]},
            {"name": SCRIPT_VOLUME, "mountPath": SCRIPT_MOUNT, "readOnly": True},
        ],
    }
    if aws_creds_secret:
        ctr["envFrom"] = [{"secretRef": {"name": aws_creds_secret}}]

    volumes = [
        copy.deepcopy(meta["nvme_volume"]),
        {"name": SCRIPT_VOLUME,
         "configMap": {"name": f"{SCRIPT_CONFIGMAP}-{run_id}",
                       "defaultMode": 0o755}},
    ]
    if need_s3mnt:
        volumes.append(copy.deepcopy(meta["s3mnt_volume"]))
        ctr["volumeMounts"].insert(0, {
            "name": meta["s3mnt_volume"]["name"],
            "mountPath": meta["s3mnt_mount"],
            "readOnly": True,
        })

    pod_spec = {
        "restartPolicy": "Never",
        "nodeSelector": dict(meta["node_selector"]),
        "affinity": copy.deepcopy(meta["affinity"]),
        "tolerations": copy.deepcopy(meta["tolerations"]),
        "volumes": volumes,
        "containers": [ctr],
    }
    if meta["service_account"]:
        pod_spec["serviceAccountName"] = meta["service_account"]

    aff = pod_spec.setdefault("affinity", {})
    aff["podAntiAffinity"] = {
        "requiredDuringSchedulingIgnoredDuringExecution": [{
            "labelSelector": {"matchLabels": {"sync-job": name}},
            "topologyKey": "kubernetes.io/hostname",
        }]
    }

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
            "completions": n,
            "parallelism": n,
            "completionMode": "Indexed",
            "backoffLimit": 0,
            "activeDeadlineSeconds": deadline,
            "template": {
                "metadata": {"labels": {"sync-job": name, "sync-task": task}},
                "spec": pod_spec,
            },
        },
    }


# ---- 派生结果自检 ----
def assert_pod_sanity(job, name, cpu_cap):
    spec = job["spec"]
    pod = spec["template"]["spec"]
    ctr = pod["containers"][0]
    res = ctr.get("resources") or {}

    for scope in ("requests", "limits"):
        block = res.get(scope) or {}
        if "cpu" not in block or "memory" not in block:
            fatal(f"{name}: resources.{scope} must declare both cpu and memory")
    req_cores = cpu_to_cores(res["requests"]["cpu"])
    lim_cores = cpu_to_cores(res["limits"]["cpu"])
    if lim_cores > cpu_cap or req_cores > cpu_cap:
        fatal(f"{name}: cpu exceeds cap {cpu_cap}")
    if req_cores > lim_cores:
        fatal(f"{name}: requests.cpu({req_cores}) > limits.cpu({lim_cores})")

    if pod.get("restartPolicy") != "Never":
        fatal(f"{name}: restartPolicy must be Never")
    if not (spec["completions"] == spec["parallelism"] >= 1):
        fatal(f"{name}: completions/parallelism invalid")
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
        fatal(f"{name}: podAntiAffinity(hostname) missing")

    mount_names = {m["name"] for m in ctr.get("volumeMounts") or []}
    vol_names = {v["name"] for v in pod.get("volumes") or []}
    if not mount_names <= vol_names:
        fatal(f"{name}: volumeMounts reference undefined volumes: "
              f"{sorted(mount_names - vol_names)}")


def sh_quote(v):
    return "'" + str(v).replace("'", "'\\''") + "'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--deadline-sec", type=int, default=3600,
                    help="每个同步任务的统一 activeDeadlineSeconds（默认 3600）")
    ap.add_argument("--task-deadline", action="append", default=[], metavar="TASK=SEC",
                    help="按任务名单独设置超时，可重复；TASK 为任务名或 env 名")
    ap.add_argument("--aws-creds-secret", default="",
                    help="同步 Pod envFrom 注入的 AWS 凭证 secret 名；不传则不注入")
    ap.add_argument("--cpu-cap", type=float, default=32)
    ap.add_argument("--mem-request-gi", type=float, default=8)
    ap.add_argument("--mem-limit-gi", type=float, default=32)
    args = ap.parse_args()

    overrides = {}
    for kv in args.task_deadline:
        if "=" not in kv:
            fatal(f"--task-deadline expects TASK=SEC, got {kv!r}")
        k, v = kv.rsplit("=", 1)
        overrides[k.strip()] = int(v)

    try:
        meta = parse(args.yaml)
    except SystemExit:
        raise
    except Exception as e:
        fatal(f"cannot parse {args.yaml}: {e}")

    tasks = build_tasks(meta)
    n_artifacts = sum(len(items) for _, _, items in tasks)
    if n_artifacts == 0:
        print("[derive][WARN] no artifacts (s3:// or fsx://) found in yaml env; "
              "no sync jobs derived", file=sys.stderr)

    res = meta["resources"] if isinstance(meta["resources"], dict) else {}
    train_req = res.get("requests") or {}
    train_lim = res.get("limits") or {}
    if "cpu" in res and "cpu" not in train_req:
        train_req = {"cpu": res.get("cpu"), "memory": res.get("memory")}
        train_lim = dict(train_req)
    cpu_req = _clamp_cpu(train_req.get("cpu"), args.cpu_cap, "requests.cpu")
    cpu_lim = _clamp_cpu(train_lim.get("cpu"), args.cpu_cap, "limits.cpu")
    if cpu_to_cores(cpu_req) > cpu_to_cores(cpu_lim):
        cpu_req = cpu_lim
    mem_req = _clamp_mem(train_req.get("memory"), args.mem_request_gi, "requests.memory")
    mem_lim = _clamp_mem(train_lim.get("memory"), args.mem_limit_gi, "limits.memory")
    sync_par = max(1, int(cpu_to_cores(cpu_lim)))

    os.makedirs(args.out, exist_ok=True)

    active_tasks, fsx_tasks = [], []
    fsx_artifacts = []
    for task, env_name, items in tasks:
        fsx_items = [it for it in items if it[0] == _KIND_FSX]
        sync_items = [it for it in items if it[0] != _KIND_FSX]
        fsx_artifacts.extend(src for _, src, _ in fsx_items)
        if not sync_items:
            fsx_tasks.append(task)
            continue

        deadline = overrides.get(task, overrides.get(env_name, args.deadline_sec))
        job_name = f"sync-{task}-{args.run_id}"
        if len(job_name) > 63 or not re.fullmatch(r"[a-z0-9-]+", job_name):
            fatal(f"derived job name {job_name!r} is not DNS-1123 or exceeds 63 chars")
        job = derive_job(
            meta, name=job_name, task=task,
            kinds=[k for k, _, _ in sync_items],
            srcs=[s for _, s, _ in sync_items],
            dsts=[d for _, _, d in sync_items],
            sync_par=sync_par,
            cpu_req=cpu_req, cpu_lim=cpu_lim, mem_req=mem_req, mem_lim=mem_lim,
            deadline=deadline,
            aws_creds_secret=args.aws_creds_secret or None,
            run_id=args.run_id,
        )
        assert_pod_sanity(job, task, args.cpu_cap)
        with open(os.path.join(args.out, f"sync_{task}.yaml"), "w") as f:
            yaml.safe_dump(job, f, sort_keys=False, default_flow_style=False,
                           allow_unicode=True)
        active_tasks.append((task, job_name, deadline))

    with open(os.path.join(args.out, "plan.env"), "w") as f:
        f.write(f"NAMESPACE={sh_quote(meta['ns'])}\n")
        f.write(f"NODES={sh_quote(' '.join(meta['nodes']))}\n")
        f.write(f"NODE_COUNT={meta['n']}\n")
        f.write(f"SYNC_TASKS={sh_quote(' '.join(t for t, _, _ in active_tasks))}\n")
        f.write(f"FSX_TASKS={sh_quote(' '.join(fsx_tasks))}\n")
        f.write(f"FSX_ARTIFACTS={sh_quote(' '.join(fsx_artifacts))}\n")
        f.write(f"FSX_MOUNT={sh_quote(meta['fsx_mount'] or '')}\n")
        f.write(f"ARTIFACT_COUNT={n_artifacts}\n")
        f.write(f"SYNC_PAR={sync_par}\n")
        f.write(f"SYNC_CPU_REQUEST={sh_quote(cpu_req)}\n")
        f.write(f"SYNC_CPU_LIMIT={sh_quote(cpu_lim)}\n")
        for task, job_name, deadline in active_tasks:
            # 任务名含连字符（如 audio-shards），不能直接做 shell 变量名后缀；
            # 统一转下划线。submit.sh 消费侧用同样的替换拼接变量名。
            var = task.replace("-", "_")
            f.write(f"WAIT_TIMEOUT_{var}={deadline + WAIT_BUFFER_SEC}\n")
            f.write(f"JOB_NAME_{var}={sh_quote(job_name)}\n")

    print(f"[derive] ns={meta['ns']} nodes={meta['n']} "
          f"cpu={cpu_req}/{cpu_lim} par={sync_par} artifacts={n_artifacts}")
    for task, job_name, deadline in active_tasks:
        print(f"[derive] sync task: {task} (job={job_name}, deadline={deadline}s)")
    for task in fsx_tasks:
        print(f"[derive] fsx-direct task (no sync job): {task}")


if __name__ == "__main__":
    main()
