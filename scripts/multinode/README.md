# Multi-node training on this HyperPod cluster

Node shape (verified on `i-049e0c04174ce5eb8`): `p5e.48xlarge`, 8x H200 fully
NVLink-meshed, 32 EFA ports, SageMaker HyperPod EKS node in `ap-southeast-3a`.
`torch 2.6.0+cu124` / NCCL 2.21.5, aws-ofi-nccl at `/opt/amazon/ofi-nccl/lib64`.

## The launcher does not rewrite the training scripts

`torchrun` reads `PET_NNODES` / `PET_NODE_RANK` / `PET_MASTER_ADDR` as defaults
for any flag the inner script leaves off its command line. 67 of the repo's 82
torchrun scripts (including `scripts/train_phase{1,2,3}*.sh`) omit exactly those
flags, so they go multi-node unchanged. The 15 that hardcode
`--nnodes 1 --node-rank 0 --master-addr 127.0.0.1` (the `simul_uniss_subsecond_*`
and `stage7a_*` families) need this three-line edit first:

```diff
-  --nnodes 1 --node-rank 0 --nproc-per-node "${nproc}" \
-  --master-addr 127.0.0.1 --master-port "${STAGE_B_MASTER_PORT}" \
+  --nnodes "${NNODES:-1}" --node-rank "${NODE_RANK:-0}" --nproc-per-node "${nproc}" \
+  --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT:-${STAGE_B_MASTER_PORT}}" \
```

## Usage

```bash
# 0. once, before the first real job
NODELIST="10.1.6.203 10.1.6.77" scripts/multinode/preflight_nccl.sh

# 1. fan out from one node over ssh
NODELIST="10.1.6.203 10.1.6.77" \
  scripts/multinode/launch_multinode.sh scripts/train_phase1.sh --train-iters 20

# 1'. or start each node yourself (k8s PyTorchJob, tmux, whatever)
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.1.6.203 \
  scripts/multinode/launch_multinode.sh scripts/train_phase1.sh
```

## Storage is the real prerequisite

`data/` (12T) and `checkpoints/` (11T) live on `/opt/dlami/nvme`, which is
node-local ephemeral NVMe at 98% full. `UniSSPackedJsonlDataset` is a map-style
dataset indexed globally by Megatron's sampler, so **every rank must be able to
read the whole packed jsonl** — and with `--use-distributed-optimizer` every DP
rank writes its own optimizer shard into `--save`. Both need one filesystem all
nodes see: FSx for Lustre (S3-linked) is the fit. Until that exists, a second
node has nothing to read and nowhere to checkpoint.
