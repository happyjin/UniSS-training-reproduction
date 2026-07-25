# Stage 7 — GRPO policy bootstrap

`run_8gpu.sh` runs the existing WAIT/WRITE policy-head GRPO bootstrap on eight
GPUs. The bounded shuffle stream is deterministically partitioned across ranks,
DDP synchronizes gradients, metrics are averaged globally, and only rank 0
writes TensorBoard/checkpoints.

Important scope: this is an engineering bootstrap for reward, rollout, and
distributed-data validation. It is not yet full Qwen-token GRPO initialized from
the Stage 6 Megatron checkpoint. The latter remains a separate formal research
adapter and must not be claimed from this bootstrap result.

