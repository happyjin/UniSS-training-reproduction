# Stage00: input audit

Run a deterministic sampled audit of the immutable 15-shard Stage-A manifest:

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python \
  experiments/uniss_streamspeech_ctc_v1/stage00_audit/audit_inputs.py \
  --output experiments/uniss_streamspeech_ctc_v1/stage00_audit/results/input_audit.json
```

The audit checks the offset index, required fields, direction/language
consistency, duration and GLM geometry, and a bounded sample of reconstructed
audio paths.  It is read-only.

