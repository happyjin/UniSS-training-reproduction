# Periodic AR eval callback tests

CPU contract test:

```bash
/opt/dlami/nvme/zeyangsong/miniconda/envs/liveinterpret/bin/python \
  -m unittest test_script.ar_eval_callback.test_ar_eval_callback
```

The test covers deterministic direction/length-stratified val selection and
the scalar EOS/length/text aggregation logged by the training callback.  A
real one-GPU smoke can use the WDS training entry with
`AR_EVAL_STEPS=1 AR_EVAL_NUM_SAMPLES=1 AR_EVAL_MAX_CODEC_FRAMES=8`.
