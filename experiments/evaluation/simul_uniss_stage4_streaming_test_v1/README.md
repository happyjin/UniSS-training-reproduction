# Simul-UniSS Stage4 end-to-end streaming UniST test evaluation v1

This experiment is isolated from the running 7,965-record dev evaluation.  It
builds the 23,369-record UniST test pseudo-streaming schedule with the frozen dev
operating point and runs the exact Stage4 checkpoint on GPU 4–7.

```text
chunk_ms=640
wait_k_chunks=2
max_phrase_tokens=16
greedy decoding, repetition_penalty=1.1
streaming BiCodec left_context=50, holdback=5, overlap=80ms
GPU data parallel=4,5,6,7
```

Prepare/verify schedules without using a GPU:

```bash
experiments/evaluation/simul_uniss_stage4_streaming_test_v1/prepare_test_schedules.sh
```

Run the complete isolated test evaluation:

```bash
experiments/evaluation/simul_uniss_stage4_streaming_test_v1/run_full_test_4gpu.sh
```

Generation keeps 512 active records per GPU because the measured 1,024-record
configuration was slower.  The runner records real utilization and power and
never duplicates computation merely to make H200 power readings look larger.
