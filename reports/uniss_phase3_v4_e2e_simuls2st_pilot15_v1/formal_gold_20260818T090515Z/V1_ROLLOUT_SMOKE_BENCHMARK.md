# V1 append-only rollout smoke and throughput benchmark

## Outcome

The persistent-KV V1 ASR rollout path is authorized for the formal train/valid cache build.

- Immutable gold/rollout alignment passed on every smoke.
- Accepted text is append-only; rollback count is zero.
- Qwen history uses persistent `past_key_values`; only new acoustic embeddings and newly generated tokens are appended.
- V1 native checkpoint fingerprint: `463ff5645ee3776f2c58343d4720cfb5beb55295972b68dd9f34cc48119fd730`.
- V1 HF export fingerprint: `e1089b6afaaf56babac717ed9a28d559a3d5b4a8d8c011354ad4a013f99772db`.
- Runtime fingerprint: `dadee468b61b9af35215c46dcb6a6015791b989299de7017734938510f93ba72`.

## Correctness results

| smoke | source language | samples | events | metric | weighted error rate | malformed WRITE | early EOS | final EOS | rollback |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| one worker/GPU | English | 32 | 542 | WER | 0.6267 | 1.67% | 1.11% | 100% | 0 |
| four workers/GPU | English | 128 | 1,767 | WER | 0.6133 | 3.38% | 0.62% | 100% | 0 |
| eight workers/GPU | English | 256 | 3,304 | WER | 0.5703 | 4.39% | 0.48% | 100% | 0 |
| sixteen workers/GPU | English | 512 | 6,494 | WER | 0.5338 | 4.19% | 0.28% | 100% | 0 |
| twenty-four workers/GPU | English | 768 | 9,632 | WER | 0.5253 | 4.16% | 0.25% | 100% | 0 |
| ranged Chinese smoke | Chinese | 64 | 1,355 | CER | 0.2870 | 0.09% | 0.07% | 100% | 0 |

The WER values in different rows must not be interpreted as a parallelism comparison because each larger run includes more records. Quality invariance was checked on overlapping records after removing only the nondeterministic `elapsed_seconds` field:

- 1 versus 4 workers/GPU: 32/32 generated rollouts exactly equal.
- 4 versus 8 workers/GPU: 128/128 exactly equal.
- 8 versus 16 workers/GPU: 256/256 exactly equal.
- 16 versus 24 workers/GPU: 512/512 exactly equal.

Thus process concurrency changes throughput only; it does not change generated tokens, text, grammar flags, EOS flags, or error counts.

## Throughput and GPU benchmark

The estimate uses the 1,325,243-record formal training split and steady aggregate throughput `records / maximum_worker_seconds`.

| processes/GPU | total workers | samples | samples/s | estimated full train time | mean SM | peak SM | mean power | peak power | peak framebuffer/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 32 | 1.516 | 242.9 h | 7.5% | 24% | 121.2 W | 133 W | 3.3 GiB |
| 4 | 32 | 128 | 4.945 | 74.4 h | 37.8% | 99% | 130.3 W | 154 W | 13.3 GiB |
| 8 | 64 | 256 | 8.489 | 43.4 h | 55.7% | 99% | 137.6 W | 165 W | 26.6 GiB |
| 16 | 128 | 512 | 11.501 | 32.0 h | 54.2% | 100% | 138.7 W | 176 W | 53.1 GiB |
| 24 | 192 | 768 | 14.908 | 24.7 h | 57.2% | 100% | 144.1 W | 181 W | 79.0 GiB |

Short-run mean utilization includes staggered model/checkpoint loading and worker teardown. During the 16-worker/GPU run, live snapshots showed 78–100% GPU utility on most cards. A formal run keeps workers active for many hours, so startup dilution becomes negligible.

Power does not approach the 700 W training limit even when instantaneous utility reaches 100%. This workload is a 0.5B autoregressive model decoding one token at a time with many short kernels; it does not have the large batched GEMMs of Phase3 training. Adding unrelated synthetic matrix multiplication would raise watts but slow the real rollout and invalidate the throughput measurement. The correct optimization is independent real-sample concurrency, not synthetic load.

## Selected formal configuration

- GPUs: 8 H200.
- Processes per GPU: 24.
- Total independent workers: 192.
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
- Input: complete immutable train split followed automatically by complete immutable valid split.
- Expected train rollout time: approximately 25 hours under smoke throughput; operational allowance 25–30 hours.
- Expected valid rollout time: approximately 15 minutes, plus model startup and merge/audit time.
- Peak smoke framebuffer: 79.0 GiB/GPU, leaving about 61 GiB/GPU headroom.

## Resolved smoke incident

The first 32-record attempt stopped before producing any rollout because cached frontend hidden states were stored as FP32 for audit while the trained V1 bridge LayerNorm was BF16. The bridge input now explicitly returns to the bridge parameter dtype before LayerNorm. A regression test covers this boundary. The successful run used a new run ID; no failed output was merged or reused.

## Formal gate meaning

This benchmark authorizes V1 rollout generation, not final joint training. `formal_training_authorized` remains false until the complete V1 rollout and the independent Phase3 teacher cache are both built and audited.
