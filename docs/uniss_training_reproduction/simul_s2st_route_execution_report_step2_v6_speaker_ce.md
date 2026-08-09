# Simul-S2ST route — Step2 v6 speaker + CE-only result

> 2026-08-09

## Settings

- 15-shard, mbs=64 / gbs=512, 3000 iters
- `ctc_weight=0`, `guided_ce_weight=1.0`, `blank_penalty=0.5`, lr=5e-4 constant
- Speaker: BiCodec-global embedding added to every NAR frame

## Train

| Metric | Start | End (valid) |
| --- | ---: | ---: |
| guided_ce | ~11.5 | ~8.61 (still ≈ log V) |
| blank_mass | ~1e-4 | ~0 |

## Decode (iter3000, 32 samples)

| Metric | v5 | **v6** |
| --- | ---: | ---: |
| Empty preds | 0 | **0** |
| Blank frames | ~68% | **0%** |
| Pred units (mean) | ~6 | **~19** |
| UER | ~99.7% | **~99.3%** |
| best nonblank prob | 0.021 | 0.015 |

## Reading

1. **Blank collapse is solved** under CE-only + speaker (argmax never blank).
2. **Content is still wrong** — guided CE plateaus near uniform over 8192 classes; text+speaker alone does not specify target BiCodec units.
3. Next: **v7 source-GLM conditioning** (acoustic discrete codes already in the joint manifest) while keeping CE-only warm.

## Artifacts

- ckpt: `checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v6_speaker_ce/`
- decode: `reports/simul_s2st_route_v1/step2_trained_nar_decode_v6_speaker_ce.md`
