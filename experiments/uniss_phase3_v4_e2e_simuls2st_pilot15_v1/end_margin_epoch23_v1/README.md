# END-margin epochs 2--3 continuation

This isolated research continuation starts from the completed one-coverage
END-margin checkpoint and adds exactly two more coverage epochs over the same
immutable 15-shard task pool.  It does not overwrite the epoch-1 run and it
does not authorize formal Phase-B training.

The continuation is justified by the identical fixed-16 free-running gate:

| metric | 100-update baseline | epoch 1 |
|---|---:|---:|
| CMN ASR error rate | 0.2066 | 0.1174 |
| ENG ASR error rate | 0.4710 | 0.3097 |
| gold-source MT coverage mean | 0.1446 | 0.1980 |
| free-source MT coverage mean | 0.1077 | 0.1775 |
| malformed S2S segments | 27 | 10 |
| non-silent S2S samples | 8/8 | 7/8 |

The positive ASR, MT and structural trends justify more optimization, while
the one silent sample requires checkpoint selection rather than blindly using
the last update.  Therefore both added epoch boundaries are retained and
evaluated independently with the same fixed-16 selection and 384 semantic
token cap.

Training geometry:

```text
parent checkpoint = epoch-1 iter_0001132
additional epochs = 2
updates           = 2264
epoch-2 boundary  = continuation iter_0001207
epoch-3 boundary  = continuation iter_0002264
MBS / GBS         = 2 / 128
sequence          = 18000
GPUs              = 8
shuffle seed      = 20260819
```

The epoch-2 boundary is defined by the 314th additional interleaved E2E block,
not by naively dividing 2264 updates in half.  Under the phase-weighted global
schedule this occurs at update 1207.  The parent weights are loaded with Megatron finetune semantics and a fresh
two-epoch optimizer/scheduler.  This is a weight continuation, not an exact
resume of the completed one-epoch cosine scheduler.  The objective remains
unchanged:

```text
semantic END CE weight       = 0.50
semantic END margin weight   = 0.25
semantic END logit margin    = 2.00
all roll-in/continue/binary  = 0.00
```

Launch with fresh immutable IDs:

```bash
RUN_ID=endmargin_epoch23_$(date -u +%Y%m%dT%H%M%SZ) \
  bash experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/\
end_margin_epoch23_v1/launch_tmux.sh
```

TensorBoard is served on port 6045 by default.  The post-training waiter
validates frozen Stage-A parameters, exports both epoch boundaries and runs
the fixed-16 free-running gate for each checkpoint.
