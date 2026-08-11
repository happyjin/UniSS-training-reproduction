# True-subsecond pilot15 dataset regression v1

This isolated folder tests the repaired pilot15 runtime on exact UniST rows
from the training scope (`train-00000`, `train-00002`) and the independent
UniST dev split.  It never edits the source parquet files, checkpoints, or
historical demo outputs.

Each result directory contains:

- `source.wav`: BiCodec reconstruction used as streaming input;
- `reference_target.wav`: dataset target BiCodec reconstruction;
- `streaming_translation.wav`: repaired pilot15 output when the quality gate
  permits audio (absent when the gate rejects all output);
- `streaming_timeline.wav`: emitted audio on its real source-time timeline
  (absent when the gate rejects all output);
- `streaming_stereo.wav`: left source / right streaming translation; the right
  channel is silent when the safety gate rejects all output;
- `offline_phase3.wav`: Phase3-v4 quality-mode control output;
- `metadata.json`: references, generated text and detailed metrics.

Run on one idle GPU without stopping the public demo:

```bash
CUDA_VISIBLE_DEVICES=1 \
  /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo/bin/python \
  eval_outputs/true_subsecond_pilot15_dataset_regression_v1/run_regression.py \
  --device cuda:0 --chunk-ms 640
```

The script creates a timestamped result directory and refuses to overwrite an
existing run.
