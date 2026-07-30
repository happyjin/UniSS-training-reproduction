# Validation record

## CPU/module validation

Run:

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo/bin/python \
  -m unittest discover -s web_demo/streaming_s2st_r2_v1/tests -v
```

The tests cover frozen model assets, audio validation/resampling, session
isolation, Stage4 prompt/action/write parsing, timeline rendering and public
access metadata.

## GPU upload smoke

Validated on physical GPU1 with R2 step 300 and the existing source sample
`magicdata_0000000001` (5.46 s, Chinese to English):

```text
translation = I want to search for text messages in Baidu.
translation audio = 5.40 s
server inference after load = 7.41 s
forced actions = 0
structural recoveries = 0
max prompt tokens = 421
CUDA OOM = 0
```

The smoke output is under the new directory's gitignored `runtime_outputs/` and
does not overwrite formal evaluation results.
