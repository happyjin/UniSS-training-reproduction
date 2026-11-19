"""Idea 2 e2e test (prereg addendum 2026-11-19): talk-job wrapper that runs the M2 voice-override path UNCHANGED except
that si_infer's --model (T1c) is replaced by the e2e Talker export (LoRA merged into a copy of T1c).
    <python> -m experiments.mist_w3po_v1.voicecond.idea2_e2e_talker --talker-model <export>/uniss --via bench|vo \
        --voice-override <table> <si_infer arguments, --model = T1c>
--via bench -> experiments.mist_w3po_v1.voicecond.bench_talker_override (OpenBroadcast-SI sidecar hook)
--via vo    -> experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo (sha1-checked token table, OOD-3)
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

T1C = "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
VIA = {"bench": "experiments.mist_w3po_v1.voicecond.bench_talker_override",
       "vo": "experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo"}


def main() -> None:
    argv = sys.argv[1:]
    i = argv.index("--talker-model")
    model = argv[i + 1]
    del argv[i:i + 2]
    i = argv.index("--via")
    via = argv[i + 1]
    del argv[i:i + 2]
    assert (Path(model) / "model.safetensors").exists() and (Path(model).parent / "export.json").exists(), model
    i = argv.index("--model")
    assert argv[i + 1].rstrip("/").endswith(T1C), f"expected the T1c --model, got {argv[i + 1]}"
    argv[i + 1] = model
    print(f"[idea2 e2e talker] --model {model} (was T1c) via {VIA[via]}", flush=True)
    sys.argv = [VIA[via]] + argv
    runpy.run_module(VIA[via], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
