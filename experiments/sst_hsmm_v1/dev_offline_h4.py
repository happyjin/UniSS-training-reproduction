"""plan v3 (change 56) H4 precheck 2: dev_offline2 (whole-audio greedy decodes of the wave2_S1 dev items, model $OFF_MODEL) with the
H4 mono tag appended to the translation instruction (h1_tags.TAG_M), as H4 is served at inference.
    OFF_MODEL=<h4 merged dir> <vllm python> -m experiments.sst_hsmm_v1.dev_offline_h4 gen --direction d ..."""
import os
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.training import e2_batch as EB  # noqa: E402
from experiments.sst_hsmm_v1.h1_tags import TAG_M  # noqa: E402
_orig = EB.inference_instruction
EB.inference_instruction = lambda *a, **k: {t: v + TAG_M for t, v in _orig(*a, **k).items()}
from experiments.sst_hsmm_v1 import offline_ceiling as OC  # noqa: E402
OC.TDATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
OC.WAVE17 = os.environ["OFF_MODEL"]
if __name__ == "__main__":
    OC.main()
