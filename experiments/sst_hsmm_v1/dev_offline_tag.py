"""en2zh analysis after change 56 (report only): whole-audio greedy decodes of the wave2_S1 dev items with model $OFF_MODEL and
the complete-audio tag p2ft_tags.TAG_C appended to the translation instruction, i.e. exactly the final-read prompt E1/H1/H3/E1+D1
used at test time (hsc22 --draft-tag). Compared with wave17 (devoff) it isolates the drafter's whole-audio translation quality from
the commit policy.
    OFF_MODEL=<merged dir> <vllm python> -m experiments.sst_hsmm_v1.dev_offline_tag gen --direction d ..."""
import os
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.training import e2_batch as EB  # noqa: E402
from experiments.sst_hsmm_v1.p2ft_tags import TAG_C  # noqa: E402
_orig = EB.inference_instruction
EB.inference_instruction = lambda *a, **k: {t: v + TAG_C for t, v in _orig(*a, **k).items()}
from experiments.sst_hsmm_v1 import offline_ceiling as OC  # noqa: E402
OC.TDATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
OC.WAVE17 = os.environ["OFF_MODEL"]
if __name__ == "__main__":
    OC.main()
