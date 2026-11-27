"""change 57 G0 (B2): whole-audio greedy decodes of the wave2_S1 dev items with model $OFF_MODEL and the PLAIN translation prompt
(no tag), i.e. the prompt the B2 adapter is served with on non-final reads (hsc22 --draft-lora without --draft-tag).
    OFF_MODEL=<merged dir> <vllm python> -m experiments.sst_hsmm_v1.dev_offline_plain gen --direction d ..."""
import os
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import offline_ceiling as OC  # noqa: E402
OC.TDATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
OC.WAVE17 = os.environ["OFF_MODEL"]
if __name__ == "__main__":
    OC.main()
