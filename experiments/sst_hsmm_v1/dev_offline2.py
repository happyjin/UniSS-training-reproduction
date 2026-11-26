"""change 47+: dev_offline.py (offline_ceiling.gen on the wave2_S1 dev items, whole audio, greedy) with the model taken from
$OFF_MODEL instead of wave17.   OFF_MODEL=<merged dir> <vllm python> -m experiments.sst_hsmm_v1.dev_offline2 gen --direction d ..."""
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
