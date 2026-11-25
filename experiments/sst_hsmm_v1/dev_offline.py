"""Dev diagnostic: offline_ceiling.gen (wave17, whole audio, greedy) on the dev items (wave2_S1 data) instead of test_robust.
    <vllm python> -m experiments.sst_hsmm_v1.dev_offline gen --direction d --shard k --num-shards n --out <jsonl>"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import offline_ceiling as OC  # noqa: E402
OC.TDATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
if __name__ == "__main__":
    OC.main()
