"""Simulation of takeover_4_6.py (no GPU, no process started): the trigger waits for the pipeline to stay gone for
GONE_S, then writes cards.json (all 8 ours, normal headroom), asks for keepers on 4-6 and leaves a done marker;
cardctl.load_cfg then returns all 8 cards.

    python experiments/mist_w3po_v1/scripts/test_takeover_4_6.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cardctl  # noqa: E402
import takeover_4_6 as T  # noqa: E402

q = Path(tempfile.mkdtemp(prefix="test_takeover_", dir="/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1"))
assert cardctl.load_cfg(q)["ours"] == (0, 1, 2, 3, 7) and cardctl.load_cfg(q)["headroom"][5] == 25.0
clock = {"t": 0.0}
seq = iter([[11, 12], [11], [], [], [], [], []])     # pipeline present twice, then gone
T.pipeline_pids = lambda: next(seq, [])
T.time.time = lambda: clock["t"]
T.time.sleep = lambda s: clock.__setitem__("t", clock["t"] + 60)
started = []
T.Card.start_keeper = lambda self, wait_s=90: started.append(self.c) or 1
sys.argv = ["takeover_4_6.py", str(q)]
T.main()
cfg = cardctl.load_cfg(q)
assert cfg["ours"] == tuple(range(8)) and cfg["headroom"] == {}, cfg
assert started == [4, 5, 6], started
assert (q / "takeover_4_6.done").exists()
assert clock["t"] >= 2 * 60 + T.GONE_S - 60, clock   # not before the pipeline stayed gone GONE_S
print("TAKEOVER_TESTS_OK")
