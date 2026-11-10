"""talker_quality refuses a directory that already holds a run's files (sim_o writes next to its output)."""
import subprocess
import sys
import tempfile
from pathlib import Path


def test_refuse():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "per_sample_sim_o.jsonl").write_text("")
        r = subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.talker.talker_quality", "--talker", d,
                            "--items", f"{d}/x", "--out", f"{d}/quality.json"], capture_output=True, text=True)
        assert r.returncode != 0 and "拒绝覆盖" in r.stderr + r.stdout


if __name__ == "__main__":
    test_refuse()
    print("ok")
