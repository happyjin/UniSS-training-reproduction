"""--hard-only (2026-11-08): the default Sampler stream is unchanged (compared with the previous train_w3po, argv[1]);
--hard-only returns only hard windows, without repeats within a pass.
    python -m experiments.mist_w3po_v1.rl.test_hard_only <old train_w3po.py>
"""
import importlib.util
import json
import sys

from experiments.mist_w3po_v1.rl.train_w3po import Sampler, hard


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def main():
    spec = importlib.util.spec_from_file_location("old_tw", sys.argv[1])
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    W = [json.loads(l) for l in open("eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_train_diar.jsonl")]
    W = [w for w in W if w.get("utts") and w["dur_ms"] <= 20500]
    a, b = OLD.Sampler(W, 20261107), Sampler(W, 20261107)
    check("default stream identical (steps 0-60, 128/step)",
          all([w["id"] for w in a.take(s, 128)] == [w["id"] for w in b.take(s, 128)] for s in range(60)))
    h = Sampler(W, 20261107, hard_only=True)
    ids = [w["id"] for s in range(50) for w in h.take(s, 128)]
    check("hard-only: all hard", all(hard(w) for s in range(3) for w in h.take(s, 128)))
    check("hard-only: no repeats within 50 steps x 128", len(ids) == len(set(ids)))
    print("ALL PASS")


if __name__ == "__main__":
    main()
