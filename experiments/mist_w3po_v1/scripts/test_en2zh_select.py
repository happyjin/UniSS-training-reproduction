"""Test of the fixed selection rule (en2zh_select.pick; CPU).

    PYTHONPATH=. python experiments/mist_w3po_v1/scripts/test_en2zh_select.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from en2zh_select import pick  # noqa: E402


def r(step, Q, O, L):
    return {"step": step, "Q": Q, "O": O, "L": L}


def main() -> None:
    # highest Q among the feasible (L <= L(last) + 0.1)
    rows = [r(500, 20.0, 15.0, 3.0), r(1000, 23.2, 16.0, 3.2), r(1500, 23.5, 17.0, 3.45), r(3000, 23.0, 18.0, 3.3)]
    assert pick(rows)["chosen"] == 1000 and pick(rows)["feasible"] == [500, 1000, 3000]
    # boundary: L exactly last + 0.1 is feasible
    rows = [r(500, 24.0, 15.0, 3.4), r(3000, 23.0, 18.0, 3.3)]
    assert pick(rows)["chosen"] == 500
    # tie on Q at 2 decimals -> higher O
    rows = [r(500, 23.004, 17.0, 3.0), r(1000, 22.996, 17.5, 3.0), r(3000, 22.0, 18.0, 3.0)]
    assert pick(rows)["chosen"] == 1000
    # tie on Q and O -> lower L, then the later step
    rows = [r(500, 23.0, 17.0, 3.1), r(1000, 23.0, 17.0, 3.0), r(3000, 22.0, 18.0, 3.05)]
    assert pick(rows)["chosen"] == 1000
    rows = [r(500, 23.0, 17.0, 3.0), r(1000, 23.0, 17.0, 3.0), r(3000, 22.0, 18.0, 3.0)]
    assert pick(rows)["chosen"] == 1000
    # the last step is always feasible
    rows = [r(400, 10.0, 10.0, 9.0), r(1200, 9.0, 9.0, 5.0)]
    assert pick(rows)["chosen"] == 1200
    print("ok")


if __name__ == "__main__":
    main()
