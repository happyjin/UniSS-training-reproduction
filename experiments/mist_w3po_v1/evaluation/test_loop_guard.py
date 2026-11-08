"""Unit tests for evaluation/loop_guard.py:  python -m experiments.mist_w3po_v1.evaluation.test_loop_guard"""
from experiments.mist_w3po_v1.evaluation.loop_guard import first_periodic_run, guard


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def main():
    A, B, C = 11, 12, 13
    check("no run", first_periodic_run([1, 2, 3, 4, 5, 6], 3, 4) == (-1, 0))
    check("p=2 run found", first_periodic_run([9, A, B, A, B, A, B, 7], 3, 4) == (1, 2))
    loop = [5, 6] + [A, B] * 50
    new, why = guard(loop, [], 6, 12, None)
    check("loop cut after first copy", new == [5, 6, A, B] and why.startswith("loop"))
    new, why = guard([A, B, C, 1, 2], [], 6, 12, None)
    check("normal block untouched", new == [A, B, C, 1, 2] and why == "")
    new, why = guard([A, B] * 3, [9] + [A, B] * 4, 6, 12, None)
    check("loop continuing from earlier reads is cut", new == [] and why.startswith("loop"))
    new, why = guard([1, 2, 3], [7] * 0, 0, 12, 2)
    check("budget cut", new == [1, 2] and why == "budget")
    new, why = guard([1, 2, 3], [], 0, 12, None)
    check("guard off: unchanged", new == [1, 2, 3] and why == "")
    yeah = [A, B, C] * 3     # three legit repeats (k=6 not reached)
    check("short legit repetition kept", guard(yeah, [], 6, 12, None)[0] == yeah)
    print("ALL PASS")


if __name__ == "__main__":
    main()
