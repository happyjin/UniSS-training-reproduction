"""Registered decision rules of reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md (CPU, read-only on the inputs).

  readperiod  §T4 读周期: T1-ctrl (A' Thinker, 0.5 s reads, full dev) text all >= A' - 0.2 and ov >= A' - 0.2 and its
              final-read word share (full dev) <= A''s -> 500 ms, else 1000 ms. With --write: logs/mist_w3po_v1/t4/
              read_period.txt (noclobber).
  rulec       rule C (user-approved): T2's dev5 cut: final-read word share > 49.7 % or the never-start (class A) share
              of the >= 6 s windows (t1_stall_diag) >= 28 % -> triggered.
    python t_decide.py readperiod [--write] | rulec
"""
import json
import re
import sys
from pathlib import Path

DEC = Path("eval_outputs/mist_w3po_v1/decode")
TE = Path("eval_outputs/mist_w3po_v1/crf/t_eval")
APRIME = "dev_crf_v2p_s3000_finalpost_s8_c2g06"
CTRL = APRIME + "_w05ctrl"
T2 = "dev_t2_latent_t0_s3000_w05"
A_FULL_FINAL = 0.5087      # final_read_share.py on A''s full dev decode (61,596 words)


def score(name):
    d = json.loads((DEC / name / "score_text.json").read_text())
    return d["tcpBLEU_all"], d["tcpBLEU_ov"]


def stall(path):
    """(>= 6 s windows: class counts) of the first block (the candidate) of t1_stall_diag's output."""
    line = Path(path).read_text().splitlines()[0]
    d = json.loads(line[line.index("{"):])
    return d["by_dur"].get(">=6s", {}), d


def main():
    mode = sys.argv[1]
    if mode == "readperiod":
        a_all, a_ov = score(APRIME)
        c_all, c_ov = score(CTRL)
        fs = json.loads((TE / f"dev5cut_{CTRL}" / "final_share_fulldev.json").read_text())["final_share"]
        ok = c_all >= a_all - 0.2 and c_ov >= a_ov - 0.2 and fs <= A_FULL_FINAL
        res = {"aprime": [a_all, a_ov, A_FULL_FINAL], "t1ctrl": [c_all, c_ov, fs], "read_period_ms": 500 if ok else 1000}
        print(json.dumps(res))
        if "--write" in sys.argv:
            with open("logs/mist_w3po_v1/t4/read_period.txt", "x") as fh:
                fh.write(f"{res['read_period_ms']}\n")
            with open("logs/mist_w3po_v1/t4/read_period_decision.json", "x") as fh:
                fh.write(json.dumps(res, indent=1))
    elif mode == "rulec":
        o = TE / f"dev5cut_{T2}"
        fs = json.loads((o / "final_share_dev5.json").read_text())["final_share"]
        b, full = stall(o / "stall_diag.txt")
        n = sum(b.values())
        ns = b.get("A", 0) / n if n else 0.0
        trig = fs > 0.497 or ns >= 0.28
        print(json.dumps({"t2_dev5_final_share": fs, "t2_ge6s_never_start": round(ns, 4), "ge6s": b,
                          "triggered": trig, "stall": full}))
    else:
        raise SystemExit(mode)


if __name__ == "__main__":
    main()
