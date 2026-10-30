"""Paper v5 (new server, shared with another user's jobs and the single-speaker branch): gstar_multi run files whose
worker count per card follows the card's free memory at launch -- one Thinker copy (~70 GB) plus ~6 GB per worker,
the card kept under CAP MiB. Shards of a system are spread round-robin over the cards that run its Thinker.
  phase "p2": Zh->En S2-dl 3k at read intervals 1 / 0.5 / 2 s on AMI and AliMeeting at 1 s (AliMeeting weighted x2:
              its windows decode about twice as slowly), on CARDS_P2
  phase "p5": the diarizer-defined En->Zh windows (871) for O2, O2 without speaker stream and the same-data cascade's
              segment decode, O2 on the first card of CARDS_P5, the no-stream Thinker on the second
  phase "p4": the mixed-interval En->Zh S2-dl-m at 0.5 / 1 / 2 s on CARDS_P4
Writes reports/gstar_si_paper_v1/jobs/v5_runs/<phase>_card<c>.json (mode "x") and prints "<card> <file>" per card.

    python3 reports/gstar_si_paper_v1/make_v5_runs.py p2 0,1,2,3,4,5
"""
import json
import subprocess
import sys
from pathlib import Path

EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v5_20261030"
D = "data/ami_multispk_si_v1"
S2DLZ = f"{EV}/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged"
S2DLM = f"{EV}/thinker_gstar_s2dlm_3k_20261030/thinker_merged"
O2 = f"{EV}/thinker_gstar_o2_20261029b/thinker_merged"
O2N = f"{EV}/thinker_gstar_o2nosd_20261029b/thinker_merged"
WIN = {"en2zh": f"{D}/e2e_gstar_stream_v1/windows_test_ref.jsonl", "zh2en": f"{D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
       "ali": "data/alimeeting_si_v1/windows_test_ref.jsonl", "diar": f"{D}/e2e_gstar_v1/windows_test_diar_offline.jsonl"}
J = Path("reports/gstar_si_paper_v1/jobs/v5_runs")
CAP, THINKER, WORKER, MAXW = 134000, 70000, 6000, 8


def capacity(card):
    used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                        "-i", str(card)]).decode().strip())
    return max(0, min(MAXW, (CAP - used - THINKER) // WORKER))


def online(name, thinker, win, d, w, n):
    return [{"module": "gstar_si_stream_online",
             "argv": ["--thinker", thinker, "--windows", win, "--out", f"{OUT}/{name}/hyp_s{k}.jsonl", "--shard", str(k),
                      "--num-shards", str(n), "--policy", "wait", "--window", w, "--post-key", "lowlat", "--dir", d],
             "log": f"{OUT}/{name}/infer_s{k}.log"} for k in range(n)]


def offline(name, thinker, win, extra, n):
    return [{"module": "gstar_si_stream",
             "argv": ["--thinker", thinker, "--windows", win, "--out", f"{OUT}/{name}/hyp_s{k}.jsonl", "--shard", str(k),
                      "--num-shards", str(n), "--inst", "gstar", "--post-key", "offline", "--dir", "en2zh"] + extra,
             "log": f"{OUT}/{name}/infer_s{k}.log"} for k in range(n)]


def spread(runs, cards, caps):
    """Round-robin over cards, at most caps[c] runs each; fails if the runs do not fit."""
    per = {c: [] for c in cards}
    slots = []
    left = dict(caps)
    while len(slots) < len(runs) and any(left[c] for c in cards):
        for c in cards:
            if left[c] and len(slots) < len(runs):
                slots.append(c); left[c] -= 1
    assert len(slots) == len(runs), f"{len(runs)} runs do not fit capacities {caps}"
    for r, c in zip(runs, slots):
        per[c].append(r)
    return per


def write(phase, per):
    J.mkdir(parents=True, exist_ok=True)
    for c, rs in per.items():
        if not rs:
            continue
        for r in rs:
            Path(r["log"]).parent.mkdir(parents=True, exist_ok=True)
        f = J / f"{phase}_card{c}.json"
        with open(f, "x") as fh:
            json.dump(rs, fh, indent=1)
        print(c, f)


phase, cards = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
caps = {c: capacity(c) for c in cards}
print("capacity", caps, file=sys.stderr)
total = sum(caps.values())
if phase == "p2":
    n_ali = max(1, round(total * 2 / 5)); n_z = max(1, (total - n_ali) // 3)
    runs = (online("zh2en_S2dl3k_d1", S2DLZ, WIN["zh2en"], "zh2en", "1", n_z)
            + online("zh2en_S2dl3k_d0.5", S2DLZ, WIN["zh2en"], "zh2en", "0.5", n_z)
            + online("zh2en_S2dl3k_d2", S2DLZ, WIN["zh2en"], "zh2en", "2", n_z)
            + online("ali_S2dl3k_d1", S2DLZ, WIN["ali"], "zh2en", "1", n_ali))
    write(phase, spread(runs, cards, caps))
elif phase == "p4":
    n = max(1, total // 3)
    runs = [r for w in ("0.5", "1", "2") for r in online(f"en2zh_S2dlm_d{w}", S2DLM, WIN["en2zh"], "en2zh", w, n)]
    write(phase, spread(runs, cards, caps))
elif phase == "p5":
    c_o2, c_o2n = cards[0], cards[1]
    n1, n2 = max(1, caps[c_o2]), max(1, caps[c_o2n] // 2)
    seg = f"{OUT}/en2zh_D-segments/segments.jsonl"
    write(phase, {c_o2: offline("en2zh_D-O2", O2, WIN["diar"], [], n1),
                  c_o2n: offline("en2zh_D-O2-nosd", O2N, WIN["diar"], ["--zero-sd"], n2)
                  + offline("en2zh_D-C-ft2_seg", O2N, seg, ["--zero-sd"], n2)})
else:
    sys.exit(f"unknown phase {phase}")
