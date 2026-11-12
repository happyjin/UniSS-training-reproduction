#!/usr/bin/env python
"""Final TEST evaluation (reports/mist_w3po_v1/final_test_prereg_v1.zh-CN.md): thinker_t_eval_orch.py, unchanged, on
a test set. Every job command is the dev one with only the set-dependent pieces swapped:
  windows        --win <the set's C2 gap-0.6 test windows> (required, must be the set's file)
  scoring        score_run --ref-windows <the set's reference windows> --set <zh2en|ali>
                   ami  data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl, zh2en (the v8 / v9 test refs)
                   ali  data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl, ali (DiariST human English, v9)
  talker prep    GSTAR_DATA / --session / --diar of the set (ali: data/alimeeting_si_v1, near_mix, nemotron_lowlat;
                 make_eval_jobs.py's ali values)
  latency        speech_latency_lit through evaluation/on_refset.py (metric_headroom.REF_WIN / DR of the set)
Decode (finalpost close_detect, v8 merged, policy wait, lag 2000, ctx 0, batch 16, 8 shards by meeting, loop guard),
talker flags, shard counts (8 talker shards per seed), seeds, ASR and placement (--cap135 etc.) are thinker_t_eval_orch's.
Every command is checked: no dev window / dev reference path may remain.

    python final_test_orch.py --set ami|ali <thinker_t_eval_orch.py arguments incl. --win> [--dry]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crf_pair_dev_orch as cp  # noqa: E402
import finalpost_dev_orch as fo  # noqa: E402
import thinker_t_eval_orch as T  # noqa: E402

WD = "eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1"
SETS = {
    "ami": {"win": f"{WD}/windows_test_online_zh2en.jsonl",
            "ref": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl", "sset": "zh2en",
            "data": "data/ami_multispk_si_v1", "sess": "zh_mix_v8", "diar": "nemotron_lowlat_zh8"},
    "ali": {"win": f"{WD}/windows_test_online_ali.jsonl",
            "ref": "data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl", "sset": "ali",
            "data": "data/alimeeting_si_v1", "sess": "near_mix", "diar": "nemotron_lowlat"},
}
DEV_PATHS = ("windows_validation_ref", "windows_dev_online", "windows_zh2en_c2g06_v1", "windows_zh2en_v1")
LAT = "-m experiments.mist_w3po_v1.evaluation.speech_latency_lit "


def fix(cmd: str, S: dict) -> str:
    if S["sset"] != "zh2en":
        cmd = cmd.replace(" --set zh2en ", f" --set {S['sset']} ")
    cmd = cmd.replace("GSTAR_DATA=data/ami_multispk_si_v1 ", f"GSTAR_DATA={S['data']} ")
    cmd = cmd.replace("--session zh_mix_v8 --diar nemotron_lowlat_zh8 ", f"--session {S['sess']} --diar {S['diar']} ")
    cmd = cmd.replace(LAT, f"-m experiments.mist_w3po_v1.evaluation.on_refset --ref-win {S['ref']} --dr {S['sset']} "
                           f"experiments.mist_w3po_v1.evaluation.speech_latency_lit ")
    bad = [p for p in DEV_PATHS if p in cmd]
    assert not bad, (bad, cmd)
    if "score_run" in cmd:
        assert f"--ref-windows {S['ref']} --set {S['sset']} " in cmd, cmd
    if "gstar_talker_v6 prep" in cmd:
        assert f"GSTAR_DATA={S['data']} " in cmd and f"--session {S['sess']} --diar {S['diar']} " in cmd, cmd
    if "speech_latency_lit" in cmd:
        assert "on_refset" in cmd and f"--win {S['win']} " in cmd, cmd
    return cmd


def main() -> None:
    argv = sys.argv[1:]
    if "--set" not in argv:
        raise SystemExit("--set ami|ali required")
    i = argv.index("--set")
    key = argv[i + 1]
    S = SETS[key]
    argv = argv[:i] + argv[i + 2:]
    if "--win" not in argv or argv[argv.index("--win") + 1] != S["win"]:
        raise SystemExit(f"--win {S['win']} required for --set {key}")
    for p in (S["win"], S["ref"]):
        if not Path(fo.REPO, p).exists():
            raise SystemExit(f"missing: {p}")
    fo.REF_WIN = cp.REF_WIN = S["ref"]
    orig = T.build_jobs

    def build_jobs(*a, **k):
        jobs = orig(*a, **k)
        for j in jobs:
            j.cmd = (lambda att, f=j.cmd: fix(f(att), S))
            j.cmd(1)   # check every command now (raises on a dev path)
        return jobs

    T.build_jobs = build_jobs
    sys.argv = [sys.argv[0]] + argv
    T.main()


if __name__ == "__main__":
    main()
