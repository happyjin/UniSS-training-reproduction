#!/usr/bin/env python
"""OpenBroadcast-SI v1 (reports/mist_w3po_v1/voicecond_bench.zh-CN.md): joint_orch.py on one benchmark condition,
exactly as experiments/mist_w3po_v1/scripts/game_orch.py does for the game excerpts -- only the set definition is
swapped, in this process:
  --dir zh2en  final_test_orch.SETS["ali"] = the condition's zh2en C2 windows / CoVoST references, data
               data/open_voicecond_v1/bench/<cond>/zh2en (session "mix", diar nemotron_lowlat), scorer set "ali"
               (English tokenisation); joint_orch.ZH_WIN["ali"] = the same windows   (internally --set ali)
  --dir en2zh  en2zh_orch.SETS["test"] = the condition's en2zh C2 windows / references; every command's
               GSTAR_DATA=data/ami_multispk_si_v1 -> data/open_voicecond_v1/bench/<cond>/en2zh (session ihm_mix, diar
               nemotron_lowlat)                                                        (internally --set test)
  --cond id2|ood1|clean   the benchmark condition (held-out MUSAN/FSD50K background | unseen-corpus background | none)
  --voice-override <json> THE HOOK for the voice-condition ideas: every talker command runs
               experiments.mist_w3po_v1.voicecond.bench_talker_override (si_infer unchanged, BiCodec's
               encode_wav_to_tokens answered from the sidecar for the listed talker items); use with --speech-from
               <the baseline text decode> --talker t1c so that text, items, blocks, shards and seeds are B's.
  --exclude <cards> is required (Idea 1 / this benchmark: 4 5 6 7; Idea 2: 0 1 2 3).
Everything else (decode flags, talkers, placement, --cap135, ledger, scoring, C2 latency) is the wrapped orchestrator's.

    python scripts/voicecond_bench_orch.py --cond <c> --dir zh2en|en2zh [--voice-override <json>] --base <merged> \\
        --run <new run dir> --name <name> --ckpt <ckpt|-> [--speech-from <name>] --talker none|t1c --window 1 --cap135 \\
        --exclude 4 5 6 7 [--dry]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments/mist_w3po_v1/scripts"))
import en2zh_orch as E  # noqa: E402
import final_test_orch as F  # noqa: E402
import joint_orch as J  # noqa: E402
import thinker_t_eval_orch as T  # noqa: E402

WD = "eval_outputs/mist_w3po_v1/data"
BD = "data/open_voicecond_v1/bench"
SI = "-m experiments.uniss_qwen3_omni_v1.evaluation.si_infer "
HOOK = "-m experiments.mist_w3po_v1.voicecond.bench_talker_override "


def win(cond: str, d: str) -> str:
    return f"{WD}/windows_obsi_{cond}_{d}_c2g06_v1/windows_test_online_{d}.jsonl"


def take(argv: list[str], key: str):
    if key not in argv:
        return argv, None
    i = argv.index(key)
    return argv[:i] + argv[i + 2:], argv[i + 1]


def main() -> None:
    argv = sys.argv[1:]
    argv, cond = take(argv, "--cond")
    argv, d = take(argv, "--dir")
    argv, vo = take(argv, "--voice-override")
    if cond not in ("id2", "ood1", "clean") or d not in ("zh2en", "en2zh"):
        raise SystemExit("--cond id2|ood1|clean and --dir zh2en|en2zh required")
    if "--set" in argv:
        raise SystemExit("--set is set by this wrapper (zh2en: ali, en2zh: test)")
    if "--exclude" not in argv or argv.index("--exclude") + 1 >= len(argv) or argv[argv.index("--exclude") + 1].startswith("--"):
        raise SystemExit("--exclude <cards> required (this agent / Idea 1: 4 5 6 7; Idea 2: 0 1 2 3)")
    W, data = win(cond, d), f"{BD}/{cond}/{d}"
    ref = f"{data}/refs_windows.jsonl"
    for p in (W, ref, f"{data}/meetings.json"):
        if not (REPO / p).exists():
            raise SystemExit(f"missing: {p}")
    if d == "zh2en":
        F.SETS["ali"] = {"win": W, "ref": ref, "sset": "ali", "data": data, "sess": "mix", "diar": "nemotron_lowlat"}
        J.ZH_WIN["ali"] = W
        key = "ali"
    else:
        E.SETS["test"] = {"win": W, "ref": ref}
        _efix = E.fix

        def efix(cmd: str, S: dict, test: bool = False) -> str:
            cmd = cmd.replace("GSTAR_DATA=data/ami_multispk_si_v1 ", f"GSTAR_DATA={data} ")
            cmd = _efix(cmd, S, test)
            assert "ami_multispk_si_v1" not in cmd and "windows_test_en2zh_c2g06" not in cmd, cmd
            return cmd

        E.fix = efix
        key = "test"
    if vo:
        if not (REPO / vo).exists() and not Path(vo).exists():
            raise SystemExit(f"--voice-override: no such file {vo}")
        if "--speech-from" not in argv or "t1c" not in argv:
            raise SystemExit("--voice-override needs --speech-from <baseline text decode> and --talker t1c")
        n_ids = len(json.load(open(vo if Path(vo).is_absolute() else REPO / vo)))
        orig = T.build_jobs

        def build_jobs(*a, **k):
            jobs = orig(*a, **k)
            n = 0
            for j in jobs:
                if j.kind == "talk":
                    j.cmd = (lambda att, f=j.cmd: f(att).replace(SI, f"{HOOK}--voice-override {vo} ", 1))
                    assert HOOK in j.cmd(1), j.id
                    n += 1
            assert n, "no talker job to override"
            return jobs

        T.build_jobs = build_jobs   # innermost: joint_orch / final_test_orch / en2zh_orch wrap it and check every command
        print(f"[voicecond_bench_orch] voice override {vo} ({n_ids} items) on every talker job", flush=True)
    J.main(argv[:0] + ["--dir", d, "--set", key] + argv)


if __name__ == "__main__":
    main()
