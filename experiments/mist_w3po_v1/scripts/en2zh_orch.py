#!/usr/bin/env python
"""En->Zh evaluation orchestrator (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §3-§5): thinker_t_eval_orch.py
(the zh->en A' dev chain), unchanged, with every direction-dependent piece swapped -- final_test_orch.py's pattern:

  decode        --thinker <en2zh v8 = thinker_gstar_s2dl_20261029b> --dir en2zh (the CRF checkpoint is the en2zh run's)
  scoring       score_run --ref-windows <set's En->Zh reference windows> --set en2zh (overlap_score, zh tokenisation)
  talker prep   --session ihm_mix --diar nemotron_lowlat --sep '' (MIST v7 en2zh streaming prep)
  talker        si_infer --direction en2zh (T1c for c1b1s2; --plem-export for plem: the En->Zh PL-EM talker)
  CTC scorers   si_ctc_scorer --lang zh (Chinese CTC, independent of the evaluation ASR)
  ASR           asr_transcribe routes tgt_lang cmn to Paraformer-zh: MODELSCOPE_CACHE added to the environment
  latency       speech_latency_lit through evaluation/en2zh_lat.py (charified copies, en2zh universe / references;
                its work dir = <out>.work)
Everything else (decode flags, loop guard, shards, talker flags and shard counts, seeds, ASR sub-shards, placement,
--cap135, the shared ledger) is thinker_t_eval_orch's. Every command is checked: no zh->en path or flag may remain, and
a test-set command may hold no dev path.

Additions (default off):
  --seeds 1        only TTS seed 1 (drops the <name>_tts2 chain and the gates): the fast dev5 speech proxy
  --heldout <dev sched dir>   one more GPU job (kind dec, id <name>__dec_heldout): crf/heldout_loss.py on the checkpoint
                   -> crf/t_eval/<name>/heldout_<name>.json (recorded only)
  --plem-export <dir>         the PL-EM export for --talker plem (required then; never the zh->en one)

    python en2zh_orch.py --set dev|dev5|test <thinker_t_eval_orch.py arguments> [--seeds 1] [--heldout D] \\
        [--plem-export D] [--dry]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crf_pair_dev_orch as cp  # noqa: E402
import finalpost_dev_orch as fo  # noqa: E402
import talker_crf_orch as tc  # noqa: E402
import thinker_t_eval_orch as T  # noqa: E402

THINKER = "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_20261029b/thinker_merged"
WD = "eval_outputs/mist_w3po_v1/data"
REF_DEV = "data/ami_multispk_si_v1/e2e_gstar_v1/windows_validation_ref.jsonl"
REF_TEST = "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl"
SETS = {
    "dev": {"win": f"{WD}/windows_en2zh_c2g06_v1/windows_dev_online.jsonl", "ref": REF_DEV},
    "dev5": {"win": f"{WD}/windows_en2zh_c2g06_v1_dev5/windows_dev_online.jsonl", "ref": REF_DEV},
    "test": {"win": f"{WD}/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl", "ref": REF_TEST},
}
ZH2EN_BITS = ("zh_mix_v8", "nemotron_lowlat_zh8", "--dir zh2en", "--direction zh2en", "--set zh2en", "e2e_gstar_zh2en_v1",
              "windows_zh2en", "s2dl_zh2en", "talker_plem_20261111T001759Z", "evaluation.speech_latency_lit --",
              "evaluation.speech_latency_lit ")
DEV_BITS = ("windows_validation_ref", "windows_en2zh_c2g06_v1/", "windows_en2zh_c2g06_v1_dev5", "windows_dev_online")
MSC = "MODELSCOPE_CACHE=/opt/dlami/nvme/jasonleeeli/cache/modelscope "
LAT = "-m experiments.mist_w3po_v1.evaluation.speech_latency_lit "
SCORER = "-m experiments.mist_w3po_v1.evaluation.si_ctc_scorer --port "


def fix(cmd: str, S: dict, test: bool = False) -> str:
    assert fo.ENV in cmd, cmd[:200]
    cmd = cmd.replace("export TMPDIR=", "export " + MSC + "TMPDIR=", 1)
    cmd = cmd.replace(f"--thinker {fo.THINKER} ", f"--thinker {THINKER} ")
    cmd = cmd.replace(" --dir zh2en ", " --dir en2zh ")
    cmd = cmd.replace(" --set zh2en ", " --set en2zh ")
    cmd = cmd.replace("--session zh_mix_v8 --diar nemotron_lowlat_zh8 --sep ' ' ",
                      "--session ihm_mix --diar nemotron_lowlat --sep '' ")
    cmd = cmd.replace(" --direction zh2en ", " --direction en2zh ")
    cmd = cmd.replace(SCORER, "-m experiments.mist_w3po_v1.evaluation.si_ctc_scorer --lang zh --port ")
    if LAT in cmd:
        out = re.search(r" --out (\S+)", cmd[cmd.index(LAT):]).group(1)
        cmd = cmd.replace(LAT, f"-m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win {S['ref']} "
                               f"--work {out}.work ")
    bad = [b for b in ZH2EN_BITS if b in cmd]
    if test:
        bad += [b for b in DEV_BITS if b in cmd]
    assert not bad, (bad, cmd)
    if "-m experiments.mist_w3po_v1.evaluation.score_run " in cmd:
        assert f"--ref-windows {S['ref']} --set en2zh " in cmd, cmd
    if "decode_crf_ckpt" in cmd:
        assert f"--thinker {THINKER} " in cmd and " --dir en2zh " in cmd and f"--windows {S['win']} " in cmd, cmd
    if "gstar_talker_v6 prep" in cmd:
        assert "--session ihm_mix --diar nemotron_lowlat --sep '' " in cmd and f"--windows {S['win']} " in cmd, cmd
    if ".si_infer " in cmd:
        assert " --direction en2zh " in cmd, cmd
        if "--scorer-url" in cmd or "--select-k" in cmd:
            assert "--select-k 2" in cmd and "--scorer-url" in cmd, cmd
    if "si_ctc_scorer" in cmd:
        assert "--lang zh" in cmd, cmd
    if "en2zh_lat" in cmd:
        assert f"--win {S['win']} " in cmd, cmd
    if "asr_transcribe" in cmd:
        assert MSC in cmd, cmd
    return cmd


def take(argv: list[str], key: str, nargs: int = 1):
    if key not in argv:
        return argv, None
    i = argv.index(key)
    val = argv[i + 1] if nargs == 1 else True
    return argv[:i] + argv[i + 1 + nargs:], val


def main() -> None:
    argv = sys.argv[1:]
    argv, key = take(argv, "--set")
    if key not in SETS:
        raise SystemExit("--set dev|dev5|test required")
    argv, seeds = take(argv, "--seeds")
    argv, heldout = take(argv, "--heldout")
    argv, plem = take(argv, "--plem-export")
    S = SETS[key]
    if "--win" in argv and argv[argv.index("--win") + 1] != S["win"]:
        raise SystemExit(f"--win must be {S['win']} for --set {key}")
    if "--win" not in argv:
        argv += ["--win", S["win"]]
    for p in (S["win"], S["ref"], THINKER):
        if not Path(fo.REPO, p).exists():
            raise SystemExit(f"missing: {p}")
    talker = argv[argv.index("--talker") + 1] if "--talker" in argv else "c1b1s2"
    if talker == "plem":
        if not plem or not Path(fo.REPO, plem, "uniss", "model.safetensors").exists():
            raise SystemExit("--talker plem needs --plem-export <En->Zh PL-EM export with uniss/model.safetensors>")
        T.PLEM = plem
    if heldout and not Path(fo.REPO, heldout, "summary.json").exists():
        raise SystemExit(f"--heldout: no sched dir {heldout}")
    fo.REF_WIN = cp.REF_WIN = S["ref"]
    T.OWNER = "en2zh"
    orig = T.build_jobs

    def build_jobs(run, name, ckpt, *a, **k):
        jobs = orig(run, name, ckpt, *a, **k)
        if seeds == "1":
            jobs = [j for j in jobs if not j.id.startswith(f"{name}_tts2__") and "__gate" not in j.id]
        elif seeds not in (None, "2"):
            raise SystemExit("--seeds 1|2")
        if heldout:
            O = f"eval_outputs/mist_w3po_v1/crf/t_eval/{name}"
            jobs.append(fo.Job(f"{name}__dec_heldout", "dec", 1, [],
                               lambda att: fo.ENV + f"mkdir -p {O} && [ -e {O}/heldout_{name}.json ] || {fo.PY} -m "
                                                    f"experiments.mist_w3po_v1.crf.heldout_loss --base {THINKER} "
                                                    f"--ckpt {ckpt} --sched {heldout} --dir en2zh --out "
                                                    f"{O}/heldout_{name}.json --mem-cap-gib 70 > {O}/heldout.log 2>&1",
                               retries=1))
            jobs[-1].order = len(jobs)
        for j in jobs:
            j.cmd = (lambda att, f=j.cmd: fix(f(att), S, key == "test"))
            j.cmd(1)   # check every command now (raises on a zh2en / dev remnant)
        return jobs

    T.build_jobs = build_jobs
    sys.argv = [sys.argv[0]] + argv
    T.main()


if __name__ == "__main__":
    main()
