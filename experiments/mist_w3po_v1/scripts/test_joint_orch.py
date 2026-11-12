"""Tests of joint_orch.py with --dry (no job runs, no GPU): the base swap on every decode / held-out command of both
directions and all sets, no separate v8 path anywhere, the held-out job for zh2en, the set windows, refusals.

    PYTHONPATH=.:third_party/Megatron-LM python experiments/mist_w3po_v1/scripts/test_joint_orch.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
ORCH = "experiments/mist_w3po_v1/scripts/joint_orch.py"
B = "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_o2_20261029b/thinker_merged"
SEP = ("thinker_gstar_s2dl_zh2en_3k_20261030", "thinker_gstar_s2dl_20261029b")
X = "eval_outputs/mist_w3po_v1/talker_crf/talker_align_20261111T001542Z/export"
CK = "/nonexistent/ckpt/step_00500"


def run(*args, ok=True) -> str:
    env = dict(os.environ, PYTHONPATH=".:third_party/Megatron-LM", CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([PY, ORCH, *args, "--dry"], cwd=REPO, env=env, capture_output=True, text=True)
    if ok:
        assert p.returncode == 0, (args, p.stderr[-2000:])
    else:
        assert p.returncode != 0, (args, p.stdout[-500:])
    return p.stdout


def common(out: str, n_dec: int, d: str) -> list[str]:
    lines = [l for l in out.splitlines() if " | " in l]
    assert not any(s in out for s in SEP), [l for l in lines if any(s in l for s in SEP)][:2]
    dec = [l for l in lines if "decode_crf_ckpt" in l]
    assert len(dec) == n_dec, (len(dec), n_dec)
    for l in dec:
        assert f"--thinker {B} " in l and f"--crf-ckpt {CK} " in l, l
        # --dry prints 700 characters; en2zh_orch.fix itself asserts " --dir en2zh " on every decode command
        assert d == "en2zh" or f" --dir {d} " in l, l
    return lines


def test_zh2en_dev5_heldout():
    out = run("--dir", "zh2en", "--set", "dev5", "--base", B, "--heldout", "eval_outputs/mist_w3po_v1/crf/sched_dev_v1",
              "--run", "/nonexistent/r", "--name", "jt_z", "--ckpt", CK, "--talker", "none", "--shards", "4",
              "--shard-by", "window", "--window", "1", "--exclude", "4", "5", "6", "7", "--cap135")
    lines = common(out, 4, "zh2en")
    h = [l for l in lines if "heldout_loss" in l]
    assert len(h) == 1 and f"--base {B} " in h[0] and " --dir zh2en " in h[0] and "sched_dev_v1" in h[0], h
    assert all("windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl" in l for l in lines if "score_run" in l)
    assert any("jt_z__textlat" in l for l in lines)


def test_en2zh_dev5_heldout():
    out = run("--dir", "en2zh", "--set", "dev5", "--base", B, "--heldout",
              "eval_outputs/mist_w3po_v1/crf/sched_en2zh_dev_v1", "--run", "/nonexistent/r", "--name", "jt_e",
              "--ckpt", CK, "--talker", "none", "--shards", "4", "--shard-by", "window", "--window", "1", "--cap135")
    lines = common(out, 4, "en2zh")
    h = [l for l in lines if "heldout_loss" in l]
    assert len(h) == 1 and f"--base {B} " in h[0] and " --dir en2zh " in h[0], h
    assert any("en2zh_lat" in l for l in lines)


def test_zh2en_no_heldout_by_default():
    out = run("--dir", "zh2en", "--set", "dev", "--base", B, "--run", "/nonexistent/r", "--name", "jt_zd",
              "--ckpt", CK, "--talker", "none", "--window", "1", "--cap135")
    lines = common(out, 8, "zh2en")
    assert not any("heldout" in l for l in lines)
    assert all("windows_zh2en_c2g06_v1/windows_dev_online.jsonl" in l for l in lines if "score_run" in l)


def test_zh2en_test_and_ali():
    for key, ref in (("test", "e2e_gstar_zh2en_v1/windows_test_ref.jsonl"), ("ali", "windows_test_ref_human.jsonl")):
        out = run("--dir", "zh2en", "--set", key, "--base", B, "--run", "/nonexistent/r", "--name", f"jt_{key}",
                  "--ckpt", CK, "--talker", "none", "--window", "1", "--cap135")
        lines = common(out, 8, "zh2en")
        assert not any("windows_validation_ref" in l or "windows_dev_online" in l for l in lines), key
        assert any(ref in l for l in lines if "score_run" in l), key


def test_en2zh_test():
    out = run("--dir", "en2zh", "--set", "test", "--base", B, "--run", "/nonexistent/r", "--name", "jt_et",
              "--ckpt", CK, "--talker", "none", "--window", "1", "--cap135")
    lines = common(out, 8, "en2zh")
    assert any("windows_test_online_en2zh" in l for l in lines)


def test_speech_only_chains():
    out = run("--dir", "zh2en", "--set", "dev", "--base", B, "--run", "/nonexistent/r", "--name", "jt_zs",
              "--ckpt", "-", "--speech-from", "jt_zd_rl", "--talker", "plem", "--window", "1", "--cap135",
              "--scorer-port0", "19310")
    common(out, 0, "zh2en")
    assert "talker_plem_20261111T001759Z" in out and "jt_zs_tts2" in out
    out = run("--dir", "en2zh", "--set", "dev", "--base", B, "--run", "/nonexistent/r", "--name", "jt_es",
              "--ckpt", "-", "--speech-from", "jt_ed_rl", "--talker", "plem", "--plem-export", X, "--window", "1",
              "--cap135", "--scorer-port0", "19320")
    common(out, 0, "en2zh")
    assert X in out and "--direction en2zh" in out


def test_refusals():
    run("--dir", "zh2en", "--set", "dev5", "--base", "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/"
        "thinker_merged", "--run", "/nonexistent/r", "--name", "x", "--ckpt", CK, "--talker", "none", ok=False)
    run("--dir", "en2zh", "--set", "ali", "--base", B, "--run", "/nonexistent/r", "--name", "x", "--ckpt", CK, ok=False)
    run("--dir", "zh2en", "--set", "dev5", "--base", B, "--win", "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/"
        "windows_dev_online.jsonl", "--run", "/nonexistent/r", "--name", "x", "--ckpt", CK, ok=False)
    run("--set", "dev5", "--base", B, "--run", "/nonexistent/r", "--name", "x", "--ckpt", CK, ok=False)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"ALL PASSED ({len(tests)} tests)")


if __name__ == "__main__":
    sys.exit(main())
