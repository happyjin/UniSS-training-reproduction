from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage00_shell_scripts_are_syntax_valid() -> None:
    import subprocess

    for path in sorted((ROOT / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_stage_a_shell_scripts_are_syntax_valid() -> None:
    import subprocess

    for name in (
        "run_stage_a_cpu_tests.sh",
        "run_stage_a_build_ctc_maps.sh",
        "launch_stage_a_ctc_maps_tmux.sh",
        "prepare_stage_a_inputs.sh",
        "run_stage_a_pack_smoke.sh",
        "run_stage_a_data_audit.sh",
        "launch_stage_a_data_audit_tmux.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)


def test_stage_a_strict_resume_restores_complete_compound_state() -> None:
    source = (ROOT / "scripts" / "run_stage_a_strict_resume_8gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "RUN_STRICTNESS=raise_all" in source
    assert "RUN_FINETUNE=0" in source
    assert "RUN_LOAD_OPTIM=1" in source
    assert "RUN_LOAD_RNG=1" in source
    assert 'RESUME_LOAD must point to a Stage A compound checkpoint root' in source


def test_stage_a_ctc_map_is_train_derived_and_validation_audited() -> None:
    source = (
        ROOT / "stage_a_causal_whisper_asr" / "build_ctc_maps.py"
    ).read_text(encoding="utf-8")
    env = (ROOT / "experiment.env").read_text(encoding="utf-8")
    assert "label-independent fixed 256-byte UTF-8 inventory" in source
    assert '"valid_oov_zero": valid_counters["oov_tokens"] == 0' in source
    assert "ctc_maps_utf8_byte_v5" in env
    assert "source_snapshot_v5.json" in env
    assert 'maps["eng"].blank_id > 0' in source
    assert 'max_keys = {"max_minimum_ctc_steps"}' in source


def test_stage00_does_not_authorize_stage_a_from_frontend_gate_only() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "FRONTEND_GATE_PASSED.json" in readme
    assert "cannot falsely authorize Stage A" in readme


def test_real_audit_uses_strict_fp32_hidden_and_exact_token_gates() -> None:
    source = (
        ROOT / "stage00_baseline" / "audit_frontend_real_pcm.py"
    ).read_text(encoding="utf-8")
    assert "rtol=2e-5, atol=2e-6" in source
    assert '"match_ratio"' in source
    assert '"future_hidden_exact_before_changed_block"' in source
    assert "forward_recomputed_reference" in source
    assert '"single_mask_numerical_diagnostic"' in source


def test_gpu_launcher_only_stops_named_synthetic_session() -> None:
    source = (ROOT / "scripts" / "launch_stage00_frontend_tmux.sh").read_text(
        encoding="utf-8"
    )
    assert "tmux kill-session -t uniss_gpu_load_60" in source
    assert "refusing to kill unknown GPU processes" in source


def test_fixed_validation_is_balanced_and_immutable_by_construction() -> None:
    source = (ROOT / "stage00_baseline" / "build_fixed_validation.py").read_text(
        encoding="utf-8"
    )
    assert "DURATION_BOUNDS_MS" in source
    assert "_stable_score" in source
    assert 'open("x"' in source
    assert "formal_manifest_sha256" in source


def test_qwen_and_bicodec_gates_are_strict() -> None:
    qwen = (ROOT / "stage00_baseline" / "audit_qwen_cache.py").read_text(
        encoding="utf-8"
    )
    codec = (ROOT / "stage00_baseline" / "audit_bicodec_streaming.py").read_text(
        encoding="utf-8"
    )
    assert "minimum_logits_cosine" in qwen
    assert ">= 0.9999" in qwen
    assert "autoregressive_runtime_gate" in qwen
    assert "all_runtime_top1_exact" in qwen
    assert 'default="eager"' in qwen
    assert "all_position_numerical_diagnostic" in qwen
    assert "semantic_gap_count" in codec
    assert "semantic_overlap_count" in codec
    assert "speaker_change_rejected" in codec


def test_native_reexport_requires_exact_weights_and_output() -> None:
    source = (ROOT / "stage00_baseline" / "audit_native_hf_reexport.py").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts" / "run_stage00_native_reexport.sh").read_text(
        encoding="utf-8"
    )
    assert "all_tensors_exact" in source
    assert "fixed_prompt_top1_exact" in source
    assert "fixed_prompt_logits_cosine_ge_0p999999" in source
    assert "--strict" in launcher
    assert "refusing to overwrite" in launcher


def test_offline_baseline_is_fixed_parallel_and_non_overwriting() -> None:
    source = (ROOT / "scripts" / "run_stage00_offline_baseline_8gpu.sh").read_text(
        encoding="utf-8"
    )
    aggregate = (
        ROOT / "stage00_baseline" / "aggregate_offline_baseline.py"
    ).read_text(encoding="utf-8")
    assert "seq 0 7" in source
    assert "--temperature 0" in source
    assert "--max-new-tokens 1500" in source
    assert "refusing to overwrite offline baseline" in source
    assert "expected-text-records 256" in source
    assert "expected-audio-records 64" in source
    assert "quality_asr_error" in aggregate
    assert "text_translation_bleu" in aggregate
    assert "audio_slc" in aggregate


def test_status_requires_full_stage00_gate() -> None:
    source = (ROOT / "scripts" / "status.sh").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "stage00_baseline/latest/FRONTEND_GATE_PASSED.json" in source
    assert "stage00_baseline/GATE_PASSED.json" in source
    assert "stage00_offline_tmux" in source
    assert "conditional" in readme.lower()
