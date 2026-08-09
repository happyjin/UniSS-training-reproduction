"""Install the Step 0 timing probes onto a live Stage09/10/11 pipeline.

Nothing here edits the Stage09-11 sources. Everything is installed through
:class:`~experiments.simul_s2st_route_v1.common.instrumentation.Patcher` and reverted when
the patcher is closed, so an instrumented process leaves no residue for a later
uninstrumented pass in the same interpreter.
"""

from __future__ import annotations

from typing import Any

from experiments.simul_s2st_route_v1.common.instrumentation import CallTreeTimer, Patcher
from experiments.uniss_streamspeech_ctc_v1.stage05_ctc_policy import policy as policy_module
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime import runtime as runtime_module
from experiments.uniss_streamspeech_ctc_v1.stage10_cached_micro_write import (
    adapter as adapter_module,
)
from experiments.uniss_streamspeech_ctc_v1.stage11_streaming_audio import engine as engine_module
from uniss.streaming import bicodec_streamer as codec_module


#: Direct children of ``session_push``; every other path is nested below one of these.
TOP_LEVEL_LABELS = (
    "source_runtime",
    "qwen_prefill_source",
    "qwen_prefill_wait",
    "qwen_ar_decode",
    "codec_stream_push",
    "result_io",
    "offline_fallback",
)

#: Human readable grouping used by the report.
LABEL_DESCRIPTIONS = {
    "session_setup": "Session construction incl. Qwen streaming prompt prefill",
    "session_push": "One Stage11Session.push call over a 160 ms ingress chunk",
    "source_runtime": "Stage09 chunk-causal source frontend (mel + Emformer + CTC + B1 + policy)",
    "src_mel_spectrogram": "torchaudio MelSpectrogram over the cumulative waveform",
    "src_feature_projection": "log-mel stacking and input projection (extract_projected minus mel)",
    "src_encoder_infer": "Emformer.infer on the 160 ms segment plus 80 ms right context",
    "src_output_norm": "Encoder output LayerNorm",
    "src_ctc_head": "CTC head linear projections (source and target)",
    "src_b1_bridge": "B1 bridge: Emformer hidden -> Qwen GLM embedding space",
    "src_policy_update": "Stage05 CTC read/write policy decision",
    "qwen_prefill_source": "Qwen KV-cache append of START_GLM + source embeddings + END_GLM",
    "qwen_prefill_wait": "Qwen KV-cache append of a single WAIT token",
    "qwen_ar_decode": "Qwen autoregressive WRITE generation (text + 50 Hz BiCodec semantic)",
    "qwen_forward_ids": "Qwen forward over discrete ids with KV cache",
    "qwen_forward_embeds": "Qwen forward over continuous source embeddings with KV cache",
    "logits_repetition_penalty": "Repetition penalty over the full expanded vocabulary",
    "logits_block_collapse": "Collapsed-semantic blocking heuristic",
    "parse_write_tokens": "Parsing one WRITE into text ids and semantic values",
    "codec_stream_push": "Streaming BiCodec wrapper (window selection, holdback, crossfade)",
    "codec_vocoder": "BiCodec detokenize over the left-context window",
    "result_io": "Final WAV/JSON writing and stereo alignment",
    "offline_fallback": "Final-only offline safety generation when no WRITE was accepted",
}


def install_pipeline_probes(timer: CallTreeTimer) -> Patcher:
    """Patch the class-level and module-level hooks shared by every session."""

    patcher = Patcher(timer)
    patcher.wrap(runtime_module.Stage09OnlineRuntime, "push_audio", "source_runtime")
    patcher.wrap(policy_module.CTCReadWritePolicy, "update", "src_policy_update")

    patcher.wrap(adapter_module.CachedMicroWriteAdapter, "append_source", "qwen_prefill_source")
    patcher.wrap(adapter_module.CachedMicroWriteAdapter, "commit_wait", "qwen_prefill_wait")
    patcher.wrap(adapter_module.CachedMicroWriteAdapter, "generate_write", "qwen_ar_decode")
    patcher.wrap(adapter_module.CachedMicroWriteAdapter, "_forward_ids", "qwen_forward_ids")
    patcher.wrap(
        adapter_module.CachedMicroWriteAdapter, "_forward_embeddings", "qwen_forward_embeds"
    )
    patcher.wrap(adapter_module, "apply_repetition_penalty", "logits_repetition_penalty")
    patcher.wrap(adapter_module, "block_collapsed_semantic", "logits_block_collapse")
    patcher.wrap(adapter_module, "parse_write_tokens", "parse_write_tokens")

    patcher.wrap(codec_module.StreamingBiCodecDecoder, "push", "codec_stream_push")

    patcher.wrap(engine_module.Stage11Session, "_write_result", "result_io")
    patcher.wrap(engine_module.Stage11Engine, "performance_fallback", "offline_fallback")
    return patcher


def install_bundle_probes(patcher: Patcher, bundle: Any) -> list[str]:
    """Patch the frozen encoder stack, which is only reachable through live instances."""

    installed: list[str] = []
    base = bundle.joint.endpoint.base
    if patcher.wrap_optional(base, "extract_projected", "src_feature_projection"):
        installed.append("src_feature_projection")
    if patcher.wrap_optional(base.mel, "forward", "src_mel_spectrogram"):
        installed.append("src_mel_spectrogram")
    if patcher.wrap_optional(base.encoder, "infer", "src_encoder_infer"):
        installed.append("src_encoder_infer")
    if patcher.wrap_optional(base.output_norm, "forward", "src_output_norm"):
        installed.append("src_output_norm")
    for head in base.heads.values():
        if patcher.wrap_optional(head, "forward", "src_ctc_head"):
            installed.append("src_ctc_head")
    if patcher.wrap_optional(bundle.joint, "b1_from_hidden", "src_b1_bridge"):
        installed.append("src_b1_bridge")
    return installed


def install_session_probes(timer: CallTreeTimer, session: Any) -> Patcher:
    """Patch the per-session BiCodec decode closure produced by ``bicodec_decode_function``.

    Returns its own patcher because the closure lives on a session-scoped object; closing it
    after each sample keeps the undo list from growing with dead references.
    """

    patcher = Patcher(timer)
    patcher.wrap_optional(session.codec, "decode", "codec_vocoder")
    return patcher
