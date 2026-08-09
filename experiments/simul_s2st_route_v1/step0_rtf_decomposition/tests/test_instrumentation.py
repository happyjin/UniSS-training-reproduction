"""CPU-only correctness tests for the Step 0 profiler and probe installation."""

import sys
import time
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
TREE = ROOT / "experiments/uniss_streamspeech_ctc_v1"
for _path in (
    ROOT,
    TREE / "stage02_ctc_probe",
    TREE / "stage03_multitask_encoder",
    TREE / "stage03_multitask_encoder/ar_s2tt_v1",
    TREE / "stage04_b2_discrete_bridge",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.simul_s2st_route_v1.common.instrumentation import (  # noqa: E402
    CallTreeTimer,
    Patcher,
)
from experiments.simul_s2st_route_v1.step0_rtf_decomposition.probe import (  # noqa: E402
    LABEL_DESCRIPTIONS,
    TOP_LEVEL_LABELS,
    install_pipeline_probes,
)


def paths(timer):
    return {stat.path: stat for stat in timer.stats()}


class Widget:
    def work(self, value):
        return value * 2


def module_level(value):
    return value + 1


class CallTreeTimerTest(unittest.TestCase):
    def test_nested_spans_split_inclusive_and_exclusive_time(self):
        timer = CallTreeTimer(synchronize=False)
        with timer.span("outer"):
            time.sleep(0.02)
            with timer.span("inner"):
                time.sleep(0.05)

        stats = paths(timer)
        self.assertEqual(set(stats), {"outer", "outer/inner"})
        outer = stats["outer"]
        inner = stats["outer/inner"]
        self.assertAlmostEqual(inner.inclusive_seconds, inner.exclusive_seconds, places=9)
        self.assertGreaterEqual(inner.inclusive_seconds, 0.045)
        self.assertGreaterEqual(outer.inclusive_seconds, inner.inclusive_seconds)
        self.assertAlmostEqual(
            outer.exclusive_seconds,
            outer.inclusive_seconds - inner.inclusive_seconds,
            places=9,
        )
        self.assertAlmostEqual(timer.total_seconds(), outer.inclusive_seconds, places=9)

    def test_same_callee_under_two_callers_stays_separable(self):
        timer = CallTreeTimer(synchronize=False)
        for _ in range(3):
            with timer.span("prefill"):
                with timer.span("forward"):
                    time.sleep(0.001)
        with timer.span("decode"):
            for _ in range(5):
                with timer.span("forward"):
                    time.sleep(0.001)

        stats = paths(timer)
        self.assertEqual(stats["prefill/forward"].calls, 3)
        self.assertEqual(stats["decode/forward"].calls, 5)
        self.assertEqual(stats["prefill"].calls, 3)
        self.assertEqual(stats["decode"].calls, 1)

    def test_span_label_rejects_path_separator(self):
        timer = CallTreeTimer(synchronize=False)
        with self.assertRaises(ValueError):
            with timer.span("a/b"):
                pass

    def test_reset_refuses_while_a_span_is_open(self):
        timer = CallTreeTimer(synchronize=False)
        with self.assertRaises(RuntimeError):
            with timer.span("outer"):
                timer.reset()

    def test_span_records_even_when_the_body_raises(self):
        timer = CallTreeTimer(synchronize=False)
        with self.assertRaises(KeyError):
            with timer.span("outer"):
                raise KeyError("boom")
        self.assertEqual(paths(timer)["outer"].calls, 1)

    def test_merge_accumulates_across_samples(self):
        aggregate = CallTreeTimer(synchronize=False)
        for _ in range(2):
            timer = CallTreeTimer(synchronize=False)
            with timer.span("push"):
                with timer.span("codec"):
                    time.sleep(0.001)
            aggregate.merge(timer)

        stats = paths(aggregate)
        self.assertEqual(stats["push"].calls, 2)
        self.assertEqual(stats["push/codec"].calls, 2)
        self.assertGreaterEqual(
            stats["push"].inclusive_seconds, stats["push/codec"].inclusive_seconds
        )

    def test_synchronize_is_disabled_without_a_cuda_device(self):
        self.assertFalse(CallTreeTimer(device=None, synchronize=True).synchronizing)
        self.assertFalse(
            CallTreeTimer(device=torch.device("cpu"), synchronize=True).synchronizing
        )


class PatcherTest(unittest.TestCase):
    def test_restores_class_module_and_instance_bindings(self):
        timer = CallTreeTimer(synchronize=False)
        module = sys.modules[__name__]
        widget = Widget()
        widget.instance_bound = lambda: "original"

        original_class = Widget.work
        original_module = module_level
        original_instance = widget.instance_bound

        with Patcher(timer) as patcher:
            patcher.wrap(Widget, "work", "class_call")
            patcher.wrap(module, "module_level", "module_call")
            patcher.wrap(widget, "instance_bound", "instance_call")
            # `work` has no per-instance override, so wrapping the instance must add one.
            patcher.wrap(widget, "work", "instance_shadow")

            self.assertEqual(widget.work(3), 6)
            self.assertEqual(module.module_level(1), 2)
            self.assertEqual(widget.instance_bound(), "original")

        self.assertIs(Widget.work, original_class)
        self.assertIs(module.module_level, original_module)
        self.assertIs(widget.instance_bound, original_instance)
        self.assertNotIn("work", vars(widget))

        stats = paths(timer)
        self.assertEqual(stats["instance_shadow/class_call"].calls, 1)
        self.assertEqual(stats["module_call"].calls, 1)
        self.assertEqual(stats["instance_call"].calls, 1)

    def test_restores_a_torch_module_forward_without_leaving_an_override(self):
        timer = CallTreeTimer(synchronize=False)
        layer = torch.nn.Linear(4, 4)
        inputs = torch.zeros(1, 4)
        expected = layer(inputs)

        with Patcher(timer) as patcher:
            patcher.wrap(layer, "forward", "linear")
            observed = layer(inputs)

        self.assertTrue(torch.allclose(observed, expected))
        self.assertNotIn("forward", vars(layer))
        self.assertEqual(paths(timer)["linear"].calls, 1)

    def test_timer_is_resolved_at_call_time(self):
        first = CallTreeTimer(synchronize=False)
        second = CallTreeTimer(synchronize=False)
        widget = Widget()
        with Patcher(first) as patcher:
            widget.work(1)
            patcher.wrap(Widget, "work", "call")
            widget.work(1)
            patcher.timer = second
            widget.work(1)

        self.assertEqual(paths(first)["call"].calls, 1)
        self.assertEqual(paths(second)["call"].calls, 1)

    def test_wrap_optional_reports_missing_attribute(self):
        timer = CallTreeTimer(synchronize=False)
        with Patcher(timer) as patcher:
            self.assertFalse(patcher.wrap_optional(Widget(), "does_not_exist", "nope"))
            self.assertFalse(patcher.wrap_optional(None, "work", "nope"))


class PipelineProbeTest(unittest.TestCase):
    def test_every_top_level_bucket_has_a_description(self):
        missing = [label for label in TOP_LEVEL_LABELS if label not in LABEL_DESCRIPTIONS]
        self.assertEqual(missing, [])

    def test_probes_install_and_revert_on_the_real_classes(self):
        from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime import (
            runtime as runtime_module,
        )
        from experiments.uniss_streamspeech_ctc_v1.stage10_cached_micro_write import (
            adapter as adapter_module,
        )
        from uniss.streaming import bicodec_streamer as codec_module

        before = {
            "push_audio": runtime_module.Stage09OnlineRuntime.push_audio,
            "generate_write": adapter_module.CachedMicroWriteAdapter.generate_write,
            "penalty": adapter_module.apply_repetition_penalty,
            "codec_push": codec_module.StreamingBiCodecDecoder.push,
        }

        timer = CallTreeTimer(synchronize=False)
        patcher = install_pipeline_probes(timer)
        try:
            self.assertIsNot(
                runtime_module.Stage09OnlineRuntime.push_audio, before["push_audio"]
            )
            self.assertIsNot(adapter_module.apply_repetition_penalty, before["penalty"])
        finally:
            patcher.close()

        self.assertIs(runtime_module.Stage09OnlineRuntime.push_audio, before["push_audio"])
        self.assertIs(
            adapter_module.CachedMicroWriteAdapter.generate_write, before["generate_write"]
        )
        self.assertIs(adapter_module.apply_repetition_penalty, before["penalty"])
        self.assertIs(codec_module.StreamingBiCodecDecoder.push, before["codec_push"])

    def test_patching_does_not_change_repetition_penalty_output(self):
        from experiments.uniss_streamspeech_ctc_v1.stage10_cached_micro_write import (
            adapter as adapter_module,
        )

        logits = torch.tensor([[1.0, -2.0, 3.0, 4.0]])
        expected = adapter_module.apply_repetition_penalty(logits, [0, 1], 1.5)

        timer = CallTreeTimer(synchronize=False)
        patcher = install_pipeline_probes(timer)
        try:
            observed = adapter_module.apply_repetition_penalty(logits, [0, 1], 1.5)
        finally:
            patcher.close()

        self.assertTrue(torch.allclose(observed, expected))
        self.assertEqual(paths(timer)["logits_repetition_penalty"].calls, 1)

    def test_streaming_codec_push_is_transparent_under_instrumentation(self):
        import numpy as np

        from uniss.streaming.bicodec_streamer import StreamingBiCodecDecoder

        def fake_decode(speaker_tokens, semantic_tokens):
            return np.arange(len(semantic_tokens) * 320, dtype=np.float32)

        def drive():
            codec = StreamingBiCodecDecoder(
                fake_decode, left_context_tokens=8, holdback_tokens=2, overlap_ms=20.0
            )
            outputs = [
                codec.push(list(range(step * 10, step * 10 + 10)), speaker_tokens=[0] * 32)
                for step in range(3)
            ]
            outputs.append(codec.push([], is_final=True))
            return np.concatenate(outputs)

        expected = drive()
        timer = CallTreeTimer(synchronize=False)
        patcher = install_pipeline_probes(timer)
        try:
            observed = drive()
        finally:
            patcher.close()

        self.assertTrue(np.array_equal(observed, expected))
        self.assertEqual(paths(timer)["codec_stream_push"].calls, 4)


if __name__ == "__main__":
    unittest.main()
