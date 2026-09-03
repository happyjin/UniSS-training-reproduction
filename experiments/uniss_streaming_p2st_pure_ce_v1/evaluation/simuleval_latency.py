#!/usr/bin/env python3
"""Score latency with SimulEval 1.1.0's own scorers, loaded from disk.

Why not just import it
----------------------
SimulEval is not installed and may not be, and it cannot be reached by putting
its tree on ``PYTHONPATH``: ``simuleval/evaluator/__init__.py`` pulls in
``SentenceLevelEvaluator`` -> ``simuleval.data.dataloader.s2t_dataloader`` ->
``pydub``, and ``latency_scorer.py`` itself imports ``textgrid`` at module
level.  Neither ``pydub``, ``textgrid``, ``yt_dlp`` nor ``tornado`` is present
in either environment.  So the single file that matters is loaded directly with
``importlib`` after stubbing exactly the names it touches -- the scorers
themselves depend on nothing but ``statistics.mean`` and the instance
attributes.

Why not reuse this repository's own latency_family
--------------------------------------------------
Because they are not the same metric.  Reading both implementations:

* ``AP``: SimulEval divides by ``source_length * reference_length``;
  ``streaming_metrics.latency_family`` divides by
  ``source_duration * hypothesis_units``.
* ``DAL``: SimulEval overrides ``target_length = len(delays)``, so its step is
  ``source / hypothesis``; ``latency_family`` uses ``source / reference``.
* the delay axis differs: SimulEval takes one delay per emitted speech chunk,
  ``latency_family`` one per semantic token.

AL and LAAL agree in form.  So SimulEval's numbers are the headline -- it is
what the field and the paper use, and using it removes any suspicion that a
metric was reimplemented favourably -- and ``latency_family`` is published
beside them as a labelled token-axis cross-check.

The digest of the scorer file goes into the output so the implementation is
pinned by hash rather than by a path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

SIMULEVAL_ROOT = Path(
    "/opt/dlami/nvme/jasonleeeli/research_sources/streamspeech/"
    "StreamSpeech-main-20260803/SimulEval"
)
SCORER_FILE = SIMULEVAL_ROOT / "simuleval" / "evaluator" / "scorers" / "latency_scorer.py"
STUB_INSTANCES = (
    "TextInputInstance",
    "TextOutputInstance",
    "SpeechOutputInstance",
    "Instance",
    "LogInstance",
)
HEADLINE = ("AL", "LAAL", "AP", "DAL")
SPEECH_EXTRA = ("StartOffset", "EndOffset", "NumChunks", "RTF")


def load_scorers(root: Path = SIMULEVAL_ROOT):
    """Import only ``latency_scorer.py``, with its unmet imports stubbed."""
    scorer_file = root / "simuleval" / "evaluator" / "scorers" / "latency_scorer.py"
    if not scorer_file.is_file():
        raise FileNotFoundError(scorer_file)
    if "uniss_simuleval_latency" in sys.modules:
        return sys.modules["uniss_simuleval_latency"]
    package = types.ModuleType("simuleval")
    package.__path__ = []  # type: ignore[attr-defined]
    evaluator = types.ModuleType("simuleval.evaluator")
    evaluator.__path__ = []  # type: ignore[attr-defined]
    instance = types.ModuleType("simuleval.evaluator.instance")
    for name in STUB_INSTANCES:
        setattr(instance, name, type(name, (), {}))
    sys.modules.setdefault("simuleval", package)
    sys.modules.setdefault("simuleval.evaluator", evaluator)
    sys.modules.setdefault("simuleval.evaluator.instance", instance)
    sys.modules.setdefault("textgrid", types.ModuleType("textgrid"))
    spec = importlib.util.spec_from_file_location(
        "uniss_simuleval_latency", str(scorer_file)
    )
    if spec is None or spec.loader is None:
        raise ImportError(str(scorer_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules["uniss_simuleval_latency"] = module
    spec.loader.exec_module(module)
    return module


def scorer_digest(root: Path = SIMULEVAL_ROOT) -> str:
    scorer_file = root / "simuleval" / "evaluator" / "scorers" / "latency_scorer.py"
    return hashlib.sha256(scorer_file.read_bytes()).hexdigest()


def reference_length(reference: str, tgt_lang: str) -> int:
    """SimulEval's own per-language unit: characters for Chinese, words else.

    ``LogInstance`` unconditionally uses ``.split(" ")``, which on a Chinese
    reference returns 1 and would wreck AL and LAAL, so the unit is set here
    explicitly rather than inherited.  The paper's own commands pass
    ``--eval-latency-unit char`` for Chinese targets.
    """
    text = reference.strip()
    if tgt_lang == "cmn":
        return max(1, len("".join(text.split())))
    return max(1, len(text.split()))


def build_instance(module, row: dict, *, computation_aware: bool):
    """A duck-typed SpeechOutputInstance carrying the four SimulEval lists."""
    base = sys.modules["simuleval.evaluator.instance"].SpeechOutputInstance

    class _Instance(base):  # type: ignore[misc, valid-type]
        pass

    instance = _Instance()
    instance.delays = [float(value) for value in row["delays"]]
    instance.elapsed = [float(value) for value in row["elapsed"]]
    instance.durations = [float(value) for value in row["durations"]]
    instance.intervals = [list(map(float, pair)) for pair in row["intervals"]]
    instance.silences = [float(value) for value in row.get("silences", [])]
    instance.source_length = float(row["source_duration_ms"])
    instance.reference = row["translation_reference"]
    instance.reference_length = reference_length(
        row["translation_reference"], row["tgt_lang"]
    )
    instance.prediction = row["target_hypothesis"]
    instance.prediction_length = len(instance.delays)
    instance.metrics = {}
    instance.source = None
    instance.finish_prediction = True
    return instance


def score(rows: list[dict], *, computation_aware: bool = False) -> dict[str, float]:
    """AL / LAAL / AP / DAL plus the speech-output family, over one arm."""
    module = load_scorers()
    registry = module.LATENCY_SCORERS_DICT
    usable = [row for row in rows if row.get("delays")]
    if not usable:
        return {}
    out: dict[str, float] = {"instances": float(len(usable)), "skipped": float(len(rows) - len(usable))}
    for name in HEADLINE + SPEECH_EXTRA:
        scorer_class = registry.get(name)
        if scorer_class is None:
            continue
        scorer = scorer_class(computation_aware=computation_aware)
        instances = {
            index: build_instance(module, row, computation_aware=computation_aware)
            for index, row in enumerate(usable)
        }
        try:
            out[name] = float(scorer(instances))
        except Exception as error:  # a scorer needing data we do not have
            out[name] = float("nan")
            out[f"{name}_error"] = f"{type(error).__name__}: {error}"
    return out


__all__ = ["load_scorers", "score", "scorer_digest", "reference_length", "SIMULEVAL_ROOT"]
