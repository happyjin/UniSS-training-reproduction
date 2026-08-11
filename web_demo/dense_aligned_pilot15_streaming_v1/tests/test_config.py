from web_demo.dense_aligned_pilot15_streaming_v1.config import load_config
from web_demo.dense_aligned_pilot15_streaming_v1.validate_samples import (
    direction_for,
)


def test_dense_runtime_is_isolated_and_validation_best() -> None:
    config = load_config()
    assert config.demo_root.name == "dense_aligned_pilot15_streaming_v1"
    assert config.checkpoint.name == "iter_0000500"
    assert config.output_root.parent == config.demo_root
    assert config.decision_chunk_ms == 320


def test_supported_directions() -> None:
    assert direction_for({"src_lang": "eng", "tgt_lang": "cmn"}) == "英文 → 中文"
    assert direction_for({"src_lang": "cmn", "tgt_lang": "eng"}) == "中文 → 英文"

