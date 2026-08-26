from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.build_episode_manifest import (
    normalized_join,
)


def test_normalized_join_uses_language_specific_boundaries():
    assert normalized_join(["你好。", "世界！"], "cmn") == "你好。世界。"
    assert normalized_join(["hello.", "world!"], "eng") == "hello. world."

