from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.trace_generator import (
    family_for_prompt,
)
from training import constants_uniss as c


def test_generation_family_routing_is_explicit():
    assert family_for_prompt([c.TOKEN_TASK_ASR]) == "asr"
    assert family_for_prompt([c.TOKEN_TASK_T2T_TRANSLATION]) == "mt"
    assert family_for_prompt([c.TOKEN_TASK_TTS]) == "tts"

