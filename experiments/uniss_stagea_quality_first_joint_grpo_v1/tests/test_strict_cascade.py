from training import constants_uniss as c

from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.strict_cascade import (
    _route_for_prompt,
)


def test_strict_cascade_routes_asr_off_and_mt_tts_on() -> None:
    assert _route_for_prompt([c.TOKEN_TASK_ASR]) is False
    assert _route_for_prompt([c.TOKEN_TASK_S2T_TRANSLATION]) is True
    assert _route_for_prompt([c.TOKEN_TASK_T2T_TRANSLATION]) is True
    assert _route_for_prompt([c.TOKEN_TASK_TTS]) is True
