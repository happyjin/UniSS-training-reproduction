from dataclasses import replace

from experiments.uniss_phase3_content_first_joint_s2st_v1.training.phrase_oracle import phrase_oracle_sessions
from experiments.uniss_phase3_event_rollout_joint_full198_v1.event_rollout import (
    OracleEvent,
    OracleSession,
    build_write_outcome,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1 import event_rollout
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import ROLE_ACTION
from training import constants_uniss as c


def test_short_write_is_coalesced_and_final_event_flushes(monkeypatch):
    source = (c.TOKEN_START_GLM, c.GLM_SEMANTIC_OFFSET, c.TOKEN_END_GLM)
    short, short_roles = build_write_outcome("eng", [11, 12], [1])
    long, long_roles = build_write_outcome("eng", [13, 14, 15], [2])
    first = OracleEvent(0, (0,), source, "WRITE", short, short_roles, c.TOKEN_START_GLM, False, 0, 2, 0, 320, 0, 0, False, True, 0)
    last = OracleEvent(1, (0,), source, "WRITE", long, long_roles, c.TOKEN_EOS, True, 2, 5, 0, 640, 0, 0, False, True, 0)
    session = OracleSession("x", "eng", (0,) * 32, (c.TOKEN_START_GLOBAL, *([c.BICODEC_GLOBAL_OFFSET] * 32), c.TOKEN_END_GLOBAL), (first, last), (11, 12, 13, 14, 15))
    monkeypatch.setattr(
        "experiments.uniss_phase3_content_first_joint_s2st_v1.training.phrase_oracle.oracle_sessions_from_pack",
        lambda _: (session,),
    )
    out = phrase_oracle_sessions({"unused": True}, minimum_tokens=4)[0]
    assert out.events[0].action == "WAIT"
    assert out.events[0].outcome_tokens == (c.TOKEN_WAIT_READ,)
    assert out.events[1].action == "WRITE"
    parsed = event_rollout.parse_write_outcome(out.events[1].outcome_tokens)
    assert parsed.text_ids == (11, 12, 13, 14, 15)
    assert parsed.semantic_codes == (1, 2)
