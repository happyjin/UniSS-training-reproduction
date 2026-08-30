"""Derive phrase-sized WRITE supervision without changing immutable packs.

The source packs are treated as immutable.  This view delays lexical WRITEs
whose accumulated target text is shorter than ``minimum_tokens`` and appends
their text and acoustic units to the next lexical WRITE.  At source end the
pending phrase is flushed.  It avoids teaching the policy to issue a stream of
tiny, individually audible fragments.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from experiments.uniss_phase3_event_rollout_joint_full198_v1.event_rollout import (
    OracleEvent,
    OracleSession,
    build_write_outcome,
    oracle_sessions_from_pack,
    parse_write_outcome,
)
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import ROLE_ACTION
from training import constants_uniss as c


def phrase_oracle_sessions(
    value: Mapping[str, object], *, minimum_tokens: int = 4
) -> tuple[OracleSession, ...]:
    """Return immutable oracle sessions with short lexical WRITEs coalesced."""

    if minimum_tokens < 1:
        raise ValueError("minimum_tokens must be positive")
    outputs: list[OracleSession] = []
    for session in oracle_sessions_from_pack(value):
        pending_text: list[int] = []
        pending_semantic: list[int] = []
        events: list[OracleEvent] = []
        for event in session.events:
            current_text: list[int] = []
            current_semantic: list[int] = []
            if event.action == "WRITE":
                parsed = parse_write_outcome(event.outcome_tokens)
                current_text = list(parsed.text_ids)
                current_semantic = list(parsed.semantic_codes)
            pending_text.extend(current_text)
            pending_semantic.extend(current_semantic)
            should_write = bool(pending_text) and (
                len(pending_text) >= minimum_tokens or event.source_finished
            )
            if should_write:
                if not pending_semantic:
                    raise ValueError("lexical phrase has no acoustic target")
                outcome, roles = build_write_outcome(
                    session.target_lang, pending_text, pending_semantic
                )
                events.append(
                    replace(
                        event,
                        action="WRITE",
                        outcome_tokens=outcome,
                        outcome_roles=roles,
                    )
                )
                pending_text.clear()
                pending_semantic.clear()
            else:
                events.append(
                    replace(
                        event,
                        action="WAIT",
                        outcome_tokens=(c.TOKEN_WAIT_READ,),
                        outcome_roles=(ROLE_ACTION,),
                    )
                )
        if pending_text or pending_semantic:
            raise ValueError("final source event did not flush a pending phrase")
        outputs.append(replace(session, events=tuple(events)))
    return tuple(outputs)


__all__ = ["phrase_oracle_sessions"]
