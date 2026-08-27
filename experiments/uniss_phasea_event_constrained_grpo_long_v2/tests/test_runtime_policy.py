from experiments.uniss_phasea_event_constrained_grpo_long_v2.runtime.event_policy_cascade import (
    micro_ready_prefixes,
)


class Tokenizer:
    def decode(self, values, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(value) for value in values)


def test_explicit_write_can_micro_flush_two_words():
    ready, remaining = micro_ready_prefixes(
        [1, 2], Tokenizer(), "eng", final=False, write_requested=True
    )
    assert ready and not remaining


def test_wait_never_forces_micro_flush():
    ready, remaining = micro_ready_prefixes(
        [1, 2], Tokenizer(), "eng", final=False, write_requested=False
    )
    assert not ready and remaining == [1, 2]
