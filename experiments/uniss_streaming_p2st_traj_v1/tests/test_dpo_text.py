"""The DPO mechanics, on a real Qwen2 of two tiny layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.uniss_streaming_p2st_traj_v1.training import lora
from experiments.uniss_streaming_p2st_traj_v1.training.dpo_text import (
    build_examples,
    stream_logprob,
)
from training import constants_uniss as c


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(ch) % 500 for ch in text]


def _model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=c.TOKEN_END_CONTENT + 64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = Qwen2ForCausalLM(config).eval()
    model.config.use_cache = False
    return model


def _steps(produced_lists):
    return [
        {
            "source_prefix": "hello there",
            "target_prefix": "你好",
            "produced": produced,
            "ended": True,
        }
        for produced in produced_lists
    ]


def test_the_prompt_is_not_scored():
    """Only the model's own output carries gradient signal.

    A longer source prefix is context, not a decision, so it must not change
    the count of scored tokens.
    """
    model = _model()
    tok = _Tokenizer()
    short = _steps([[7, 8]])
    long = [dict(short[0], source_prefix="hello there and then some more")]
    _, n_short = stream_logprob(
        model, tok, short, tgt_lang="cmn", device=torch.device("cpu")
    )
    _, n_long = stream_logprob(
        model, tok, long, tgt_lang="cmn", device=torch.device("cpu")
    )
    assert n_short == n_long == 3  # two produced tokens plus the stop token


def test_a_silent_step_still_costs_one_token():
    """Falling silent is a decision, and it is scored as one."""
    model = _model()
    _, n = stream_logprob(
        model,
        _Tokenizer(),
        [{"source_prefix": "a", "target_prefix": "", "produced": [], "ended": True}],
        tgt_lang="cmn",
        device=torch.device("cpu"),
    )
    assert n == 1


def test_steps_are_additive():
    """A stream's score is the sum over its read steps."""
    model = _model()
    tok = _Tokenizer()
    device = torch.device("cpu")
    with torch.no_grad():
        one, n1 = stream_logprob(model, tok, _steps([[7, 8]]), tgt_lang="cmn", device=device)
        two, n2 = stream_logprob(model, tok, _steps([[9]]), tgt_lang="cmn", device=device)
        both, n3 = stream_logprob(
            model, tok, _steps([[7, 8], [9]]), tgt_lang="cmn", device=device
        )
    assert n3 == n1 + n2
    assert torch.allclose(both, one + two, atol=1e-3)


def test_the_margin_is_zero_before_any_update():
    """Policy and reference are the same module, so step 0 is exactly log 2.

    This is the invariant that a second copy of the model can only approximate
    and that switching adapters off makes exact.
    """
    model = _model()
    tok = _Tokenizer()
    device = torch.device("cpu")
    layers = lora.inject(model, rank=4, alpha=8.0)
    chosen = _steps([[7, 8], [9, 10, 11]])
    rejected = _steps([[7], [8]])

    scores = {}
    for name, steps in (("chosen", chosen), ("rejected", rejected)):
        with torch.no_grad(), lora.adapters_disabled(layers):
            ref, length = stream_logprob(model, tok, steps, tgt_lang="cmn", device=device)
        with torch.no_grad():
            pol, _ = stream_logprob(model, tok, steps, tgt_lang="cmn", device=device)
        scores[name] = (pol, ref, length)

    cp, cr, cl = scores["chosen"]
    rp, rr, rl = scores["rejected"]
    margin = (cp - cr) / cl - (rp - rr) / rl
    assert abs(float(margin)) < 1e-4
    assert abs(float(-F.logsigmoid(0.1 * margin)) - 0.6931) < 1e-3


def test_length_normalisation_does_not_reward_saying_less():
    """A stream must not win on token count alone.

    In the real pairs the preferred candidate spreads the same text over more
    read steps, so it emits more tokens in total; unnormalised, its summed log
    probability is the more negative of the two and the objective would push
    against it.  This is that arithmetic in miniature.
    """
    model = _model()
    tok = _Tokenizer()
    device = torch.device("cpu")
    talkative = _steps([[7, 8, 9, 10], [11, 12, 13, 14]])
    terse = _steps([[7], [11]])
    with torch.no_grad():
        loud, n_loud = stream_logprob(model, tok, talkative, tgt_lang="cmn", device=device)
        quiet, n_quiet = stream_logprob(model, tok, terse, tgt_lang="cmn", device=device)
    assert float(quiet) > float(loud), "unnormalised, saying less scores higher"
    assert abs(float(loud) / n_loud - float(quiet) / n_quiet) < abs(
        float(loud) - float(quiet)
    ), "normalising removes most of that length advantage"


def test_pairs_without_a_recorded_stream_are_dropped():
    arms = {
        "k02": {"s1": {"mt_steps": [{"produced": [1]}], "tgt_lang": "cmn"}},
        "k03": {"s1": {"mt_steps": [{"produced": [2]}], "tgt_lang": "cmn"}},
    }
    pairs = [
        {"sample_id": "s1", "chosen": {"arm": "k02"}, "rejected": {"arm": "k03"}},
        {"sample_id": "s1", "chosen": {"arm": "k02"}, "rejected": {"arm": "k01"}},
        {"sample_id": "s9", "chosen": {"arm": "k02"}, "rejected": {"arm": "k03"}},
    ]
    got = build_examples(pairs, arms)
    assert len(got) == 1 and got[0]["sample_id"] == "s1"


def test_a_saved_adapter_reproduces_the_trained_model():
    """Save, reload into a fresh model, merge, unwrap -- same outputs.

    This is the path that produces the checkpoint the evaluator measures, so a
    silent failure here would look exactly like 'the training had no effect'.
    """
    from torch import nn

    trained = _model()
    layers = lora.inject(trained, rank=4, alpha=8.0)
    with torch.no_grad():
        for layer in layers:
            layer.b.normal_(std=0.02)
    ids = torch.tensor([[5, 9, 13, 21]])
    with torch.no_grad():
        want = trained(input_ids=ids).logits

    blob = {"state": lora.state_dict(layers), "rank": 4, "alpha": 8.0}

    fresh = _model()
    reloaded = lora.inject(fresh, rank=blob["rank"], alpha=blob["alpha"])
    lora.load_state_dict(reloaded, blob["state"])
    lora.merge(reloaded)
    for parent in fresh.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, lora.LoRALinear):
                setattr(parent, name, child.base)
    assert not any(isinstance(m, lora.LoRALinear) for m in fresh.modules())
    assert all(isinstance(m, nn.Linear) for m in fresh.modules() if "proj" in str(type(m)).lower() or False)
    with torch.no_grad():
        got = fresh(input_ids=ids).logits
    assert torch.allclose(want, got, atol=2e-2), (want - got).abs().max()


def test_loading_an_adapter_with_missing_layers_is_refused():
    model = _model()
    layers = lora.inject(model, rank=4, alpha=8.0)
    state = lora.state_dict(layers)
    partial = {k: v for k, v in state.items() if "layers.0" not in k}
    with __import__("pytest").raises(SystemExit):
        lora.load_state_dict(layers, partial)
