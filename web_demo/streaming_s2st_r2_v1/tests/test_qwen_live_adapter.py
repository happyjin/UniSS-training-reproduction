import unittest
from types import SimpleNamespace

import torch

from training import constants_uniss as c
from uniss.streaming.policy import PolicyDecision
from web_demo.streaming_s2st_r2_v1.engine.qwen_live_adapter import (
    QwenLiveAdapter,
    SemanticAntiCollapseProcessor,
    semantic_rejection_reason,
)


class FakeTokenizer:
    def decode(self, values, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(str(value) for value in values)


class FakeModel:
    def __init__(self, action_token):
        self.action_token = action_token
        self.config = SimpleNamespace(vocab_size=c.VOCAB_SIZE + 3)

    def __call__(self, input_ids, attention_mask):
        del attention_mask
        logits = torch.full(
            (1, input_ids.shape[1], self.config.vocab_size), -100.0
        )
        logits[0, -1, self.action_token] = 10.0
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, **kwargs):
        del kwargs
        tail = torch.tensor(
            [[
                c.TOKEN_ENG,
                c.TOKEN_START_CONTENT,
                10,
                11,
                c.TOKEN_END_CONTENT,
                c.TOKEN_START_SEMANTIC,
                c.bicodec_semantic_id(3),
                c.bicodec_semantic_id(4),
                c.TOKEN_END_SEMANTIC,
            ]],
            dtype=torch.long,
        )
        return torch.cat([input_ids.cpu(), tail], dim=1)


class QwenLiveAdapterTest(unittest.TestCase):
    def adapter(self, action_token=c.TOKEN_WRITE_GENERATE):
        return QwenLiveAdapter(
            model=FakeModel(action_token),
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
            target_language="eng",
            speaker_tokens=list(range(32)),
        )

    def test_write_matches_stage4_prompt_and_parser(self):
        adapter = self.adapter()
        adapter.append_source([1, 2])
        self.assertEqual(adapter.choose_action(), PolicyDecision.WRITE)
        result = adapter.generate_write()
        self.assertEqual(list(result.target_text_ids), [10, 11])
        self.assertEqual(list(result.semantic_tokens), [3, 4])
        self.assertEqual(adapter.translation, "10 11")
        self.assertEqual(adapter.structural_recoveries, 0)

    def test_final_wait_is_forced_to_write(self):
        adapter = self.adapter(c.TOKEN_WAIT_READ)
        adapter.append_source([1])
        self.assertEqual(adapter.choose_action(is_final=True), PolicyDecision.WRITE)
        self.assertEqual(adapter.last_action.forced_reason, "final_flush")
        self.assertEqual(adapter.forced_actions, 1)

    def test_semantic_quality_gate_rejects_static_collapse(self):
        self.assertIsNotNone(semantic_rejection_reason([848] * 64))
        self.assertIsNone(semantic_rejection_reason(list(range(64))))

    def test_anti_collapse_processor_masks_repeated_semantic_token(self):
        repeated = c.bicodec_semantic_id(848)
        prompt = [c.TOKEN_WRITE_GENERATE]
        generated = [c.TOKEN_START_SEMANTIC, *([repeated] * 6)]
        input_ids = torch.tensor([[*prompt, *generated]], dtype=torch.long)
        scores = torch.zeros((1, c.VOCAB_SIZE), dtype=torch.float32)
        adjusted = SemanticAntiCollapseProcessor(len(prompt))(input_ids, scores)
        self.assertTrue(torch.isneginf(adjusted[0, repeated]))
        self.assertEqual(float(adjusted[0, c.bicodec_semantic_id(1)]), 0.0)


if __name__ == "__main__":
    unittest.main()
