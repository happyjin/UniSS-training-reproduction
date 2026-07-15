import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from training import constants_uniss as c
from training import initialize_uniss_hf_checkpoint as init


def tiny_qwen2(*, vocab_size: int = 16, tie_word_embeddings: bool = True) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = Qwen2ForCausalLM(config)
    model.tie_weights()
    return model


class InitializeUniSSHFCheckpointTest(unittest.TestCase):
    def test_resize_tied_embeddings_preserves_qwen_rows(self):
        model = tiny_qwen2(vocab_size=16, tie_word_embeddings=True)
        old_input = model.get_input_embeddings().weight.detach().clone()

        summary = init.resize_model_to_vocab(model, target_vocab_size=20, seed=7, initializer_range=0.02)

        self.assertEqual(summary.base_vocab_size, 16)
        self.assertEqual(summary.target_vocab_size, 20)
        self.assertEqual(summary.added_tokens, 4)
        self.assertEqual(summary.input_embedding_shape, (20, 8))
        self.assertEqual(summary.output_embedding_shape, (20, 8))
        self.assertTrue(summary.tied_word_embeddings)
        self.assertEqual(model.config.vocab_size, 20)
        self.assertTrue(torch.equal(model.get_input_embeddings().weight[:16], old_input))
        self.assertFalse(torch.equal(model.get_input_embeddings().weight[16:], torch.zeros_like(model.get_input_embeddings().weight[16:])))

    def test_resize_untied_output_preserves_qwen_rows(self):
        model = tiny_qwen2(vocab_size=16, tie_word_embeddings=False)
        old_input = model.get_input_embeddings().weight.detach().clone()
        old_output = model.get_output_embeddings().weight.detach().clone()

        summary = init.resize_model_to_vocab(model, target_vocab_size=20, seed=11)

        self.assertFalse(summary.tied_word_embeddings)
        self.assertTrue(torch.equal(model.get_input_embeddings().weight[:16], old_input))
        self.assertTrue(torch.equal(model.get_output_embeddings().weight[:16], old_output))
        self.assertEqual(tuple(model.get_output_embeddings().weight.shape), (20, 8))

    def test_resize_rejects_shrinking_vocab(self):
        model = tiny_qwen2(vocab_size=16)
        with self.assertRaisesRegex(ValueError, "smaller than base vocab"):
            init.resize_model_to_vocab(model, target_vocab_size=15)

    def test_target_vocab_defaults_to_paper_size_without_tokenizer(self):
        target = init.resolve_target_vocab_size(
            explicit_vocab_size=None,
            tokenizer_path=Path("/path/that/does/not/exist"),
        )
        self.assertEqual(target, c.VOCAB_SIZE)

    def test_nonstandard_vocab_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "UniSS paper config requires"):
            init.resolve_target_vocab_size(
                explicit_vocab_size=20,
                tokenizer_path=Path("/path/that/does/not/exist"),
            )
        target = init.resolve_target_vocab_size(
            explicit_vocab_size=20,
            tokenizer_path=Path("/path/that/does/not/exist"),
            allow_nonstandard_vocab=True,
        )
        self.assertEqual(target, 20)

    def test_ensure_output_dir_requires_overwrite_for_nonempty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "checkpoint"
            output.mkdir()
            (output / "existing.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                init.ensure_output_dir(output, overwrite=False)
            init.ensure_output_dir(output, overwrite=True)


if __name__ == "__main__":
    unittest.main()
