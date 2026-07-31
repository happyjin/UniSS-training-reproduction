from __future__ import annotations

import unittest

import torch

from training.simul_uniss.subsecond_v2.neural_word_aligner import links_from_similarity


class NeuralWordAlignerTest(unittest.TestCase):
    def test_mutual_and_exact_links_are_preserved(self) -> None:
        similarity = torch.tensor(
            [
                [0.8, 0.1, 0.0],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.3],
            ]
        )
        links = links_from_similarity(
            similarity,
            ["I", "Beijing", "2026"],
            ["我", "北京", "2026"],
        )
        pairs = {(value["source_index"], value["target_index"]): value for value in links}
        self.assertIn((0, 0), pairs)
        self.assertIn((1, 1), pairs)
        self.assertEqual(pairs[(2, 2)]["method"], "exact_lexical_anchor")
        self.assertEqual(pairs[(2, 2)]["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
