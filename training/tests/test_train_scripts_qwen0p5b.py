import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(*args: str) -> str:
    env = os.environ.copy()
    env.setdefault("USER_ROOT", "/opt/dlami/nvme/jasonleeeli")
    result = subprocess.run(
        [str(REPO_ROOT / args[0]), *args[1:]],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


class Qwen0p5BTrainScriptsTest(unittest.TestCase):
    def test_download_target_dry_run_uses_qwen0p5b(self):
        output = run_script("scripts/download_hf_assets.sh", "--dry-run", "qwen0p5b")
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", output)
        self.assertIn("pretrained_models/Qwen2.5-0.5B-Instruct", output)

    def test_phase1_qwen0p5b_dry_run_shape_and_paths(self):
        output = run_script("scripts/train_phase1_qwen0p5b.sh", "--dry-run")
        self.assertIn("--num-layers 24", output)
        self.assertIn("--hidden-size 896", output)
        self.assertIn("--ffn-hidden-size 4864", output)
        self.assertIn("--num-attention-heads 14", output)
        self.assertIn("--num-query-groups 2", output)
        self.assertIn("--vocab-size 180407", output)
        self.assertIn("--seq-length 18000", output)
        self.assertIn("--global-batch-size 128", output)
        self.assertIn("checkpoints/qwen2_0p5b_uniss_vocab", output)
        self.assertIn("data/megatron/phase1_unist13/packed_train.jsonl", output)
        self.assertNotIn("qwen2_1p5b_uniss_vocab", output)

    def test_phase2_qwen0p5b_dry_run_shape_and_paths(self):
        output = run_script("scripts/train_phase2_qwen0p5b.sh", "--dry-run")
        self.assertIn("--num-layers 24", output)
        self.assertIn("--hidden-size 896", output)
        self.assertIn("--ffn-hidden-size 4864", output)
        self.assertIn("checkpoints/uniss_qwen0p5b_phase1", output)
        self.assertIn("checkpoints/uniss_qwen0p5b_phase2", output)
        self.assertIn("data/megatron/phase2_unist13_mix/packed_train.jsonl", output)
        self.assertNotIn("qwen2_1p5b_uniss_vocab", output)

    def test_phase3_qwen0p5b_reuses_current_phase2_data(self):
        output = run_script("scripts/train_phase3_qwen0p5b.sh", "--dry-run")
        self.assertIn("--num-layers 24", output)
        self.assertIn("--hidden-size 896", output)
        self.assertIn("--ffn-hidden-size 4864", output)
        self.assertIn("--lr 5e-5", output)
        self.assertIn("--lr-decay-style cosine", output)
        self.assertIn("checkpoints/uniss_qwen0p5b_phase2", output)
        self.assertIn("checkpoints/uniss_qwen0p5b_phase3", output)
        self.assertIn("data/megatron/phase2_unist13_mix/packed_train.jsonl", output)

    def test_existing_1p5b_phase1_dry_run_is_unchanged(self):
        output = run_script("scripts/train_phase1.sh", "--dry-run")
        self.assertIn("--num-layers 28", output)
        self.assertIn("--hidden-size 1536", output)
        self.assertIn("--ffn-hidden-size 8960", output)
        self.assertIn("--num-attention-heads 12", output)
        self.assertIn("--num-query-groups 2", output)
        self.assertIn("checkpoints/qwen2_1p5b_uniss_vocab", output)
        self.assertNotIn("qwen2_0p5b_uniss_vocab", output)


if __name__ == "__main__":
    unittest.main()
