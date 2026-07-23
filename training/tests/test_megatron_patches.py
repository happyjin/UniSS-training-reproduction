import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_SCRIPT = REPO_ROOT / "scripts/apply_megatron_full_validation_patch.sh"
PATCH_FILE = REPO_ROOT / "training/patches/megatron_full_validation_scalar_eval_iters.patch"


class MegatronPatchesTest(unittest.TestCase):
    def test_full_validation_scalar_eval_iters_patch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            megatron_root = Path(tmp) / "Megatron-LM"
            target = megatron_root / "megatron/training/training.py"
            target.parent.mkdir(parents=True)
            target.write_text(
                ("\n" * 4266)
                + "        # with full validation we need to distribute eval_iters to all ranks\n"
                "        if mpu.get_tensor_model_parallel_rank() == 0:\n"
                "            eval_iters = torch.tensor(args.eval_iters, dtype=torch.long, device='cuda')\n"
                "        else:\n"
                "            eval_iters = torch.tensor([0] * len(eval_iters), dtype=torch.long, device='cuda')\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init", "-q", str(megatron_root)], check=True)
            env = os.environ.copy()
            env.update(
                {
                    "MEGATRON_ROOT": str(megatron_root),
                    "MEGATRON_FULL_VALIDATION_PATCH": str(PATCH_FILE),
                }
            )

            first = subprocess.run(
                [str(PATCH_SCRIPT)], env=env, text=True, capture_output=True, check=True
            )
            second = subprocess.run(
                [str(PATCH_SCRIPT)], env=env, text=True, capture_output=True, check=True
            )

            self.assertIn("Applied Megatron", first.stdout)
            self.assertIn("already applied", second.stdout)
            rendered = target.read_text(encoding="utf-8")
            self.assertIn("torch.tensor(eval_iters,", rendered)
            self.assertNotIn("torch.tensor(args.eval_iters,", rendered)


if __name__ == "__main__":
    unittest.main()
