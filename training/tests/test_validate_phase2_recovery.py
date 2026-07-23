import tempfile
import unittest
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from training.validate_phase2_recovery import validate


class ValidatePhase2RecoveryTest(unittest.TestCase):
    def _write_events(self, root: Path, *, last_validation: float = 4.7) -> None:
        writer = SummaryWriter(root)
        for step in range(10, 501, 10):
            writer.add_scalar("lm loss", 4.8 - step / 10000, step)
            writer.add_scalar("grad-norm", 2.0, step)
            writer.add_scalar("learning-rate", 5e-5, step)
        for step in range(50, 501, 50):
            value = last_validation if step == 500 else 4.8
            writer.add_scalar("lm loss validation", value, step)
        writer.close()

    def test_accepts_finite_stable_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensorboard = root / "tensorboard"
            log = root / "pilot.log"
            self._write_events(tensorboard)
            log.write_text(
                "number of skipped iterations:   0 | number of nan iterations:   0 |\n",
                encoding="utf-8",
            )
            result = validate(tensorboard, log, 500, 5.3, 5.0, 20.0, 4)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["last_validation_step"], 500)

    def test_rejects_bad_final_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensorboard = root / "tensorboard"
            log = root / "pilot.log"
            self._write_events(tensorboard, last_validation=5.2)
            log.write_text("healthy\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "last validation"):
                validate(tensorboard, log, 500, 5.3, 5.0, 20.0, 4)


if __name__ == "__main__":
    unittest.main()
