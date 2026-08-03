import unittest
from pathlib import Path


class AuditLayoutTest(unittest.TestCase):
    def test_experiment_root_is_isolated(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(root.name, "uniss_streamspeech_ctc_v1")
        self.assertTrue((root / "stage00_audit" / "audit_inputs.py").is_file())


if __name__ == "__main__":
    unittest.main()
