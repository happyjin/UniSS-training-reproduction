import tempfile
import unittest
from pathlib import Path

import numpy as np

from web_demo.streaming_s2st_r2_v1.session_manager import SessionRegistry


class SessionRegistryTest(unittest.TestCase):
    def test_sessions_are_isolated_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SessionRegistry(Path(directory), 1.0, limit=2)
            first = registry.create()
            second = registry.create()
            first.ingress.append((16_000, np.ones(800, dtype=np.float32)))
            self.assertEqual(first.ingress.sample_count, 800)
            self.assertEqual(second.ingress.sample_count, 0)
            self.assertNotEqual(first.ensure_request_dir(), second.ensure_request_dir())
            with self.assertRaises(RuntimeError):
                registry.create()
            registry.discard(first.session_id)
            self.assertEqual(len(registry), 1)

    def test_microphone_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            session = SessionRegistry(Path(directory), 0.05).create()
            with self.assertRaises(ValueError):
                session.ingress.append((16_000, np.ones(801, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
