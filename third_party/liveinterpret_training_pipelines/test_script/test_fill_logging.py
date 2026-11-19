#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FillRate next() timer — no Lance, no GPU, no cluster."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ["FILL_LOG_ENABLED"] = "1"
os.environ["FILL_LOG_RANK"] = "0"
os.environ["RANK"] = "0"

from custom import fill_logging as fl  # noqa: E402


class FillLoggingTest(unittest.TestCase):
    def setUp(self) -> None:
        fl.reset_wait()
        fl._state["last_step_t"] = time.monotonic()
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["FILL_LOG_PATH"] = os.path.join(self._tmpdir.name, "fill_log.jsonl")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("FILL_LOG_PATH", None)

    def test_timed_iterator_records_wait(self) -> None:
        class _Slow:
            def __init__(self):
                self.n = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.n >= 2:
                    raise StopIteration
                self.n += 1
                time.sleep(0.03)
                return self.n

        it = fl._TimedIterator(_Slow())
        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertGreaterEqual(fl._state["wait_s"], 0.05)
        self.assertEqual(fl._state["n_next"], 2)

    def test_attach_next_timer_on_dataloader(self) -> None:
        try:
            import torch.utils.data as tud
        except ImportError:
            self.skipTest("torch not installed in this environment")

        class _DS(tud.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, i):
                time.sleep(0.01)
                return i

        loader = tud.DataLoader(_DS(), batch_size=2, num_workers=0)
        timed = fl.attach_next_timer(loader)
        timed2 = fl.attach_next_timer(timed)  # idempotent
        self.assertIs(timed, timed2)
        n = 0
        for _ in timed:
            n += 1
            if n >= 2:
                break
        self.assertEqual(n, 2)
        self.assertGreaterEqual(fl._state["wait_s"], 0.015)
        self.assertIsInstance(timed, fl._FillTimedLoader)

    def test_wraps_concrete_trainer_loader_override(self) -> None:
        class _Loader:
            def __iter__(self):
                return fl._TimedIterator(iter([1]))

            def __len__(self):
                return 1

        class _Trainer:
            def get_train_dataloader(self):
                return _Loader()

        trainer = _Trainer()
        fl._wrap_trainer_loader_instance(trainer)
        fl._wrap_trainer_loader_instance(trainer)
        loader = trainer.get_train_dataloader()
        self.assertIsInstance(loader, fl._FillTimedLoader)
        self.assertEqual(list(loader), [1])

    def test_flush_writes_fill_rate(self) -> None:
        fl.note_wait(0.1)
        time.sleep(0.2)
        rec = fl.flush_step(7)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["step"], 7)
        self.assertGreaterEqual(rec["t_wait_ms"], 90.0)
        self.assertGreaterEqual(rec["t_step_ms"], 190.0)
        self.assertLess(rec["fill_rate"], 0.7)
        self.assertGreater(rec["fill_rate"], 0.0)
        path = os.environ["FILL_LOG_PATH"]
        with open(path) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_next"], 1)


if __name__ == "__main__":
    unittest.main()
