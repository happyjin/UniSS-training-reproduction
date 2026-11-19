#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DataLoader multiprocessing_context injection (Lance fork-safety)."""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("DATALOADER_CPU_BIND", "0")

from custom import dataloader_cpu_bind as dcb  # noqa: E402
import torch.utils.data as tud  # noqa: E402


class DataloaderMpContextTest(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("DATALOADER_MP_CONTEXT", None)
        dcb._MP_CONTEXT_LOGGED = False
        dcb._DL_INDEX.clear()

    def test_injects_forkserver(self) -> None:
        os.environ["DATALOADER_MP_CONTEXT"] = "forkserver"
        loader = tud.DataLoader([0, 1, 2, 3], batch_size=2, num_workers=2)
        ctx = loader.multiprocessing_context
        name = getattr(ctx, "get_start_method", lambda: ctx)()
        self.assertEqual(name, "forkserver")
        self.assertIsInstance(loader.worker_init_fn, dcb._WorkerInit)

    def test_num_workers_zero_skips(self) -> None:
        os.environ["DATALOADER_MP_CONTEXT"] = "forkserver"
        loader = tud.DataLoader([0, 1, 2, 3], batch_size=2, num_workers=0)
        self.assertIsNone(loader.multiprocessing_context)

    def test_explicit_context_not_overridden(self) -> None:
        os.environ["DATALOADER_MP_CONTEXT"] = "forkserver"
        loader = tud.DataLoader(
            [0, 1, 2, 3],
            batch_size=2,
            num_workers=2,
            multiprocessing_context="spawn",
        )
        ctx = loader.multiprocessing_context
        name = getattr(ctx, "get_start_method", lambda: ctx)()
        self.assertEqual(name, "spawn")

    def test_unset_env_keeps_default(self) -> None:
        os.environ.pop("DATALOADER_MP_CONTEXT", None)
        loader = tud.DataLoader([0, 1, 2, 3], batch_size=2, num_workers=2)
        ctx = loader.multiprocessing_context
        self.assertTrue(ctx is None or getattr(ctx, "get_start_method", lambda: None)() == "fork")

    def test_worker_init_discards_inherited_lance_reader(self) -> None:
        from custom import wds_audio
        inherited = object()
        wds_audio._LANCE_READER = inherited
        wds_audio._LANCE_READER_PID = os.getpid()
        dcb._WorkerInit(None, 0, 1)(0)
        self.assertIsNone(wds_audio._LANCE_READER)
        self.assertIs(wds_audio._FORK_ORPHANS[-1], inherited)
        wds_audio._FORK_ORPHANS.clear()
        wds_audio._LANCE_READER = None
        wds_audio._LANCE_READER_PID = None


if __name__ == "__main__":
    unittest.main()
