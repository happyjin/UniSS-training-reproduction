#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for schema-aware S2ST Lance task filters."""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from custom.lance_manifest import (  # noqa: E402
    default_s2st_task_filter,
    resolve_s2st_task_filter,
)


class TaskFilterHelperTest(unittest.TestCase):
    def test_legacy_direction_tables(self) -> None:
        filt = default_s2st_task_filter(
            ["direction", "target_audio_utmos_score", "codes"]
        )
        self.assertIn("direction", filt)
        self.assertIn("target_audio_utmos_score", filt)

    def test_s2st_train_view_tables(self) -> None:
        filt = default_s2st_task_filter(
            ["src_lang", "tgt_lang", "source_input_features", "codes"]
        )
        self.assertEqual(filt, "src_lang='zh' AND tgt_lang='en'")

    def test_explicit_overrides_schema(self) -> None:
        self.assertEqual(
            resolve_s2st_task_filter(
                explicit="src_lang='en'",
                column_names=["direction"],
            ),
            "src_lang='en'",
        )

    def test_repairs_ray_stripped_unquoted_eq(self) -> None:
        self.assertEqual(
            resolve_s2st_task_filter(explicit="split=train"),
            "split = 'train'",
        )
        self.assertEqual(
            resolve_s2st_task_filter(explicit="split='train'"),
            "split='train'",
        )


if __name__ == "__main__":
    unittest.main()
