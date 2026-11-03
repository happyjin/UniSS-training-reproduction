#!/usr/bin/env python3
"""Compatibility entry point for train_mist_workload.py."""
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, "-B", "-u",
                             str(Path(__file__).with_name("train_mist_workload.py")), *sys.argv[1:]])
