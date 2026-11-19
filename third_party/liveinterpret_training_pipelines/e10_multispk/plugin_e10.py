"""swift `--external_plugins` entry for E10 (instead of custom/plugin.py, which also pulls in unrelated model
families). Registers the streaming_si model/template/datasets/S5 tuner exactly as custom/plugin.py does, then the
E10 patches, the `e10_ms` tuner and the `e10_grouped` optimizer."""
import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUSTOM = os.path.join(os.path.dirname(_HERE), "custom")
for p in (_CUSTOM, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

importlib.import_module("streaming_si.integration.register_streaming_si").register_all()

from e10 import patch as _patch  # noqa: E402
from e10 import tuner as _tuner  # noqa: E402

_patch.install()
_tuner.register()
