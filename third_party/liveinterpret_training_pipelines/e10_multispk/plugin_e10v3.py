"""swift `--external_plugins` entry for E10 v3 (e10.v3): streaming_si registration as plugin_e10.py, the S5
metrics patch, the v3 data/model patches and the `e10v3` tuner. The speaker-slot token id is resolved here from
the tokenizer of $MODEL_PATH (the same add_s4_stream_tokens the swift template applies)."""
import importlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUSTOM = os.path.join(os.path.dirname(_HERE), "custom")
for p in (_CUSTOM, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

importlib.import_module("streaming_si.integration.register_streaming_si").register_all()
importlib.import_module("token_accuracy_metrics").install_qwen_omnispeak_accuracy_patch()

from e10 import v3 as _v3  # noqa: E402


def _slot_id():
    from transformers import AutoTokenizer
    from streaming_si.data.s4_pipeline import add_s4_stream_tokens
    mp = os.environ["MODEL_PATH"]
    tok = AutoTokenizer.from_pretrained(mp)
    add_s4_stream_tokens(tok, capacity=int(json.load(open(os.path.join(mp, "config.json")))["vocab_size"]))
    return _v3.slot_id(tok)


_v3.install()
_v3.register(_slot_id())
