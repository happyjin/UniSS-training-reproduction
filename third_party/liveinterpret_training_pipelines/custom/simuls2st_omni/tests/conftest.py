"""把 custom/ 放进 sys.path，使测试可以 `import simuls2st_omni`（插件运行时由 custom/plugin.py 负责）。"""
import sys
from pathlib import Path

_CUSTOM_DIR = str(Path(__file__).resolve().parents[2])
if _CUSTOM_DIR not in sys.path:
    sys.path.insert(0, _CUSTOM_DIR)
