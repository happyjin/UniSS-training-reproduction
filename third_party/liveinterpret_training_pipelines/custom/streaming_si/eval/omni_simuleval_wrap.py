"""官方 `python -m simuleval.cli` 的薄包装：只给语音输出实例的日志补上计算感知时间戳。

SimulEval 的 `SpeechOutputInstance.receive_prediction` 每收到一段语音都会记 `elapsed`
（已读源音频 ms + 自本句开始的墙钟 ms，即 SimulEval 的 computation-aware 口径），
但 `summarize()` 写日志时把它写成了 `"elapsed": []`，计算感知延迟因此无法事后计算。
这里在 summarize 返回值里补回 `elapsed`，并加一个 `wall_sec`（本句总墙钟，用于 RTF）。
推理逻辑、agent、调度循环全部不动，其余参数原样传给官方 CLI。

可选：环境变量 `SIMULS2ST_MERGE_ADAPTERS_FROM=<Stage 2 目录>` 时，在官方 `from_pretrained`
加载 `--checkpoint-path`（Stage 3 adapter）之前，先把该 Stage 2（thinker LoRA r32 + talker 全量）
合进基座——与训练时 Stage 3 的起点（`merge_adapters_from`）一致。用于评测复现的 Stage 3：
官方 agent 没有这个参数，直接叠 Stage 3 的 r8 LoRA 到未合 Stage 2 的基座上是错的。见 `omni_merge.py`。

用法：python omni_simuleval_wrap.py <官方 simuleval.cli 的全部参数>
"""
import os
import runpy
import sys
import time

from simuleval.evaluator import instance as _inst

_orig_summarize = _inst.SpeechOutputInstance.summarize


def _summarize(self):
    info = _orig_summarize(self)
    info['elapsed'] = list(getattr(self, 'elapsed', []) or [])
    start = getattr(self, 'start_time', None)
    if start is not None and 'wall_sec' not in info:
        info['wall_sec'] = round(time.time() - start, 4)
    return info


_inst.SpeechOutputInstance.summarize = _summarize

if os.environ.get('SIMULS2ST_MERGE_ADAPTERS_FROM'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from omni_merge import install_stage2_merge
    install_stage2_merge(os.environ['SIMULS2ST_MERGE_ADAPTERS_FROM'])

if __name__ == '__main__':
    sys.argv = ['simuleval.cli'] + sys.argv[1:]
    runpy.run_module('simuleval.cli', run_name='__main__', alter_sys=True)
