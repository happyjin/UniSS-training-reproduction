"""Stage 3 延迟倍数 m 的稳定抽取（替换源 `StreamingS2STDataset._pick_latency` 的哈希）。

源实现：`draw = (hash((seed, index, "m")) & 0xFFFFFFFF) / 2**32`。Python 的字符串哈希随进程随机化
（PYTHONHASHSEED），且 3.11 起算法由 SipHash-2-4 换成 SipHash-1-3，因此：

* 同一份数据、同一 seed，每次运行抽到的 m 都不同（续训后同一样本的 m 也会变）；
* 源环境（Python 3.10）与本仓库镜像（3.12）即使固定 PYTHONHASHSEED 也抽不到同一个 m。

这里改用 sha256 派生同样的 [0, 1) 均匀数，仍是「对 (seed, index) 稳定、在配置的倍数集合上均匀」，
分布与源一致，但跨进程、跨 Python 版本可复现。`SIMULS2ST_LATENCY_HASH=python` 恢复源的行为。
本模块不依赖 swift / torch，源环境里的对拍脚本也直接复用。
"""
import hashlib
import os

ENV = 'SIMULS2ST_LATENCY_HASH'


def stable_draw(seed: int, index: int) -> float:
    digest = hashlib.sha256(f'{int(seed)}:{int(index)}:m'.encode()).digest()
    return int.from_bytes(digest[:4], 'little') / 0x100000000


def _pick_latency(self, index: int) -> int:
    draw = stable_draw(self.seed, index)
    return self.latency_multipliers[min(int(draw * len(self.latency_multipliers)), len(self.latency_multipliers) - 1)]


def install(dataset_cls) -> bool:
    """给 `StreamingS2STDataset`（源或迁入版本）换上稳定抽取；返回是否已替换。"""
    if os.environ.get(ENV, 'stable') == 'python':
        return False
    dataset_cls._pick_latency = _pick_latency
    return True
