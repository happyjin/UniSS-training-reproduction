"""SimulS2ST-Omni 的 ms-swift 接入层（register / plan / swift_args）。

源训练代码（`simuls2st_omni/train/`，自 SimulS2ST-Omni-stage123 原样迁入）不改动行为；本包只负责把它接到
`swift sft` 上：model_type / template / 数据集 / tuner / checkpoint 格式。设计说明见
`pipelines/simuls2st_omni/README.md`。
"""
