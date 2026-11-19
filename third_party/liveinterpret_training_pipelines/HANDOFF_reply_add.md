# Reply-Add 交接文档（总览）

本分支 `reply-add-public` 基于 public 原始版本（`archive` 分支），移植了 **reply-add**
语音到语音翻译（S2S）训练所需的核心代码。本文件是导航总览，机制细节按主题拆分在
`docs/` 下的多个文档中。

> **默认方法：本分支默认使用 `reply_add`。** full-add 与 ICL 为同源可选路径，通过环境变量
> 切换，默认关闭，不影响 reply-add 的运行。

---

## 一句话说清我们改了什么

public 原版的 talker 用的是 **DualRoute Bridge**（把 thinker 语义块**拼接**到 codec 序列前）。
本分支改用一条**并列的新路径**：

- **不走 bridge，改用轻量 `Qwen3TTSThinkerAdapter`**；
- thinker 输出的隐状态经 adapter 投影后，**逐帧"叠加"到 talker 的 codec 序列上**
  （而不是拼接）——这就是 "reply-add / full-add" 的含义；
- 训练时**只训 adapter**（`model.thinker_adapter`），thinker 与 talker 主体冻结。

两条路径共存于同一份 `modeling_qwen_omnispeak.py`，由 `use_upstream_talker_input` 开关选择。

---

## 分支关系

```
archive              纯 public 原始代码（对照基线，无任何本改动）
   │  移植 custom/ 核心 + upstream 入口脚本 + S2S 数据脚本（剔除全部实验产物）
   ▼
reply-add-public     ← 本分支
```

查看本分支相对 public 版的全部改动：
```bash
git diff --stat archive reply-add-public
```

---

## 文档索引（按机制拆分）

| 文档 | 内容 |
|---|---|
| [`docs/01_architecture.md`](docs/01_architecture.md) | **架构对比**：Bridge（拼接）vs Adapter（叠加），为什么改用 adapter、各模块职责、可训练参数。 |
| [`docs/02_reply_add_dataflow.md`](docs/02_reply_add_dataflow.md) | **reply-add 完整数据流**：thinker → adapter → text_projection → 逐帧叠加到 codec；reply-add vs full-add 布局差异；为什么流式友好。 |
| [`docs/03_dataloader.md`](docs/03_dataloader.md) | **数据管线改动**：dataset preprocessor、template encode、`_data_collator` 构造了哪些新张量（`target_text_ids` / `codec_bos_index` / `full_add_prompt_len` 等）。 |
| [`docs/04_env_and_training.md`](docs/04_env_and_training.md) | **环境变量总表 + 入口脚本用法 + 超参 + 实测稳态速度**。 |
| [`docs/05_wds_eos_history_recipe.md`](docs/05_wds_eos_history_recipe.md) | **新版 WDS 训练配方**：indexed-WDS、标准 prompt、10× EOS loss、5% history perturbation、预检与启动命令。 |

---

## 快速开始

```bash
# reply_add（默认），adapter-only，冻结 thinker
bash pipelines/bridge_alignment/train_speech_to_speech_translation_upstream.sh
```

详见 [`docs/04_env_and_training.md`](docs/04_env_and_training.md)。

新版 indexed-WDS 数据使用带预检的公开配方：

```bash
MODEL_PATH=/path/to/checkpoint \
WDS_MANIFEST_DIR=/path/to/manifests \
PYTHON_BIN=/path/to/training-env/python \
  bash pipelines/bridge_alignment/run_reply_add_wds_eos_history.sh
```

该配方是 checkpoint 权重 warm-start，不恢复旧 optimizer/scheduler。详见
[`docs/05_wds_eos_history_recipe.md`](docs/05_wds_eos_history_recipe.md)。

## 快速验证代码可用

```bash
cd <repo>
python3 -c "
import sys, os; sys.path.insert(0, os.path.abspath('custom'))
import importlib; importlib.import_module('qwen_omnispeak.register_qwen_omnispeak')
from qwen_omnispeak.configuration_qwen_omnispeak import QwenOmniSpeakConfig
c = QwenOmniSpeakConfig()
print('use_upstream_talker_input', c.use_upstream_talker_input,
      '| full_add_layout', c.full_add_layout,
      '| use_speaker_prompt', c.use_speaker_prompt)
"
```
