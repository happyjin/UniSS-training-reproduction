# OpenBroadcast-SI v1：音色条件两条线共用的开源全链路测试集

方案依据：`reports/mist_w3po_v1/game_voice_condition_ideas_plan.zh-CN.md` §1.4、§7、§7.2d–§7.2f。

**这是测试集。** 两条线（Idea 1 ZNE、Idea 2 CA-Kalman）都不得在上面做任何选择、调参或训练。只用开源数据构建；电竞数据只贡献了一个统计量，即提示 BVR 的分位数。

---

## 0. 预登记：各评测条件"见过"哪些语料（在运行任何 baseline 之前写定）

### 0.1 三个评测条件

三个条件的语音流**完全相同**（同一批 CoVoST 2 test 说话人、同样的句子和时间轴），只有背景不同。

| 条件 | 会议 ID | 语音语料 | 背景语料 | 与两条线的关系 |
|---|---|---|---|---|
| **ID-2**（领域内） | `obz001…`（zh→en）、`obe001…`（en→zh） | CoVoST 2 test（Common Voice v4 test 划分） | MUSAN 音乐和噪声中留出的 15%，加 FSD50K eval 中留出的 15%（`HELDOUT_BACKGROUNDS*.txt`） | 背景语料与两条线相同，但文件不重叠；说话人不重叠 |
| **OOD-1**（背景语料领域外） | `ooz001…`、`ooe001…` | 同上，逐条相同 | FMA-small 的 Instrumental 类（音乐床）、DEMAND 的 5 个人多的公共环境（人群/环境段）、ESC-50 的 8 类音效 | 三个背景语料两条线都**完全不使用** |
| **clean**（无背景对照） | `ocz001…`、`oce001…` | 同上，逐条相同 | 无 | 上限和"无害"对照 |

ID-2 和 OOD-1 共用同一份背景事件时间表（每段的起止、增益、在文件中的相对位置）、同一个 BVR 目标和同样的压低深度，只把文件换成另一个语料里的。所以两者之差只来自背景语料。

### 0.1b 两个方向是两种结构，结果分开报告

- **zh→en = 长流：** 37 条流、2.0 h；每条 2–4 人，每人多句，流长 61–336 s（中位数 214 s）。说话人取自 CoVoST 2 zh-CN test 中至少有 8 条的说话人。
- **en→zh = 短流，也就是"首句阶段"：** 148 条流、2.0 h；每条 2–4 人，每人恰好 3 句，流长 17–82 s（中位数 47 s）。
  - 原因：CoVoST 2 en→zh test 每个说话人最多只有 3 条（9472 人中，1 条的 4959 人、2 条的 2967 人、3 条的 1546 人），拼不出每人多句的 5 分钟流。
  - 后果：音色提示大多只能从每人最初一两句中取得，所以这个方向衡量的是"刚开口、可用语音很少"时的音色条件。
- 协调者 2026-11-19 同意这一偏差。两个方向的结果分开报告，不合并。

### 0.2 两条线见过的语料（来自方案 §7.1–§7.2c 和各自的预登记）

| 线 | 语音 | 背景 | 其他 |
|---|---|---|---|
| Idea 1（ZNE，不训练） | LibriSpeech test/dev（英）、AISHELL-3 test 说话人（中），只用于诊断和外推族选择 | MUSAN、FSD50K eval（**排除留出文件**） | HDemucs（MUSDB18-HQ 训练）、冻结的 BiCodec 和 Talker |
| Idea 2（CA-Kalman，训练） | LibriSpeech train-clean-100/360、AISHELL-3 train、Common Voice zh（**排除 `HELDOUT_SPEECH.txt` 中的说话人和整个 CoVoST 2 test 划分**） | MUSAN、FSD50K eval（**排除留出文件**），按文件 8:1:1 划分 | WenetSpeech 等真实带噪数据（无监督项） |
| 本基准 | CoVoST 2 test，两条线都没用过 | ID-2：上述两个语料中的留出文件；OOD-1：FMA、DEMAND、ESC-50 | — |

另外说明：
- 被测系统（Thinker `joint_v8h6k s06000`、T1c Talker）的训练数据里没有 CoVoST 2 test。
- FMA 与 MUSAN 音乐中的 fma 子集都来自 Free Music Archive 网站，但不是同一个语料发布。构建时已排除与 MUSAN 音乐同艺术家的曲目，文件不重叠。
- ESC-50 与 FSD50K 都来自 Freesound；Freesound id 出现在 FSD50K（dev 或 eval）中的 ESC-50 片段已排除。

### 0.3 留出清单（已提交）

| 文件 | 内容 |
|---|---|
| `data/open_voicecond_v1/bench/HELDOUT_BACKGROUNDS.txt` | FSD50K eval 留出 1560 / 10231 条，并写明规则 |
| `data/open_voicecond_v1/bench/HELDOUT_BACKGROUNDS_MUSAN.txt` | MUSAN 音乐和噪声，同一规则（下载完成后写入） |
| `data/open_voicecond_v1/bench/HELDOUT_SPEECH.txt` | 基准用到的 553 个 Common Voice 说话人（client_id）和 2609 条片段 |

留出规则：`int(sha1('OpenBroadcast-SI-v1|' + 路径).hexdigest(), 16) % 1000 < 150`，即固定的 15%。路径是 MUSAN 的 tar 成员名（如 `musan/music/fma/music-fma-0003.wav`），或 `FSD50K.eval_audio/<fname>.wav`。

FMA、DEMAND、ESC-50 整体保留给本基准。

---

（以下各节在构建和运行完成后填写。）
