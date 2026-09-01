# B:均匀 CE —— 只改一个从未被改过的权重

## 动机

**`boundary_eos` 在本仓库的每一个脚本里都是 CLI 默认值 0.10,从未被改过。**
而在交织家族里:

| kind | 占监督 token | 权重 | **梯度占比** | **本次(均匀)** |
|---|---:|---:|---:|---:|
| semantic | 60.2% | 1.00 | **85.8%** | **60.2%** |
| **boundary**(WAIT/WRITE、TASK_*、END_CONTENT、END_SEMANTIC、语言、速度) | **32.8%** | **0.10** | **4.7%** | **32.8%** |
| asr | 3.8% | 1.00 | 5.4% | 3.8% |
| mt | 2.9% | 1.00 | 4.1% | 2.9% |

UniSS 自己的论文报告的是**纯 next-token CE、无任何辅助 loss**
(`ℒ_AR = −∑ log P_θ`)。本次就是把那个配方用在决策 token 上。

顺带:声学 token 的梯度占比从 85.8% 降到 60.2% —— 这是 SimulS2ST-Omni 用
Thinker-Talker 双流解决的 modality interference 的廉价部分版本
(他们的架构改动带来 +4.0 ASR-BLEU)。

## 与父运行的差别

| | `iter_0002264`(父) | **本次** |
|---|---|---|
| 数据 | `task_pool_formal_p4_20260820T154500Z`,5 家族 | **完全相同** |
| 几何 | MBS 2 / GBS 128 / seq 18000 / seed 20260819 | **完全相同** |
| `asr_ce` / `mt_ce` / `semantic_ce` | 1.0 | **相同** |
| `replay_ce` / `v1_asr_kl` / `phase3_kl` / `commit` | 0.50 / 0.30 / 0.25 / 0.20 | **相同** |
| **`boundary_eos`** | **0.10** | **1.0** |
| `semantic_end_ce` / `semantic_end_margin` | 0.50 / 0.25 | **0 / 0** |
| 全部 roll-in / binary / prefix-corruption | 部分开启 | **全部 0** |
| dataloader workers | 0 | **8** |

**保留两个 KL 的理由:** 实测 `phase3_kl` −0.181、`v1_asr_kl` −0.013,一直在降,
是有效的防遗忘。本次只改一件事。

**撤掉 margin 家族的理由:** 三次独立实验把推理侧决策 gap 单调推向错误方向
(−2.88 → −3.75 → −4.97),同时 gold 行分离度每次都朝正确方向动。

## 为什么 workers 从 0 改成 8

`runtime_dataset.py::__getitem__` 每次访问都 `with self.path.open("rb")`,
**不跨调用持有文件句柄**,所以多进程 dataloader 是安全的。

而既有默认本来就是 8(`run_e2e_megatron.sh:32`),是本血脉的三个续训脚本
把它硬写成了 0。父运行 `endmargin_epoch23` 的 `gpu.csv`(60,440 个采样点)显示
**平均利用率 47%、46% 的采样点为 0%、平均功率 281 W/卡**。

**这会改变步时,因此与父运行的墙钟不可比,但不改变任何训练数学。**

## 代码改动

**零新 Python。** 本次用既有 trainer `pretrain_e2e_megatron.py` 原封不动。

| 文件 | 与既有的差别 |
|---|---|
| `scripts/run_e2e_megatron_uniform.sh` | 既有 `run_e2e_megatron.sh` 的副本,**只加 2 处**:`RUN_BOUNDARY_EOS_WEIGHT` 声明 + `--e2e-boundary-eos-weight` 传递(该参数既有脚本从不传,所以一直吃 CLI 默认 0.10) |
| `scripts/run_8gpu.sh` | 权重配置 + workers |

10 项单测锁住:2-hunk 差异、boundary_eos=1.0、全部 margin 为 0、
**不安装任何 objective 扩展**、KL 未被覆盖、父 checkpoint、workers≠0、
dataset 无跨调用句柄、数据与几何未变。

## 门禁

同一份 frozen fixed-16 `SELECTION.json`,δ=5 + rp=1.1 w8 配置下与
`continue_end` 的 6/6 对照;并用 `family_logit_probe` 测推理侧决策 gap。

**falsification:** gap 从 −2.88 往 0 移动 → 方向对;仍不动或反向 →
teacher-forced 无论怎么加权都修不了这个决策,转 C(prefix-to-prefix 重构)。

顺带看 eng→cmn 的 chrF 是否从 15.0 上升 —— `incremental_mt_event` 家族有
34.3% 的监督 token 是 boundary,均匀 CE 会把它解放出来。
