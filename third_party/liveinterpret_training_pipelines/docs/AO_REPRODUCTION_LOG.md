# Lance 数据源实验记录：AO 复刻 + zero0/zero1 对照 + WDS 基准对比

> 记录日期：2026-09-05 ~ 2026-09-07。
> 分支：`feature/deferzhou/merge-training-metrics` @ `85b6d75`。
> 目的：复刻 `docs/TRAINING_METRICS_MERGED.md` 里的 **AO**（Lance + ray+deepspeed
> zero0+liger + 双节点 + GBS=4096）任务，验证其报告数值，并进一步做
> zero0/zero1 对照、Lance/WDS 单变量对照，回答"Lance 数据源在 ray 双节点
> GBS=4096 场景下到底有没有性能提升"。

---

## 1. 第一次复刻（失败）：`ao-repro-a4bq`

原样用 deferzhou 的 `ktrain/ktrainjob_sts_zeyang_custom_data_lance_train.yaml`
（未修改）提交，节点 028+08e3，ray 双节点，zero0+liger+Lance，GBS=4096。

**结果明显偏离 AO 报告值**：

| 指标 | AO报告值 | 复刻值 | 差异 |
|---|---|---|---|
| loss@200 | 5.5979 | 5.76694 | **+0.169，明显更高** |
| 显存@200 | 100.86 GiB | 97.49 GiB | −3.3% |
| 200步耗时 | 1h8m40s | 46m50s | 快32% |
| 稳态窗口单步耗时方差 | 中位10.610/均值10.769（差1.5%，很稳） | 4.973~42.996s（**同窗口内最快最慢差8.6倍**） | 方差完全不是一个量级 |

**根因排查**：用 `git log -p -L` 查 `common_args.sh` 里 `GROUP_BY_LENGTH` 默认值的
历史，发现：

```
commit 85b6d750（本次复刻所用 commit）
+: "${GROUP_BY_LENGTH:=true}"
```

这一行是**纯新增**（diff `@@ -181,0 +181,1 @@`），说明 AO 原始跑实验时这条
默认值语句根本不存在，实际生效的是更早期（master）的 `false`。yaml 头部注释
声称"GROUP_BY_LENGTH=false 由预设/common_args 给出"在她当时是对的，但检出
`85b6d750` 之后这个默认值已经被静默改成了 `true`——**我们复刻时检出的代码比
AO 实际跑实验时的代码新，中间偷偷改了脚本默认值**，不是配置抄错，也不是
AO 数据有问题。

## 2. 第二次复刻（成功）：`ao-repro2-txy1`

在 yaml 里显式加 `GROUP_BY_LENGTH: "false"`（不依赖脚本默认值），其余原样，
重新提交。

**结果与 AO 报告值几乎精确吻合**：

| 指标 | AO报告值 | 复刻值 | 差异 |
|---|---|---|---|
| loss@200 | 5.5979 | 5.60770 | +0.0098（噪声范围内） |
| 显存@200 | 100.86 GiB | **100.86 GiB** | **完全一致** |
| 200步耗时 | 1h8m40s | **1h8m49s** | 差9秒（0.2%） |
| 稳态单步（中位/均值） | 18.983 / 19.410 | 19.150 / 19.445 | 差0.9% / 0.18% |
| 稳态吞吐（样本/s） | 215.8 | 213.9（中位）/ 210.6（均值） | 差0.9%~2.4% |
| 稳态窗口方差 | 低（median≈mean） | 低（17.5~22.8s，PadWaste稳定~60%） | 方差量级一致 |

**结论：AO 的原始数据真实可信，问题完全出在脚本默认值随 commit 漂移。**
`GROUP_BY_LENGTH` 从 false→true 会启用长度分桶，改变每个 micro-batch 的样本
构成，直接导致 loss 轨迹偏离、单步 token 数暴涨式波动（因为不同长度桶之间的
计算量差异被放大暴露）。

## 3. Lance vs WDS 单变量对照：`lance-zero1-cagi`

第二次复刻用的是 zero0，而 GBS4096 家族里能直接横比的 WDS 基准列是 **S**
（ray+**zero1**+liger+WDS+GBS4096+双节点，来自 `GBS512_GBS4096_BASELINE_ABCD`
文档 D 列 / MERGED 文档 S 列）。为了跟 S 做真正单变量对照（只换数据源，
不换 ZeRO stage），另建 `ktrainjob_sts_zeyang_custom_data_lance_train_zero1.yaml`
——在 AO 原 yaml 基础上把 `DEEPSPEED` 从 `zero0` 改成 `zero1`，其余
（Lance数据源/mel/speaker预计算/liger/GROUP_BY_LENGTH=false/per_device=32×ga8）
逐项对齐，提交为 `lance-zero1-cagi`，跑满200步。

**结果：Lance 明显慢于 WDS，不是提升，是倒退**：

| 指标 | S（ray+zero1+liger+**WDS**） | 本次（ray+zero1+liger+**Lance**） | 差异 |
|---|---|---|---|
| 稳态 s/it（191-200） | **12.22s** | 19.42s（中位）/ 19.67s（均值） | **慢 59-61%** |
| 累计 s/it@200 | 13.59 | 20.85 | 慢53% |
| 200步耗时 | 45m18s | 1h9m30s | 慢53% |
| 显存@200 | 82.77 GiB | 81.13 GiB | 基本一致（低2%） |
| loss@200 | 5.59102 | 5.59434 | 差0.0033，噪声范围内 |

loss 几乎完全重合（证明两边吃到的是同一批数据、数学结果一致），但稳态单步
耗时慢了近60%。**在这个 ray + GBS4096 + 双节点场景下，Lance 数据路径本身
（mel/speaker预计算的批量take延迟，或16个ray actor同时读6.2TiB全表的IO
竞争）拖慢了训练，跟ZeRO stage、liger、GROUP_BY_LENGTH都无关。**

## 4. zero0 vs zero1（Lance 数据路径内部对照）

用第2节（zero0+Lance）和第3节（zero1+Lance）两次实验直接对比——两者都是
ray+Lance+GBS4096+双节点+GROUP_BY_LENGTH=false，唯一变量=ZeRO stage：

| 指标 | zero0（`ao-repro2-txy1`） | zero1（`lance-zero1-cagi`） | 差异 |
|---|---|---|---|
| 稳态s/it（中位/均值） | 19.150 / 19.445 | 19.416 / 19.672 | zero0快约1.4%（中位）/1.2%（均值） |
| 200步耗时 | 1h8m49s | 1h9m30s | zero0快41秒（~1%） |
| 显存@200 | **100.86 GiB** | **81.13 GiB** | zero0多用24.3% |
| loss@200 | 5.60770 | 5.59434 | 差0.0134，噪声范围内 |

**zero0 比 zero1 略快、但显存开销明显更大——方向跟 WDS/torchrun 场景下的
D→G 对照（zero1→zero0：12.22s→11.57s快5.3%，显存82.77→100.04GiB+20.9%）
完全一致**，只是在 Lance+ray 这个组合下 zero0 的速度优势被大幅稀释
（1.4% vs 5.3%），因为此时训练的瓶颈已经不在 ZeRO 通信开销上，而是
Lance 数据路径本身的开销主导了单步耗时（两者稳态都卡在~19.5s附近，
远慢于 WDS 的 12.22s）。

## 5. 四方横向对比（GBS=4096 双节点 + liger，全部 DP=16，per_device=32×ga8）

| 编号/任务名 | 框架 | ZeRO | 数据源 | 稳态s/it | 吞吐(样本/s) | 显存@200(GiB) | 200步耗时 | loss@200 |
|---|---|---|---|---|---|---|---|---|
| D/S（历史数据） | ray | zero1 | WDS | **12.22** | 335 | 82.77 | **45m18s** | 5.59102 |
| G/V（历史数据） | torchrun | zero0 | WDS | **11.57** | 354 | 100.04 | **42m50s** | 5.59615 |
| `ao-repro2-txy1` | ray | zero0 | **Lance** | 19.150(中位) | 213.9 | 100.86 | 1h8m49s | 5.60770 |
| `lance-zero1-cagi` | ray | zero1 | **Lance** | 19.416(中位) | 211.0 | 81.13 | 1h9m30s | 5.59434 |

**速度排序（快→慢）**：G/V（torchrun+zero0+WDS，11.57s）> D/S（ray+zero1+WDS，
12.22s，慢5.6%）> `ao-repro2`（ray+zero0+Lance，19.15s，慢比D/S 57%、比G/V
65.6%）> `lance-zero1`（ray+zero1+Lance，19.42s，全场最慢）。

**交叉观察**：

- **数据源单变量对照（控制ray+zero1）**：D/S(12.22) vs `lance-zero1`(19.42)，
  **Lance 慢 59%**——四组里唯一干净的单变量对照，样本量最大、控制最严格。
- **数据源对照（控制zero0，但框架从torchrun变ray）**：G/V(11.57) vs
  `ao-repro2`(19.15)，差65.6%，但混了框架+数据源两个变量，无法拆分。
- **框架单变量对照（ray+zero0+WDS）缺失**：该组合此前直接 OOM 崩溃（step2），
  从未产出有效吞吐数字，是四方对比里唯一的空白格。
- **显存**：与 ZeRO stage 强相关（zero0 恒高于 100GiB，zero1 恒低于 83GiB），
  与数据源/框架基本无关，符合 DeepSpeed 引擎预期。
- **数学结果**：4 组 loss@200 落在 5.591~5.608，跨度仅 0.017，全部收敛一致，
  说明上述所有速度差异都是效率问题，不涉及正确性问题。

## 6. 单机场景 Lance vs WDS 对照：`lance-1n-z1-z5is`（颠覆双节点结论的方向）

第3节的"Lance 慢 59%"是在 **ray + 双节点 + 16 actor** 场景下测的，怀疑根因
可能是"Lance 数据路径本身的开销"，也可能是"ray 多 actor 并发读 6.2TiB 全表
的 IO 竞争"。为拆分这两个可能性，去掉 ray、去掉双节点，做纯**单机单进程**
Lance vs WDS 对照：另建 `ktrainjob_junyaoyang_gbs4096_liger_zero1_lance_1node.yaml`
——对齐历史 **R 列**（`sts-cmp200-zero1-liger-gbs4096-1node`：deepspeed 单机
+zero1+liger+**WDS**+GBS4096，稳态24.99s/it，已在85b6d750收敛yaml时被删除，
参数从 `git show 85b6d750^:ktrain/...` 还原），唯一变量=数据源，其余
（zero1/liger/GBS4096/per_device=64×ga8/单机8卡/GROUP_BY_LENGTH=false）
逐项对齐，提交为 `lance-1n-z1-z5is`，跑满200步。

**结果：单机场景下 Lance 反而比 WDS 快，方向跟双节点完全相反**：

| 指标 | R（deepspeed单机+zero1+liger+**WDS**） | 本次（deepspeed单机+zero1+liger+**Lance**） | 差异 |
|---|---|---|---|
| 稳态s/it（191-200） | **24.99s** | 18.52s（中位）/19.24s（均值） | **Lance快23-26%** |
| 200步耗时 | 1h23m18s | 1h3m23s | **Lance快24%** |
| 显存@200 | 136.88 GiB | 137.25 GiB | 基本一致（差0.27%） |
| loss@200 | 5.58948 | 5.58757 | 差0.0019，噪声范围内 |
| PadWaste | — （R年代未埋点） | 64.6%（中位） | 跟双节点Lance的61-65%量级一致 |

loss几乎完全重合（数学结果一致），稳态窗口方差也很低（CV=5.4%，175个样本
里最快17.5s最慢22.6s），跟双节点lance-zero1-cagi的方差量级（CV≈9%）接近，
**不是偶发的，是稳定可重复的"单机更快"**。

**结论（第一版，后被第7节推翻/修正）：双节点场景下 Lance 变慢的根因更可能是
"ray 多 actor 并发读全表的 IO 竞争"，而不是 Lance 数据路径本身的固有开销**
——因为把 ray 和多节点并发都去掉后，Lance 的固有开销完全不足以拖累它，反而
比 WDS 更快。

## 7. 重大修正：R 列（24.99s/it）本身不可信——发现同分支 WDS 单机也有性能回归

用户追问"为什么以前能跑的WDS现在跑不动/跑很慢"后，做了更细致的排查，发现
第6节的对照本身**方法论上有漏洞**：

### 7.1 双节点 WDS+ray+zero1 同commit补跑：卡在预处理，非节点/数据问题

补跑 `ktrainjob_sts_zeyang_custom_data_train_zero1_ray2n.yaml`（WDS版
lance-zero1-cagi，同commit 322ef57）两次，均卡在训练前的"Map"预处理阶段
数小时（单进程700-870 examples/s，1919万行，进度条无`(num_proc=N)`标记，
证明`num_proc`实际生效值是1，理论上脚本默认应是135）。排查并排除：
① 节点028数据被删——用`kubectl debug node`直接查证028上音频shards（2.2T）
和FSX manifest（198个train分片，140G）均完整存在，非数据问题；
② 节点硬件问题——单机WDS在08e3上跑通（见7.2），历史上028+08e3这两个节点
组合跑过同款任务是快的（D/S列45m18s），硬件本身没问题。
**该bug根因仍未定位**（ray driver进程的CPU探测/`DATASET_NUM_PROC`环境变量
传递可能有问题），但已确认只在 ray+双节点场景出现，单机不触发，跟数据/
节点硬件无关。

### 7.2 单机 WDS 同commit重跑：不卡预处理，但稳态慢近一倍

为验证7.1的bug是否代码级（而非ray/双节点特有），另建单机WDS对照
`ktrainjob_junyaoyang_gbs4096_liger_zero1_wds_1node.yaml`（对齐R列，同
commit 322ef57），先后在028（Pending，GPU被无关任务`chaser-test`占用未能
跑）、08e3（成功）上提交：

- **预处理正常**（11分钟，不卡7.1那个bug，证明bug是ray-specific）。
- **但稳态单步耗时 ~45.6s/it（中位，43个样本）**，比R列的24.99s/it慢近
  **83%**——这是一个第6节对照完全没发现的新问题：**R列的24.99s/it用的是
  老commit数据，跟当前commit的单机WDS表现根本不是同一个基准**，第6节的
  "Lance单机快23-28%"实际上是"当前commit的Lance" vs "老commit的WDS"，
  不是苹果对苹果。

### 7.3 根因定位：`85b6d750` 顺手改小了 DataLoader 预取队列深度（8倍）

`git diff eaaa3a88（R列yaml创建/运行时的commit）..HEAD` 精确定位到：

```diff
common_args.sh:
- : "${DATALOADER_PREFETCH_FACTOR:=8}"
+ : "${DATALOADER_PREFETCH_FACTOR:=2}"

train_speech_to_speech_translation.sh（SCENE=batch4_8_adapter_cmp）:
-         : "${DATALOADER_NUM_WORKERS:=16}"
-         : "${DATALOADER_PERSISTENT_WORKERS:=true}"
-         : "${DATASET_NUM_PROC:=16}"
```

全部来自同一个commit `85b6d750`（deferzhou，"合并训练实验对照文档并收敛
任务yaml"），commit message自述"SCENE预设不再硬编码workers/persistent/
dataset_num_proc"——是一次代码整洁化重构（把散落在SCENE分支里的硬编码值
收拢到`common_args.sh`的动态公式），**不是故意调慢**，但产生了副作用：
单机场景下 `DATALOADER_NUM_WORKERS` 从硬编码16退化为动态公式算出的8
（`CONTAINER_CPUS/(2×LOCAL_WORLD_SIZE)=135/16≈8`），叠加`PREFETCH_FACTOR`
8→2，**DataLoader总"在飞batch数"（workers×prefetch）从16×8=128缩水到
8×2=16，缩水8倍**——WDS现场解码音频这种IO重活，预取深度不够就喂不饱GPU。

`PREFETCH_FACTOR`默认改2很可能参照了`TRAINING_METRICS_MERGED.md`3.3节
"DataLoader队列专项"（W-AG那组，结论"8×prefetch2已喂满，加深无收益"），
但**该实验明确在200K热缓存子集上测的**（文档4.3节自认"测不出6.2TiB全表
的随机读压力"），拿小样本的结论去改全量1900万行的默认值，是这次slowdown
的根源。

### 7.4 修复尝试：直接照抄老参数会导致 CPU 超配，需按当前核数重算

**第一次尝试**（显式设 `DATALOADER_NUM_WORKERS=16` + `PREFETCH_FACTOR=8`）
**翻车更严重**：稳态耗时在28~142s间剧烈跳动，比缩水默认值的45s还差。
根因：`common_args.sh`自己写的约束"总worker = NUM_WORKERS×2(train+eval)×
LOCAL_WORLD_SIZE"——16×2×8=**256个进程抢135核**，严重超配，进程互相抢核
的上下文切换开销盖过了预取深度的收益。**说明"R列用16个worker"这个历史
数字本身可能是在不同的容器CPU配额/eval开关条件下才成立的，不能无脑照抄**。

**第二次尝试**（只显式覆盖 `PREFETCH_FACTOR=8`，`NUM_WORKERS`交给动态公式
留在8，8×2×8=128不超配）：稳态收敛到 **32.1~32.9s/it**（86步样本，很稳，
CV低），比45s好很多，但仍比R的24.99s慢约30%——**总在飞batch数
`8×8=64`，只有R当年`16×8=128`的一半**，差距应该出在这里，理论上把
`PREFETCH_FACTOR`再提到16（`8×16=128`，跟R的总缓冲深度对齐，同时8个
worker不超配）能进一步逼近，但本轮未跑完这组消融就转向了下一步。

### 7.5 关键发现：这个回归完全没有进 master，只存在于未合并的 feature 分支

`git merge-base HEAD origin/master` = `3e88d11`，**当前分支
`feature/deferzhou/merge-training-metrics`只比master多两个commit**：
`85b6d750`（本节讨论的回归源头）和`322ef57`（本文档+GBL修复，我们自己的
commit）。`git diff origin/master HEAD -- common_args.sh
train_speech_to_speech_translation.sh custom/dataset.py custom/wds_audio.py`
显示**只有前两个脚本文件有差异，`dataset.py`/`wds_audio.py`完全没有
diff**（Lance代码本身master早就有，经MR !66合并）。

**在master（3e88d11）上跑单机WDS验证**（未显式覆盖DataLoader参数，用
master原生的16 worker+prefetch8+dataset_num_proc16）：第一次
（`wds-master-cu59`）因为复用了跟其他实验相同的`OUTPUT_DIR`意外从已有
checkpoint（step374）续训、且被周期性存ckpt的耗时尖峰（688~1296s）污染，
数据作废；换独立`OUTPUT_DIR`重新提交（`wds-master3-em8k`），**跑满200步，
结果与 R 历史值精确吻合**：

| 指标 | R历史值 | master重跑（`wds-master3-em8k`） | 差异 |
|---|---|---|---|
| 稳态s/it（191-200，中位/均值） | 24.99s | **23.89s / 24.61s** | 差1.2%~4.4%，噪声范围内 |
| 显存@200 | 136.88 GiB | **136.88 GiB** | **完全一致** |
| loss@200 | 5.58948 | 5.59904 | 差0.0096，噪声范围内 |
| PadWaste（中位） | — | 63.9% | 跟Lance路径的64.6%量级一致 |

显存精确到小数点后两位完全一致，loss/速度差异都在噪声范围内——这是本轮
除AO复刻外最干净的一次数字对齐，**证实master的代码状态与R历史跑的时候
完全等价**。

**结论：这个DataLoader回归被完全限制在defer这条还没提MR/还没合并的
feature分支里，master是干净的**——用master原生配置（不需要任何显式
覆盖）直接复现了R的24.99s/it。这次排查本质上是在给这条未合并分支做
提前的回归测试，目前还没有机会影响到master或其他人。

### 7.6 用双边可信数据重新核算单机 Lance vs WDS

有了master的干净WDS数据后，重新做单机场景的Lance vs WDS对比（两边都是
可信配置，不再有跨代码状态问题）：

| 数据源 | 稳态s/it（中位） | 数据来源 |
|---|---|---|
| WDS | **23.89s** | `wds-master3-em8k`（master，本节） |
| Lance | **18.52s** | `lance-1n-z1-z5is`（feature分支，第6节；Lance相关代码
  `dataset.py`/`wds_audio.py`在master和feature分支间无diff，见第7.5节，
  不受DataLoader回归影响，可直接跟WDS的master数据横比） |

**Lance仍然比WDS快约22.5%**（(23.89-18.52)/23.89），跟第6节用R历史值算出
的23-28%基本一致——**第6节"单机场景Lance更快"这个方向性结论确认成立**，
只是对比基准从"R历史值"更新为更可信的"master重跑值"，数字略有收窄但
方向不变。

## 9. 单机快、双节点慢的根因排查：混淆了"多节点"和"ray框架"两个变量

用户提出"为什么Lance单机比WDS快、但多机比WDS慢"后，检查已跑过的全部
Lance实验的`framework`字段，发现一个此前未被察觉的方法论漏洞：

| 任务 | 节点数 | framework | 稳态s/it |
|---|---|---|---|
| `lance-1n-z1-z5is`（单机） | 1 | **pytorch**（直连deepspeed，无ray） | 18.52 |
| `lance-zero1-cagi`（双节点） | 2 | **ray** | 19.42 |
| `ao-repro2-txy1`（双节点） | 2 | **ray** | 19.15 |

**本文档目前所有"多节点"样本都同时是"ray"样本，从未测过"多节点+不用
ray（deepspeed直连/torchrun）"或反过来的组合**——"单机快、多机慢"这个
现象里，节点数和是否用ray两个变量被死死绑在一起，之前第5/6/7节的所有
"双节点Lance慢"结论，严格来说只能说是"ray+双节点Lance慢"，不能排除
是ray框架本身的问题、还是单纯节点数增加（跨节点网络/IO竞争）的问题。

### 10.1 反证：GBS512老配置里，双节点场景 ray 对 Lance 反而更快

翻查`TRAINING_METRICS_MERGED.md`的H–M家族（GBS512，无liger，双节点场景
下ray与直连各有一份Lance数据，是本文档范围外唯一现成的可比样本）：

| 数据源 | ray+deepspeed | deepspeed直连(torchrun) | ray vs 直连 |
|---|---|---|---|
| WDS | D: 2.76s | F: 2.69s | ray慢2.6%（符合预期，ray有框架开销） |
| **Lance** | **J: 1.82s** | **L: 2.35s** | **ray反而快22.6%** |

WDS这边ray比直连慢一点符合直觉（ray本身有调度开销）；但**Lance这边
方向完全反过来，ray反而更快**——这跟"ray多actor并发读全表引入IO竞争
导致变慢"的猜测**直接矛盾**。当然这组数据是老配置（GBS512、无liger、
ga=1、未必有precomputed mel/speaker预取），不能直接套用到本文档主线
的GBS4096+liger+zero1场景，但至少说明"ray导致Lance变慢"这个假设本身
缺乏跨配置的普适性支撑，不该被当作确定结论。

### 10.2 待补实验：拆分节点数与ray框架两个变量

要真正回答"是节点数问题还是ray框架问题"，需要在当前GBS4096+liger+zero1+
Lance这套主线配置下，补一组**双节点 + deepspeed直连（不用ray，对齐F/L
的torchrun模式）**的实验，跟已有数据组成完整2×2矩阵：

| | 单机 | 双节点 |
|---|---|---|
| pytorch直连 | 18.52s（`lance-1n-z1-z5is`，已有） | **缺失，待补** |
| ray | （单机无需ray，意义不大，可不补） | 19.42s（`lance-zero1-cagi`，已有） |

**判断逻辑**：若"双节点+直连"接近18.52s（跟单机一样快）→ 问题出在ray
框架本身；若"双节点+直连"也慢（接近或超过19.42s）→ 问题出在多节点本身
（跨节点网络/NVMe IO竞争），跟ray无关。**本轮会话未执行此实验，留作
下一步**。

1. **AO 的原始实验数据是真实、可信的**——用还原后的正确配置（显式
   `GROUP_BY_LENGTH=false`）能在 0.2% 精度内精确复现她报告的 200步耗时/
   显存/稳态吞吐。
2. **`85b6d750`（"合并整理"提交）在当前 feature 分支上引入了至少两处
   意外的默认值回归**：① `GROUP_BY_LENGTH` false→true（第1节）；
   ② `DATALOADER_PREFETCH_FACTOR` 8→2 + SCENE硬编码的
   `DATALOADER_NUM_WORKERS=16`/`DATASET_NUM_PROC=16`被删（第7节，单机WDS
   慢近一倍）。**这两处回归都只存在于未合并的
   `feature/deferzhou/merge-training-metrics` 分支，master 是干净的**——
   在master上用原生配置重跑单机WDS，200步稳态23.89s/it、显存136.88GiB
   （小数点后两位完全一致）、loss@200差0.0096，跟R历史值精确吻合（第7.5
   节），需要提醒 deferzhou 在合并前修正这两处默认值。
3. **单机场景 Lance vs WDS 的最终可信对比（第7.6节，双边均为可信配置）：
   Lance 比 WDS 快约 22.5%**（18.52s vs 23.89s）——第6节最初用R历史值算出
   的"快23-28%"这个方向性结论得到确认，数字基本一致（因为Lance相关代码
   本身在master和feature分支间无diff，不受DataLoader回归影响，可直接
   跟WDS的master数据横比）。
4. **双节点场景"Lance慢59%"这个结论（第3节）仍未被同commit数据验证**：
   D/S历史WDS数据是老commit跑的，同commit的双节点WDS补跑因7.1节的
   ray预处理卡死bug两次均未跑出有效数据。考虑到单机场景下WDS本身在
   feature分支上确认存在近一倍的DataLoader回归，**不能排除双节点历史
   WDS数据同样存在未被发现的可比性问题**；但由于Lance代码本身两分支间
   无差异、且第7.6节已证实单机Lance vs WDS方向不受影响，双节点结论的
   风险更多在"D/S具体数值是否可比"而非"方向是否反转"，仍建议以此为
   风险提示、暂不作为最终定论。
5. **ZeRO stage 的 zero0 vs zero1 规律（zero0快、显存贵）在 WDS 和 Lance 下
   方向一致**（第4节），这条结论不受本节发现的DataLoader回归影响（两次
   对照都在同一套Lance配置内部做的，未涉及WDS）。
6. **下一步排查方向**：
   - 双节点WDS+ray+zero1的预处理卡死bug（第7.1节）根因仍未定位，需要
     py-spy或添加debug日志确认`args.dataset_num_proc`实际解析值，以及
     ray driver进程的`_get_container_cpus()`探测结果——修复后可补跑同
     commit的双节点WDS，才能给结论4的"Lance双节点慢59%"做最终验证。
   - Lance 变慢的具体机制（mel/speaker批量take延迟 vs ray多actor IO
     竞争 vs LANCE_DATALOADER_BATCH_RESOLVE本身开销）仍需nsys/py-spy
     profiling才能最终定位（第6节遗留问题）。
   - 提醒deferzhou在合并`feature/deferzhou/merge-training-metrics`到
     master前，修复`85b6d750`引入的`GROUP_BY_LENGTH`和DataLoader两处
     默认值回归，避免污染master。

## 10. 任务清单

| 任务名 | 配置 | 状态 | 原始日志 |
|---|---|---|---|
| `ao-repro-a4bq` | ray+zero0+liger+Lance+GBS4096，GBL意外为true | Stopped（数据已作废，仅作根因排查证据） | `/tmp/ao_repro_full.log` |
| `ao-repro2-txy1` | ray+zero0+liger+Lance+GBS4096，GBL=false（显式） | Stopped，200步完整数据已采集 | `/tmp/ao_repro2_full.log` |
| `lance-zero1-cagi` | ray+zero1+liger+Lance+GBS4096，GBL=false（显式） | Stopped，200步完整数据已采集 | `/tmp/lance_zero1_repro_full.log` |
| `lance-1n-z1-z5is` | 单机deepspeed+zero1+liger+Lance+GBS4096，GBL=false（显式），对齐R列 | Stopped，200步完整数据已采集，**可信度待第7.5节master数据复核** | `/tmp/lance_1node_zero1_full.log` |
| `wds-z1-ray2-2g90` | 双节点ray+zero1+liger+WDS+GBS4096，同commit重跑（第7.1节） | Stopped，卡在预处理数小时，无有效数据 | `/tmp/wds_zero1_ray2_full.log` |
| `wds-1n-z1b-58ly` | 单机WDS+zero1+liger+GBS4096，同commit，无DataLoader覆盖（第7.2节） | Stopped，44步样本，确认稳态~45.6s/it慢近一倍 | `/tmp/wds_1node_zero1_diag2.log` |
| `wds-1n-olddl-tcpv` | 单机WDS，显式16worker+prefetch8（照抄老值，第7.4节第一次尝试） | Stopped，CPU超配导致28~142s剧烈抖动，数据作废 | `/tmp/wds_1node_olddl.log`、`/tmp/wds_1node_olddl3.log` |
| `wds-1n-pf8-wgvt` | 单机WDS，只显式prefetch=8（第7.4节第二次尝试） | Stopped，86步样本，稳态收敛32.1~32.9s/it | `/tmp/wds_1node_pf8.log` |
| `wds-master-cu59` | 单机WDS，master分支代码，第一次（第7.5节） | Stopped，复用OUTPUT_DIR意外续训+ckpt尖峰污染，数据作废 | `/tmp/wds_master.log` |
| `wds-master3-em8k` | 单机WDS，master分支代码，独立OUTPUT_DIR（第7.5/7.6节干净重跑） | Stopped，200步完整数据已采集，与R历史值精确吻合 | `/tmp/wds_master3.log` |

以上原始日志均为本机会话临时文件，未纳入版本库；本文档已摘录全部关键数值。
