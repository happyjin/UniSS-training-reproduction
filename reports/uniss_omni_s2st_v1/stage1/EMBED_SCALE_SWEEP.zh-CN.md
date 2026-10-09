# 码嵌入尺度:三臂受控对照

**日期** 2026-10-09 · Megatron · 三臂除 `--omni-embed-scale` 外**完全相同**
(MBS 48 / GBS 384 / `train_iters 5000` / LR 3.5e-4 / 同种子 / 同 dev 子集)

`train_iters` 三臂都设 5000,所以 cosine 曲线在**每一个被比较的步上数值相同** ——
这正是上一次 scale=8 实验失败的地方,它用了 2500 步的调度去比 15000 步的基线。

## 结果

| iter | scale=1 | scale=8 | scale=32 |
|---|---|---|---|
| 300 | 8.7807 | **8.3436** | 8.3274 |
| 600 | 8.0924 | **7.8255** | 7.8617 |
| 900 | 7.7793 | **7.6711** | 7.6866 |
| 1200 | 7.6106 | **7.5838** | 7.6089 |
| 1500 | **7.3879** | 7.5277 | 7.5431 |

## 结论:尺度不是那个杠杆

放大码嵌入**只买到更快的早期收敛,不改变终点**。领先从 iter 300 的 0.44 nats
一路收窄,在 **iter 1300 附近交叉**,到 1500 步 scale=1 反超 0.14 且差距仍在扩大。

scale=32 与 scale=8 基本重合(早期互有胜负,后期同样落后),说明这不是"再拉大一点
就更好"的单调关系。

### 这推翻了我自己的两个说法

1. 早先那个 scale=8 对照的 −0.289,我已标注为被 LR 调度混淆。现在可以说得更确切:
   **方向对、但结论错** —— 那个优势是暂时的。
2. 正式训练我选了 `embed_scale=8`,理由是实测的 36 倍失衡加上早期优势。
   按这组数据,**对一个 5000 步的长跑,这个选择是错的**;scale=1 的终值会更低。
   正式训练的 7.3556 因此不是这套配置能达到的最好值。

### 失衡本身是真的,只是不是瓶颈

`CONDITIONING_SCALE.zh-CN.md` 测到的 173.2 : 4.78 没有问题,受控消融显示的
offset 5–20 区间文本有害也没有问题。但把码行放大到 38 或 153 都没能改变终点 ——
说明这个失衡影响的是优化路径,不是模型能学到的上限。

真正的候选瓶颈另有其一:Talker **从未被告知说话人**。本仓库自己跑通的
`build_tts_sample` 把 32 个 `bicodec_global` 放进提示再预测语义码,而 Stage 1
把它们直接送给了声码器。对照实验正在跑。

## 复现

```bash
for S in 1.0 8.0 32.0; do
  MBS=48 GBS=384 TRAIN_ITERS=5000 WARMUP_ITERS=200 LR=3.5e-4 \
    EMBED_SCALE=$S HEAD_SCALE=0.1 EXIT_INTERVAL=1500 \
    EVAL_INTERVAL=100 EVAL_ITERS=16 SAVE_INTERVAL=100000 \
    SAVE_DIR=.../probe_scale$S TB_DIR=.../probe_scale$S \
    bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_megatron.sh
done
```

曲线从各自 `TB_DIR` 的 `code_ce validation` 标量读取。
