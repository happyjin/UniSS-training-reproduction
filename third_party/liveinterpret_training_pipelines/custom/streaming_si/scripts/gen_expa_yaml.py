"""由 `ktrain/ktrainjob_streaming_si_s4.yaml` 派生实验 A（talker 改接 thinker 中间层）各臂的任务 yaml。

设计与判读规则见 `pipelines/streaming_si/docs/talker_para_experiments.md` §3。

    python custom/streaming_si/scripts/gen_expa_yaml.py            # 生成全部臂
    python custom/streaming_si/scripts/gen_expa_yaml.py a2 a2_joint # 只生成指定臂

各臂唯一的实验变量是起点目录里 `config.talker_bridge_layer`；其余超参、数据、LR 调度逐字相同。
起点权重**逐位相同**（a1/a2 由 a0 的起点目录复制后只改 config.json 得到，`diff -rq` 验证过
只有 config.json 一个文件不同）——不要对每个臂单独跑 `reinit_talker_bridge.py`，那会得到
不同的随机桥接初值，臂间不可比。

两个阶段，必须串行：
* `s4-w`  只训 `talker.thinker_proj` + `talker.fusion`。桥接刚被重初始化，随机桥接的输出量级
          约为 text 路的 40 倍（`pipelines/streaming_si/docs/paralinguistic_preservation.md` §4.6
          实测 CE 3.51 -> 4.95），不预热就 joint 等于拿已收敛的 talker 去适应噪声。
* `s4`    joint：`audio_proj` + thinker + talker 全开。MODEL_PATH = 该臂 s4-w 的最终 checkpoint，
          且**必须先过门 A**（`s4w_bridge_gate.py`，要求每方向 full 的 speech CE < text_only）。

全局 batch 必须跨臂一致，否则有效学习率动态不同、臂间不可比：

    阶段    卡数              per_device x 卡数 x GA = 全局
    s4-w    16（2 节点）      16 x 16 x 1 = 256
    s4-w     8（1 节点）      16 x  8 x 2 = 256
    s4      16（2 节点）      32 x 16 x 1 = 512
    s4       8（1 节点）      32 x  8 x 2 = 512

即 per_device 固定（s4-w 用 train_s2st.sh 默认 16，joint 用 yaml 的 32），只用
`GRADIENT_ACCUMULATION_STEPS` 补节点数差。`train_simul.sh` L95 的
`SI_S4_GLOBAL_BATCH = per_device x WORLD x GA` 即此口径。

节点白名单的三条硬约束（都踩过）：
1. **机型不能混**：集群有 ml.p5e.48xlarge（efa=32）与 ml.p5en.48xlarge（efa=16）两种，
   一个任务内混用会让 worker 永久 Pending
   （事件：`Insufficient vpc.amazonaws.com/efa` + `didn't match Pod's node affinity/selector`）。
   `efa_devices_per_worker` 与 `instance-type` 必须与白名单节点实际一致。
2. **EFA 可能被别人占满**：节点 GPU 空着但 EFA 配额被同节点其他任务独占时同样调度不上。
   提交前查 `vpc.amazonaws.com/efa` 的实际占用，不是只看 GPU。
3. **磁盘要留足**：P2 数据同步到节点 NVMe 实测占 **2.1 TB**，kubelet 驱逐阈值是
   `nodefs.available<10%`。可用空间低于约 3 TB 的节点会在同步途中被打上
   `disk-pressure:NoSchedule` 并驱逐 Pod。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# custom/streaming_si/scripts -> custom/streaming_si -> custom -> repo 根
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(REPO, 'ktrain', 'ktrainjob_streaming_si_s4.yaml')
OUT_DIR = os.path.join(REPO, 'ktrain')
S3_ROOT = ('s3://stevehguo-ap-southeast-3/speech_model_backup/stevehguo/model_checkpoint/'
           'streaming_si_swift/trained_streaming_si_4b')

# 取层换算：SI_TALKER_BRIDGE_LAYER=k -> hidden_states[k+1] -> 探针口径 L(k+1)
#   （`model/model_si.py` pick_bridge_hidden；实验 4 实锤 取层 3 = 探针 L4 / 取层 7 = 探针 L8）
# 4B thinker 共 32 层（权重实测 thinker_model.model.layers 下标 0..31，
# layer_types = 24 linear_attention + 8 full_attention）。
#   a0 = -1  末层（探针 L32），对照臂
#   a1 =  6  探针 L7，落在探针峰值区 L4~L8（韵律 R² L4 0.203 vs 末层 0.092）
#   a2 = 15  探针 L16 = 半深度，与 Qwen3-Omni 同口径
#            （`custom/qwen_omnispeak`：accept_hidden_layer=24 / 共 48 层 = 0.50，
#             其 `mcore_talker_loss.py` 的 idx = accept - 1）
P5EN = ['hyperpod-i-07e7a9b74bdcbb8b0', 'hyperpod-i-0aaea3b1ebac8d256']   # efa=16
P5E_A1 = ['hyperpod-i-09a68c454800140d6', 'hyperpod-i-08e3257c716579618']  # efa=32
# a2 单节点。选 07d384 的理由（2026-10-19 核实）：
#   * P2 数据**已在该节点 NVMe 上**，与 a0/a1 节点逐项一致（parquet 70/64/64、tar 70/64/206），
#     不需要新同步 2.1 TB，a2 的增量只有起点权重 15G + 中间产物 8G + checkpoint 约 28G；
#     故该节点 88% 用量（可用 3.3 TB）不会触发 `nodefs.available<10%` 驱逐。
#   * 8 卡 + 32 EFA 全空、无污点（原先占卡占满 EFA 的 zzj S3 任务已 Succeeded）。
# 曾用 0a3ebd369de798192（可用 5.12 TB），但其 Pod 持续 ContainerCreating，
# 报 `Failed to find corresponding MountpointS3PodAttachment`（S3 CSI 时序问题），故改用本节点。
P5E_A2 = ['hyperpod-i-07d384c1277ead0cd']                                  # efa=32，单节点

# 实验 b 用四台：07d384 + 09a68 是 p5e（EFA 32），07e7 + 0aaea 是 p5en（EFA 16）。
# **混机型**：a2j7（07d384 p5e + 0aaea p5en）实测 NCCL/OFI 建连失败
#   `fi_addr_t nccl_ofi_cm::endpoint::av_insert_address ... Unable to insert address`
#   36 分钟零进度。故这里按 mixed 模式渲染：去掉 instance-type 选择器、EFA 取两者较小的 16。
# 若起不来（同样报错），退回 p5e 对（07d384 + 09a68）跑 16 卡。
# b2 用 3 台 p5e（07d384 + 09a68 + 08e3）= 24 卡。24 卡凑不出 512（512/24=21.33），
# 故全局 batch 取 768 = 32 x 24 x GA1，lr 按线性缩放 5e-5 -> 7.5e-5（1.5 倍）。
# 代价：与 a0j/a1j（512 / 5e-5 / 37470 步）不再是严格同口径，1 epoch 步数变为
# 1928万/768 ≈ 25110 步（少 1/3），报告里必须标注这一条。
# 选 768 而不是 384：384 会让步数变成 50196（多 1/3），比 16 卡还慢。
P5E_B2 = ['hyperpod-i-08e3257c716579618', 'hyperpod-i-09a68c454800140d6']      # efa=32，同机型
# 消融（幅值通道）两臂：单节点各一台，互不重叠，便于同时提交、各自独立跑。
# fs0 原绑 0a3ebd，2026-10-22 预处理完成后 NCCL/EFA socket 初始化失败（rank7 abort）；
# 改绑 09a68（b2j 全程训练过的已验证好节点，p5e/efa=32）。
P5E_FS_A = ['hyperpod-i-09a68c454800140d6']
# fs1 改到 p5en（0aaea）：2026-10-22 实测 p5en 的 FSx / S3 / NVMe 三卷均正常挂载，
# 之前的 ContainerCreating 是 CPU/内存不足，不是挂载问题。efa=16（p5en 实际值）。
P5EN_FS_B = ['hyperpod-i-0aaea3b1ebac8d256']
# b2_joint 用 2 节点 16 卡（512，与 a0j/a1j 同口径）；3 节点 768 版已被磁盘驱逐废弃。
# 4 节点版：必须同机型（p5e），07e7 是 p5en，混搭实测 NCCL 建连失败两次，故不入列。
P5E_B2_4N = ['hyperpod-i-03e7d1f6d95864e1a', 'hyperpod-i-08e3257c716579618',
             'hyperpod-i-09a68c454800140d6', 'hyperpod-i-07d384c1277ead0cd']
# 上面四个节点混机型（p5e + p5en）实测 NCCL 建连失败，故 b2 只取 p5e 对、a2 改到 p5en 对。

ARMS = {
    # ---- 预热阶段 s4-w ----
    # a0 / a1 已跑完（各 2500 步，25 分钟）且**两个方向都过门 A**：
    #   a0  cmn->eng full-text_only = -0.2415   eng->cmn = -0.2657
    #   a1  cmn->eng full-text_only = -0.2821   eng->cmn = -0.3522
    # a1 两个方向都优于 a0，**推翻了实验 4「换层无收益」**——实验 4 未重初始化桥接，
    # fusion 门控停在为末层调好的旧解上，掩盖了取层差距（详见 docs §3 的预注册判读）。
    'a0': dict(stage='s4-w', nodes=P5EN, itype='ml.p5en.48xlarge', efa=16,
               model=f'{S3_ROOT}/expA/a0/',
               desc='实验 A 对照臂：桥接重初始化，取层末层(-1)=探针 L32'),
    'a1': dict(stage='s4-w', nodes=P5E_A1, itype='ml.p5e.48xlarge', efa=32,
               model=f'{S3_ROOT}/expA/a1/',
               desc='实验 A：桥接重初始化 + 取层 6（= 探针 L7，峰值区 L4~L8 内）'),
    'a2': dict(stage='s4-w', nodes=P5E_A2, itype='ml.p5e.48xlarge', efa=32, ga=2,
               model=f'{S3_ROOT}/expA/a2/',
               desc='实验 A：桥接重初始化 + 取层 15（= 探针 L16，半深度，对标 Omni 24/48）'),

    # ---- joint 阶段 s4 ----
    # MODEL_PATH = 各臂 s4-w 的最终 checkpoint。预热产物写在**各自 master 节点的本地 NVMe**
    # （hostPath /opt/dlami/nvme，非共享存储），Pod 退出后就取不到，所以跑完要立刻
    # `aws s3 sync` 到下面这些前缀；换节点提交 joint 时才读得到。
    'a0_joint': dict(stage='s4', nodes=P5EN, itype='ml.p5en.48xlarge', efa=16,
                     model=f'{S3_ROOT}/expA/a0_w2500/',
                     desc='实验 A 对照臂 joint：取层末层(-1)，1 epoch'),
    'a1_joint': dict(stage='s4', nodes=P5E_A1, itype='ml.p5e.48xlarge', efa=32,
                     model=f'{S3_ROOT}/expA/a1_w2500/',
                     desc='实验 A joint：取层 6（探针 L7），1 epoch'),
    # a2 单节点 8 卡：用 per_device 64 x GA 1 而不是 32 x GA 2 —— 全局 batch 同为 512，
    # 优化过程与步数完全一致，但每步只做一次前反向，省掉一半 kernel launch 开销。
    # 再关掉 gradient_checkpointing：它在反向时重算一遍前向，关掉省 20~30% 计算，
    # 且**梯度逐位等价**（只是存 vs 重算），对臂间可比性零影响。
    # 两项都只影响速度不影响数值，故可单独给 a2 用而不破坏与 a0/a1 的可比性。
    # ---- 实验 b：改 talker 条件取哪些位置（代码见 model_si.build_cond_mask）----
    # b2 = 音频帧位置 + 译文位置，取层 -1（与 a0 对照同层）。起点复用 expA/a0（桥接已重初始化、
    # 取层 -1），只靠 SI_TALKER_COND_POSITIONS 切换条件位置，故与 a0j 逐项可比。
    # 32 卡 -> 预热全局 batch 256 = 8 x 8 x 4 x GA1；joint 512 = 16 x 8 x 4 x GA1。
    # b1 = 只取音频帧位置（text 路在该模式下全零，talker 拿不到"说什么"）。
    # 这是**消融臂**：用来验证"副语言信息确实来自音频位置"，而不是主推方案。
    # 若它把翻译搞崩，门 A（每方向 full 的 speech CE < text_only）会直接拦下，不会白跑 joint。
    'b1': dict(stage='s4-w', nodes=P5EN, itype='ml.p5en.48xlarge', efa=16,
               cond_positions='audio',
               model=f'{S3_ROOT}/expA/a0/',
               desc='实验 B 消融：talker 条件 = 只取音频帧位置（取层 -1）'),
    'b1_joint': dict(stage='s4', nodes=P5EN, itype='ml.p5en.48xlarge', efa=16,
                     per_device=32, cond_positions='audio',
                     model=f'{S3_ROOT}/expA/b1_w2500/',
                     desc='实验 B joint：只取音频帧位置，跑满 1 epoch'),

    # 24 卡：预热也取 768（32 x 24），与 joint 同批；lr 仍用 s4-w 默认的 3e-4（放大到 9e-4 过激）。
    # 预热步数沿用 2500 步 —— 比 a 臂（256 x 2500）多看 3 倍样本，但桥接换了新的条件位置，
    # 多预热只有好处；门 A 是 pass/fail，不影响判定。
    'b2': dict(stage='s4-w', nodes=P5E_B2, itype='ml.p5e.48xlarge', efa=32, per_device=32,
               cond_positions='audio+answer',
               model=f'{S3_ROOT}/expA/a0/',
               desc='实验 B：talker 条件 = 音频帧位置 + 译文位置（取层 -1）'),
    # 续跑版（checkpoint-26000 起步，剩余约 11470 步）：**开检查点**。
    # 关检查点 + per_device 16 在 26675/37470 步撞到长音频批 OOM（139.8G 显存仅剩 5.6G，
    # 反向要 6.66G）；检查点对梯度数学透明（只差存激活 vs 重算），中途切换安全。
    # 全局 512 = 16 x 8 x 2 x GA2 与 a0j/a1j 逐位同口径；lr 5e-5。
    # 2 节点可做**精确续跑**（优化器分片 DP=16 与 world size 一致，已互拷到两节点各 16 片）。
    'b2_joint': dict(stage='s4', nodes=P5E_B2, itype='ml.p5e.48xlarge', efa=32,
                     per_device=16, ga=2,
                     lr='5e-5', cond_positions='audio+answer',
                     model=f'{S3_ROOT}/expA/b2_w2500/',
                     resume_ckpt='/nvme/output/stevehguo/streaming_si/s4_s2st/v4-20261020-150421/checkpoint-26000',
                     desc='实验 B joint：条件 = 音频帧 + 译文位置，跑满 1 epoch（从 26000 精确续跑）'),

    # ---- 4 节点 32 卡版（2026-10-21）----
    # 全局 batch 仍 512 = 16 x 8 x 4 x GA1（GA 从 2 降到 1 抵掉卡数翻倍），与 a0j/a1j 同口径；
    # 每卡微批仍 16，显存占用与 2 节点版相同（开检查点），不会重演 OOM。
    # **只能热启动、不能精确续跑**：ckpt-26000 的 ZeRO-2 优化器分片是 **DP=16**，
    # 32 卡 world size 不匹配，DeepSpeed 会在 _load_zero_checkpoint 处报错（除非先用
    # ds_to_universal 转 universal checkpoint，成本更高）。故加 --resume_only_model true：
    #   * 模型权重：从 ckpt-26000 加载（26000 步的训练成果全部保留）
    #   * global_step / lr 调度 / 数据跳过：仍按 trainer_state 恢复到 26000
    #     （resume_only_model 只在与 ignore_data_skip 同时为真时才跳过读 trainer_state，
    #      见 swift/trainers/mixin.py L146；我们不设 ignore_data_skip）
    #   * 代价：Adam 一阶/二阶动量重置 → 续跑开头几十步 loss 可能轻微抖动，
    #     lr 已衰减到后段，影响有限；报告需注明"26000 步处优化器状态重置"。
    'b2_joint_4n': dict(stage='s4', nodes=P5E_B2_4N, itype='ml.p5e.48xlarge', efa=32,
                        per_device=16, ga=1,
                        lr='5e-5', cond_positions='audio+answer',
                        model=f'{S3_ROOT}/expA/b2_w2500/',
                        resume_ckpt='/nvme/output/stevehguo/streaming_si/s4_s2st/'
                                    'v4-20261020-150421/checkpoint-26000',
                        resume_only_model=True,
                        desc='实验 B joint（4 节点 32 卡）：从 26000 热启动跑满 1 epoch'),

    'a2_joint': dict(stage='s4', nodes=P5EN, itype='ml.p5en.48xlarge', efa=16, ga=1,
                     per_device=32,
                     model=f'{S3_ROOT}/expA/a2_w2500/',
                     desc='实验 A joint：取层 15（探针 L16），1 epoch'),

    # ---- 消融：TalkerFusion 幅值通道（2026-10-22）----
    # 依据：SUMMARY §10 的 Bridge 四点探针显示 C→D（Fusion 末尾 RMSNorm）损失了
    #   F0 起伏 / 停顿 / 能量起伏（−0.03～−0.07 R²，两方向多数显著），而 §10 的探针级
    #   幅值通道（D ⊕ log-RMS(C)）几乎补不回来（DS−D≈0）——说明损失不是"幅值这个标量"，
    #   而是"幅值 × 方向"的交互（归一化抹掉逐 token 尺度，池化后权重被抹平）。
    # 设计：两臂**完全相同**，唯一差别 SI_TALKER_FUSION_SCALE（0 / 1）。
    #   同起点 S4/checkpoint-24980（正式 S4 终点）、同数据、同全局 batch 512
    #   （32 x 8 x GA2）、同 lr 5e-5、同 3000 步、同随机种子。
    #   开通道臂只多 1024 个参数（talker.fusion.scale_gain，**零初始化**，
    #   已用 tests/check_fusion_scale_equiv.py 在真实 ckpt 上验证起步逐位等价）。
    # 注意：这是**短续训**（3000/37470 步 ≈ 8% epoch），不与 a0j/b2j 的跑满 1 epoch 直接比绝对值，
    #   只在两臂之间比差值；报告需注明 lr 从 5e-5 常量重新起步（非余弦尾段）。
    'fs0': dict(stage='s4', nodes=P5E_FS_A, itype='ml.p5e.48xlarge', efa=32,
                per_device=32, ga=2, lr='5e-5', max_steps=3000,
                model=f'{S3_ROOT}/S4/checkpoint-24980/',
                env={'SAVE_STEPS': '1000'},
                desc='消融对照：S4 续训 3000 步，Fusion 幅值通道关（与原模型逐位一致）'),
    'fs1': dict(stage='s4', nodes=P5EN_FS_B, itype='ml.p5en.48xlarge', efa=16,
                per_device=32, ga=2, lr='5e-5', max_steps=3000,
                model=f'{S3_ROOT}/S4/checkpoint-24980/',
                env={'SAVE_STEPS': '1000', 'SI_TALKER_FUSION_SCALE': '1'},
                desc='消融：S4 续训 3000 步，Fusion 幅值通道开（log-RMS × 零初始化 scale_gain）'),
}


def render(arm: str) -> str:
    cfg = ARMS[arm]
    nn, ga, stage = len(cfg['nodes']), cfg.get('ga', 1), cfg['stage']
    s = open(SRC, encoding='utf-8').read()
    # 1) 节点白名单 + numNodes（两者必须一致）
    s = re.sub(r'"values": \[[^\]]*\]',
               '"values": ["' + '", "'.join(cfg['nodes']) + '"]', s, count=1)
    s = re.sub(r'^  numNodes: \d+.*$', f'  numNodes: {nn}                  # = 白名单节点数',
               s, count=1, flags=re.M)
    # 2) 机型与 EFA 数必须与白名单节点一致（混机型会导致 worker 永久 Pending）
    if cfg.get('itype') is None:
        # 混机型（p5e + p5en）：不能限定 instance-type，只靠白名单主机名钉死节点；
        # EFA 取两者较小的 16。若 NCCL 仍建连失败，改回同机型对。
        s = re.sub(r'"node_selector": \{[^}]*\},?\s*', '', s, count=1)
    else:
        s = s.replace('"node.kubernetes.io/instance-type": "ml.p5e.48xlarge"',
                      f'"node.kubernetes.io/instance-type": "{cfg["itype"]}"')
    s = re.sub(r'"efa_devices_per_worker": \d+', f'"efa_devices_per_worker": {cfg["efa"]}',
               s, count=1)
    # 3) 阶段与起点
    s = re.sub(r'^    SI_STAGE: "s4"$', f'    SI_STAGE: "{stage}"', s, count=1, flags=re.M)
    s = re.sub(r'^    MODEL_PATH: ".*"$', f'    MODEL_PATH: "{cfg["model"]}"', s, count=1, flags=re.M)
    # 4) 超参：s4-w 清掉 joint 的覆盖项，改用 train_s2st.sh 的 s4-w 默认
    #    （每卡 16 / lr 3e-4 / 2500 步 / 文本权重 0）；joint 保留 yaml 原值
    if stage == 's4-w':
        pd_w = cfg.get('per_device')
        for k in ('LEARNING_RATE', 'SI_TEXT_LOSS_WEIGHT'):
            s = re.sub(rf'^    {k}: ".*"$',
                       f'    # {k} 由 train_s2st.sh 的 s4-w 默认值决定', s, count=1, flags=re.M)
        if pd_w:                      # 非 2 节点时要显式给，才能凑出 256
            s = re.sub(r'^    PER_DEVICE_TRAIN_BATCH_SIZE: ".*"$',
                       f'    PER_DEVICE_TRAIN_BATCH_SIZE: "{pd_w}"', s, count=1, flags=re.M)
        else:
            s = re.sub(r'^    PER_DEVICE_TRAIN_BATCH_SIZE: ".*"$',
                       '    # PER_DEVICE_TRAIN_BATCH_SIZE 由 train_s2st.sh 的 s4-w 默认值决定',
                       s, count=1, flags=re.M)
        gbs = (pd_w or 16) * 8 * nn * ga
    else:
        pd = cfg.get('per_device', 32)
        if pd != 32:
            s = re.sub(r'^    PER_DEVICE_TRAIN_BATCH_SIZE: ".*"$',
                       f'    PER_DEVICE_TRAIN_BATCH_SIZE: "{pd}"', s, count=1, flags=re.M)
        if cfg.get('lr'):             # 全局 batch 变了才动 lr（线性缩放）
            s = re.sub(r'^    LEARNING_RATE: ".*"$',
                       f'    LEARNING_RATE: "{cfg["lr"]}"   # 随全局 batch 线性缩放（512 -> {pd * 8 * nn * ga}）',
                       s, count=1, flags=re.M)
        gbs = pd * 8 * nn * ga
    if ga != 1:
        s = re.sub(r'^(    SI_STAGE: ".*"\n)',
                   rf'\1    GRADIENT_ACCUMULATION_STEPS: "{ga}"   # 单节点：补齐全局 batch\n',
                   s, count=1, flags=re.M)
    if cfg.get('max_steps'):
        s = re.sub(r'^(    SI_STAGE: ".*"\n)',
                   rf'\1    MAX_STEPS: "{cfg["max_steps"]}"\n', s, count=1, flags=re.M)
    if cfg.get('no_grad_ckpt'):
        # common_args.sh 不传 --gradient_checkpointing（用 swift 默认 True），
        # 经 ktrain_entry.sh 的 SI_EXTRA_ARGS 追加（L318/L323 透传给训练脚本）。
        s = re.sub(r'^(    SI_STAGE: ".*"\n)',
                   r'\1    SI_EXTRA_ARGS: "--gradient_checkpointing false"'
                   '   # 省重算；梯度逐位等价\n',
                   s, count=1, flags=re.M)
    if cfg.get('resume_ckpt'):
        # 显式续跑：swift 的版本链自动续跑（dlrover_latest.txt）只在 use_flash_ckpt=true
        # 时启用（pipelines/train/sft.py L239-241），我们不开 flash ckpt，故必须显式传。
        # 路径字符串在两个节点各自解析到本地 /nvme 的同名目录（save_on_each_node 布局，
        # 08e3=rank0-7 分片、09a68=rank8-15 分片），正好各取各的优化器分片。
        # 2026-10-21 b2_joint 从 checkpoint-26000 续跑（26675 步 OOM 后；开检查点重启）。
        _extra = f'--resume_from_checkpoint {cfg["resume_ckpt"]}'
        if cfg.get('resume_only_model'):
            # world size 与 ckpt 的 ZeRO DP 不一致时必须加（见 b2_joint_4n 注释）：
            # 只加载权重，不读优化器分片；步数/lr/数据跳过仍由 trainer_state 恢复。
            _extra += ' --resume_only_model true'
        s = re.sub(r'^(    SI_STAGE: ".*"\n)',
                   rf'\1    SI_EXTRA_ARGS: "{_extra}"'
                   '   # 续跑点；见 gen_expa_yaml.py 的臂注释\n',
                   s, count=1, flags=re.M)
    if cfg.get('env'):                # 任意额外环境变量（消融开关 / SAVE_STEPS 等）
        add = ''.join(f'    {k}: "{v}"\n' for k, v in cfg['env'].items())
        s = re.sub(r'^(    SI_STAGE: ".*"\n)', lambda m: m.group(1) + add, s, count=1, flags=re.M)
    if cfg.get('cond_positions'):     # 实验 b：talker 条件取哪些位置
        s = re.sub(r'^(    SI_STAGE: ".*"\n)',
                   rf'\1    SI_TALKER_COND_POSITIONS: "{cfg["cond_positions"]}"'
                   '   # answer / audio / audio+answer\n',
                   s, count=1, flags=re.M)
    gate = ('# 预热完成后必须用门 A 验收，通过才提交 joint：\n'
            '#   python custom/streaming_si/scripts/s4w_bridge_gate.py --model <ckpt> \\\n'
            '#       --data \'<P2 parquet glob>\' --wav_dir <p2_source_wav 目录>\n'
            '# 判据：每方向 full 的 speech CE < text_only（margin 0.005）。\n'
            '# 注意：预热阶段 thinker 冻结，换层的浅层优势按定义用不上，门 A 略差于对照臂\n'
            '# **不代表该臂失败**（实验 4 即如此）；真正的判据是 joint 后的 AutoPCP / SIM-o delta。\n'
            if stage == 's4-w' else
            '# 本臂的 s4-w 预热已过门 A（两个方向 full - text_only < -0.005）。\n'
            '# joint 跑满 1 epoch（1928 万样本）；实测吞吐约 26.4 样本/秒/卡，\n'
            '# 16 卡约 12.7 小时、8 卡约 25.3 小时。\n')
    head = ('# ' + cfg['desc'] + '\n'
            '# 由 custom/streaming_si/scripts/gen_expa_yaml.py 生成，勿手改；\n'
            '# 设计与判读见 pipelines/streaming_si/docs/talker_para_experiments.md §3。\n'
            f'# 阶段 = {stage}；{nn} 节点 x 8 卡，GA={ga}，**全局 batch = {gbs}**（跨臂必须一致）。\n'
            f'# 机型 {cfg["itype"]}（efa={cfg["efa"]}）。各臂机型可不同（同为 8xH200，仅 EFA 数不同），\n'
            '# 但 per_device x 卡数 x GA 必须相同，否则有效学习率动态不同、臂间不可比。\n'
            '# 各臂起点权重逐位相同，仅 config.talker_bridge_layer 不同（a0 -1 / a1 6 / a2 15）。\n'
            + gate)
    return head + s, nn, ga, gbs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    unknown = [a for a in argv if a not in ARMS]
    if unknown:
        raise SystemExit(f'未知臂 {unknown}；可选：{list(ARMS)}')
    for arm in (argv or list(ARMS)):
        s, nn, ga, gbs = render(arm)
        p = os.path.join(OUT_DIR, f'ktrainjob_streaming_si_expA_{arm}.yaml')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(s)
        print(f'写出 {os.path.relpath(p, REPO)}  (numNodes={nn} ga={ga} 全局batch={gbs})')


if __name__ == '__main__':
    main()
