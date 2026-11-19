# E10：LiveInterpret S5 按 B 的形式接入多说话人同传 baseline

方案文档：`reports/mist_w3po_v1/e10_liveinterpret_multispk_plan.zh-CN.md`（§11 记录了实现与冒烟结果）。

本目录只新增文件，不修改 `custom/`、`pipelines/` 里同事的任何代码。所有改动都通过 swift 的 `--external_plugins plugin_e10.py` 在运行时挂上。

## 文件

| 文件 | 作用 |
|---|---|
| `setup/setup_simuls2s.sh`、`setup/setup_simuls2s_part2.sh` | 建 venv `/opt/dlami/nvme/jasonleeeli/venvs/simuls2s`，版本记录在 `simuls2s.freeze.txt` |
| `setup/fetch_assets.sh` | 下载 S4 底座、声码器 `speech_decoder/`、临时 speech tokenizer，并 clone CosyVoice（commit 074ca6d） |
| `build_li_records.py` | 把 B 的 AMI 窗口和截止时间标签转成 S5 的 `si-rw-m1-v0` 训练记录（640 ms 一步、带说话人 tag、两个方向数量对齐） |
| `e10/common.py` | 共用常量与工具：说话人 tag（复用 `<|dur:1..25|>`，id 248080–248087）、指令、后验 13 帧延迟、B 的扰动 |
| `e10/patch.py` | 运行时补丁：record 的指令与后验 → `spk_post` → collate → `e10_sd_proj` 注入音频帧 |
| `e10/tuner.py` | `--tuner_type e10_ms`：从 checkpoint-29773 热启动（零初始化时与原模型逐位等价），只训 Thinker；`--optimizer e10_grouped` |
| `e10/load.py` | swift 之外加载模型（测试与推理用） |
| `plugin_e10.py` | swift 插件入口 |
| `train_e10.sh` | 训练脚本（前台运行；支持 `DRY_RUN=1` / `SMOKE=1`；GA 自动调整，保证 GBS=64） |
| `launch_e10_detached.sh` | 正式训练的 detached 启动包装。只用 `bash -n` 或 `DRY_RUN=1` 检查，不要试跑 |
| `run_e10.py` | 在线窗口解码。E10 模式用后验注入，增量按 tag 切开、分说话人送 Talker，音色用 B 的因果规则；E10-0 用原始 S5 加滚动音色 |
| `build_e10.py` | 转成 E8 的输出格式（`hyp_liveinterpret.jsonl`、`talker/results.jsonl`；说话人可选模型自带的或因果重标） |
| `eval_e10.py` | 评测编排：解码 → build → 与 E8 相同的打分链。默认只打印命令，加 `--run` 才执行 |
| `tests/check_data_pipeline.py` | 数据管线检查（CPU） |
| `tests/smoke_tuner_equivalence.py` | tuner 检查（GPU）：等价性、训练范围、梯度、保存/加载 |

`data/`、`runs/`、`output/` 都已加入 `.gitignore`。

## 命令

所有命令都在 `third_party/liveinterpret_training_pipelines/` 下执行。

```bash
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python      # 数据构建与 build_e10 用
S2S=/opt/dlami/nvme/jasonleeeli/venvs/simuls2s/bin/python            # 训练与解码用

# 1. 训练数据（已生成：e10_multispk/data/li_s5_records_v1）
PYTHONPATH=/opt/dlami/nvme/jasonleeeli/projects/UniSS $PY e10_multispk/build_li_records.py --out e10_multispk/data/li_s5_records_v1

# 2. 正式训练（未启动）。8 卡：GBS 64 = 8×8；4 卡时自动变成 8×4×2
DRY_RUN=1 bash e10_multispk/launch_e10_detached.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash e10_multispk/launch_e10_detached.sh
#    产物：e10_multispk/output/e10_thinker_v1/v*/checkpoint-*，用最后一步

# 3. 评测（未启动；先写预登记）。不加 --run 只打印命令
$PY e10_multispk/eval_e10.py --mode e10   --adapter <最后一步 checkpoint> --sets ami,ali,en
$PY e10_multispk/eval_e10.py --mode e10_0 --adapter checkpoints/p4mix_2n_20261117/checkpoint-29773 --sets ami,ali,en
```

## 临时项

- `speech_tokenizer_v3`：目前用的是 S3 上的 `speech_tokenizer_v3.batch.onnx`，冒烟时可以正常工作。拿到确认的文件后，设置 `E10_SPEECH_TOKENIZER=<路径>`，或给 `run_e10.py` 传 `--prompt_tokenizer`。
