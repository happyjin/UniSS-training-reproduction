# 新服务器运行指南：单说话人论文、多说话人论文、TTS 实验

写于 2026-10-30，旧服务器关机之前。本文回答一个问题：新服务器上拷贝哪些东西，三套实验就能直接运行。

三套实验分别是：
1. 单说话人同传论文（QABC）：`paper/bayes_simul_s2st_v7`
2. 多说话人会议同传论文（G-STAR / MIST）：`paper/gstar_si_v4`
3. TTS 实验（LLM × VITS 手册方案）：`experiments/llm_vits_route_v1`

记号：
- **J** = `/opt/dlami/nvme/jasonleeeli`
- **R** = `J/projects/UniSS`（代码仓库）
- **S3** = `s3://stevehguo-ap-southeast-3/speech_model_backup/jasonleeeli`，与 J 一一对应

---

## 0. 先读这一节：迁移的原则和五个坑

### 原则

**新服务器上保持完全相同的绝对路径 `/opt/dlami/nvme/jasonleeeli`。** 如果盘挂在别处，就建一个软链接：`ln -s <新盘目录> /opt/dlami/nvme/jasonleeeli`。原因有三：
- 大量脚本写死了 J 下的路径（每节的"写死路径"列出了是哪些）；
- 数据 jsonl 里的音频路径是绝对路径；
- 几个环境的解释器是指向 `J/conda_envs/uniss-eval` 的软链接。

路径不变，就不用改任何代码。

### 五个坑

1. **S3 备份不含软链接。** 全量备份用了 `--no-follow-symlinks`。从 S3 拉回之后：
   - 环境的 `bin/python`、HF 缓存的 snapshot 链接、`uniss-train` 环境的 1380 个库链接都缺失；
   - **必须运行 `bash docs/migration/restore_symlinks.sh` 补回来**；
   - 清单在 `docs/migration/symlinks_jasonleeeli.tsv`，共 6,972 条，2026-10-30 导出；
   - 脚本只补缺失的链接，不覆盖任何文件。
2. **`git clone` 拿不到数据。** 以下目录被 .gitignore 排除，只能从 S3 拉：
   - `data/`、`pretrained_models/`、`checkpoints/`、`eval_outputs/`
   - `third_party/Megatron-LM`
   - `experiments/llm_vits_route_v1/{data,runs}`
3. **几个环境共用 `uniss-eval` 的解释器。** `qwen3omni`、`vllm_omni_030`、`nemo_diar`、`tts_schemes` 都建在 `J/conda_envs/uniss-eval` 的 Python 3.12 上，必须一起拷，路径也要一致。另外：
   - `nemo_diar` 的 NeMo 以 editable 方式装在 `J/softwares/NeMo-Speech`；
   - `uniss-eval` 里的 stopes 以 editable 方式装在 `J/evaluation_deps/stopes-a4e75e8`。
4. **有几样东西在 `/root/.cache`，不在 J 里，S3 上也没有备份。**
   - Whisper-large-v3：J 里已有一份，见 `J/cache/huggingface/hub/models--openai--whisper-large-v3`。
   - Paraformer：J 里已有一份，见 `J/cache/modelscope`。
   - tectonic 的编译缓存：已经复制到 `J/cache/root_dot_cache/Tectonic`，并上传到 S3。
   - 新服务器上设置下面这几个变量，或者把它们复制回 `~/.cache`：
     ```
     export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
     export MODELSCOPE_CACHE=/opt/dlami/nvme/jasonleeeli/cache/modelscope
     export TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch
     mkdir -p ~/.cache && cp -a /opt/dlami/nvme/jasonleeeli/cache/root_dot_cache/Tectonic ~/.cache/
     ```
5. **任务清单 json 里写死了命令。** `reports/*/jobs/*.json` 记录的是旧服务器上的完整命令，重跑时应该用 `make_*_jobs.py` 重新生成。生成脚本用 `open(..., "x")`，目标文件已存在时会拒绝写，需要先改名旧文件。

### 只在别处、需要额外处理的东西

- **Hy-MT2-30B-A3B-FP8**：在共享存储 `/mnt/speech-h200-fsx/data/data/models/tencent-Hy-MT2-30B-A3B-FP8`，约 29 GB，不在 J 里。只有重新生成 AliMeeting 参考译文和口语开发集参考译文时才需要。现成的参考译文已经在 `data/` 里。
- **AMI 原始数据的来源**：内部 bucket `s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/...`，下载脚本是 `data/ami_multispk_si_v1/tools/ami_download.sh`。只有从零重建数据时才需要；处理好的数据在 `R/data/ami_multispk_si_v1`，已经在 S3 上。
- **旧会话临时目录 `/tmp/claude-0/...`**：只有旧的 v1/v2 启动脚本会引用它，新服务器上没有这个目录：
  - `reports/gstar_si_paper_v1/run_speech*.sh`、`run_v2*.sh`、`make_speech_jobs.py`、`make_v2_jobs.py`；
  - `experiments/uniss_qwen3_omni_v1/scripts/run_gstar_night*.sh`、`run_gstar_demo.sh`。

  里面调用的 gpuq6.py 和 holder 脚本，仓库里都有正式版本（`scripts/gpu_queue/`），要用的话把路径改过去即可。当前两篇论文的脚本已经不依赖 `/tmp`：v7 用到的临时文件已存进 `reports/uniss_si_v4/scratch_inputs_v7`。

---

## 1. 环境：三套实验共用

| 环境 | 大小 | 版本 | 用途 |
|---|---|---|---|
| `J/conda_envs/uniss-eval` | 9.7G | Py 3.12.13，torch 2.6.0+cu124，transformers 4.53.1，vllm 0.8.5.post1，funasr 1.3.29，sacrebleu 2.5.1，modelscope 1.38.1，jieba、opencc | 打分、ASR、A.PCP、SIM-O、UTMOS、GPU 队列（gpuq6/gpuq7）、论文脚本、数据构建；**也是下面几个环境的底座** |
| `J/conda_envs/qwen3omni` | 161M | 建在 uniss-eval 上（`--system-site-packages`）的 venv，加 transformers 5.19.0、transformer-engine 2.6.0、flash-attn 2.8.3 | Thinker 的 Megatron 训练（`ENV_ROOT`）、合并、HF 解码（`gstar_si_stream*`、`gstar_multi`） |
| `J/venvs/vllm_omni_030` | 8.7G | torch 2.13.0+cu132，vllm 0.30.0，vllm-omni 0.30.0，transformers 5.14.1 | `si_thinker_stream`（Thinker）、`si_infer`（Talker）、Hy-MT2 |
| `J/venvs/nemo_diar` | 9.8G | torch 2.11.0+cu128，nemo_toolkit 3.1.0（editable → `J/softwares/NeMo-Speech`，208M） | Nemotron-3 说话人分离后验（多说话人） |
| `J/conda_envs/uniss-train` | 7.5G | Py 3.12.13，torch 2.6.0+cu124，transformers 4.53.1，numba，librosa 0.11 | TTS 实验的主要训练和评测 |
| `J/venvs/tts_schemes` | 6.5G | torch 2.6.0+cu124，transformers 4.57.3，voxcpm 2.0.3，qwen-tts 0.1.1（底座是 uniss-eval） | TTS 的 S3/S4 方案、`render_backend.py`、编码缓存 |
| `J/venvs/uniss-formal-align-v1` | 470M | — | 仅 `si_extra_metrics.py` 在 `EXTRA=1` 时使用，可不拷 |
| `J/evaluation_deps/stopes-a4e75e8` | 277M | — | A.PCP（AutoPCP）依赖 |

**工具：**
- tectonic：`J/tools/tectonic-0.15.0/tectonic`（另有 `J/tools/tectonic/tectonic`），编译论文用
- s5cmd：`/usr/local/bin/s5cmd`，仅备份脚本用
- 系统自带的 `tmux`、`nvidia-smi`：占卡程序 `scripts/gpu_queue/holder_on.sh` 要用

**常用环境变量：**
```
# vLLM 作业：PATH 里没有这个 bin，引擎会因为找不到 ninja 而启动失败
export PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH
export VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030   # 15G 编译缓存，可重建
export VLLM_USE_FLASHINFER_SAMPLER=0
# 打分（.tmp/pylib_jieba/jieba 是指向 uniss-eval 里 jieba 的软链接）
export PYTHONPATH=.:.tmp/pylib_jieba
# 训练和 HF 解码
export PYTHONPATH=$R:$R/third_party/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1 TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch
# GPU 队列
GPUQ_CARDS=0,1,2,3,4,5,6,7   # gpuq7 还读 OUR_MAX、CAP、PIGGY_MIN、GPUQ7_LEDGER
```
经验值：vLLM TP2 在 `--gpu-mem 0.3` 时每张卡实际占 50–55 GB，交给 gpuq7 时按 56000 MB 申报。

**两篇论文共用的模型**（都在 R 里，已同步到 S3）：

| 路径 | 大小 | 用途 |
|---|---|---|
| `pretrained_models/Qwen3-Omni-30B-A3B-Instruct` | 66G | 所有 Thinker 训练和解码的底座 |
| `pretrained_models/UniSS`（BiCodec、glm4 tokenizer） | 5.4G | Talker 语音合成前端 |
| `eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss`（Talker T1c） | 3.5G | 所有语音合成 |
| `third_party/Megatron-LM` | 94M | 训练 |

**评测模型**（都不在 R 里，在 J 里）：

| 模型 | 位置 | 用途 |
|---|---|---|
| Whisper-large-v3 | `J/cache/huggingface/hub/models--openai--whisper-large-v3` | 英文 ASR |
| Paraformer-zh | `J/cache/modelscope` | 中文 ASR |
| WavLM-SV（base-plus） | `J/evaluation_models/wavlm-base-plus-sv` | 说话人相似度 |
| AutoPCP-multilingual-v2、wav2vec2-large-xlsr-53 | `J/evaluation_models/`（共 1.7G） | A.PCP |
| WavLM-Large | `J/cache/huggingface/hub/models--microsoft--wavlm-large`（1.2G） | 只是自监督预训练模型，**没有说话人验证（x-vector）头，不能用来算 SIM-O**（2026-10-30 更正）；TTS 的 S5 判别器用它 |
| UTMOS | `J/cache/torch/hub/tarepan_SpeechMOS_v1.2.0` + `hub/checkpoints`（393M） | 语音质量 |

---

## 2. 从 S3 恢复：命令

先把代码仓库拉回来，加上三套实验都要的东西：

```bash
export AWS_REGION=ap-southeast-3
S3=s3://stevehguo-ap-southeast-3/speech_model_backup/jasonleeeli
J=/opt/dlami/nvme/jasonleeeli
# 1) 代码仓库：S3 上的 R 包含 .git 和所有未提交的文件；排除大目录，后面按需拉
s5cmd --numworkers 64 cp --exclude 'data/*' --exclude 'pretrained_models/*' --exclude 'checkpoints/*' \
      --exclude 'eval_outputs/*' --exclude 'experiments/llm_vits_route_v1/runs/*' "$S3/projects/UniSS/*" $J/projects/UniSS/
#    也可以从 GitHub 拉：git clone https://github.com/happyjin/UniSS-training-reproduction.git
#    但 GitHub 上没有被 .gitignore 排除的目录，也没有论文仓库（paper/* 各自是独立的 git 仓库，没有远端）
# 2) 环境
for e in conda_envs/uniss-eval conda_envs/qwen3omni conda_envs/uniss-train venvs/vllm_omni_030 venvs/nemo_diar \
         venvs/tts_schemes softwares/NeMo-Speech evaluation_deps/stopes-a4e75e8 tools evaluation_models; do
  s5cmd --numworkers 64 cp "$S3/$e/*" "$J/$e/"; done
# 3) 缓存（只拉用到的模型）
for m in models--openai--whisper-large-v3 models--microsoft--wavlm-large models--openbmb--VoxCPM2 models--Qwen--Qwen3-TTS-Tokenizer-12Hz; do
  s5cmd --numworkers 64 cp "$S3/cache/huggingface/hub/$m/*" "$J/cache/huggingface/hub/$m/"; done
s5cmd cp "$S3/cache/modelscope/*" $J/cache/modelscope/
s5cmd cp "$S3/cache/torch/*" $J/cache/torch/
s5cmd cp "$S3/cache/root_dot_cache/*" $J/cache/root_dot_cache/
# 4) 补回软链接（必须做）
bash $J/projects/UniSS/docs/migration/restore_symlinks.sh
# 5) 检查环境
$J/conda_envs/uniss-eval/bin/python -c "import torch, transformers, sacrebleu, funasr; print(torch.__version__)"
$J/venvs/vllm_omni_030/bin/python -c "import vllm; print(vllm.__version__)"
```

**注意事项：**
- 第 1 步的 `--exclude` 用 s5cmd 的通配符语法。如果版本不支持，就按子目录分别 cp。
- S3 上的 UniSS 有两个来源：
  - 2026-10-30 08:21 UTC 之后有实时同步（每 60 秒），关机前的最后改动都在；
  - J 里其余的东西靠原来的全量同步（每轮约 3 小时）。
- 关机前同步状态的日志在 `J/s3_live_sync/sync.log`，S3 上也有一份。

每套实验要额外拉的数据见下面三节，表里的路径都相对于 R，直接拼到 `$S3/projects/UniSS/` 后面即可。

---

## 3. 单说话人同传论文（QABC，`paper/bayes_simul_s2st_v7`）

### 3.1 数据

| 路径 | 大小 | 用途 |
|---|---|---|
| `data/si_thinker_oracle_prefix_v1/`（`rows_train.jsonl`、`rows_val`、`dev_items_{d}.jsonl`） | 156M | 训练适配器 s0/s1/s2；UniST 开发集 |
| `data/si_thinker_oracle_prefix_v7_spont/` | 83M | 训练 A2 |
| `data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/` | 183G | 训练和开发集的源语音（rows 的 `audio` 字段指向这里） |
| `data/si_spont_v1/`（`audio/`、`items_*`、`dev_{d}_ref.jsonl`） | 3.9G | 口语开发集、A2 训练 |
| `data/raw/realsi/RealSI_data/data/{en2zh,zh2en}/{json,wav}` | 812M | **RealSI 测试集，只能测试** |
| `data/acl6060_dev_v1/`（`items.jsonl`、`refs.jsonl`、`talks.jsonl`、`seg/`） | 107M | **ACL60/60 测试集，只能测试** |
| `data/cvss_t/si_items_v1/items_{en2zh,zh2en}.jsonl` | 小 | CVSS-T 测试条目和参考译文 |
| `J/CVSS/canonical_16k/cvss_t_zh_en_test/{source_zh,target_en}`（**不在 R 里**） | 1.6G | CVSS-T 输入音频，以及 SIM-O 的参考语音 |
| 只在重建训练 rows 时需要：`data/si_bayes_lr_v1`（3.0G）、`data/si_belief_v1`（689M）、`data/unist_simul_thinker_v1`（3.7G）、`data/raw/acl6060`（2.0G） | 9.4G | 可不拷 |

### 3.2 模型

**训练好的适配器：**
- 都有 LoRA 版本，体积很小，拷 LoRA 再合并就可以省掉多份 66G 的合并模型。
- LoRA 在 `checkpoints/uniss_qwen3_omni_v1/` 下：

  | 适配器 | LoRA 目录 | 大小 | 合并后目录 |
  |---|---|---|---|
  | s0（预注册主检验用的就是它） | `thinker_op_v3_20261023T213631Z` | 69M | `eval_outputs/uniss_qwen3_omni_v1/thinker_op_v3_20261023T213631Z/thinker_merged_i200` |
  | s1（分档表用） | `thinker_op_v3s1_20261025T143042Z` | 18M | `.../thinker_op_v3s1_20261025T143042Z/thinker_merged` |
  | s2 | `thinker_op_v3s2_20261025T143042Z` | 18M | `.../thinker_op_v3s2_20261025T143042Z/thinker_merged` |
  | A2 | `thinker_op_v7sp_20261029` | 18M | `.../thinker_op_v7sp_20261029/thinker_merged_i200` |
  | bma3（s0/s1/s2 权重平均） | 无 LoRA，由三个合并模型平均得到 | — | `.../thinker_op_v3_bma3/thinker_merged` |

- 每个合并模型 66G。
- 合并命令：`merge_thinker_sd.py --checkpoint-dir <lora>/iter_0000200 --rank 32 --alpha 64 --out <merged>`。
- bma3：`python -m experiments.uniss_qwen3_omni_v1.training.average_merged --out .../thinker_op_v3_bma3/thinker_merged <s0> <s1> <s2>`。

**Talker：**
- T1c 的导出版本（3.5G）见第 1 节。
- 只有重新导出时才需要 checkpoint `checkpoints/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/iter_0001500`（4.6G）。

### 3.3 不重跑、只重建论文表格需要的结果

都在 `eval_outputs/uniss_qwen3_omni_v1/`（下表记为 O）下面。语音目录很大，**只拷 json/jsonl 就够了**（约 4.5G，全拷要约 560G）：

| 路径 | 说明 |
|---|---|
| `O/si_v4/`（除 `speech/` 以外，<1G） | 开发集、RealSI 文本网格、离线结果、CVSS、`cvss_table1/` |
| `O/si_v4/speech/`（全部 95G，json 约 0.6G） | 语音结果，只拷 json |
| `O/si_t1c_20261021T181440Z/full_iter0001500/system_*`（全部 467G，json 约 3.8G） | v3 语音结果，只拷 json |
| `O/realsi_thinker_offline_check`、`realsi_thinker_stream_v1`、`_la2dense_v1`、`_swbayes_v1`、`_full_thinker_op_v3_20261023T213631Z_iter200`、`acl6060_dev_v1*`、`acl6060_dev_v2_thinker_op_v3s{1,2}_*`、`bayesrisk_v1/dev_collect2/dump`、`v2_ca_latency` | 合计约 250M。**其中几个目录没有保存过启动脚本，必须拷**，见 3.5 节 |
| `reports/uniss_si_v4/`（含 `diag/`、`scratch_inputs_v7/`） | 诊断、按方向选配置、P2 的结果 |
| `reports/uniss_m2_fluency_bayes_v3cfg`、`reports/uniss_m2_bayesrisk_{sf,v1,v2}` | 早期附录用到的数字 |
| `paper/bayes_simul_s2st_v4/`（4.6M） | **v7 的脚本会 import v4 的 scripts**，必须一起拷 |
| `paper/bayes_simul_s2st_v7/` | 论文本身，有独立的 git 仓库 |

只拷 json 的命令（s5cmd 支持 `--include`）：
```bash
s5cmd --numworkers 64 cp --include '*.json' --include '*.jsonl' \
  "$S3/projects/UniSS/eval_outputs/uniss_qwen3_omni_v1/si_v4/speech/*" $R/eval_outputs/uniss_qwen3_omni_v1/si_v4/speech/
```

### 3.4 运行顺序（在 R 下执行，`EPY=$J/conda_envs/uniss-eval/bin/python`）

1. **训练适配器**（8 卡）：
   ```
   CUDA_VISIBLE_DEVICES=0,...,7 RUN_ID=thinker_op_v3_20261023T213631Z ROWS=data/si_thinker_oracle_prefix_v1/rows_train.jsonl \
     NPROC=8 MBS=32 GBS=256 NUM_WORKERS=12 TRAIN_ITERS=800 SAVE_INTERVAL=200 EXIT_INTERVAL=200 WARMUP_ITERS=30 \
     MERGED=$PWD/eval_outputs/uniss_qwen3_omni_v1/<RUN_ID>/thinker_merged_i200 \
     bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_sd.sh
   ```
   - 用第 200 步的存档。s0 的种子是 20261022。
   - s1、s2 用同样的命令，加 `TRAIN_SEED=SD_SEED=20261101` 或 `20261102`。
   - A2：先用 `make_p1_jobs.py` 跑 P1 队列（生成 v7_spont 训练数据），再按 `reports/gstar_si_paper_v1/run_v3_eval8.sh` 里 `train_a2()` 的设置训练。
2. **文本端**（RealSI、ACL、开发集）：
   - 生成作业清单：`python3 reports/uniss_si_v4/make_<q>_jobs.py reports/uniss_si_v4/jobs/<q>_jobs.json`。
   - 运行：`GPUQ_CARDS=0,...,7 $EPY -u scripts/gpu_queue/gpuq7.py reports/uniss_si_v4/jobs/<q>_jobs.json logs/si_v4/<q>`。
   - 队列顺序：`p0` → `c1`（`launch_c1_now.sh`）→ `d1` → `d1b` → `d1_test` → `p2`/`p2fix` → `tier` → `tier2_text`。
   - ACL 的 StreamLAAL 评分：`$EPY -m experiments.uniss_qwen3_omni_v1.evaluation.si_streamlaal score --refs data/acl6060_dev_v1/refs.jsonl --items data/acl6060_dev_v1/items.jsonl --blocks "<dir>/blocks_en2zh_*.jsonl" --dump <dir>/seg --procs 48`。
3. **语音合成和 ASR**：
   - 生成作业：`python3 reports/uniss_si_v4/make_speech_jobs.py <jobs.json> <spec.json>`，用 gpuq7 运行。现成的 spec 有 `s1_spec`、`s3_spec`、`tier2_speech_jobs_spec` 等；`resume_speech_jobs.py` 用来续跑。
   - 每个作业三步：`run_si_probe.sh`（Talker）→ `speech_only.py` → `evaluation.asr_transcribe`。
   - 预注册检验要求每个系统 3 个语音种子 × 15 个窗口。
4. **CVSS-T Table 1**：
   - 文本：p0 里的 `cvss_*` 作业。
   - 冻结 Thinker 的语音：`jobs/s2_jobs.json`。
   - s0 的语音和各项指标：`bash reports/uniss_si_v4/run_cvss_table1.sh`（固定用卡 0–3）。
   - 出表：`paper/bayes_simul_s2st_v7/scripts/cvss_table.py`。
5. **打分和检验**（`PYTHONPATH=. $EPY`）：
   1. `reports/uniss_si_v4/{select_d1,test_d1,test_p2,commit_errors,zh2en_diag}.py`
   2. v7 的 scripts：`select_dev` → `select_seed` → `realsi_text` → `realsi_speech` → `realsi_speech_v4count` → `c1_points` → `offline` → `cvss_table` → `tiers` → `collect_results` → `make_tables` → `v2_extras` / `v3_extras` / `v5_extras` / `v6_extras` / `v7_extras`
6. **编译论文**：`cd paper/bayes_simul_s2st_v7 && $J/tools/tectonic-0.15.0/tectonic --keep-logs --keep-intermediates main.tex`

### 3.5 本论文要注意的地方

- 以下目录没有保存过启动脚本，**只能拷贝**：
  - `realsi_thinker_stream_v1`、`_la2dense_v1`、`_swbayes_v1`、`_full_..._iter200`
  - `realsi_thinker_offline_check`
  - `acl6060_dev_v1`、`acl6060_dev_v2_*`

  要重跑的话，参数可以从文件名后缀还原，例如 `_swop1b1q0.191` 对应 `--policy selfwait --wait-biases=1 --sw-queue-rate 0.191`。
- **RealSI 和 ACL60/60 只能用作测试**，不能用于训练、调参或选择。
- **SIM-O：** v8 已用 `wavlm-base-plus-sv` 补算（另一个尺度，不可与论文那一列比较）。论文定义的 WavLM-Large 说话人验证 checkpoint 本机和 S3 上都没有。

---

## 4. 多说话人会议同传论文（`paper/gstar_si_v4`）

### 4.1 数据（都在 R 里；`data/ami_multispk_si_v1` 共 188G，下面只列需要的子集）

| 路径（`data/ami_multispk_si_v1/` 下） | 大小 | 用途 |
|---|---|---|
| `session_audio/ihm_mix`（171 个 AMI 混合音频） | 11G | 构建数据、语音 |
| `session_audio/zh_mix_v8`（重新配音的中文会议，即中译英的源语音） | 11G | 构建中译英数据、语音 |
| `shards/{source_ihm,target_ihm_en_zh}` | 26G | 构建数据；评测时的参考音色 |
| `index/`、`annotations/`（含 `ami_public_manual_1.6.2.zip`）、`meetings.json`、`en_norm_v1.jsonl`、`reports/audio_check_rows.jsonl` | 约 0.3G | 构建数据 |
| `diar/nemotron_{offline,lowlat}[_zh3..zh8]`（后验和 RTTM） | 586M | 构建数据、语音 |
| `e2e_v0` | 7.2G | 英译中训练和测试音频 |
| `e2e_gstar_v1` | 1.9G | 离线英译中（O2、级联、分离表） |
| `e2e_gstar_stream_v1` | 351M | 英译中截止期标签（S2-dl）、LA2 |
| `e2e_gstar_zh2en_v1` | 12G | 中译英全部 |
| `e2e_gstar_bidir_v2/rows_train.jsonl` | 39M | 训练 O2 和 O2-nosd |
| `e2e_gstar_stream_zh2en_v1` | 100M | 中译英截止期标签 |
| `e2e_gstar_abc_v1` | 70M | S2（ABC，v4 正文不用） |
| `zh_revoice_v1` | 41G | 只在重建中文配音时需要 |
| `zh_mix_v8`、`zh_mix_v6/exclude.jsonl` | 0.15G | 构建数据 |
| `session_audio/{zh_mix,zh_mix_v2..v7}`、`sdm/` | ~75G | **不需要**（旧迭代；`sdm` 是指向 `J/datasets/auden_ami_sdm` 的软链接） |

其他：

| 路径 | 大小 | 用途 |
|---|---|---|
| `data/si_thinker_selfdistill_v2/rows_train.jsonl` | 69M | 训练数据里 15% 的单说话人行 |
| `data/processed/simul_uniss_subsecond_v1`（全部 230G，**只用到其中 4,285 个文件**） | — | 训练行的音频；可以只拷 rows 里引用到的文件 |
| `data/alimeeting_si_v1`（音频、分离结果、Hy-MT2 参考译文、4,243 个测试窗口） | 2.1G | **AliMeeting 测试集，只能测试** |
| `data/raw/alimeeting/Test_Ali/Test_Ali_near` | 3.5G | 只在重建 AliMeeting 测试集时需要 |
| `pretrained_models/Nemotron-3-Diarization` | 190M | 说话人分离 |
| `pretrained_models/Qwen2.5-1.5B-Instruct` | 2.9G | 英文参考规范化（`normalize_ami_en.py`） |

### 4.2 模型

合并后的 Thinker 在 `eval_outputs/uniss_qwen3_omni_v1/` 下，每个 66G：

| 模型 | 目录 |
|---|---|
| O2 | `thinker_gstar_o2_20261029b` |
| O2-nosd（也是级联 C-ft2 的翻译器） | `thinker_gstar_o2nosd_20261029b` |
| S2（ABC） | `thinker_gstar_s2abc_20261029b` |
| 英译中 S2-dl | `thinker_gstar_s2dl_20261029b` |
| 中译英 S2-dl（只训了 1000 步） | `thinker_gstar_s2dl_zh2en_1k_20261030` |

- 省空间的办法：拷 `checkpoints/uniss_qwen3_omni_v1/<run>` 的 LoRA（每个 0.26–0.79G），再用 `merge_thinker_gstar.py` 合并。
- v1/v2 的旧模型 `thinker_gstar_{o1,o1nosd,s1}_20261026T195403Z` 只在重跑旧基线时需要。

### 4.3 不重跑、只重建论文表格需要的结果（`eval_outputs/uniss_qwen3_omni_v1/` 下）

| 目录 | 大小 | 说明 |
|---|---|---|
| `gstar_v3_20261029` | 2.2G | 主结果：两个方向 + AliMeeting |
| `gstar_v4_20261030` | 小 | 英译中 S2-dl 在 Δ=0.5/2 s 的解码；中译英和 AliMeeting 的 S2-dl 以关机前实际跑完的为准 |
| `gstar_v4_speech_20261030` | 52G | 语音表 |
| `gstar_offline_eval_o1_20261026T195403Z`、`gstar_cascade_20261028`、`gstar_stream_eval_o1_20261026T195403Z`、`gstar_v2b_20261028` | 约 12G | 英译中的旧基线（B-off、C-van、B-LA2）和分离表 |
| `gstar_speech_full_20261028` | 31G | 英译中 B-off 的语音 |

- **v3 论文引用的 `gstar_v3_speech_20261029` 已经不存在**，v3 的语音表无法重新生成；v4 不受影响。
- 论文：`paper/gstar_si_v4`（2.7M，有独立 git 仓库，含 `generated/bootstrap.json` 和 `numbers.json`）。

### 4.4 运行顺序（在 R 下执行）

1. **构建数据**：
   - **AMI：**
     1. `ami_download.sh`（需要内部 bucket）；
     2. `setup_nemo_diar.sh`；
     3. `experiments/uniss_qwen3_omni_v1/scripts/run_nemotron_dump.sh`；
     4. `python -m …data_clean.build_ami_ms_rows --manual data/ami_multispk_si_v1/annotations/ami_public_manual_1.6.2.zip`；
     5. `build_ami_gstar_si_rows`；
     6. `build_ami_gstar_stream_rows`。
   - **中译英会议：**
     1. `normalize_ami_en`；
     2. `build_ami_zh_revoice prep`；
     3. `make_revoice_jobs.py`；
     4. `run_revoice.sh`；
     5. `V=v8 bash reports/gstar_si_paper_v1/run_zh_data_final.sh`；
     6. `build_ami_gstar_stream_rows_zh2en`。
   - **AliMeeting：** `build_alimeeting_test prep --raw data/raw/alimeeting/Test_Ali --out data/alimeeting_si_v1`。分离后验、Hy-MT2 参考译文和窗口切分是 `make_v3_jobs.py` 里最前面的几个作业。
2. **训练**（`run_thinker_gstar.sh`，Megatron，8 卡，GBS 64）：
   - `run_bidir_pipeline.sh` 依次训练 O2（2000 步）、S2（3000 步）、O2-nosd（2000 步）。
   - 英译中 S2-dl：`run_v3_eval3.sh` 里的训练部分（基于 O2，3000 步，MBS 8）。
   - 中译英 S2-dl：`run_v4c.sh`（1000 步，4 卡，MBS 16）。**建议有时间时补训到 3000 步，与英译中对齐。**
   - 级联 C-ft2 不用另外训练：它是 O2-nosd 加上按分离结果切段。
3. **解码**：
   - 主结果：`python reports/gstar_si_paper_v1/make_v3_jobs.py > jobs/v3_jobs.json`，然后 `GPUQ_CARDS=0,…,7 $EPY scripts/gpu_queue/gpuq6.py jobs/v3_jobs.json logs/...`。
     - `ali_mt` 作业要在 PATH 里加上 vllm venv 的 bin。
     - 可选 `make_v3_multi_jobs.py`：同一张卡上跑 4 个作业，共享模型权重。
   - v4 新增：`make_v4_runs.py` 和 `make_v4c_runs.py`，配合 `run_v4.sh`、`run_v4c.sh`。
4. **语音和 ASR**：
   - 生成并运行：`make_v4_speech_jobs.py jobs/v4_speech_systems_{a,b}.json`，用 gpuq6 运行。
   - 每个系统依次执行：Talker `si_infer` → `p80_speechonly.py` → `asr_transcribe`（英文用 Whisper，中文用 Paraformer）→ `gstar_talker stitch` → `gstar_score --asr` → `gstar_listen` → `gstar_voice`（WavLM）→ `utmos_metrics`。
   - 补跑：`run_v4_speechfix.sh` 和 `run_v4_utmos.sh`（要设 `TORCH_HOME`）。
5. **打分和显著性检验**：`gstar_score` 已经包含在作业里。然后在 `paper/gstar_si_v4` 下运行 `scripts/analyze.py`、`scripts/bootstrap.py`。
6. **编译论文**：`python scripts/make_tables.py && python scripts/make_example.py && tectonic main.tex`。

注意：`run_v3_eval*.sh`、`run_bidir_pipeline.sh` 等启动脚本会等待旧日志里的完成标记，例如 `logs/bidir_pipeline_v8b.log`。新服务器上从头跑时去掉这些等待。

---

## 5. TTS 实验（LLM × VITS，`experiments/llm_vits_route_v1`，下面记为 V）

**V 下的 `data/` 和 `runs/` 被 .gitignore 排除，只能从 S3 拉。**
- 整个 V 共 24G。
- 最少需要约 4G：代码 + `data/`（2.9G） + E0 主干的 `G_60000.pth` 和 `D_60000.pth`（约 1G）。
- 手册原文已存进仓库：`docs/llm_vits_route_v1/handbook.md`（原来只在会话的临时目录里）。

### 5.1 数据

只用到 LJSpeech 1.1，按官方 VITS 的划分，另外留出 390 句作为校准集。

| 路径 | 大小 | 用途 |
|---|---|---|
| `J/datasets/ljspeech_download/LJSpeech-1.1`（`metadata.csv`、`wavs/`；**不在 R 里**） | 3.6G | 所有训练和参考录音。**数据清单 manifest 里写的是这个绝对路径** |
| `J/research_sources/vits_official/filelists/ljs_audio_text_{train,val,test}_filelist.txt`（**不在 R 里**） | 15M | 只用来定义数据划分 |
| `V/data/manifest.jsonl`、`manifest_v2.jsonl` | 各 9M | 训练、验证、测试、校准列表和参考文本 |
| `V/data/qwen_prefix/*.npz` | 2.2G | Qwen3-1.7B 前缀状态（E2、S6、S1 方案） |
| `V/data/future_cands/` | 108M | vLLM 生成的后续候选（S6-future、S1、T2、T5） |
| `V/data/s3_targets/`、`V/data/s4_targets/` | 531M + 37M | 编码缓存（VoxCPM2 AudioVAE；Qwen3-TTS 12Hz） |
| `V/runs/final/align`（`durations.npz`、`unit_prosody.jsonl`） | 12M | 时长标签，所有方案都要 |

### 5.2 模型

| 模型 | 位置（都在 J/cache 或 J/hf_models，不在 R 里） | 大小 | 用途 |
|---|---|---|---|
| VoxCPM2（只用 AudioVAE） | `J/cache/huggingface/hub/models--openbmb--VoxCPM2` | 4.7G | S3 方案 |
| Qwen3-TTS-Tokenizer-12Hz | `J/cache/huggingface/hub/models--Qwen--Qwen3-TTS-Tokenizer-12Hz` | 651M | S4 编码缓存 |
| WavLM-Large | `J/cache/huggingface/hub/models--microsoft--wavlm-large` | 1.2G | S5 判别器 |
| Qwen3-1.7B | `J/hf_models/Qwen3-1.7B` | 3.8G | 只在重建 `qwen_prefix` 和 `future_cands` 时需要 |
| Whisper-large-v3 | 见第 1 节 | 2.9G | WER |
| UTMOS | `J/cache/torch/hub` | 393M | 语音质量 |

**训练结果**（都在 `V/runs`）：
- `e0_backbone_v2`：12G，其实只需要 `G_60000.pth` 和 `D_60000.pth`。
- `final`：4.6G，S6 系列的微调和策略。
- `all_schemes`：4.6G，S1–S5、T4 的 checkpoint 和策略 t1/t2/t5。

只想跳过训练直接评测时才需要拷这些。

### 5.3 环境

- 用 `uniss-train` 的：准备数据、E0、S6、S1、S2、S5、T4 的训练，以及评测。
- 用 `tts_schemes` 的：编码缓存、S3/S4 训练（`PY_BIN=J/venvs/tts_schemes/bin`）、`render_backend.py`。
- 用 `vllm_omni_030` 的：只有 `sample_future_candidates`。

### 5.4 运行顺序（cwd = V）

1. 准备数据：`prepare_ljspeech.py --ljs <LJS> --filelists <vits_official>/filelists --out data`，然后 `units_v2.py data/manifest.jsonl data/manifest_v2.jsonl`。
2. E0 主干：`launch_backbone.sh <cards> runs/e0_backbone_v2`（60k 步）。
3. LLM 特征：`extract_qwen_prefix.py --model J/hf_models/Qwen3-1.7B --out data/qwen_prefix`，然后 `launch_candidates.sh <shard> 2 <card>`。
4. S6/T3 主路线：`final_driver.sh runs/e0_backbone_v2/G_60000.pth runs/final <cards>`，再跑 `final_extend.sh`、`final_audio.sh` 和 `make_report.py`。
5. 编码缓存（tts_schemes 环境）：`cache_codec_targets.py --what s3|s4 --out data/s3_targets|s4_targets --shard k --num-shards 4`。
6. 各方案训练，都用 `launch_ddp.sh`：

   | 方案 | 脚本 | 步数 | 环境 |
   |---|---|---|---|
   | S5 | `train_s5_wavlm.py` | 20k | uniss-train |
   | T4 | `train_t4_causal.py --part flow`，再 `--part dec` | 各 30k | uniss-train |
   | S2 | `train_s2_cfm.py` | 60k | uniss-train |
   | S3 | `train_codec_prior.py --scheme s3` | 40k | tts_schemes |
   | S4 | `train_codec_prior.py --scheme s4` | 40k | tts_schemes |
   | S1 | `train_stream.py --variant s1` | 8k | uniss-train |

   具体参数见各自的 `runs/all_schemes/<x>/args.json`。
7. 策略：`fit_policies.py --kind t1|t2|t5`。
8. 评测：
   1. `all_schemes_driver.sh <8 cards>`；
   2. `run_best.sh`、`run_best_la.sh`、`run_seam.sh`；
   3. `make_report_all.py --stage final`；
   4. `make_listen_page_all.py`。

**当前状态：**
- 所有方案都已训练并评测完毕，结果在 `V/README.md`。
- 最好的系统是 S6-future + T5 + S5 解码器，等待时间比 E1 k=1 少 43%。
- 解码器加 16 帧前瞻（R=16）可以修复接缝，UTMOS 4.20。
- S3 方案失败了。

**写死路径的脚本**（保持 J 路径不变就不用改）：
- `final_driver.sh`、`final_extend.sh`、`final_audio.sh`、`all_schemes_driver.sh`；
- `launch_*.sh`、`make_jobs.py`、`cache_codec_targets.py`、`train_s5_wavlm.py`；
- `src/audiovae.py`、`eval_audio.py`、`s3_diagnose.py`；
- 另外 `runs/` 下有 32 个作业 json。

### 5.5 其他 TTS 相关（简要）

- Omni Talker 第一阶段 TTS 预热：
  - 代码：`experiments/uniss_omni_s2st_v1/training/{stage1_talker_warmup.py,tts_data.py}`、`scripts/run_stage1_megatron.sh`；
  - 数据：`data/raw/UniST`、`data/cvss_t`；
  - checkpoint：`checkpoints/uniss_omni_s2st_v1`，**735G**，按需拷贝。
- `fluency_test/`：597M，不在 git 里。

---

## 6. 换机器后要继续的实验

以下实验因为旧服务器 2026-10-30 关机没有做完。v4 和 v7 两篇论文里都已标为"未完成"。

**规则：**
- 补完之后写**新版本**的论文文件夹（`paper/gstar_si_v5`、`paper/bayes_simul_s2st_v8`），不要改 v4 和 v7。
- 新实验的输出写到新目录，不要覆盖旧结果。

### 6.1 多说话人论文（优先级从高到低）

1. **中译英 MIST-Simul 补训到 3000 步。这是最重要的一项。**
   - **为什么必须补训：** 英译中训了 3000 步；中译英只训了 1000 步（`thinker_gstar_s2dl_zh2en_1k_20261030`）。所以中译英同传现在只和 LA2 持平（24.78 对 24.81），可能是训练不足，不一定是方法本身的问题。
   - **为什么不能接着第 1000 步续训：** 训练时用了 `--no-save-optim`，没有保存优化器状态；另外学习率调度是按 1000 步设定的。所以要**从 O2 重新训 3000 步**，并换一个新的 RUN_ID。`run_thinker_gstar.sh` 发现目标目录已有 checkpoint 时会拒绝写入，所以旧目录不会被覆盖。
   - 已停掉的 `thinker_gstar_s2dl_zh2en_20261030` 只训到第 40 步，没有用。
   - 命令（8 张卡；4 张卡时用 NPROC=4，MBS 16 显存不够就用 8，GBS 保持 64）：
     ```
     cd $R
     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 MBS=8 GBS=64 RUN_ID=thinker_gstar_s2dl_zh2en_3k_<日期> \
       BASE=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_o2_20261029b/thinker_merged INIT_MERGED=1 \
       ROWS=data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1/rows_train.jsonl TRAIN_ITERS=3000 MASTER_PORT=29855 \
       bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/thinker_gstar_s2dl_zh2en_3k_<日期>.log 2>&1
     ```
     旧服务器上 4 卡跑 1000 步约 50 分钟，8 卡跑 3000 步预计约 1.5 小时。
   - 训完后解码：参照 `reports/gstar_si_paper_v1/make_v4c_runs.py` 和 `run_v4c.sh`，复制一份。把 `S2DLZ` 改成新模型，`OUT` 改成新目录，并去掉 `run_v4c.sh` 里的截止时间判断（`DEADLINE`）。
     - 在 Δ=1 s 下解码 AMI 中译英（3,317 个窗口）和 AliMeeting（4,243 个窗口）；
     - 再在 Δ=0.5 s 和 2 s 下解码 AMI 中译英，参照 `make_v4_runs.py` 里英译中的做法。
2. **AliMeeting 同传解码。** 现在只跑了 608/4243 个窗口，用第 1 项的新模型完整重跑。
3. **中译英语音端。**
   - O2、C-ft2、O2+LA2 三个系统的 Whisper 识别停在约 40%。补跑命令（会接着已有的 `asr_speech.jsonl` 续跑）：
     `bash reports/gstar_si_paper_v1/run_v4_speechfix.sh zh2en_O2 zh2en_C-ft2 zh2en_O2_LA2_d1`
     脚本里的卡号和线程数要按新机器调整。旧机器上识别这一步主要被 CPU 卡住。
   - 中译英 MIST-Simul（3k 新模型）的语音合成加识别：参照 `make_v4_speech_jobs.py`。
4. **多读入间隔训练（S2-dl-m，审稿第二轮第 3 条）。**
   - 数据已经构造好：`data/ami_multispk_si_v1/e2e_gstar_stream_multid_v1`。
   - 训练和解码在 `reports/gstar_si_paper_v1/run_v4b.sh` 里：3000 步，在 0.5/1/2 s 下解码。
   - 这个脚本会等 `run_v4.sh` 的日志，新机器上要去掉这段等待；RUN_ID 也要换成新的。
5. **v4 模型在说话人分离自切窗口上的评测。** 命令：`python3 reports/gstar_si_paper_v1/make_v4_diar_jobs.py > reports/gstar_si_paper_v1/jobs/<新文件名>.json`，然后用 gpuq6 运行。目前附录沿用的是 v2 模型的结果。
6. **算完后更新论文。** 在 `paper/gstar_si_v5`（从 v4 复制）里改 `scripts/systems.py`，指向新的输出目录，然后运行 `analyze.py`、`bootstrap.py`、`make_tables.py`，再编译。
7. **（做不了，或者需要人工的）**
   - 人工核对测试子集；
   - AMI-SDM 远场评测；
   - MOS 主观听测。

### 6.2 单人论文

1. **s1 的 10 个语音点没跑**（只影响分档表，那张表本来就是补充结果）：
   - 英译中：b0=1 @6 s，b0=2 @5 s；
   - 中译英：b0=1 @0.6/0.9/1.1/1.75/5 s，b0=2 @4/5 s。

   用 `reports/uniss_si_v4/make_speech_jobs.py` 加上 `tier2_speech_jobs_spec` 里对应的条目生成作业，用 gpuq7 运行。然后在 v8 里重跑 `tiers.py`。
2. **SIM-O（音色相似度）。** v8 已用 `wavlm-base-plus-sv` 补算（2026-10-30，见 `paper/bayes_simul_s2st_v8/docs/response_to_review_v7.zh-CN.md`）。缓存里的 `microsoft/wavlm-large` 没有说话人验证头，不能用；论文定义的 SIM-O 需要 UniSpeech 的 WavLM-Large SV checkpoint，目前没有。
3. **ACL60/60 上的 LA2 补到预注册的 15 个窗口。** 现在只有 8 个。这是协议缺口，补上后预注册的 ACL 检验才完整。
4. **（需要另外决策的）**
   - 在同一底座上加一个学习型读写策略作对照；
   - 人工听测；
   - 换一个 Thinker 验证；
   - 长篇 RealSI（Doc）。

---

## 7. 关机前的最后状态（2026-10-30）

- **代码仓库：** 已推送到 GitHub `private` 的 main 分支；UniSS 目录每 60 秒实时同步到 S3（`scripts/backup/uniss_s3_live_sync.sh`）。
- **论文仓库：** `paper/bayes_simul_s2st_v7`、`paper/gstar_si_v4` 各有独立的 git 仓库，没有远端，靠 S3 同步（包括 `.git`）。
- **Claude 对话记录和记忆：** 每 3 分钟同步到 `$S3/claude_backup/root_dot_claude/`（`scripts/backup/claude_s3_live_sync.sh`）。恢复时放回 `~/.claude/`，并把项目放在相同路径 `/opt/dlami/nvme/jasonleeeli/projects/UniSS`。
- **软链接清单：** `docs/migration/symlinks_jasonleeeli.tsv`；恢复脚本 `docs/migration/restore_symlinks.sh`。
