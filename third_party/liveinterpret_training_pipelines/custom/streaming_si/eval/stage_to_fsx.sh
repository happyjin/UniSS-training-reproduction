#!/bin/bash
# 把评测所需的模型 / 代码 / 数据从 07d3 的节点本地盘搬到共享 FSX，供多节点 Pod 共用。
set -uo pipefail
E=/fsx/stevehguo/eval
C=/nvme/output/stevehguo/streaming_si/s3_s2st/v0-20261012-034209/checkpoint-24980
mkdir -p $E/{models,code,work}

echo "[1/9] S4 checkpoint（跳过 33G 的优化器状态 global_step*）"
mkdir -p $E/models/s4_ck
rsync -a --exclude 'global_step*' --exclude 'rng_state_*' --exclude 'optimizer*' \
      --exclude 'scheduler.pt' $C/ $E/models/s4_ck/

echo "[2/9] S5 adapter"
rsync -a /nvme/stevehguo_infer/kic/s5ck/ $E/models/s5_adapter/

echo "[3/9] SimulS2ST-Omni 模型包（18G）"
rsync -a /nvme/kenxsong/SimulS2ST-Omni-repro/models/SimulS2ST-Omni/ $E/models/SimulS2ST-Omni/

echo "[4/9] Whisper-large-v3（只要 safetensors + 配置）"
mkdir -p $E/models/whisper-large-v3
W=/nvme/stevehguo_eval/models/whisper-large-v3
cp -f $W/model.safetensors $W/*.json $W/merges.txt $W/vocab.json $E/models/whisper-large-v3/ 2>/dev/null
ls $E/models/whisper-large-v3

echo "[5/9] COMET wmt22-comet-da"
rsync -a /nvme/stevehguo_eval/models/comet/ $E/models/comet/

echo "[6/9] CosyVoice3 speech tokenizer"
mkdir -p $E/models/cosyvoice3
cp -f /nvme/zeyangsong/projects/model/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx $E/models/cosyvoice3/

echo "[7/9] 代码：CosyVoice 仓库 / Omni 官方仓库 / 本仓库 custom"
rsync -a --exclude '.git' /nvme/zeyangsong/projects/CosyVoice/ $E/code/CosyVoice/
rsync -a --exclude '__pycache__' /nvme/stevehguo_infer/simul/SimulS2ST-Omni-main/ $E/code/SimulS2ST-Omni-main/
rm -f $E/code/SimulS2ST-Omni-main/models
rsync -a --exclude '__pycache__' /nvme/stevehguo_eval/custom/ $E/code/custom/

echo "[8/9] Python 依赖（Omni 用 transformers 4.52.3；COMET 包）"
rsync -a /nvme/stevehguo_infer/simul/site/ $E/code/site_omni/
rsync -a /nvme/stevehguo_eval/site_comet/ $E/code/site_comet/

echo "[9/9] 评测数据（句级 wav + 清单）"
rsync -a /nvme/stevehguo_eval/work/ $E/work/

echo "STAGE_DONE"
du -sh $E/models/* $E/code/* $E/work 2>/dev/null
