"""Fun-CosyVoice3 flow + HiFT（speech token -> 24kHz 波形）随模型目录分发。

flow / HiFT 只在推理 / 评测中把 speech token 转成波形，**不参与训练、不进 state_dict**
（`SI.attach_speech_decoder` 用 object.__setattr__ 挂载，与源项目一致）。这里只负责把
官方权重文件原样拷进模型目录的 `speech_decoder/` 子目录，使一个目录即可推理：

    <model_dir>/
      config.json / model.safetensors / tokenizer* / voxtral_processor_config.json
      speech_decoder/
        flow.pt  hift.pt  cosyvoice3.yaml  [campplus.onnx]  manifest.json

* 导出：`export_streaming_si.py --cosyvoice_dir <Fun-CosyVoice3-0.5B>` 调用 `bundle_speech_decoder`；
* 训练：模型注册的 `additional_saved_files` 含本目录，swift 保存 S1-S3 checkpoint 时整目录复制，
  下一阶段 / 续训的 checkpoint 仍自带声码器（S4 只存 adapter，不复制；推理时回到 base_model_dir 找）；
* 推理：`resolve_speech_decoder` 按「显式参数 > 模型目录 > S4 adapter 的 base_model_dir」定位。

`campplus.onnx`（CAM++ 说话人向量）是可选项：缺失时推理用零向量（中性音色）。
yaml 里 `llm` / `get_tokenizer` 等在加载时被覆盖为 None，因此不需要 `CosyVoice-BlankEN/`。
"""
import json
import os
import shutil
from typing import Optional, Tuple

SPEECH_DECODER_DIR = 'speech_decoder'
MANIFEST = 'manifest.json'
REQUIRED_FILES = ('flow.pt', 'hift.pt')
OPTIONAL_FILES = ('campplus.onnx',)
YAML_CANDIDATES = ('cosyvoice3.yaml', 'cosyvoice2.yaml')


def find_config_yaml(cosy_dir: str, explicit: Optional[str] = None) -> str:
    """flow/HiFT 的官方 yaml 名。Fun-CosyVoice3 是 cosyvoice3.yaml；显式参数优先，其次 manifest。"""
    if explicit:
        if not os.path.isfile(os.path.join(cosy_dir, explicit)):
            raise FileNotFoundError(f'{cosy_dir} 下没有 {explicit}')
        return explicit
    manifest = os.path.join(cosy_dir, MANIFEST)
    if os.path.isfile(manifest):
        with open(manifest, encoding='utf-8') as f:
            name = json.load(f).get('config_yaml')
        if name and os.path.isfile(os.path.join(cosy_dir, name)):
            return name
    for name in YAML_CANDIDATES:
        if os.path.isfile(os.path.join(cosy_dir, name)):
            return name
    raise FileNotFoundError(f'{cosy_dir} 下没有 {" / ".join(YAML_CANDIDATES)}')


def check_decoder_dir(cosy_dir: str) -> None:
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(cosy_dir, f))]
    if missing:
        raise FileNotFoundError(f'{cosy_dir} 缺 {missing}（需要 Fun-CosyVoice3-0.5B 目录）')


def bundle_speech_decoder(cosy_dir: str, model_dir: str, config_yaml: Optional[str] = None) -> str:
    """把 flow.pt / hift.pt / yaml / campplus.onnx 拷到 `<model_dir>/speech_decoder/`，返回该目录。"""
    check_decoder_dir(cosy_dir)
    yaml_name = find_config_yaml(cosy_dir, config_yaml)
    dst = os.path.join(model_dir, SPEECH_DECODER_DIR)
    tmp = dst + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    files = {}
    for name in REQUIRED_FILES + (yaml_name,) + OPTIONAL_FILES:
        src = os.path.join(cosy_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tmp, name))
            files[name] = os.path.getsize(src)
    with open(os.path.join(tmp, MANIFEST), 'w', encoding='utf-8') as f:
        json.dump({'config_yaml': yaml_name, 'source_dir': os.path.abspath(cosy_dir), 'files': files,
                   'has_campplus': 'campplus.onnx' in files}, f, indent=2, ensure_ascii=False)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return dst


def resolve_speech_decoder(model_dir: str, explicit: Optional[str] = None,
                           config_yaml: Optional[str] = None) -> Tuple[str, str]:
    """返回 (声码器目录, yaml 名)。找不到时报错并说明三种来源。"""
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join(model_dir, SPEECH_DECODER_DIR))
    meta = os.path.join(model_dir, 's4_metadata.json')          # S4 adapter 目录只存 adapter
    if os.path.isfile(meta):
        with open(meta, encoding='utf-8') as f:
            base = json.load(f).get('base_model_dir')
        if base:
            candidates.append(os.path.join(base, SPEECH_DECODER_DIR))
    for cand in candidates:
        if cand and all(os.path.isfile(os.path.join(cand, f)) for f in REQUIRED_FILES):
            return cand, find_config_yaml(cand, config_yaml)
    if explicit:
        check_decoder_dir(explicit)                               # 给出具体缺哪个文件
    raise FileNotFoundError(
        f'找不到 flow/HiFT：已查 {candidates}。用 --cosyvoice_dir 指向 Fun-CosyVoice3-0.5B，'
        f'或导出时加 --cosyvoice_dir 把声码器打包进模型目录')
