"""ms-swift 注册：streaming_si 模型、template（S1/S2/S3）与数据集。

由 `custom/plugin.py` 导入即完成注册。命名：

  * model_type: `streaming_si`（Thinker Qwen3-0.6B / Talker 0.1B）、`streaming_si_4b`（Thinker Qwen3.5-4B /
    Talker 0.6B），见 variants.py；template 共用 `streaming_si`
  * 数据集（`--dataset <名>`，路径由环境变量给出，见各 `register_*`）。数据集名与行内 `si_stage`
    沿用源项目编号，表示样本格式；括号内为使用它的训练阶段（2026-10 改名后的编号）：
      - `streaming_si_s1`  Emilia-FSQ parquet（S1 / S3-w / S3，TTS）
      - `streaming_si_s2`  UniST P2 parquet（S2，源音频 -> 译文）
      - `streaming_si_s3`  UniST P2 parquet（S4-w / S4，源音频 -> 译文 + 目标 FSQ）

      - `streaming_si_s4` / `streaming_si_s4_val`  流式轨迹 records（S5，见 register_streaming_si_s4）

阶段差异通过 **数据集名 -> 行内 `si_stage` 字段** 传给 template，而不是靠不同 template：
同一次训练只会用一个阶段的数据，template 在 `_encode` 里按行分派到 `si_encode.encode_*`
（S4 分派到 `s4_pipeline.S4Encoder`，其内部复用源 S4StreamingDataset）。
S4 另需 `--tuner_type streaming_si_s4`（s4_peft.StreamingSIS4Tuner）。

必须的 swift 参数（`pipelines/streaming_si/common_args.sh` 已固定）：
  --remove_unused_columns false   # 否则 source_wav / target_fsq 等列在 map 后被删
  --padding_free false --packing false   # SI 的 audio_pad 注入按样本对齐，不能拼包
"""
import copy
import hashlib
import io
import json
import os
from typing import Any, Dict, List, Optional

import torch
from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset
from swift.model import (Model, ModelGroup, ModelLoader, ModelMeta, MultiModelKeys, register_model,
                         register_model_arch)
from swift.template import Template, TemplateMeta, register_template
from swift.template.base import MaxLengthError
from swift.utils import get_logger

from ..model.base_model import VoxtralAudioProcessor
from ..model.speech_decoder import SPEECH_DECODER_DIR
from ..model.modeling_streaming_si import StreamingSIConfig, StreamingSIForConditionalGeneration
from ..data.s4_pipeline import add_s4_stream_tokens, s4_stream_token_ids
from ..data.si_encode import (SIEncodeError, SIOverlongError, collate_si, encode_s1, encode_st, normalize_fsq,
                        st_instruction, st_num_frames, st_overlong_reason)
from ..model.variants import VARIANTS, variant_of
from ..model.voxtral_compat import VoxtralFeatureExtractor

logger = get_logger()

MODEL_TYPE = 'streaming_si'
TEMPLATE_TYPE = 'streaming_si'
STAGES = ('s1', 's2', 's3')

# ---------------------------------------------------------------------------
# 1. 模型结构分组（freeze_llm / freeze_vit / freeze_aligner / LoRA 目标按此生效）
# ---------------------------------------------------------------------------
register_model_arch(
    MultiModelKeys(
        MODEL_TYPE,
        language_model=['thinker_model'],
        vision_tower=['audio_encoder'],            # 冻结的 Voxtral 塔
        aligner=['audio_proj'],
        generator=['talker'],
    ))


# ---------------------------------------------------------------------------
# 2. Loader
# ---------------------------------------------------------------------------
class StreamingSILoader(ModelLoader):
    """从 `export_streaming_si.py` 导出的目录加载（config.json + safetensors + tokenizer）。"""

    # 阶段开关：由 pipelines/streaming_si/train_*.sh export，覆盖导出目录 config.json 中的值。
    # 这些字段不改变参数结构，只影响 forward（loss 组成 / talker 条件），故可按阶段切换。
    STAGE_ENV = {
        'SI_TEXT_LOSS_WEIGHT': ('text_loss_weight', float),
        'SI_SPEECH_LOSS_WEIGHT': ('speech_loss_weight', float),
        'SI_COND_MODE': ('cond_mode', str),
        'SI_HIDDEN_NOISE': ('hidden_noise', float),
        # 1/true = cond_mode='text_only' 且 text_w=0 时跳过 thinker 主干前向（S1 省 4B 空转）。
        # 条件与等价性见 model/model_si.py::SI.can_skip_thinker；不满足条件时自动不生效。
        'SI_SKIP_THINKER': ('skip_thinker_when_text_only', lambda v: str(v).lower() in ('1', 'true', 'yes')),
        # 1/true = text loss 走 liger fused_linear_cross_entropy（S2/S4 省掉 151936 词表的
        # 全序列 logits 物化）。代价：无 text_acc 指标。见 model/model_si.py::SI.use_fused_text_ce
        'SI_FUSED_TEXT_CE': ('fused_text_ce', lambda v: str(v).lower() in ('1', 'true', 'yes')),
        'SI_S4_DECISION_WEIGHT': ('s4_decision_weight', float),   # 设置即启用 S4 目标
        # talker 条件取 thinker 第几层（0 起；-1 = 最后一层，已过 final norm）。只改 forward 取哪层，
        # 不改参数结构；会写进保存的 config，推理（infer_s2st / 门 A）自动沿用。换层宜在 S4-w 这类只训桥接的阶段
        'SI_TALKER_BRIDGE_LAYER': ('talker_bridge_layer', int),
        # 实验 b：talker 条件取哪些位置（answer / audio / audio+answer），默认 answer = 历史行为。
        # 与取层同理，只改 forward 的掩码、不改参数结构；会写进保存的 config，推理自动沿用。
        'SI_TALKER_COND_POSITIONS': ('talker_cond_positions', str),
        # 消融：Fusion 幅值通道（1/true 开）。**会新增参数** talker.fusion.scale_gain（零初始化），
        # 故必须在建模前设置；写进保存的 config，推理自动沿用。
        'SI_TALKER_FUSION_SCALE': ('talker_fusion_scale', lambda v: str(v).lower() in ('1', 'true', 'yes')),
    }

    def get_config(self, model_dir: str):
        config = StreamingSIConfig.from_pretrained(model_dir)
        for env, (attr, cast) in self.STAGE_ENV.items():
            if os.environ.get(env, '') != '':
                setattr(config, attr, cast(os.environ[env]))
        if config.cond_mode not in ('full', 'text_only'):
            raise ValueError(f'SI_COND_MODE 应为 full / text_only，收到 {config.cond_mode!r}')
        cond_pos = str(getattr(config, 'talker_cond_positions', 'answer') or 'answer')
        if cond_pos not in ('answer', 'audio', 'audio+answer'):
            raise ValueError(f'SI_TALKER_COND_POSITIONS 应为 answer / audio / audio+answer，收到 {cond_pos!r}')
        expected = self.model_meta.model_type if self.model_meta is not None else None
        actual = variant_of(config)
        if expected in VARIANTS and expected != actual:
            raise ValueError(f'{model_dir} 是 {actual}（thinker_arch={config.thinker_arch}），'
                             f'与 --model_type {expected} 不符')
        logger.info(f'streaming_si 变体: {actual}（{VARIANTS[actual]["desc"]}；talker_preset={config.talker_preset}, '
                    f'talker {config.talker_num_hidden_layers}L/{config.talker_hidden_size}d）')
        logger.info(f'streaming_si 阶段开关: text_w={config.text_loss_weight} speech_w={config.speech_loss_weight} '
                    f'cond_mode={config.cond_mode} hidden_noise={config.hidden_noise} '
                    f'talker_bridge_layer={config.talker_bridge_layer} '
                    f'talker_cond_positions={config.talker_cond_positions}')
        return config

    def get_processor(self, model_dir: str, config):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # 导出时已按 build_qwen_tokenizer 注册 <|audio_pad|> 并保存；这里只校验，
        # 避免加载到未经对齐的原始 Qwen3 tokenizer 而静默错位。
        ids = tokenizer.encode(config.audio_special_token, add_special_tokens=False)
        if ids != list(config.audio_ids):
            raise RuntimeError(
                f'{model_dir} 的 tokenizer 中 {config.audio_special_token} -> {ids}，'
                f'与 config.audio_ids={config.audio_ids} 不一致；请用 export_streaming_si.py 重新导出')
        if config.s4_decision_weight is not None:
            add_s4_stream_tokens(tokenizer, capacity=int(config.vocab_size))
        return tokenizer

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        # liger 融合核必须在建模型之前挂：RMSNorm 那半是替换 HF 模块里的类对象，
        # 只影响此后新建的实例（详见 integration/liger_ops.py 的 docstring）。
        # 不走 swift/transformers 的 --use_liger_kernel：那条路按 config.model_type
        # 查表，streaming_si 必然 miss 且静默空转。
        from .liger_ops import apply_liger_to_streaming_si
        apply_liger_to_streaming_si()
        # model_kwargs 已由 ModelLoader.__init__ 注入 dtype（transformers>=4.56 用 `dtype`），勿重复传
        model = StreamingSIForConditionalGeneration.from_pretrained(model_dir, config=config, **model_kwargs)
        if model.audio_encoder is None and config.speech_loss_weight + config.text_loss_weight > 0:
            logger.warning('streaming_si: 导出目录未含 Voxtral 结构，音频通路不可用（S1 纯 TTS 可忽略）')
        if config.s4_decision_weight is not None:
            model.config.s4_stream_token_ids = s4_stream_token_ids(processor)
            if model.audio_encoder is None:
                raise RuntimeError('S4 需要 Voxtral 音频塔，导出目录未包含')
            logger.info(f'streaming_si S4: decision_weight={config.s4_decision_weight} '
                        f'[wait, W, chunk, src_end]={model.config.s4_stream_token_ids}')
        return model


# ---------------------------------------------------------------------------
# 3. 音频读取（P2 的 tar 内偏移引用）与 mel 前端
# ---------------------------------------------------------------------------
_P2_FPS: Dict[str, Any] = {}


# pipelines/streaming_si/p2_parquet_prep.py --lengths 写入的元数据列（与 limits 无关的原始量）
P2_META_COLS = ('src_samples', 'instr_tokens', 'tgt_tokens')


def _drop_overlong() -> bool:
    """SI_DROP_OVERLONG=1：S2/S4-w/S4 超长样本丢弃而不截断（train_st.sh / train_s2st.sh 默认打开）。"""
    return os.environ.get('SI_DROP_OVERLONG', '0').lower() in ('1', 'true')


def _keep_noise_neg() -> bool:
    """SI_KEEP_NOISE_NEG=1：S3/S4 保留「译文为空的噪声负样本」（train_s2st.sh 的 s4 分支默认打开）。

    文本 CE 学「噪声+指令 -> 立即 eos」（与 S2 负样本同款信号），speech 监督为空。
    S4-w 不打开：thinker 冻结 + text_loss_weight=0，这类行没有任何梯度，纯属浪费。
    """
    return os.environ.get('SI_KEEP_NOISE_NEG', '0').lower() in ('1', 'true')


# 噪声负样本的来源标记（匹配 `dataset_name` 或 `id` 的小写子串）：esc50_* / musan_noise / dns5_noise，
# id 形如 `esc50:xxx.wav#0` / `musan_noise:...` / `dns5:...`。新增噪声源（如 mssnsd）时在此补标记，
# 或用 SI_NOISE_TAGS（逗号分隔）覆盖。
_NOISE_TAGS = ('esc50', 'musan', 'dns5', 'mssnsd')


def _noise_tags() -> tuple:
    env = os.environ.get('SI_NOISE_TAGS', '').strip()
    return tuple(t.strip().lower() for t in env.split(',') if t.strip()) or _NOISE_TAGS


def _is_noise_row(row: Dict[str, Any], tags: tuple) -> bool:
    """行是否来自已登记的噪声源。只靠「译文为空」识别太脆弱：译文缺失的真实语音行
    （翻译失败 / 未对齐）若被放行，会教模型「听到人说话也直接 eos」，比丢弃更糟。"""
    key = f"{row.get('dataset_name') or ''}|{row.get('id') or ''}".lower()
    return any(t in key for t in tags)


def _p2_limits() -> Dict[str, Any]:
    """S2/S3 的样本口径。template（编码时）与预处理器（预先过滤）共用，保证两处判定一致。"""
    env = os.environ
    return dict(max_length=int(env.get('SI_MAX_LENGTH', 512)),
                max_target_len=int(env.get('SI_MAX_TARGET_LEN', 256)),
                max_speech_len=int(env.get('SI_MAX_SPEECH_LEN', 750)),
                max_audio_sec=float(env.get('SI_MAX_AUDIO_SEC', 20.0)),
                min_audio_sec=float(env.get('SI_MIN_AUDIO_SEC', 0.5)))


def read_p2_wav(ref: str, min_sec: float = 0.5, max_sec: float = 20.0,
                drop_long: bool = False) -> Optional[torch.Tensor]:
    """'<abs_tar>::offset:length' -> 16k 单声道 1D tensor（与源 `S2STDataset._read_wav` 同口径）。

    短于 min_sec 返回 None（该样本被丢弃）；超过 max_sec 默认截断，`drop_long=True` 时抛
    `SIOverlongError`（丢弃该样本）。
    """
    import numpy as np
    import soundfile as sf
    if not ref or '::' not in ref:
        return None
    path, loc = ref.split('::', 1)
    if not os.path.exists(path):
        return None
    fp = _P2_FPS.get(path)
    if fp is None:
        fp = _P2_FPS[path] = open(path, 'rb')
    try:
        off, ln = (int(x) for x in loc.split(':'))
        fp.seek(off)
        wav, sr = sf.read(io.BytesIO(fp.read(ln)), dtype='float32')
    except Exception:
        return None
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr, 16000).numpy()
    dur = len(wav) / 16000
    if dur > max_sec:
        if drop_long:
            raise SIOverlongError(f'音频 {dur:.2f}s > SI_MAX_AUDIO_SEC={max_sec:g}')
        wav = wav[:int(max_sec * 16000)]
    elif dur < min_sec:
        return None
    return torch.from_numpy(np.ascontiguousarray(wav))


VOXTRAL_FE_CONFIG = 'voxtral_processor_config.json'


def _mel_processor(model_dir: str) -> VoxtralAudioProcessor:
    """Voxtral log-mel 前端，读导出目录里的 `voxtral_processor_config.json`。

    导出时改名存放（不用 processor/preprocessor_config.json）：swift 基类 get_processor 见到
    这两个名字会走 AutoProcessor，而 transformers 4.57 不认识 voxtral_realtime。
    global_log_mel_max 取该配置的值（官方为 1.5，即源训练实际生效值），缺失即报错而非回退默认。
    """
    return VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(model_dir))


# ---------------------------------------------------------------------------
# 4. Template
# ---------------------------------------------------------------------------
class StreamingSITemplate(Template):
    """SI 的序列是自研布局（audio_pad×n + 指令 + 译文 + eos），不经过 chat 模板；
    `_encode` 直接调用 `si_encode`，与源项目 dataset 逐 token 一致（见 test_si_encode.py）。

    `_post_encode` 保持基类的原样返回，因此自定义字段（answer_mask / audio_inputs /
    speech_tokens ...）会一路透传到 `StreamingSIForConditionalGeneration.forward`。
    """

    def _cfg(self) -> Dict[str, Any]:
        if not hasattr(self, '_si_cfg'):
            env = os.environ
            self._si_cfg = dict(
                max_length=int(env.get('SI_MAX_LENGTH', self.max_length or 512)),
                max_target_len=int(env.get('SI_MAX_TARGET_LEN', 256)),
                max_speech_len=int(env.get('SI_MAX_SPEECH_LEN', 750)),
                max_prompt_speech_len=int(env.get('SI_MAX_PROMPT_SPEECH_LEN', 125)),
                max_audio_sec=float(env.get('SI_MAX_AUDIO_SEC', 20.0)),
                min_audio_sec=float(env.get('SI_MIN_AUDIO_SEC', 0.5)),
                ref_drop_ratio=float(env.get('SI_REF_DROP_RATIO', 0.5)),
                strip_prompt=env.get('SI_STRIP_PROMPT_TEXT', 'false').lower() == 'true',
                text_scheduled_sampling=float(env.get('SI_TEXT_SCHEDULED_SAMPLING', 0.0)),
                task=env.get('SI_ST_TASK', 'translate'),
                drop_overlong=_drop_overlong(),
            )
        return self._si_cfg

    def _mel(self):
        if not hasattr(self, '_si_mel'):
            self._si_mel = _mel_processor(self.model_info.model_dir)
        return self._si_mel

    def _s4(self):
        if not hasattr(self, '_si_s4'):
            from ..data.s4_pipeline import S4Encoder
            self._si_s4 = S4Encoder(self.tokenizer, self._mel())
        return self._si_s4

    def _encode(self, inputs) -> Dict[str, Any]:
        row = inputs.extra_kwargs
        stage = row.get('si_stage')
        if stage == 's4':
            enc = self._s4().encode(row)
            if not self.is_training:
                enc.pop('labels', None)
            return enc
        cfg = self._cfg()
        if stage == 's1':
            enc = encode_s1(
                self.tokenizer, json.loads(row['conversations']) if isinstance(row['conversations'], str)
                else row['conversations'], row.get('answer_speech_tokens'), row.get('ref_speech_tokens'),
                max_length=cfg['max_length'], max_speech_len=cfg['max_speech_len'],
                max_prompt_speech_len=cfg['max_prompt_speech_len'],
                ref_drop_ratio=cfg['ref_drop_ratio'] if self.is_training else 1.0,
                strip_prompt=cfg['strip_prompt'],
                text_scheduled_sampling=cfg['text_scheduled_sampling'] if self.is_training else 0.0,
                # 与源 S1TalkerDataset 一致：基础词表大小（不含新增特殊 token），噪声不会采到流式 token
                text_vocab_size=int(getattr(self.tokenizer, 'vocab_size', 151936)))
        elif stage in ('s2', 's3'):
            try:
                wav = read_p2_wav(row['source_wav'], cfg['min_audio_sec'], cfg['max_audio_sec'],
                                  drop_long=cfg['drop_overlong'])
                if wav is None:
                    raise SIEncodeError(f'读不到源音频或时长越界: {row.get("source_wav")}')
                mel = self._mel()(wav, sr=16000)
                target = row['translation'] if cfg['task'] == 'translate' else row['source_transcript']
                enc = encode_st(
                    self.tokenizer, mel, target, row.get('tgt_lang'), max_length=cfg['max_length'],
                    max_target_len=cfg['max_target_len'], task=cfg['task'],
                    speech_tokens=normalize_fsq(row.get('target_fsq')) if stage == 's3' else None,
                    max_speech_len=cfg['max_speech_len'], with_answer_mask=(stage == 's3'),
                    drop_overlong=cfg['drop_overlong'])
            except SIOverlongError as e:
                # 预处理已按同一判定过滤（见 StreamingSIPreprocessor），这里只是兜底；
                # 转成 MaxLengthError 让 LazyLLMDataset 静默换一条
                raise MaxLengthError(f'streaming_si 丢弃超长样本: {e}') from e
        else:
            raise SIEncodeError(f'未知 si_stage={stage!r}（应为 {STAGES}）')
        if not self.is_training:
            enc.pop('labels', None)
        return enc

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        if self.padding_free or self.packing:
            raise ValueError('streaming_si 不支持 padding_free / packing（audio_pad 注入按样本对齐）')
        n_s4 = sum(bool(b.get('_s4')) for b in batch)
        if n_s4:
            if n_s4 != len(batch):
                raise ValueError('同一 batch 不能混用 S4 与 S1/S2/S3 样本')
            from ..data.s4_pipeline import collate_s4
            return collate_s4(batch)
        return collate_si(batch, pad_token_id=0)

    def data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        # 行内原始列（source_wav / conversations ...）已在 _encode 消费完，不再回灌到 batch
        for b in batch:
            b.pop('_extra_kwargs', None)
        return self._data_collator(batch, padding_to=padding_to)


def register_streaming_si_template() -> None:
    register_template(
        TemplateMeta(
            template_type=TEMPLATE_TYPE,
            # 不使用 chat 包装：SI 的 prompt 由 si_encode 固定生成
            prefix=[], prompt=['{{QUERY}}'], chat_sep=None, suffix=[],
            template_cls=StreamingSITemplate,
        ),
        exist_ok=True)


# ---------------------------------------------------------------------------
# 5. 数据集
# ---------------------------------------------------------------------------
_CODE_DEPS = ('integration/register_streaming_si.py', 'data/si_encode.py')   # 相对 streaming_si/


def _code_version() -> str:
    """并入 datasets map 缓存指纹：改动编码逻辑后缓存必须失效（原因见 custom/dataset.py）。

    只把文件名（不含子目录）并入指纹，目录调整本身不改变指纹；读不到文件直接报错 ——
    静默跳过会让编码逻辑改动后复用旧缓存。
    """
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha256()
    for rel in _CODE_DEPS:
        h.update(os.path.basename(rel).encode())
        with open(os.path.join(pkg, rel), 'rb') as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def resolve_p2_wav_dir(data_glob: str, wav_dir: Optional[str]) -> str:
    """一个 P2 数据集对应**唯一**的 tar 目录。

    缺省与源项目一致：`<parquet 所在目录>/../p2_source_wav`。

    为什么不支持一个数据集跨多个 tar 目录：datasets 读多文件的顺序取决于 fsspec glob
    （datasets 2.x 自身不排序），无法可靠地把每行映射回来源 parquet；错配时 speech 监督
    会接到别的样本上且不报错。多机数据（m1/m2 的 tar 同名不同内容）请分别训练，或把 tar 合并到
    同一目录后用 SI_P2_WAV_DIR 指定。
    """
    files = _parquet_files(data_glob)
    if not files:
        raise RuntimeError(f'P2 数据为空: {data_glob!r}')
    if wav_dir:
        return wav_dir
    dirs = {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(p))), 'p2_source_wav')
            for p in files}
    if len(dirs) != 1:
        raise RuntimeError(
            f'{data_glob!r} 跨 {len(dirs)} 个 p2_source_wav 目录 {sorted(dirs)[:3]}；'
            f'请按目录拆成多个数据集，或显式设置 SI_P2_WAV_DIR')
    return dirs.pop()


class StreamingSIPreprocessor(RowPreprocessor):
    """把 S1 / P2 parquet 行规整成 swift 行：补 `messages`（swift 校验必需、template 不读），
    打 `si_stage` 标签，P2 的 `source_wav` 改写为 tar 绝对路径。

    S1/P2 的原始列经 `--remove_unused_columns false` 以 `_extra_kwargs` 进入 template。
    返回 None = 丢弃该行，对应源 dataset 的「跳到下一条」（无语音监督 / 无音频引用）。

    SI_DROP_OVERLONG=1 且行内有 p2_parquet_prep.py 写入的元数据列（`P2_META_COLS`）时，
    超长样本在这里就丢掉，判定与编码时相同（`st_overlong_reason`）。必须在这里丢：按长度分桶后
    若到编码时才丢，LazyLLMDataset 会随机换一条别的长度的样本补进来，打乱分桶。
    """

    def __init__(self, stage: str, wav_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if stage not in STAGES:
            raise ValueError(f'stage 应为 {STAGES}，收到 {stage!r}')
        if stage != 's1' and not wav_dir:
            raise ValueError(f'{stage} 需要 wav_dir（P2 源音频 tar 目录）')
        self.stage = stage
        self.wav_dir = wav_dir
        # 构造时固定（map 子进程里读到的 env 与主进程一致，且进入缓存指纹）
        self.drop_overlong = stage != 's1' and _drop_overlong()
        self.keep_noise_neg = _keep_noise_neg()
        self.noise_tags = _noise_tags()
        self.limits = _p2_limits()
        self.task = os.environ.get('SI_ST_TASK', 'translate')
        self._preprocess_code_version = _code_version()

    def _overlong(self, row: Dict[str, Any]) -> bool:
        if not self.drop_overlong or any(row.get(c) is None for c in P2_META_COLS):
            return False
        lim = self.limits
        sec = int(row['src_samples']) / 16000
        if not lim['min_audio_sec'] <= sec <= lim['max_audio_sec']:
            return True
        if self.task != 'translate':      # 元数据只统计了 translation 的 token 数；transcribe 由编码时兜底
            return False
        return st_overlong_reason(
            n_frames=st_num_frames(row['src_samples']), n_instr=int(row['instr_tokens']),
            n_tgt=int(row['tgt_tokens']), max_length=lim['max_length'], max_target_len=lim['max_target_len'],
            n_speech=len(normalize_fsq(row.get('target_fsq'))) if self.stage == 's3' else None,
            max_speech_len=lim['max_speech_len']) is not None

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        out = {k: copy.deepcopy(v) for k, v in row.items()}
        out['si_stage'] = self.stage
        if self.stage == 's1':
            conv = row.get('conversations')
            conv = json.loads(conv) if isinstance(conv, str) else conv
            if not conv or not row.get('answer_speech_tokens'):
                return None
            out['messages'] = [{'role': t['role'], 'content': t['content']} for t in conv
                               if t.get('role') in ('user', 'assistant')]
        else:
            if self.stage == 's3' and not normalize_fsq(row.get('target_fsq')):
                # 噪声负样本（有效译文为空）保留进 S4：文本 CE 学「噪声+指令 -> 立即 eos」，
                # speech 监督为空。译文非空但 fsq 缺失仍是真异常，照旧丢弃。
                # SI_KEEP_NOISE_NEG=0（默认）恢复旧行为；S4-w 不应打开（无梯度，见 _keep_noise_neg）。
                tgt = row.get('source_transcript') if self.task != 'translate' else row.get('translation')
                # 三个条件缺一不可：开关打开 + 译文为空 + 来自已登记的噪声源（dataset_name / id）
                if (not self.keep_noise_neg or str(tgt or '').strip()
                        or not _is_noise_row(row, self.noise_tags)):
                    return None
            ref = row.get('source_wav') or ''
            if '::' not in ref:
                return None
            if self._overlong(row):
                return None
            tar_name, loc = ref.split('::', 1)
            out['source_wav'] = f'{os.path.join(self.wav_dir, tar_name)}::{loc}'
            out['messages'] = [
                {'role': 'user', 'content': st_instruction(row.get('tgt_lang')).strip()},
                {'role': 'assistant', 'content': str(row.get('translation') or '')},
            ]
        return out


_ENV = {'s1': 'SI_S1_DATA', 's2': 'SI_P2_DATA', 's3': 'SI_P2_DATA'}


def _parquet_files(spec: str) -> List[str]:
    """与源 `S2STDataset._expand` 同语义：逗号分隔的目录 / glob / 单文件。"""
    import glob
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if os.path.isdir(part):
            out += sorted(glob.glob(os.path.join(part, '*.parquet')))
        elif any(c in part for c in '*?['):
            out += sorted(glob.glob(part))
        elif part.endswith('.parquet'):
            out.append(part)
    return out


def _check_lengths_column(stage: str, spec: str) -> None:
    """`--group_by_length` 读数据集的 `lengths` 列（swift get_train_dataloader）。缺列要等十几分钟的
    预处理之后才 KeyError，这里在插件注册时就检查所有 parquet 的 schema。"""
    if os.environ.get('GROUP_BY_LENGTH', 'false').lower() != 'true':
        return
    import pyarrow.parquet as pq
    missing = [p for p in _parquet_files(spec) if 'lengths' not in pq.read_schema(p).names]
    if missing:
        raise RuntimeError(
            f'GROUP_BY_LENGTH=true 但 {stage} 有 {len(missing)} 个 parquet 缺 lengths 列（如 {missing[0]}）；'
            f'S2/S3 经 ktrain_entry.sh 提交时由 p2_parquet_prep.py --lengths 生成，'
            f'本地运行请先对合并目录跑该脚本，或关闭 GROUP_BY_LENGTH')


def register_streaming_si_datasets() -> None:
    """按环境变量注册（插件导入时读取，训练脚本须先 export）。

    `SI_S1_DATA` / `SI_P2_DATA` 必须是**一个以 .parquet 结尾的文件或 glob**：swift 按
    `dataset_path` 的扩展名选择 HF 读取器（`hf_load_dataset('parquet', data_files=...)`）。
    未设置时跳过注册，不影响本插件内的其他任务。
    """
    for stage in STAGES:
        spec = os.environ.get(_ENV[stage], '').strip()
        if not spec:
            continue
        if not spec.endswith('.parquet') or ',' in spec:
            raise ValueError(f'{_ENV[stage]}={spec!r} 须为单个 .parquet 文件或以 .parquet 结尾的 glob')
        wav_dir = None if stage == 's1' else resolve_p2_wav_dir(spec, os.environ.get('SI_P2_WAV_DIR'))
        if wav_dir and not os.path.isdir(wav_dir):
            raise RuntimeError(f'P2 源音频目录不存在: {wav_dir}')
        _check_lengths_column(stage, spec)
        register_dataset(
            DatasetMeta(
                dataset_name=f'streaming_si_{stage}', ms_dataset_id=f'streaming_si_{stage}',
                dataset_path=spec,
                preprocess_func=StreamingSIPreprocessor(stage, wav_dir=wav_dir),
                tags=['audio', 'speech-generation', 'streaming-si'],
            ),
            exist_ok=True)


# ---------------------------------------------------------------------------
# 6. 模型
# ---------------------------------------------------------------------------
def register_streaming_si_model() -> None:
    """每个变体一个 swift model_type（见 variants.py），共用 loader / template / 数据集 / tuner / 模块分组。"""
    for model_type, variant in VARIANTS.items():
        register_model(
            ModelMeta(
                model_type,
                [ModelGroup([Model(model_type, model_type)], TEMPLATE_TYPE)],
                StreamingSILoader,
                template=TEMPLATE_TYPE,
                is_multimodal=True,
                model_arch=MODEL_TYPE,        # 模块分组相同：thinker_model / audio_encoder / audio_proj / talker
                architectures=['StreamingSIForConditionalGeneration'],
                requires=['transformers>=4.57', 'soundfile'],
                tags=['audio', 'speech-generation', variant['thinker_arch']],
                # swift 保存 checkpoint 时从 model_dir 复制，保证续训 / 下一阶段仍有 mel 前端配置；
                # speech_decoder/（flow/HiFT，仅推理用）存在时整目录带上，使每个 checkpoint 可直接推理
                additional_saved_files=[VOXTRAL_FE_CONFIG, SPEECH_DECODER_DIR],
            ),
            exist_ok=True)


def register_all() -> None:
    register_streaming_si_model()
    register_streaming_si_template()
    register_streaming_si_datasets()
    from .register_streaming_si_s4 import register_streaming_si_s4_datasets
    from ..model.s4_objective import install_s4_metric_patch
    from .s4_peft import register_s4_tuner
    register_streaming_si_s4_datasets()
    register_s4_tuner()
    install_s4_metric_patch()
