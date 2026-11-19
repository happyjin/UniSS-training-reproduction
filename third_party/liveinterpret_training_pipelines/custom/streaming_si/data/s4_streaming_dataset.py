"""S4 数据集（chunk-as-turn 流式同传：H1 源音频按 640 ms 基础块进入 -> 每步决定等/说）。

与 S3 的关系
------------
S3 = 「整段源音频 -> 整段译文 + 整段 speech token」的离线 S2ST。
S4 = 把**同一个** thinker/talker 拆到时间轴上：源音频不再一次性给完，而是按
`trajectory.steps` 一块一块地插进 thinker 的输入序列，每块之后强制一个读写决策
token（`<|wait|>` / `<|W|>`），只有 WRITE 步才吐译文增量。

**关键：talker 侧与 S3 逐位一致，一个字节都不用改。**
  * `answer_mask`仍然只标「译文文本 token」位置 —— 各 WRITE 步的 `text_delta`
    按序拼起来就是完整译文，故 gather 出来的 cond 序列与 S3 的
    「译文 token 序列」同分布（只是分词边界按步切开，见 §分词）。
  * `speech_tokens` 仍然是**无缝**的 `target_codec_codes`（25Hz 单码本），
    **不含**任何插入静音、**不需要** SILENCE_TOKEN。

为什么 WAIT 步什么都不发（推翻了早期的 `<|pause:N|>` 方案）
--------------------------------------------------------
论文 §3.2 明确："Pink WAIT intervals indicate steps where the agent receives
additional source audio but **emits no speech**"。即 WAIT 步只是「多听一块」，
不在 speech 流上产生任何 token。于是：
  * 目标时间轴上那 17%~30% 的「插入静音」帧**不是**监督目标，它们只是
    「WRITE 步之间隔了几个 WAIT 步」这件事的另一种记法—— 而这件事已经由
    thinker 侧的 `<|wait|>` 序列完整表达了。
  * 因此 `<|pause:N|>` 是冗余的（而且分桶上限 2.4s 根本盖不住 zh→en 最长 15s 的
    停顿，8.3% 会溢出）；`SILENCE_TOKEN`（data_usage.md §5 的"唯一硬阻塞项"）
    也随之不需要了。
  * 推理时的时序由**驱动循环**保证：H1 每 640 ms 送一个基础块 -> 采一个决策 -> WRITE 就
    把该步的 speech token 交给 vocoder 播出，WAIT 就什么都不播。静音是"没播"，
    不是"播了静音 token"。

序列布局（`n_i` = 第 i 步的音频帧数，见 §帧数对齐）
--------------------------------------------------
    [instr]
    <|chunk|> <|audio_pad|>×n_0 [<|src_end|>] <|wait|>
    <|chunk|> <|audio_pad|>×n_1                <|W|>  译文增量_1
    <|chunk|> <|audio_pad|>×n_2                <|W|>  译文增量_2
    ...
    <|chunk|> <|audio_pad|>×n_K                <|W|>  译文增量_K
    <|chunk|> <|audio_pad|>×n_{K+1} ... [<|src_end|>] <|wait|>  <eos>
    │         └── 由 inject_audio_features 替换 ─┘    │      │        │
    │                │      │          └计 loss
    │                                                │      └ 计 loss + answer_mask=1
    └ labels=-100（仅第 0 步）      └ 计 loss（读写决策）

最后一行表示“译文先结束”的可选尾部 WAIT；若最后一个 WRITE 本身已经读完 source，
`<|src_end|>` 和 Thinker `<eos>` 就落在该 WRITE 所在的最后一步，不额外增加 WAIT。

两个位置**不**计 loss：`<|audio_pad|>`、`<|src_end|>` —— 它们是驱动循环喂进去的
**环境输入**，不是模型的输出，让模型去预测它们等于学噪声。

§每步的终止符（`<|chunk|>` 从第 1 步起被监督）
---------------------------------------------
`<|chunk|>` 是个**双重身份**的 token：对第 i 步它是「环境要开始喂第 i 块音频」的输入，
但对第 i-1 步它同时是「上一步已经说完/等完，把控制权交回驱动循环」的**终止符**。
所以除第 0 步（紧跟 instruction，驱动循环必然会喂，模型无需预测）之外，
`<|chunk|>` 位置一律计 loss。

不加这一条会怎样：按 shift 约定 `Y[j] = labels[j+1]` 逐位推 ——

    source 未结束的 WRITE delta 末 token -> 下一位是 <|chunk|> 的 label = -100  ✗ 无梯度
    source 最后一步的动作/译文末 token   -> 下一位是 <eos>                      ✓

于是「译文 token 之后」这个上下文里**唯一被标注过的结果就是 eos**（只来自 source 最后一步），
中间 WRITE 一个反例都不提供，模型先验偏向 eos -> 推理时第一个 WRITE 步就可能吐 eos，
翻译在第 1 秒结束。而且驱动循环连「让模型停下来交还控制权」都无法表达。

S3 只有一个 delta，eos 就是完整且正确的终止符（`s3_s2st_dataset.py:131-133`），
per-step 边界这个概念 S3 根本不需要，所以没有可继承的机制 —— 这是 S4 新增的。

**WAIT 步同样监督**（`<|wait|>` -> `<|chunk|>`），这样驱动循环对两种 action 的
终止规则完全一致：**采样直到 `<|chunk|>` 或 `<|im_end|>`**；采到 `<|chunk|>` 就把
下一块音频特征接在它后面继续 —— 那个 token 本就是下一步的开头，**零额外 token**。

代价：`<|chunk|>` 进入 loss 后会混进「译文 CE」污染与 S3 基线的可比性，因此
`train_s4_stream.py` 把它单独拆成 `boundary` 一路统计（权重与决策同为
`decision_weight`，`w=1` 时总 loss 仍等于「所有监督位一起按 token 平均」）。

§帧数对齐（最容易静默出错的地方）
--------------------------------
Voxtral: 100Hz mel -> conv stride 2 -> 50Hz -> adapter /4 -> 12.5Hz(80ms/帧)，
即 `MEL_PER_ADAPTER_FRAME = 8`。1 秒 chunk = 100 mel 帧 = **12.5** 个adapter 帧，
**不整除**。流式 encoder（`encode_audio_chunk`）用 `mel_remainder` 把余数留到下一块，
所以真实的逐步帧数是 12,13,12,13,… 而**不是**每步 12 或 13。

本数据集按「**累计** mel 帧数做整除」来复刻这个行为：

    b_i = min(T_mel, source_end_ms_i // 10)          # 第 i 步末尾的累计 mel 帧
    n_i = b_i // 8 - b_{i-1} // 8                    # 第 i 步该放几个 audio_pad

这与 `encode_audio_chunk` 逐块调用的结果**逐帧相同**；训练时因此可以直接对
**整段** mel 调一次 `encode_audio`（因果 encoder，实测整段 vs 逐块 max|diff|
2.2e-06），把产出的 `floor(T_mel/8)` 帧按 `n_i` 顺序填进各个 audio_pad 段。
若改成「每步都按 1000ms//80=12 帧」，总帧数就会比 encoder 实际产出少~4%，
`inject_audio_features` 会静默丢掉尾部音频特征 —— 不报错、只是模型听不全。

**模型侧配合**：`inject_audio_features` 已支持多段占位符（按出现顺序跨段连续消费
同一份特征）。本数据集给出的 `audio_seg_lens` 应一路传到 `forward(audio_seg_lens=...)`，
它会硬校验「段数 / 逐段长度 / 总长」——帧数错位是这条链上唯一的失效模式，
而它天然是静默的（形状照样对得上，只是模型听到的内容错位或缺尾）。

§translation_eos 与 Thinker EOS
-------------------------------
`translation_eos` 只表示最后一个 WRITE 已经提交完目标译文，不是 Thinker 的序列终止符。
若 source 仍未读完，后续 WAIT 必须保留，直到包含 `source_stream_done` / `<|src_end|>` 的
最后一个 source step；随后才追加 Thinker EOS（token 为 `<|im_end|>`）。这些尾部 WAIT
不产生 speech token，也不添加 trailing codec silence。

§分词
-----
监督目标是**逐步的 `text_delta`**，不是 `target_text`。`text_delta` 是空格分隔的
词序列，还原规则按目标语分两种：
  * 中文/日文等无空格语：词直接相连；仅当两侧都是 ASCII 字母数字时补一个空格
    （否则 "progressive house" 会粘成 "progressivehouse"）。
  * 英文等空格语：词与词、步与步之间都补一个空格 —— 于是第 k(>0) 步的首词自带
    前导空格，BPE 切分与自然文本一致。
逐步拼接与 `target_text` 有 0.4%(en→zh) / 2.4%(zh→en) 的**纯空白差异**（引号、
中英混排处），不影响训练；`--strict_text` 可把这些样本剔掉。
"""
import hashlib
import io
import json
import os
import random
from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .s4_context_views import (
    S4ViewIndex,
    _eligible_indices,
    context_view_replacement_seed,
)
from .si_traj_ops import merge_steps, steps_until_thinker_eos
from ..model.base_model import MEL_FRAMES_PER_ADAPTER_FRAME, MEL_HOP_LENGTH, SR

# 一个 adapter 帧 = 8 个 mel 帧 = 80ms（与 model.base_model 同源，不重复定义常量）
MEL_PER_ADAPTER_FRAME = MEL_FRAMES_PER_ADAPTER_FRAME
# mel 帧率 100Hz -> 1 帧 = 10ms，故「毫秒 // MS_PER_MEL_FRAME」= mel 帧下标
MS_PER_MEL_FRAME = 1000 // (SR // MEL_HOP_LENGTH)

LANG_NAME = {'cmn': 'Chinese', 'zho': 'Chinese', 'zh': 'Chinese',
             'eng': 'English', 'en': 'English', 'yue': 'Cantonese',
             'jpn': 'Japanese', 'ja': 'Japanese', 'kor': 'Korean', 'ko': 'Korean'}
# 书写系统不用空格分词的语言：还原 text_delta 时词间不补空格
NO_SPACE_LANGS = {'zh', 'cmn', 'zho', 'yue', 'ja', 'jpn', 'th', 'tha'}


class S4ViewBuildError(RuntimeError):
    """A direct context-view candidate could not satisfy its descriptor."""

    def __init__(self, reason, descriptor, replacement_reasons=None):
        self.reason = str(reason)
        self.descriptor = descriptor
        self.replacement_reasons = list(replacement_reasons or [])
        super().__init__(
            f'S4 view candidate {descriptor.index} rejected: {self.reason} '
            f'(designated={descriptor.designated_kind}, applied={descriptor.applied_kind})')


@dataclass(frozen=True)
class _PreparedS4Record:
    flat_index: int
    record_id: str
    steps: list
    mel: torch.Tensor
    audio_seg_lens: list[int]
    speech_tokens: list[int]
    speech_len: int
    text_ids: list[list[int]]
    speech_seg_lens: list[int]
    instr_ids: list[int]


def _is_word_char(ch):
    return ch.isascii() and ch.isalnum()


def source_key(rec):
    """样本的**源文本**指纹，用作 train/val 划分的分组键。

    为什么不按 `record_id` 或按 part 文件切 val：同一句源文本在库里会出现多次
    （en→zh 实测重复率 21.52%），且重复项散落在不同 part 里。按 id/文件切，val 里的
    句子有很大概率也出现在 train 里 —— BLEU 会**虚高**且不报错。按源文本分组后
    整组一起划走，才是真正的 held-out。

    归一化只做「小写 + 折叠空白」：标点差异是真实的内容差异，不该合并。
    """
    text = (rec.get('source_transcript') or '').lower()
    return hashlib.md5(' '.join(text.split()).encode('utf-8')).hexdigest()[:16]


class JsonlIndex:
    """records/part-*.jsonl 的字节偏移索引（惰性按行 seek 读）。

    为什么不整表读进内存：一条记录 ~4.5KB（`target_codec_codes` 占大头），
    单个 part 50,000 行 = 223MB，两个方向共 5.8GB。DataLoader 每个 worker 都
    fork 一份就会 OOM。索引本身只有 `行数 × 8B`（50k 行 = 400KB），可忽略。

    偏移表缓存到 `<part>.offsets.npy`；缓存的 mtime/size 与 jsonl 不符时自动重建
    （否则改过数据还用旧索引 = 读到错位的字节，json 解析失败或读到别的样本）。
    """

    def __init__(self, path, cache=True):
        self.path = os.path.abspath(path)
        self.offsets = self._load_or_build(cache)
        self._fp = None

    def _cache_path(self):
        return self.path + '.offsets.npy'

    def _load_or_build(self, cache):
        import numpy as np
        st = os.stat(self.path)
        cp = self._cache_path()
        if cache and os.path.exists(cp):
            try:
                arr = np.load(cp)
                # 末位存(size, mtime) 指纹，不匹配就重建
                if arr.size >= 2 and int(arr[-2]) == st.st_size:
                    return arr[:-2]
            except Exception:
                pass
        offsets, pos = [], 0
        with open(self.path, 'rb') as f:
            for line in f:
                if line.strip():
                    offsets.append(pos)
                pos += len(line)
        arr = np.asarray(offsets, dtype=np.int64)
        if cache:
            try:
                np.save(cp, np.concatenate([arr, np.asarray(
                    [st.st_size, int(st.st_mtime)], dtype=np.int64)]))
            except Exception:
                pass          # 只读目录：不缓存也能跑，只是每次启动多扫一遍
        return arr

    def __len__(self):
        return int(self.offsets.shape[0])

    def read(self, i):
        # fp惰性打开：__init__ 在 fork 之前执行，句柄跨 fork 共享会让多个 worker
        # 抢同一个文件位点（seek/read 非原子）-> 读到撕裂的行。故每进程各自打开。
        if self._fp is None:
            self._fp = open(self.path, 'rb')
        self._fp.seek(int(self.offsets[i]))
        return json.loads(self._fp.readline())


class S4StreamingDataset(Dataset):
    """si_s4 / rw_data records/part-*.jsonl -> chunk-as-turn 训练样本（dict）。

    Args:
        data_path: records的 jsonl 路径 / 目录 / glob /逗号分隔的多项。
            目录会展开成其中的 `part-*.jsonl`（跳过 `_manifest.json`）。
        data_root: 记录里音频相对路径（`path_schema: relative-v1`）的根目录。
            None 时按 records 目录上推两级猜（`<root>/<dir>/records/part-*.jsonl`）。
        tokenizer: 必须已 `build_qwen_tokenizer(..., add_stream_tokens=True)`，
            否则 `<|wait|>`/`<|W|>`/`<|chunk|>` 不是单token（会静默切成多段）。
        audio_processor: `wav(1D) -> (128, T)` 的 mel 前端（VoxtralAudioProcessor）。
        max_length: thinker 序列上限；超限的样本**跳过**而不截断 —— 截断会把译文
            或步序列切一半，产生自相矛盾的监督。
        max_speech_len: speech token 上限（25Hz，750≈ 30s）；超限跳过，同理。
        use_src_end: 是否在源流结束的那一步插 `<|src_end|>`。data_usage.md §4建议
            显式喂"源结束"，否则模型只能从"最后一块特别短"这个假线索去猜，而推理
            时按固定 640 ms 基础块切根本看不到这个线索。
        strict_text: 逐步 `text_delta` 拼接与 `target_text` 不等（含纯空白差异）时跳过。
        exclude_keys: 要排除的 `source_key(rec)` 集合（训练集用它剔掉 val 的源文本组）。
            按源文本而非 record_id 排除，见 `source_key` 的说明。
    """

    def __init__(self, data_path, tokenizer, data_root=None, audio_processor=None,
                 max_length=640, max_speech_len=750, max_audio_sec=20.0, min_audio_sec=0.5,
                 rank=0, world_size=1, use_src_end=True, strict_text=False,
                 instruction_template=None, index_cache=True, align_ranks=False,
                 max_scan=64, exclude_keys=None,
                 latency_multiplier=1, m_max=4, m_schedule='uniform', seed=0,
                 prefix_context_turns=16):
        files = self._expand(data_path)
        if not files:
            raise RuntimeError(f'没有找到 records jsonl: {data_path}')
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.max_length = int(max_length)
        self.max_speech_len = int(max_speech_len)
        self.max_audio_sec = float(max_audio_sec)
        self.min_audio_sec = float(min_audio_sec)
        self.use_src_end = bool(use_src_end)
        self.strict_text = bool(strict_text)
        self.instruction_template = instruction_template or \
            '\nTranslate the streaming speech into {tgt} simultaneously:\n'
        self.max_scan = int(max_scan)
        self.exclude_keys = set(exclude_keys) if exclude_keys else None
        self.skip_stats = Counter()
        self.prefix_context_turns = int(prefix_context_turns)
        if self.prefix_context_turns < 1:
            raise ValueError('prefix_context_turns 必须 >= 1')

        # ---- 延迟倍数 m（P1-1）----
        # latency_multiplier: 'sample' 表示每次取样随机抽；int 表示固定值（评测用）。
        # m_schedule: 'uniform' = {1..m_max} 均匀；'linear' = 论文的截断线性
        #   w_m = max(a - b(m-1), 0)，a=0.18/b=0.017，强烈偏向小 m。
        # 论文 §4 实测 schedule 之间「negligible differences」，所以默认 uniform；
        # 但 m_max 压到 4 —— 我们句长下 m>=6 的首写延迟已饱和于整句长度，纯属重复
        # S3 的离线监督。
        self.latency_multiplier = latency_multiplier
        self.m_max = max(1, int(m_max))
        self.m_schedule = m_schedule
        self._m_rng = random.Random(seed + 1315423911 * rank)
        self._m_probs = self._build_m_probs()
        self.m_stats = Counter()

        # ---- rank 分片：文件数够就按文件切（IO 局部性好），否则按行号跨步切 ----
        self.file_sharded = world_size > 1 and len(files) >= world_size
        if self.file_sharded:
            files = files[rank::world_size]
        self._sharded_files = list(files)

        self.roots, self.indexes = [], []
        for f in files:
            self.indexes.append(JsonlIndex(f, cache=index_cache))
            self.roots.append(self._infer_root(f) if data_root is None
                              else os.path.abspath(data_root))
        missing = sorted({r for r in self.roots if not os.path.isdir(r)})
        if missing:
            raise RuntimeError(f'音频根目录不存在: {missing[:3]}（data_root={data_root}）')

        # (file_idx, line_idx) 扁平索引：跨 part 均匀混合由 sampler 的 shuffle 负责
        self.items = [(fi, li) for fi, ix in enumerate(self.indexes)
                      for li in range(len(ix))]
        if world_size > 1 and not self.file_sharded:
            self.items = self.items[rank::world_size]
        if align_ranks:
            self._align_ranks()

        # ---- 特殊 token（全部必须是单 token，否则布局会静默错位）----
        self.audio_token = getattr(tokenizer, 'audio_special_token', '<|audio_pad|>')
        self.audio_token_id = self._single_id(self.audio_token)
        from ..model.base_model import (STREAM_CHUNK_TOKEN, STREAM_SRC_END_TOKEN,
                STREAM_WAIT_TOKEN, STREAM_WRITE_TOKEN)
        self.chunk_token_id = self._single_id(STREAM_CHUNK_TOKEN)
        self.wait_token_id = self._single_id(STREAM_WAIT_TOKEN)
        self.write_token_id = self._single_id(STREAM_WRITE_TOKEN)
        self.src_end_token_id = self._single_id(STREAM_SRC_END_TOKEN) if use_src_end else None
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            raise RuntimeError('tokenizer 没有 eos_token_id，S4 无法标记译文结束')
        self._instr_cache = {}
        self._fps = {}
        self._context_view_pools = None
        self.context_view_pool_digests = None

    # ------------------------------------------------------------------ 构造辅助
    def _build_m_probs(self):
        """m 的采样分布。`latency_multiplier != 'sample'` 时返回 None（固定 m）。"""
        if self.latency_multiplier != 'sample':
            return None
        if self.m_schedule == 'linear':
            # 论文附录：w_m = max(a - b(m-1), 0)，a=0.18, b=0.017
            w = [max(0.18 - 0.017 * (m - 1), 0.0) for m in range(1, self.m_max + 1)]
        else:
            w = [1.0] * self.m_max
        s = sum(w) or 1.0
        return [x / s for x in w]

    def _sample_m(self):
        """取一个 m。固定值时直接返回，避免每条样本白算一次随机数。"""
        if self._m_probs is None:
            m = max(1, int(self.latency_multiplier))
        else:
            m = self._m_rng.choices(range(1, self.m_max + 1),
                                    weights=self._m_probs, k=1)[0]
        self.m_stats[m] += 1
        return m

    def _single_id(self, token):
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f'{token} 不是单 token（得到 {ids}）。S4 依赖它做序列布局；'
                f'请用 build_qwen_tokenizer(..., add_stream_tokens=True) 构造 tokenizer')
        return ids[0]

    @staticmethod
    def _expand(data_path):
        import glob
        out = []
        for part in str(data_path).split(','):
            part = part.strip()
            if not part:
                continue
            if os.path.isdir(part):
                out += sorted(glob.glob(os.path.join(part, 'part-*.jsonl')))
            elif any(c in part for c in '*?['):
                out += sorted(p for p in glob.glob(part) if p.endswith('.jsonl'))
            elif part.endswith('.jsonl'):
                out.append(part)
        return out

    @staticmethod
    def _infer_root(jsonl_path):
        """`<root>/en2zh/records/part-000xx.jsonl` -> `<root>`。

        记录里的 `source_audio` 形如 `en2zh/shard000/run/<id>/src.wav`，
        含方向前缀，所以根要上推两级（records -> en2zh -> root）。
        """
        return os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(jsonl_path))))

    def _align_ranks(self):
        """把各 rank 的样本数补齐到全局 max（**防 DDP 静默死锁**）。

        按文件分片时各 part 行数不等（最后一片只有 27,807 / 49,982 行），步数不等
        会让最短的 rank 先跑完并**以 exit code 0 退出**，torchrun 不报错，其余 rank
        永久阻塞在 allreduce —— 表现为"无报错、日志停增、1 卡满载其余 0"。
        """
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return
        dev = (torch.device(torch.cuda.current_device()) if dist.get_backend() == 'nccl'
               else torch.device('cpu'))
        n = torch.tensor([len(self.items)], dtype=torch.long, device=dev)
        dist.all_reduce(n, op=dist.ReduceOp.MAX)
        target, own = int(n.item()), len(self.items)
        while len(self.items) < target:      # 差值可能 > 自身长度，故循环补
            self.items += self.items[:target - len(self.items)]
        if dist.get_rank() == 0 and target > own:
            print(f'[S4 data] rank 长度对齐: {own} -> {len(self.items)} 条（全局 max）',
                  flush=True)

    def __len__(self):
        return len(self.items)

    def configure_context_view_pools(self, weights, thinker_eligible):
        """Freeze rank-local weighted full/Thinker pools for lazy joint admission."""
        try:
            weight_tensor = torch.as_tensor(
                weights, dtype=torch.double, device='cpu').clone().contiguous()
        except (TypeError, ValueError) as exc:
            raise ValueError('context-view weights must be a numeric 1-D vector') from exc
        if weight_tensor.ndim != 1 or weight_tensor.numel() != len(self):
            raise ValueError(
                f'context-view weights must have length {len(self)}')
        if not bool(torch.isfinite(weight_tensor).all()) or bool((weight_tensor < 0).any()):
            raise ValueError('context-view weights must be finite and nonnegative')
        full_indices = torch.nonzero(weight_tensor > 0).flatten().to(torch.long)
        thinker_indices = torch.as_tensor(
            _eligible_indices(thinker_eligible, len(self), 'thinker_eligible'),
            dtype=torch.long)
        if full_indices.numel() == 0:
            raise ValueError('full context-view pool must have positive weight')
        if (thinker_indices.numel() == 0
                or float(weight_tensor[thinker_indices].sum()) <= 0):
            raise ValueError('Thinker context-view pool must have positive weight')

        def freeze_pool(indices):
            pool_weights = weight_tensor[indices].clone().contiguous()
            return indices.clone().contiguous(), pool_weights

        self._context_view_pools = {
            'full': freeze_pool(full_indices),
            'thinker_crop': freeze_pool(thinker_indices),
        }
        digests = {}
        for kind, (indices, pool_weights) in self._context_view_pools.items():
            digest = hashlib.sha256()
            digest.update(kind.encode('ascii') + b'\0')
            digest.update(indices.numpy().astype('<i8', copy=False).tobytes())
            digest.update(pool_weights.numpy().astype('<f8', copy=False).tobytes())
            digests[kind] = digest.hexdigest()
        self.context_view_pool_digests = digests

    # ------------------------------------------------------------------ 读取
    @staticmethod
    def _resolve_audio_path(root, rel):
        """相对路径拼 data_root；绝对路径若已失效，改走 si_s4/audio 下同相对尾。

        统一布局后删掉了 rw_data*，但一部分 jsonl 仍写着
        `/opt/.../data/rw_data/{en2zh,zh2en}/shardNNN/...`。文件已 rename 到
        `$SI_S4/audio/` 下同一相对路径。
        """
        if not os.path.isabs(rel):
            return os.path.join(root, rel)
        if os.path.exists(rel):
            return rel
        for marker in ('/en2zh/', '/zh2en/'):
            i = rel.find(marker)
            if i >= 0:
                return os.path.join(root, rel[i + 1:])
        return rel

    def _read_wav(self, root, rel):
        import soundfile as sf
        path = self._resolve_audio_path(root, rel)
        if not os.path.exists(path):
            return None
        try:
            wav, sr = sf.read(path, dtype='float32')
        except Exception:
            return None
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SR:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
        return torch.from_numpy(wav)

    def _instruction(self, tgt_lang):
        key = (tgt_lang or '').lower()
        if key not in self._instr_cache:
            tgt = LANG_NAME.get(key, tgt_lang or 'English')
            self._instr_cache[key] = self.tokenizer.encode(
                self.instruction_template.format(tgt=tgt), add_special_tokens=False)
        return self._instr_cache[key]

    def _step_texts(self, write_steps, tgt_lang):
        """各 WRITE 步的 `text_delta` -> 可直接送tokenizer 的字符串（见类docstring §分词）。"""
        space_lang = (tgt_lang or '').lower() not in NO_SPACE_LANGS
        pieces, prev = [], None
        for step in write_steps:
            buf = []
            for word in step['text_delta'].split():
                if prev is not None and (space_lang or
                                         (_is_word_char(prev[-1]) and _is_word_char(word[0]))):
                    buf.append(' ')
                buf.append(word)
                prev = word
            pieces.append(''.join(buf))
        return pieces

    # ------------------------------------------------------------------ 主逻辑
    def __getitem__(self, index):
        if isinstance(index, S4ViewIndex):
            return self._build_direct_view(index)
        # 跳过用**有界扫描**而不是 `return self[(index+1) % len(self)]` 递归：
        # 递归在"连续一片坏样本"时会爆 RecursionError（S2/S3 里的已知隐患）。
        for attempt in range(self.max_scan):
            item = self._build((index + attempt) % len(self))
            if item is not None:
                return item
        raise RuntimeError(
            f'连续 {self.max_scan} 条样本都不可用（起点 {index}）；'
            f'跳过原因统计: {dict(self.skip_stats)}')

    def _skip(self, reason):
        self.skip_stats[reason] += 1
        return None

    def _direct_error(self, reason, descriptor, replacement_reasons=None):
        raise S4ViewBuildError(reason, descriptor, replacement_reasons)

    def _direct_reason(self, before):
        delta = self.skip_stats - before
        return next(iter(delta), 'unavailable')

    def _thinker_cut(self, n_steps, draw_seed, descriptor):
        context = self.prefix_context_turns
        if n_steps < context + 2:
            self._direct_error('insufficient_turns', descriptor)
        live_turn = context + 1 + int(draw_seed) % (n_steps - context - 1)
        return live_turn, live_turn - context

    def _validate_direct_descriptor(self, descriptor):
        if not 0 <= descriptor.index < len(self):
            self._direct_error('index_out_of_range', descriptor)
        if descriptor.designated_kind not in ('full', 'thinker_crop'):
            self._direct_error('unsupported_designated_view', descriptor)
        if descriptor.applied_kind not in ('full', 'thinker_crop'):
            self._direct_error('unsupported_applied_view', descriptor)
        if (descriptor.designated_kind == 'full'
                and descriptor.applied_kind != 'full'):
            self._direct_error('descriptor_kind_mismatch', descriptor)
        if (descriptor.designated_kind == 'thinker_crop'
                and self.latency_multiplier != 1):
            self._direct_error('latency_multiplier_not_one', descriptor)

    def _build_direct_candidate(self, descriptor):
        self._validate_direct_descriptor(descriptor)

        before = self.skip_stats.copy()
        prepared = self._prepare_record(descriptor.index, forced_m=1)
        if prepared is None:
            self._direct_error(self._direct_reason(before), descriptor)

        live_turn = retained_start = -1
        crop = None
        full = self._build_view(prepared, kind='full', include_view_fields=True)
        if full is None:
            self._direct_error(self._direct_reason(before), descriptor)
        if descriptor.designated_kind == 'thinker_crop':
            live_turn, retained_start = self._thinker_cut(
                len(prepared.steps), descriptor.draw_seed, descriptor)
            crop = self._build_view(
                prepared, kind='thinker_crop', live_turn=live_turn,
                include_view_fields=True)
            if crop is None:
                self._direct_error(self._direct_reason(before), descriptor)

        item = crop if descriptor.applied_kind == 'thinker_crop' else full
        item.update({
            'record_index': descriptor.index,
            'designated_view_kind': descriptor.designated_kind,
            'view_kind': descriptor.applied_kind,
            'draw_seed': descriptor.draw_seed,
            'retained_start_turn': retained_start,
            'live_turn': live_turn,
        })
        return item

    def _build_direct_view(self, descriptor):
        self._validate_direct_descriptor(descriptor)
        pools = getattr(self, '_context_view_pools', None)
        if pools is None:
            item = self._build_direct_candidate(descriptor)
            item.update({
                'scheduled_record_index': descriptor.index,
                'replacement_count': 0,
                'replacement_reasons': [],
            })
            return item

        pool_kind = descriptor.designated_kind
        pool_indices, pool_weights = pools[pool_kind]
        generator = torch.Generator(device='cpu')
        generator.manual_seed(context_view_replacement_seed(
            descriptor.draw_seed, descriptor.designated_kind))
        replacements = torch.multinomial(
            pool_weights, max(self.max_scan - 1, 0), replacement=True,
            generator=generator)
        candidates = [descriptor.index] + [
            int(pool_indices[position]) for position in replacements]
        reasons = []
        for attempt, candidate_index in enumerate(candidates):
            candidate = S4ViewIndex(
                candidate_index, descriptor.designated_kind,
                descriptor.applied_kind, descriptor.draw_seed)
            try:
                item = self._build_direct_candidate(candidate)
            except S4ViewBuildError as error:
                reasons.append(error.reason)
                continue
            item.update({
                'record_index': candidate_index,
                'scheduled_record_index': descriptor.index,
                'designated_view_kind': descriptor.designated_kind,
                'view_kind': descriptor.applied_kind,
                'draw_seed': descriptor.draw_seed,
                'replacement_count': attempt,
                'replacement_reasons': reasons,
            })
            return item
        self._direct_error('admission_exhausted', descriptor, reasons)

    def _build(self, flat_index):
        prepared = self._prepare_record(flat_index)
        if prepared is None:
            return None
        return self._build_view(prepared, kind='full')

    def _prepare_record(self, flat_index, forced_m=None):
        fi, li = self.items[flat_index]
        rec = self.indexes[fi].read(li)
        if self.exclude_keys is not None and source_key(rec) in self.exclude_keys:
            return self._skip('val_holdout')
        traj = rec['trajectory']
        steps = traj['steps']

        # ---- 1. translation_eos 只结束翻译；Thinker 继续读到 source_stream_done ----
        steps = steps_until_thinker_eos(steps)
        if steps is None:
            return self._skip('invalid_termination')

        # ---- 1b. 延迟倍数 m：把相邻基础步合并成 m 个 640 ms 块粒度（P1-1）----
        # 论文附录 §Latency multiplier sampling。合并只做相邻归并，保持生产期的全部
        # 规则（文本顺序 / span 平铺 / 因果性 / ready 单调），实测 R1~R7 全过。
        # 注意 m 变大会**稀释** WAIT 监督（m=2 时零 WAIT 样本从 ~30% 涨到 ~80%，
        # m>=8 只剩 1 步 = S3 离线翻译），所以默认 M_max=4 且大部分概率留给 m=1。
        m = self._sample_m() if forced_m is None else int(forced_m)
        if m > 1:
            steps = merge_steps(steps, m)

        # ---- 2. 音频与 mel ----
        dur_ms = int(rec['source_duration_ms'])
        if not (self.min_audio_sec * 1000 <= dur_ms <= self.max_audio_sec * 1000):
            return self._skip('duration_out_of_range')
        wav = self._read_wav(self.roots[fi], rec['source_audio'])
        if wav is None:
            return self._skip('wav_missing')
        mel = self.audio_processor(wav, sr=SR)
        t_mel = mel.size(-1)

        # 逐步 audio_pad 数：累计 mel 帧整除，复刻 encode_audio_chunk 的余数行为
        audio_seg_lens, prev_b, b = [], 0, 0
        for s in steps:
            b = min(t_mel, int(s['source_end_ms']) // MS_PER_MEL_FRAME)
            audio_seg_lens.append(b // MEL_PER_ADAPTER_FRAME - prev_b // MEL_PER_ADAPTER_FRAME)
            prev_b = b
        n_frames = b // MEL_PER_ADAPTER_FRAME
        if n_frames < 1:
            return self._skip('no_audio_frame')
        # mel 裁到 Thinker EOS 前最后一个 source step 的边界。
        mel = mel[..., :b]
        assert sum(audio_seg_lens) == n_frames, (sum(audio_seg_lens), n_frames)

        # ---- 3. speech 监督：无缝 target_codec_codes，按 WRITE 步切段 ----
        codes = rec['target_codec_codes']
        raw_frames = int(traj['raw_target_num_frames'])
        if len(codes) != raw_frames:
            return self._skip('codec_len_mismatch')
        if raw_frames > self.max_speech_len:
            return self._skip('speech_too_long')
        write_steps = [s for s in steps if s['action'] == 'write']
        speech_seg_lens, cursor = [], 0
        for s in write_steps:
            n = 0
            for lo, hi in s['source_codec_spans']:
                if lo != cursor:            # span 必须严格顺序平铺 [0, raw_frames)
                    return self._skip('span_not_contiguous')
                cursor, n = hi, n + (hi - lo)
            speech_seg_lens.append(n)
        # 最后一个 WRITE 必须已经铺满整条语音；后续 WAIT 不再承载 speech span。
        if cursor != raw_frames:
            return self._skip('span_not_complete')

        # ---- 4. 文本增量 ----
        pieces = self._step_texts(write_steps, rec.get('target_lang'))
        if self.strict_text:
            joined = ''.join(pieces)
            if joined != rec.get('target_text'):
                return self._skip('text_mismatch')
        text_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in pieces]
        if any(len(t) == 0 for t in text_ids):
            return self._skip('empty_text_delta')

        return _PreparedS4Record(
            flat_index=flat_index,
            record_id=rec['record_id'],
            steps=steps,
            mel=mel.to(torch.float32),
            audio_seg_lens=audio_seg_lens,
            speech_tokens=[int(c) for c in codes],
            speech_len=raw_frames,
            text_ids=text_ids,
            speech_seg_lens=speech_seg_lens,
            instr_ids=list(self._instruction(rec.get('target_lang'))),
        )

    def _build_view(self, prepared, kind, live_turn=None,
                    include_view_fields=False):
        if kind == 'full':
            retained_start = 0
            kept_steps = prepared.steps
            kept_audio_seg_lens = prepared.audio_seg_lens
            write_start = 0
            speech_token_start = 0
            speech_loss_weight = 1.0
            audio_feature_offset = 0
        elif kind == 'thinker_crop':
            context = self.prefix_context_turns
            if live_turn is None or not context + 1 <= live_turn < len(prepared.steps):
                return self._skip('invalid_crop_cut')
            retained_start = live_turn - context
            kept_steps = prepared.steps[retained_start:]
            kept_audio_seg_lens = prepared.audio_seg_lens[retained_start:]
            write_start = sum(
                s['action'] == 'write' for s in prepared.steps[:retained_start])
            speech_token_start = sum(prepared.speech_seg_lens[:write_start])
            speech_loss_weight = 0.0
            audio_feature_offset = sum(
                prepared.audio_seg_lens[:retained_start])
        else:
            return self._skip('unsupported_view_kind')

        kept_text_ids = prepared.text_ids[write_start:]
        kept_speech_seg_lens = prepared.speech_seg_lens[write_start:]
        kept_speech_tokens = prepared.speech_tokens[speech_token_start:]
        if (sum(kept_speech_seg_lens) != len(kept_speech_tokens)
                or len(kept_text_ids) != len(kept_speech_seg_lens)):
            return self._skip('write_suffix_mismatch')
        if (audio_feature_offset + sum(kept_audio_seg_lens)
                != sum(prepared.audio_seg_lens)):
            return self._skip('audio_suffix_mismatch')

        # ---- 5. 拼未 shift 序列；crop mask 完成后只 shift 一次 ----
        ids = list(prepared.instr_ids)
        labels = [-100] * len(prepared.instr_ids)
        answer = [0] * len(prepared.instr_ids)
        src_end_done = False
        w = 0
        live_chunk_position = None
        for i, s in enumerate(kept_steps):
            original_turn = retained_start + i
            if original_turn == live_turn:
                live_chunk_position = len(ids)
            ids.append(self.chunk_token_id)            # 步边界（0帧的步也发，保持一致）
            # i>0 的 chunk 同时是「上一步说完/等完」的终止符 -> 必须监督（见 §每步的终止符）。
            # i==0 紧跟 instruction，驱动循环必然会喂，模型无需预测。
            labels.append(-100 if i == 0 else self.chunk_token_id)
            answer.append(0)
            n = kept_audio_seg_lens[i]
            if n:
                ids += [self.audio_token_id] * n
                labels += [-100] * n
                answer += [0] * n
            if self.use_src_end and not src_end_done and s.get('source_stream_done'):
                ids.append(self.src_end_token_id)
                labels.append(-100)
                answer.append(0)
                src_end_done = True
            is_write = s['action'] == 'write'
            dec = self.write_token_id if is_write else self.wait_token_id
            ids.append(dec)
            labels.append(dec)                         # 读写决策：唯一要学的"动作"
            answer.append(0)
            if is_write:
                tid = kept_text_ids[w]
                w += 1
                ids += tid
                labels += tid
                answer += [1] * len(tid)               # answer_mask 只标译文文本
        # Thinker EOS：只有完整 source 已读完并监督最后一个动作后才结束序列。
        ids.append(self.eos_token_id)
        labels.append(self.eos_token_id)
        answer.append(0)
        if kind == 'thinker_crop':
            if live_chunk_position is None:
                return self._skip('invalid_crop_cut')
            labels[len(prepared.instr_ids):live_chunk_position + 1] = \
                [-100] * (live_chunk_position + 1 - len(prepared.instr_ids))
        if len(ids) > self.max_length:
            return self._skip('too_long' if kind == 'full' else 'crop_too_long')

        # shift 约定与 S1/S2/S3 **逐位一致**：X=ids[:-1] / Y=labels[1:] /
        # answer_mask=answer[:-1]（与 X 同下标，见 s3_s2st_dataset 的说明）
        item = {
            'record_id': prepared.record_id,
            'input_ids': torch.tensor(ids[:-1], dtype=torch.long),
            'labels': torch.tensor(labels[1:], dtype=torch.long),
            'answer_mask': torch.tensor(answer[:-1], dtype=torch.long),
            'mel': prepared.mel,
            'mel_len': prepared.mel.size(-1),
            'audio_seg_lens': torch.tensor(kept_audio_seg_lens, dtype=torch.long),
            'speech_tokens': torch.tensor(kept_speech_tokens, dtype=torch.long),
            'speech_len': len(kept_speech_tokens),
            'text_seg_lens': torch.tensor(
                [len(t) for t in kept_text_ids], dtype=torch.long),
            'speech_seg_lens': torch.tensor(kept_speech_seg_lens, dtype=torch.long),
        }
        if include_view_fields:
            item.update({
                'audio_feature_offset': audio_feature_offset,
                'speech_loss_weight': speech_loss_weight,
            })
        return item


def s4_collate_fn(batch):
    """右pad 文本轴 / mel 时间轴 / speech token 轴，返回 **dict**。

    为什么用 dict 而不是 S2/S3 的 tuple：S4 多出 3 个分段长度轴
    （`audio_seg_lens` / `text_seg_lens` / `speech_seg_lens`），tuple 到 9 元就极易
    在解包处错位，而错位全部是**静默**的（形状恰好都能对上）。key 名已对齐
    `MiniMindOmniCosy.forward` 的形参，可直接 `model(**batch_without_meta)`。

    分段长度**不 pad**：各样本步数差异大（1~21），pad 成矩阵还要再传 mask，
    而消费方（多段音频注入、数据驱动交错）本来就是逐样本 python 循环。
    """
    def pad1d(seqs, value):
        L = max(s.numel() for s in seqs)
        out = seqs[0].new_full((len(seqs), L), value)
        for i, s in enumerate(seqs):
            out[i, :s.numel()] = s
        return out

    mels = [b['mel'] for b in batch]
    T = max(m.size(-1) for m in mels)
    mel_batch = mels[0].new_zeros(len(mels), mels[0].size(0), T)
    for i, m in enumerate(mels):
        mel_batch[i, :, :m.size(-1)] = m

    result = {
        'input_ids': pad1d([b['input_ids'] for b in batch], 0),
        'labels': pad1d([b['labels'] for b in batch], -100),
        'answer_mask': pad1d([b['answer_mask'] for b in batch], 0),
        'audio_inputs': mel_batch,
        'audio_lens': torch.tensor([b['mel_len'] for b in batch], dtype=torch.long),
        'speech_tokens': pad1d([b['speech_tokens'] for b in batch], 0),
        'speech_token_lens': torch.tensor([b['speech_len'] for b in batch], dtype=torch.long),
        'audio_seg_lens': [b['audio_seg_lens'] for b in batch],
        'text_seg_lens': [b['text_seg_lens'] for b in batch],
        'speech_seg_lens': [b['speech_seg_lens'] for b in batch],
        'record_ids': [b['record_id'] for b in batch],
    }
    is_view = ['view_kind' in b for b in batch]
    if any(is_view):
        if not all(is_view):
            raise ValueError('不能在同一个 batch 混用 plain-int 与 S4ViewIndex 样本')
        result.update({
            'audio_feature_offsets': torch.tensor(
                [b['audio_feature_offset'] for b in batch], dtype=torch.long),
            'speech_loss_weights': torch.tensor(
                [b['speech_loss_weight'] for b in batch], dtype=torch.float32),
            'view_kinds': [b['view_kind'] for b in batch],
            'designated_view_kinds': [b['designated_view_kind'] for b in batch],
            'record_indices': torch.tensor(
                [b['record_index'] for b in batch], dtype=torch.long),
            'scheduled_record_indices': torch.tensor(
                [b['scheduled_record_index'] for b in batch], dtype=torch.long),
            'draw_seeds': torch.tensor(
                [b['draw_seed'] for b in batch], dtype=torch.long),
            'retained_start_turns': torch.tensor(
                [b['retained_start_turn'] for b in batch], dtype=torch.long),
            'live_turns': torch.tensor(
                [b['live_turn'] for b in batch], dtype=torch.long),
            'replacement_counts': torch.tensor(
                [b['replacement_count'] for b in batch], dtype=torch.long),
            'replacement_reasons': [b['replacement_reasons'] for b in batch],
        })
    return result
