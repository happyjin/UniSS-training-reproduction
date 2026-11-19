"""SI trajectory 的纯数据操作：延迟倍数合并、NIR、延迟指标。

抽成独立模块是因为同一套逻辑有三个调用方，各写一份必然漂移：
  * `dataset/s4_streaming_dataset.py` —— 训练时按 m 做 on-the-fly 合并（P1-1）；
  * `scripts/build_record_index.py`   —— 全量算 NIR / 句长 / 延迟，产出采样索引；
  * `scripts/verify_latency_multiplier.py` —— 规则保持性与延迟评估。

这里的函数都**不碰音频、不碰模型**，只在 records 的 `trajectory` 上做变换，
因此可以在 CPU 上大规模跑。
"""

MS_PER_MEL_FRAME = 10          # 100Hz mel
MEL_PER_ADAPTER_FRAME = 8      # 8 mel 帧 = 1 adapter 帧 = 80ms
BASE_CHUNK_MS = 1000           # records 的生产粒度（si-rw-m1-v0）


def steps_until_thinker_eos(steps):
    """校验终止标记并保留直到 source 结束的全部 step；不合法返回 None。

    ``translation_eos`` 只标记最后一个 WRITE，也就是目标译文已经提交完；它不等于
    Thinker 序列结束。若 source 仍未读完，后续 WAIT 仍是读写策略监督。训练序列在
    最后一个 ``source_stream_done`` step 后再追加 Thinker EOS（``<|im_end|>``）。
    """
    at = [i for i, s in enumerate(steps) if s.get('translation_eos')]
    last_write = max((i for i, s in enumerate(steps) if s.get('action') == 'write'),
                     default=-1)
    if (len(at) != 1 or at[0] != last_write
            or not steps or not steps[-1].get('source_stream_done')):
        return None
    return list(steps)


def merge_steps(steps, m, base_chunk_ms=BASE_CHUNK_MS, with_provenance=False):
    """把按 1 秒生产的步合并成 m 秒粒度（论文 §3.2 Step 2 的 latency multiplier）。

    分组键 `(source_end_ms - 1) // (m * base_chunk_ms)`：减 1 是为了让恰好落在
    边界上的 end（m=2 时 end=2000）归入前一组，而不是单独开一组。

    合并**只做相邻归并**，因此保持生产期定下的全部规则（`verify_latency_multiplier.py`
    的 R1~R7 实测全过）：文本顺序不变、span 仍按序平铺、
    `commit_ms ≤ source_end_ms` 的因果性保持、ready 单调性保持。

    注意合并**不会**增加 WAIT 监督，反而稀释：句子只有 4.5–6.2 秒，m=2 时零 WAIT
    样本就从 ~15% 涨到 ~80%，m≥8 时只剩 1 步（退化为 S3 离线翻译）。m 提供的是
    延迟-质量连续谱，不是等待监督。

    `with_provenance=True` 时每个新步附带 `_from`（它覆盖的原始步下标列表），
    供验证脚本核对「只做相邻归并、不跨组重排」。训练路径不需要，默认关闭。
    """
    if m <= 1:
        return ([dict(s, _from=[i]) for i, s in enumerate(steps)] if with_provenance
                else list(steps))

    win = m * base_chunk_ms
    groups, cur, cur_key = [], [], None
    for i, s in enumerate(steps):
        key = (int(s['source_end_ms']) - 1) // win
        if cur_key is None or key == cur_key:
            cur.append((i, s))
            cur_key = key
        else:
            groups.append(cur)
            cur, cur_key = [(i, s)], key
    if cur:
        groups.append(cur)

    out = []
    for g in groups:
        gs = [s for _, s in g]
        writes = [s for s in gs if s['action'] == 'write']
        spans = []
        for s in writes:
            for lo, hi in s['source_codec_spans']:
                # 首尾相接的 span 合并成一段：纯表示层压缩，不改变覆盖范围
                if spans and spans[-1][1] == lo:
                    spans[-1][1] = hi
                else:
                    spans.append([lo, hi])
        step = {
            'source_start_ms': gs[0]['source_start_ms'],
            'source_end_ms': gs[-1]['source_end_ms'],
            'action': 'write' if writes else 'wait',
            # 用空格拼接：_step_texts 会按目标语言规则重新决定要不要加空格
            'text_delta': ' '.join(s['text_delta'] for s in writes if s['text_delta']),
            'source_codec_spans': spans,
            'commit_ms': max((s['commit_ms'] for s in writes),
                             default=gs[-1]['source_end_ms']),
            'source_stream_done': gs[-1].get('source_stream_done', False),
            'translation_eos': any(s.get('translation_eos') for s in gs),
        }
        if with_provenance:
            step['_from'] = [i for i, _ in g]
        out.append(step)
    return out


def audio_seg_lens(steps, t_mel):
    """逐步的 adapter 帧数：累计 mel 帧整除，复刻 `encode_audio_chunk` 的余数行为。

    返回 `(seg_lens, n_frames, mel_cut)`。`mel_cut` 是 Thinker EOS 前最后一个 source
    step 的 mel 边界。
    """
    segs, prev_b, b = [], 0, 0
    for s in steps:
        b = min(t_mel, int(s['source_end_ms']) // MS_PER_MEL_FRAME)
        segs.append(b // MEL_PER_ADAPTER_FRAME - prev_b // MEL_PER_ADAPTER_FRAME)
        prev_b = b
    return segs, b // MEL_PER_ADAPTER_FRAME, b


def inversion_count(seq):
    """逆序对数量。句子词数 < 100，O(n²) 足够且比归并排序常数小。"""
    n, inv = len(seq), 0
    for i in range(n):
        si = seq[i]
        for j in range(i + 1, n):
            if si > seq[j]:
                inv += 1
    return inv


def compute_nir(words):
    """NIR = 2·I(Â) / (|Â|·(|Â|−1)) × 100%（论文附录 §Monotonicity-Aware Selection）。

    `Â` 取各目标词依赖源词下标的 **max**，与 `ready_max` 的边界口径一致。
    未对齐词（`dep_src_word_idxs` 为空）跳过 —— 它们的位置由 `bridge_next` 补，
    不代表真实对齐关系，计入会虚增逆序。

    NIR 是 **pre-trajectory** 属性：只依赖词对齐，与 chunk 大小 m 无关。
    这正是它的价值 —— 一次计算对所有 m 档通用。不要把 WAIT 折进来，
    否则会变成 m 依赖，且污染「纯单调性」信号（实测 NIR=0 的样本前导等待仍有 ~830ms，
    因为等待还有「听够音频才能定词边界」这个与重排无关的来源）。
    """
    seq = [max(w['dep_src_word_idxs']) for w in words if w.get('dep_src_word_idxs')]
    if len(seq) < 2:
        return None
    return 200.0 * inversion_count(seq) / (len(seq) * (len(seq) - 1))


def latency_metrics(steps, dur_ms):
    """延迟指标。`steps` 应已通过 steps_until_thinker_eos（并按需 merge_steps）。

    - `lead_wait_ms` / `inter_wait_ms`：首个 WRITE **之前** / **之后** 的等待。
      实测两个方向的等待位置相反：en→zh 以前导为主（先等再说），
      zh→en 以内部为主（80.9% 首步即 WRITE，先说再等）。
    - `first_write_ms`：听了多久才开口。恒等式（实测 100% 成立）：
          first_write_ms == lead_wait_ms + 首个 WRITE 步自身的读窗宽度
      所以首写延迟由前导 WAIT 承载，m 增大时前导被挤成 0、改由 chunk 宽度承载
      （两者相加，不是替代）。
    - `laal`：策略口径 `mean_k(d_k − k·|X|/N)`，`d_k` = 发射帧 k 所属 WRITE 步的
      `source_end_ms`（必须听完整块才能决策），目标单元取 codec 帧。
      **不要**用 `target_start_ms − commit_ms` —— 那两个字段从原始步继承、
      不随 m 变化，评估 m 时恒定无意义。
      WAIT 已通过「推后 WRITE 步 → d_k 变大」计入，不需额外加项。
    """
    writes = [s for s in steps if s['action'] == 'write']
    waits = [s for s in steps if s['action'] == 'wait']
    width = lambda s: int(s['source_end_ms']) - int(s['source_start_ms'])

    fw = next((i for i, s in enumerate(steps) if s['action'] == 'write'), None)
    lead = steps[:fw] if fw is not None else []
    inter = [s for s in steps[fw:] if s['action'] == 'wait'] if fw is not None else []

    d_per_frame = []
    for s in writes:
        nf = sum(hi - lo for lo, hi in s['source_codec_spans'])
        d_per_frame.extend([float(s['source_end_ms'])] * nf)
    n = len(d_per_frame)
    laal = None
    if n:
        ideal = dur_ms / n
        laal = sum(d - k * ideal for k, d in enumerate(d_per_frame)) / n

    return {
        'n_step': len(steps),
        'n_wait': len(waits),
        'n_write': len(writes),
        'wait_ms': sum(width(s) for s in waits),
        'lead_wait_ms': sum(width(s) for s in lead),
        'inter_wait_ms': sum(width(s) for s in inter),
        'n_lead_wait': len(lead),
        'n_inter_wait': len(inter),
        'first_write_ms': int(writes[0]['source_end_ms']) if writes else None,
        'first_step_is_write': (fw == 0),
        'laal': laal,
        'speech_frames': n,
        'frames_per_write': (n / len(writes)) if writes else 0.0,
    }
