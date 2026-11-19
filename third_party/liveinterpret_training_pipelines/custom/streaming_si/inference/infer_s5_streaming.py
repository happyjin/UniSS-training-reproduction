"""S5（流式同传）推理：源音频按 640ms 分块进入，每块后由模型决定等/说，说出时立刻发声。

与 S5 训练布局逐 token 一致（见 `s4_streaming_dataset.py` 的类 docstring）：

    [instr]
    <|chunk|> <|audio_pad|>×n_0 [<|src_end|>] <|wait|>
    <|chunk|> <|audio_pad|>×n_1                <|W|>  译文增量_1
    ...
    <|W|>  译文增量_K  <eos>

驱动循环：每步先喂一块音频（`<|chunk|>` + audio_pad，源读完再补 `<|src_end|>`），然后采一个决策
token —— `<|wait|>` 就继续喂下一块，`<|W|>` 就采一段译文增量，用这段增量的 thinker hidden 做
talker 条件、立刻生成语音，再交回控制权。

用法（单条 / 批量二选一）：

    python -m streaming_si.inference.infer_s5_streaming --model_dir <S4 底座> --adapter <S5 adapter> \
        --input_jsonl eval.jsonl --out_dir out --cosyvoice_repo /path/to/CosyVoice

eval.jsonl 每行：`{"id": "...", "audio": "/path/x.wav", "tgt_lang": "en"}`
产出：`out/<id>.wav`（24kHz）、`out/results.jsonl`（逐步时间线）、`out/summary.json`。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.model.base_model import VoxtralAudioProcessor  # noqa: E402
from streaming_si.model.model_si import TalkerStreamState  # noqa: E402
from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration  # noqa: E402
from streaming_si.data.s4_pipeline import add_s4_stream_tokens, s4_stream_token_ids  # noqa: E402
from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor  # noqa: E402

from streaming_si.inference.infer_s2st import (CampplusSpeaker, add_source_prompt_args,  # noqa: E402
                                               build_source_prompt, load_wav16)
from streaming_si.inference.source_prompt import load_wav24  # noqa: E402

SR = 16000
OUT_SR = 24000
# S5 训练口径（train_simul.sh：SI_MAX_AUDIO_SEC=30、max_length=1024；数据 rechunk_D，640ms 基础块）。
# 训练数据实际最长 24.9s（> 20s 的约 0.1%），25～30s 的输入超出训练分布。
CHUNK_MS = 640
MS_PER_MEL_FRAME = 10            # 16kHz / hop 160 → 100Hz mel
DEFAULT_MAX_AUDIO_SEC = 30.0     # Voxtral 编码器上限（1500 帧 @50Hz）
DEFAULT_MAX_LENGTH = 1024
MAX_TEXT_TOKENS = 256            # 训练 max_target_len
MAX_TAIL_STEPS = 8               # 源读完后最多再走几个 0 帧步等模型说完
INSTRUCTION = '\nTranslate the streaming speech into {tgt} simultaneously:\n'
LANG_NAME = {'zh': 'Chinese', 'cmn': 'Chinese', 'zho': 'Chinese', 'yue': 'Cantonese',
             'en': 'English', 'eng': 'English', 'ja': 'Japanese', 'ko': 'Korean'}
NO_SPACE_LANGS = ('zh', 'cmn', 'zho', 'yue', 'ja', 'jpn', 'th', 'tha')


def log(msg):
    print(msg, file=sys.stdout, flush=True)


def collapse_ok(res):
    """True = 生成正常。判据取自 docs/talker_collapse.md：静音码字占比高（token 种类占比低）
    且语音长度明显超过源音频，就是塌缩。"""
    t = res.get('speech_tokens') or []
    if not t:
        return False
    uniq = len(set(t)) / len(t)
    over = res['speech_sec'] > 1.6 * res['source_sec'] + 1.0
    return not (uniq < 0.3 and over)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def load_model(model_dir, adapter_dir, dtype='bfloat16'):
    """S4 底座 + S5 adapter（合并成一个普通模型，推理不再依赖 peft）。"""
    from transformers import AutoTokenizer
    import peft

    meta_path = os.path.join(adapter_dir, 's4_metadata.json')
    if os.path.isfile(meta_path):
        log(f'[infer] adapter metadata: {json.load(open(meta_path))}')
    model = StreamingSIForConditionalGeneration.from_pretrained(
        model_dir, dtype=getattr(torch, dtype))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # 与训练一致：容量取 config.vocab_size（4B 变体 248k，见 register_streaming_si.py）
    add_s4_stream_tokens(tokenizer, capacity=int(model.config.vocab_size))
    ids = s4_stream_token_ids(tokenizer)
    if os.path.isfile(meta_path):
        ckpt_ids = json.load(open(meta_path)).get('stream_token_ids')
        if ckpt_ids and [int(x) for x in ckpt_ids] != ids:
            raise SystemExit(f'tokenizer 的流式 token {ids} 与 adapter 记录的 {ckpt_ids} 不一致')
    model = peft.PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload().eval()
    # 实验 b 的 talker_cond_positions 只实现到 S4：本文件的流式条件是逐 WRITE 步自建 am
    # （见 stream_translate 里的 `am[0, -len(delta):] = 1` → gather_answer_hidden），
    # 绕过了 build_cond_mask，所以 audio / audio+answer 的 checkpoint 在这里会**静默**
    # 退回"只取译文位置"，与训练口径不符。宁可报错也不要出一份看似正常的错结果。
    cond_pos = str(getattr(model.config, 'talker_cond_positions', 'answer') or 'answer')
    if cond_pos != 'answer':
        raise SystemExit(
            f'该 checkpoint 的 talker_cond_positions={cond_pos!r}，但 S5 流式推理尚未支持\n'
            '（需同步改 stream_translate 的条件构造与 build_stream_talker_condition，\n'
            ' 并解决音频 12.5Hz 与译文 token 粒度在交错下的对齐；见\n'
            ' pipelines/streaming_si/docs/talker_para_experiments.md §4 实现要点 3）。\n'
            '实验 b 的评测请走 S4 离线路径 infer_s2st.py。')
    return model, tokenizer, ids


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def _pick(logits, temperature, top_p, ban_above=None, allow=()):
    """ban_above：屏蔽 >= 该 id 的 token（特殊 token 区）；allow：其中的例外。"""
    scores = logits.float()
    if ban_above is not None:
        saved = {t: scores[t].clone() for t in allow}
        scores[ban_above:] = -float('inf')
        for t, v in saved.items():
            scores[t] = v
    if top_p and top_p > 0 and temperature and temperature > 0:
        probs = (scores / temperature).softmax(-1)
        sp, idx = probs.sort(descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p
        sp = sp * keep
        return int(idx[torch.multinomial(sp / sp.sum(), 1)])
    return int(scores.argmax())


class _HiddenSoFar:
    """增量模式下 thinker 输出的替身：`last_hidden_state` 是到目前为止全部位置的末层 hidden。"""

    def __init__(self, pieces):
        self._pieces = list(pieces)
        self._cat = None

    @property
    def last_hidden_state(self):
        if self._cat is None:
            self._cat = torch.cat(self._pieces, dim=1)
        return self._cat


class ThinkerRunner:
    """对 `embeds`（只追加、不回退的 embedding 列表）做 thinker 前向，返回 (末位 logits, thinker 输出)。

    `use_cache=True`：每次只喂上次之后新追加的部分，线性注意力状态与 KV 留在缓存里
    （Qwen3.5 兼容实现见 `qwen3_5_compat.py`）；返回的输出覆盖全部位置（talker 条件要取译文增量的 hidden）。
    `use_cache=False`：每次整段重算（旧行为，对照用）。
    `graph=True`（需 use_cache 且 CUDA、thinker 为 Qwen3.5）：改用定长缓存 + CUDA Graph
    （`qwen3_5_static.Qwen3_5StaticRunner`，按模型懒建、跨句复用）。
    talker 条件取中间层（`talker_bridge_layer >= 0`）时需要逐层 hidden，退回整段重算。
    """

    def __init__(self, model, embeds, use_cache=True, graph=True):
        layer = getattr(model.config, 'talker_bridge_layer', -1)
        self.model, self.embeds = model, embeds
        self.use_cache = bool(use_cache) and (layer is None or layer < 0)
        self.cache, self.fed, self.hidden, self.last = None, 0, [], None
        self.n_forward = 0
        self.static = None
        if self.use_cache and graph:
            self.static = thinker_static_runner(model)
            if self.static is not None:
                self.static.reset()

    def __call__(self):
        if not self.use_cache:
            self.n_forward += 1
            out = self.model.thinker(inputs_embeds=torch.cat(self.embeds, dim=1))
            return self.model.lm_head(out.last_hidden_state[:, -1])[0], out
        if self.fed == len(self.embeds):          # 没有新输入：结果不变
            return self.last
        self.n_forward += 1
        new = torch.cat(self.embeds[self.fed:], dim=1)
        self.fed = len(self.embeds)
        if self.static is not None:
            h = self.static(new).clone()           # 图的输出缓冲下次会被覆盖
        else:
            out = self.model.thinker(inputs_embeds=new, past_key_values=self.cache, use_cache=True)
            self.cache = out.past_key_values
            h = out.last_hidden_state
        self.hidden.append(h)
        self.last = (self.model.lm_head(h[:, -1])[0], _HiddenSoFar(self.hidden))
        return self.last


def thinker_static_runner(model, max_len=2048, graph_lens=(1, 9, 10)):
    """模型上懒建一个 `Qwen3_5StaticRunner`（首次建时录图）；非 CUDA 或非 Qwen3.5 thinker 返回 None。"""
    th = model.thinker
    if next(th.parameters()).device.type != 'cuda' or not hasattr(th.layers[0], 'layer_type'):
        return None
    r = model.__dict__.get('_thinker_static')
    if r is None or r.max_len < max_len:
        from streaming_si.model.qwen3_5_static import Qwen3_5StaticRunner
        r = Qwen3_5StaticRunner(th, max_len=max_len, graph_lens=graph_lens)
        r.prepare()
        model.__dict__['_thinker_static'] = r
    return r


def sample_delta(embeds, run, tok_emb, pick, chunk_id, eos_id, wait_id, *,
                 ban_above, temperature, top_p, max_tokens=256):
    """逐 token 生成一段译文增量，采到终止符（`<|chunk|>` / eos / `<|wait|>`）为止。

    终止符不进 embeds（下一步由驱动循环喂 `<|chunk|>`，与训练布局一致）。
    返回 (delta, out, term)：out 是最后一个译文 token 已进入序列后的 thinker 输出，
    其末尾 len(delta) 个位置就是本段译文的 hidden（talker 条件用）；term 是命中的终止符
    （触顶时为 None）。`run` 是 `ThinkerRunner`（增量缓存或整段重算）。
    """
    delta, logits, out, term = [], None, None, None
    for _ in range(max(0, int(max_tokens))):
        logits, out = run()           # 序列已含此前全部译文 token
        tok = pick(logits, temperature, top_p, ban_above=ban_above,
                   allow=(wait_id, chunk_id, eos_id))
        if tok in (chunk_id, eos_id, wait_id):
            term = tok
            break
        delta.append(tok)
        embeds.append(tok_emb(tok))
    if delta and term is None:        # 触顶退出：最后一个 token 还没进 out
        logits, out = run()
    return delta, out, term


# ---------------------------------------------------------------------------
# 单条流式推理
# ---------------------------------------------------------------------------
@torch.inference_mode()
def stream_translate(model, tokenizer, ids, mel_fn, wav16, tgt_lang, *,
                     device, chunk_ms=CHUNK_MS, max_length=DEFAULT_MAX_LENGTH,
                     max_speech_tokens=750, talker_top_k=25, talker_top_p=0.8,
                     temperature=1.0, top_p=0.0, speaker=None, ode_steps=5,
                     decode=True, max_text_tokens=MAX_TEXT_TOKENS,
                     cond_mode='step', seed=0, max_tail_steps=MAX_TAIL_STEPS, thinker_cache=True,
                     talker_graph=True, thinker_graph=True, source_prompt=None):
    """返回 dict：译文、逐步时间线、speech token、波形。

    `cond_mode`：'step' = 每段只用本步增量的 hidden（默认）；
    'cum' = 用截至本步的全部译文 hidden（对照用）。
    `max_tail_steps`：源读完后、模型还没输出 eos 时最多再走几个 0 帧步（训练布局里
    0 帧步同样带 `<|chunk|>` 与决策）。
    `thinker_cache`：thinker 用增量缓存（默认）；False = 每个 token 整段重算（对照用）。
    `talker_graph`：talker 单 token 步用 StaticCache + CUDA Graph（默认，仅 CUDA）；False = eager。
    `thinker_graph`：thinker 增量前向用定长缓存 + CUDA Graph（默认，仅 CUDA）；False = 动态缓存 eager。
    `source_prompt`：(prompt_tokens, prompt_feat)，源语音编成的 flow 声学前缀（`source_prompt.py`）；
    None = 只用说话人向量。
    """
    wait_id, write_id, chunk_id, src_end_id = ids
    t_start = time.perf_counter()              # 计算感知延迟用：每步记 elapsed_sec（墙钟，自本句开始）
    audio_pad_id = int(model.config.audio_ids[0])
    eos_id = int(tokenizer.eos_token_id)
    # thinker 侧只屏蔽特殊 token 区（tokenizer 基础词表之后：<|im_end|>、audio_pad、流式 token…），
    # 需要的终止符 / 决策 token 由 allow 放开。注意不能用 talker_vocab_size（那是语音码本大小）。
    ban_above = int(tokenizer.vocab_size)
    lang = (tgt_lang or 'en').lower()
    space_lang = lang not in NO_SPACE_LANGS
    emb_layer = model.thinker.embed_tokens
    dtype = emb_layer.weight.dtype

    def tok_emb(i):
        return emb_layer(torch.tensor([[int(i)]], device=device))

    mel = mel_fn(torch.from_numpy(wav16), sr=SR).unsqueeze(0).to(device)   # (1,128,T)，100Hz
    t_mel = mel.size(-1)
    embeds = [emb_layer(torch.tensor(
        [tokenizer.encode(INSTRUCTION.format(tgt=LANG_NAME.get(lang, tgt_lang or 'English')),
                          add_special_tokens=False)], device=device))]

    run = ThinkerRunner(model, embeds, use_cache=thinker_cache, graph=thinker_graph)

    chunk_mel = max(1, int(chunk_ms // MS_PER_MEL_FRAME))
    n_steps = max(1, int(np.ceil(t_mel / chunk_mel)))
    # 决定 talker 采样路径（塌缩重试靠它）；放在这里，驱动循环的决策不受影响
    torch.manual_seed(seed)
    state = TalkerStreamState(model.talker, cuda_graph=talker_graph)
    vcache = None
    steps, speech_tokens, pieces = [], [], []
    write_hidden, cum_delta = [], []
    n_text, stop = 0, 'max_steps'

    i, tail_steps, src_done = 0, 0, False
    while True:
        b = min(t_mel, (i + 1) * chunk_mel)
        n_frames = (b // 8 - (i * chunk_mel) // 8) if not src_done else 0
        is_last = b >= t_mel
        step = dict(step=i, source_end_ms=round(b * MS_PER_MEL_FRAME / 1000.0, 3),
                    n_audio_frames=int(n_frames), src_end=bool(is_last and not src_done),
                    tail=bool(src_done))

        # ---- 喂一块音频：<|chunk|> + audio_pad×n（+ 源读完的 <|src_end|>，只发一次）----
        embeds.append(tok_emb(chunk_id))
        if n_frames > 0:
            feats, vcache = model.encode_audio_chunk(mel[..., i * chunk_mel:b], cache=vcache)
            if feats.size(1) > 0:
                pad_ids = torch.full((1, int(feats.size(1))), audio_pad_id, device=device)
                filled = model.inject_audio_features(pad_ids, emb_layer(pad_ids), [feats[0]])
                embeds.append(filled.to(dtype))
        if is_last and not src_done:
            embeds.append(tok_emb(src_end_id))
            src_done = True
        logits, out = run()

        # ---- 决策：只允许 <|wait|> / <|W|>；源读完后也允许 eos（训练布局：src_end 后 wait + eos）----
        allow = (wait_id, write_id, eos_id) if src_done else (wait_id, write_id)
        dec = _pick(logits, temperature, top_p, ban_above=ban_above, allow=allow)
        if dec == eos_id:
            step['decision'] = 'eos'
            step['elapsed_sec'] = round(time.perf_counter() - t_start, 4)
            steps.append(step)
            stop = 'eos'
            break
        embeds.append(tok_emb(dec))
        step['decision'] = 'write' if dec == write_id else 'wait'
        term = None

        if dec == write_id:
            # 译文增量：逐 token 采到终止符（<|chunk|> / eos / <|wait|>）
            delta, out, term = sample_delta(
                embeds, run, tok_emb, _pick, chunk_id, eos_id, wait_id,
                ban_above=ban_above, temperature=temperature, top_p=top_p,
                max_tokens=int(max_text_tokens) - n_text)
            n_text += len(delta)
            step['term'] = {chunk_id: 'chunk', eos_id: 'eos', wait_id: 'wait', None: 'cap'}[term]
            if delta:
                # talker 条件：本步译文增量所在位置的 thinker hidden（末层）
                am = torch.zeros((1, out.last_hidden_state.size(1)), dtype=torch.long, device=device)
                if cond_mode == 'cum':
                    write_hidden.append(model.pick_bridge_hidden(out)[0, -len(delta):])
                    cond_h = torch.cat(write_hidden, dim=0).unsqueeze(0)
                    cum_delta += delta
                else:
                    am[0, -len(delta):] = 1
                    cond_h, lens = model.gather_answer_hidden(model.pick_bridge_hidden(out), am)
                    cond_h = cond_h[:, :int(lens[0])]
                    cum_delta = list(delta)
                cond = model.build_stream_talker_condition(
                    cond_h, torch.tensor([cum_delta], device=device))
                seg = list(model.talker.stream_generate_s4(
                    state, cond_chunk=cond, max_speech_tokens=max_speech_tokens,
                    top_k=talker_top_k, top_p=talker_top_p))
                speech_tokens += seg
                text = tokenizer.decode(delta, skip_special_tokens=True)
                pieces.append(((' ' if space_lang else '') if pieces else '') + text.strip())
                step.update(text_delta=text, speech_tokens=len(seg),
                            speech_stop=state.last_stop_reason)
        step['elapsed_sec'] = round(time.perf_counter() - t_start, 4)   # 本步（含本段 speech token）完成时刻
        steps.append(step)
        i += 1

        if term == eos_id:                     # 译文写完且模型给出 Thinker EOS
            stop = 'eos'
            break
        if src_done and dec == wait_id:        # 源读完后 WAIT：训练布局里紧跟 eos
            stop = 'eos'
            break
        if src_done:
            tail_steps += 1
            if tail_steps > max_tail_steps:
                stop = 'tail_cap'
                break
        if n_text >= max_text_tokens:
            stop = 'max_text_tokens'
            break
        if i > n_steps + max_tail_steps + 1:   # 兜底，正常不会走到
            stop = 'max_steps'
            break

    # 尾部 flush：放开 eos
    tail = list(model.talker.stream_generate_s4(
        state, cond_chunk=None, finalize=True, max_speech_tokens=max_speech_tokens,
        top_k=talker_top_k, top_p=talker_top_p))
    speech_tokens += tail
    t_tail = time.perf_counter() - t_start

    res = dict(text=''.join(pieces).strip(), text_pieces=pieces, text_token_total=n_text,
               steps=steps, speech_tokens=speech_tokens, speech_stop=state.last_stop_reason,
               tail_speech_tokens=len(tail), tail_elapsed_sec=round(t_tail, 4),
               n_write=sum(1 for s in steps if s.get('decision') == 'write'),
               n_wait=sum(1 for s in steps if s.get('decision') == 'wait'),
               fill_rate=round(state.fill_rate, 3), segcap_rate=round(state.segcap_fallback_rate, 3),
               stop=stop, source_sec=round(wav16.size / SR, 2),
               speech_sec=round(len(speech_tokens) / 25.0, 2), n_forward=run.n_forward,
               thinker_cache=run.use_cache, thinker_graph=run.static is not None,
               talker_graph=state.graph is not None,
               seq_len=sum(int(e.size(1)) for e in embeds))
    # 训练序列上限（max_length）：超过不截断（会丢后半段音频），只标记为训练分布外
    res['over_train_length'] = res['seq_len'] > max_length
    if decode and speech_tokens:
        t_voc = time.perf_counter()
        emb = speaker(wav16) if speaker is not None else None
        p_tok, p_feat = source_prompt if source_prompt is not None else (None, None)
        wav = model.decode_speech(speech_tokens, spk_emb=emb, ode_steps=ode_steps,
                                  prompt_speech_tokens=p_tok, prompt_feat=p_feat)
        res['wav'] = wav.detach().float().cpu().numpy().reshape(-1)
        res['vocoder_sec'] = round(time.perf_counter() - t_voc, 4)   # 整句一次合成；延迟评测按 token 比例摊到各段
    res['source_prompt_tokens'] = 0 if source_prompt is None else int(len(source_prompt[0]))
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_dir', required=True, help='S4 底座（checkpoint-24980）')
    ap.add_argument('--adapter', required=True, help='S5 adapter 目录（checkpoint-17309 / 16000）')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--audio', help='单条源音频')
    src.add_argument('--input_jsonl', help='批量：每行 {"id","audio","tgt_lang"}')
    ap.add_argument('--tgt_lang', default=None, help='单条模式的目标语言（en / zh）')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--cosyvoice_repo', default=None, help='CosyVoice 官方仓库（--recursive）')
    ap.add_argument('--cosyvoice_dir', default=None, help='缺省用 <model_dir>/speech_decoder')
    ap.add_argument('--no_vocoder', action='store_true', help='不合成波形，只输出译文与 speech token')
    ap.add_argument('--no_speaker', action='store_true', help='不提取说话人向量（零向量）')
    ap.add_argument('--chunk_ms', type=int, default=CHUNK_MS)
    ap.add_argument('--max_audio_sec', type=float, default=DEFAULT_MAX_AUDIO_SEC,
                    help='源音频上限（S5 训练 30s；训练数据实际 ≤ 24.9s），超出部分截掉')
    ap.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH,
                    help='训练序列上限，超过只标记 over_train_length，不截断')
    ap.add_argument('--no_thinker_cache', action='store_true',
                    help='thinker 每个 token 整段重算（旧行为，对照增量缓存用）')
    ap.add_argument('--no_thinker_graph', action='store_true',
                    help='thinker 不用定长缓存 + CUDA Graph（动态缓存 eager）')
    ap.add_argument('--no_talker_graph', action='store_true',
                    help='talker 不用 CUDA Graph（eager + DynamicCache，旧行为）')
    ap.add_argument('--max_speech_tokens', type=int, default=750)
    ap.add_argument('--cond_mode', default='step', choices=['step', 'cum'],
                    help="talker 条件：step = 只用本步增量（默认）；cum = 截至本步的全部译文")
    ap.add_argument('--retry', type=int, default=3,
                    help='生成结果判定为塌缩时换 seed 重试的次数（0 = 不重试）')
    ap.add_argument('--ode_steps', type=int, default=5)
    ap.add_argument('--talker_top_k', type=int, default=25)
    ap.add_argument('--talker_top_p', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    add_source_prompt_args(ap)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    model, tokenizer, ids = load_model(args.model_dir, args.adapter, dtype=args.dtype)
    model = model.to(args.device)
    log(f'[infer] 流式 token: wait={ids[0]} write={ids[1]} chunk={ids[2]} src_end={ids[3]}')
    mel_fn = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(args.model_dir))
    speaker, decode = None, not args.no_vocoder
    if decode:
        from streaming_si.model.speech_decoder import resolve_speech_decoder
        dec_dir, yaml = resolve_speech_decoder(args.model_dir, args.cosyvoice_dir, None)
        repo = args.cosyvoice_repo or os.environ.get('COSYVOICE_REPO')
        if not repo:
            raise SystemExit('声码器需要 --cosyvoice_repo（或 COSYVOICE_REPO）')
        model.config.talker_config_yaml = yaml
        model.attach_speech_decoder(dec_dir, device=args.device, cosyvoice_repo=repo, strict=True)
        onnx = os.path.join(dec_dir, 'campplus.onnx')
        if not args.no_speaker and os.path.isfile(onnx):
            speaker = CampplusSpeaker(onnx)
        log(f'[infer] 声码器: {dec_dir}')
    prompter = build_source_prompt(args.source_prompt and decode, dec_dir if decode else None,
                                   args.prompt_tokenizer)

    if args.audio:
        if not args.tgt_lang:
            raise SystemExit('单条模式需要 --tgt_lang')
        rows = [dict(id=os.path.splitext(os.path.basename(args.audio))[0],
                     audio=args.audio, tgt_lang=args.tgt_lang)]
    else:
        rows = [json.loads(l) for l in open(args.input_jsonl) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    results = []
    summary = dict(model_dir=args.model_dir, adapter=args.adapter, chunk_ms=args.chunk_ms,
                   n=len(rows), completed=0, failed=0, stop={}, text_stop={})
    for r in rows:
        t0 = time.time()
        try:
            wav16 = load_wav16(r['audio'], max_sec=args.max_audio_sec)
            sp = None
            if prompter is not None:
                sp = prompter(wav16, max_sec=args.prompt_sec, position=args.prompt_position,
                              wav24=load_wav24(r['audio'], max_sec=args.max_audio_sec))
            attempts, res = [], None
            for k in range(max(1, args.retry + 1)):
                torch.manual_seed(args.seed + k)
                res = stream_translate(
                    model, tokenizer, ids, mel_fn, wav16, r.get('tgt_lang', 'en'),
                    device=args.device, chunk_ms=args.chunk_ms, max_length=args.max_length,
                    max_speech_tokens=args.max_speech_tokens, talker_top_k=args.talker_top_k,
                    talker_top_p=args.talker_top_p, speaker=speaker, ode_steps=args.ode_steps,
                    decode=decode, cond_mode=args.cond_mode,
                    seed=args.seed + k, thinker_cache=not args.no_thinker_cache,
                    talker_graph=not args.no_talker_graph, thinker_graph=not args.no_thinker_graph,
                    source_prompt=sp)
                ok = collapse_ok(res)
                log(f'[infer] {r["id"]} seed={args.seed + k}: 译音 {res["speech_sec"]}s / 源 '
                    f'{res["source_sec"]}s，token 种类 {len(set(res["speech_tokens"]))}'
                    f'（{len(set(res["speech_tokens"])) / max(len(res["speech_tokens"]), 1):.2f}）→ '
                    f'{"正常" if ok else "判定塌缩"}')
                if ok:
                    break
            res['seed'] = args.seed + k
            res['retried'] = k
        except Exception as exc:                                   # noqa: BLE001
            import traceback
            log(f'[infer] {r["id"]} 失败: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            summary['failed'] += 1
            results.append(dict(id=r['id'], status='failed', error=f'{type(exc).__name__}: {exc}'))
            continue
        res.update(id=r['id'], status='ok', tgt_lang=r.get('tgt_lang') or args.tgt_lang,
                   wall_sec=round(time.time() - t0, 1))
        results.append(res)
        summary['completed'] += 1
        summary['stop'][res['stop']] = summary['stop'].get(res['stop'], 0) + 1
        if res.get('wav') is not None:
            import soundfile as sf
            wav_path = os.path.join(args.out_dir, f'{r["id"]}.wav')
            wav_arr = np.asarray(res['wav'], dtype=np.float32)
            sf.write(wav_path, wav_arr, OUT_SR)
            res.pop('wav')
            # 记下路径与时长（与 infer_s2st 的字段一致，评测脚本按 wav / output_sec 取）
            res['wav'] = os.path.abspath(wav_path)
            res['output_sec'] = round(wav_arr.size / OUT_SR, 2)
        log(f'[infer] {r["id"]}: 源 {res["source_sec"]}s → 译音 {res["speech_sec"]}s '
            f'（WRITE {res["n_write"]} / WAIT {res["n_wait"]}，fill {res["fill_rate"]}，'
            f'{res["wall_sec"]}s）| {res["text"][:200]}')

    with open(os.path.join(args.out_dir, 'results.jsonl'), 'w') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
