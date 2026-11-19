"""全链路风格探针：thinker（LLM）与 talker（语音生成）各层的风格类别可分性。

背景：编码器侧探针（probe_voxtral_style.py / probe_omni_audio_tower.py）已给出
audio encoder 各层（Voxtral 峰值 L15 85.1%、Omni audio_tower 峰值 L27 87.1%）。
本脚本把同协议（Expresso 7 风格 × 4 说话人、mean 池化、L2 逻辑回归、LOSO）延伸到
下游两条链，补全信息漏斗。

**池化位置（2026-10-21 修订，关键）**：
首版只在 thinker 上池化**音频位置**，却在 talker 上池化**cond 段**，而 cond 段来自
`gather_answer_hidden(bridge_hidden, answer_mask)` —— 即**译文位置**。两者位置不同，
拼成一条"漏斗"是错的（首版出现 thinker L32 音频位 58% 却 talker L1 cond 段 79%
的矛盾，根因就是这个）。现在两侧都同时记录：

  thinker_aud : 音频 token 位置（源音频表示逐层如何被改写）
  thinker_ans : **译文位置**（= 喂给 talker 的那一路，**主漏斗**）
  talker      : talker 各层在条件段上的池化
  talker_aud  : 仅 Omni —— 其 talker 前缀里**包含音频位置**（见下），单独记一份

**两侧 talker 的条件结构不同（这本身是结论，不是噪声）**：
* S4：talker 输入 = `[sos] + cond + [task] + speech`，cond **只有译文位置**
  （`build_talker_condition` → `gather_answer_hidden(..., answer_mask)`）；
* Omni：talker 输入 = `prefill_cond(整个 prompt 前缀，**含音频 token 位置**)
  + [text_bos] + reply_cond[0]`，即 talker 直接看得到音频位置的条件。
  这正是实验 b（audio+answer 取位）想补的那一刀。

**协议**：
* 固定同一段**中文**译文（`FIXED_TEXT`）——文本恒定，故各层可分性差异只能归因于音频。
* 方向 **en→zh**：Expresso 是英语表达性语音（实测 text 列为英文），两个对比模型都做
  中英互译，故 ref=English / tgt=Chinese。Omni 侧 ref≠tgt 才会走
  `TASK_TTS_PAIRED_CROSS_LANG`（同语种会落到另一个任务模板，口径不对）。
* talker 只跑前缀单次前向：因果模型前缀位置的 hidden 与后面有什么 token 无关，
  无需教师强制的语音码，也不用自回归生成。

用法（8 卡并行 = 8 进程各占一卡，按行号取模分片；跑完 merge 再分类）：
  python probe_style_fullfunnel.py extract-si   --parquet '...train-*.parquet' \
      --ck /fsx/.../s4_ck --out /nvme/ff_si --shard $i --nshards 8
  python probe_style_fullfunnel.py extract-omni --parquet '...' --out /nvme/ff_omni --shard $i --nshards 8
  python probe_style_fullfunnel.py merge --out /nvme/ff_si
  python probe_style_fast.py --cache /nvme/ff_si/thinker_ans --jobs 24
"""
import argparse
import glob
import io
import os

import numpy as np

# 固定中文译文（内容恒定；长度固定 → cond 段长度也固定）
FIXED_TEXT = '这是语音翻译系统的一次测试。'
REF_LANG, TGT_LANG = 'English', 'Chinese'      # Expresso 音频是英语 → 译中文（en2zh）

_OMNI = '/fsx/stevehguo/eval/models/SimulS2ST-Omni'
STYLES = ('confused', 'default', 'enunciated', 'happy', 'laughing', 'sad', 'whisper')


def load_rows(pattern):
    import pyarrow.parquet as pq
    t = []
    for p in sorted(glob.glob(pattern)):
        t += pq.read_table(p).to_pylist()
    rows = [r for r in t if r.get('style') in STYLES
            and isinstance(r.get('audio'), dict) and r['audio'].get('bytes')]
    print(f'clips: {len(rows)}', flush=True)
    return rows


def wav16_of(row):
    import soundfile as sf
    import librosa
    wav, sr = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav[:16000 * 30]


def _mean_masked(h, mask):
    """h: (L, D) float, mask: (L,) bool → 掩码位置均值 → (D,) float16"""
    m = mask.unsqueeze(-1).to(h.dtype)
    v = (h * m).sum(0) / m.sum().clamp(min=1)
    return v.cpu().numpy().astype(np.float16)


# ---------------------------------------------------------------- A. streaming_si

def extract_si(args):
    import torch
    from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
    from streaming_si.data.si_encode import encode_st, audio_token_id
    from transformers import AutoTokenizer

    model = StreamingSIForConditionalGeneration.from_pretrained(
        args.ck, dtype=torch.bfloat16).to(args.device).eval()
    # forward_thinker 只在 bridge_layer>=0 时返回全部层，故临时设 0；**算 cond 前必须
    # 恢复原值**（部署口径 a0/a2/b2 均为 -1=末层，s4 阶段 ckpt 也是 -1），否则 cond
    # 会错误地取自 L1 —— 2026-10-21 v2 的 talker@cond 曲线即踩此坑（已定位）。
    orig_bridge = model.config.talker_bridge_layer
    model.config.talker_bridge_layer = 0
    tok = AutoTokenizer.from_pretrained(args.ck)
    fe = VoxtralFeatureExtractor.from_pretrained(args.ck)
    a_id = audio_token_id(tok)
    n_thk = model.config.num_hidden_layers
    n_tkr = len(model.talker.backbone.layers)
    print(f'thinker {n_thk}L/{model.config.hidden_size}d  talker {n_tkr}L', flush=True)

    rows = load_rows(args.parquet)[args.shard::args.nshards]
    feats = {'thinker_aud': {f'L{i}': [] for i in range(1, n_thk + 1)},
             'thinker_ans': {f'L{i}': [] for i in range(1, n_thk + 1)},
             'talker': {f'L{i}': [] for i in range(1, n_tkr + 1)}}
    labels, speakers, ids = [], [], []
    with torch.no_grad():
        for n, r in enumerate(rows, 1):
            try:
                wav = wav16_of(r)
                mel = torch.as_tensor(fe(wav, sampling_rate=16000)['input_features'],
                                      dtype=torch.float32)
                if mel.dim() != 2:
                    mel = mel[0]
                enc = encode_st(tok, mel, FIXED_TEXT, TGT_LANG,
                                with_answer_mask=True, speech_tokens=[0])
                ids_t = torch.tensor([enc['input_ids']], device=args.device)
                am_t = torch.tensor([enc['answer_mask']], device=args.device)
                model.config.talker_bridge_layer = 0   # 每条样本都要重置：上一条末尾已恢复为 orig_bridge
                tk_out = model.forward_thinker(
                    input_ids=ids_t,
                    audio_inputs=mel.unsqueeze(0).to(args.device),
                    audio_lens=torch.tensor([enc['audio_lens']], device=args.device))
                aud_mask = (ids_t[0] == a_id)
                ans_mask = am_t[0].bool()
                hs = tk_out.hidden_states          # tuple(n+1): [0]=emb, [i]=第 i 层
                for i in range(1, n_thk + 1):
                    h = hs[i][0].float()
                    feats['thinker_aud'][f'L{i}'].append(_mean_masked(h, aud_mask))
                    feats['thinker_ans'][f'L{i}'].append(_mean_masked(h, ans_mask))
                # talker：cond 段（由译文位置 hidden + 恒定文本 embedding 融合而来）
                model.config.talker_bridge_layer = orig_bridge   # 恢复部署口径（-1=末层）
                cond, cond_lens = model.build_talker_condition(tk_out, ids_t, am_t)
                inputs, attn, _ = model.talker.build_inputs(cond, cond_lens)
                t_out = model.talker.backbone(inputs_embeds=inputs, attention_mask=attn,
                                              output_hidden_states=True)
                c_len = int(cond_lens[0])
                cmask = torch.zeros(inputs.size(1), dtype=torch.bool, device=args.device)
                cmask[1:1 + c_len] = True        # [sos] 之后的 cond 段
                for i in range(1, n_tkr + 1):
                    feats['talker'][f'L{i}'].append(
                        _mean_masked(t_out.hidden_states[i][0].float(), cmask))
                labels.append(r['style']); speakers.append(r['speaker_id']); ids.append(r['id'])
            except Exception as exc:
                if os.environ.get('PROBE_DEBUG'):
                    import traceback
                    traceback.print_exc()
                    raise
                print(f'  [FAIL] {r.get("id")}: {type(exc).__name__}: {exc}', flush=True)
            if n % 100 == 0 or n == len(rows):
                print(f'[si shard {args.shard}] {n}/{len(rows)} ok={len(labels)}', flush=True)
    _save(args, feats, labels, speakers, ids)


# ---------------------------------------------------------------- B. SimulS2ST-Omni

def extract_omni(args):
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from simuls2st_omni.inference.infer_omni_talker_tts import _get_feat_extract_output_lengths
    from simuls2st_omni.train.prompt_formats import (build_prompt_bundle, build_prompt_texts,
                                                     resolve_tts_task_type)

    base = f'{_OMNI}/offline'
    proc = Qwen2_5OmniProcessor.from_pretrained(base)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16).to(args.device)
    if not hasattr(model, 'talker'):
        model.enable_talker()
    if args.lora:
        from peft import PeftModel
        model.thinker = PeftModel.from_pretrained(model.thinker, args.lora).merge_and_unload()
    model.eval()
    tok = proc.tokenizer
    a_idx = model.config.thinker_config.audio_token_index
    thinker, talker = model.thinker, model.talker
    n_thk, n_tkr = len(thinker.model.layers), len(talker.model.layers)
    h_cfg = getattr(thinker.config, 'text_config', thinker.config)
    print(f'omni thinker {n_thk}L/{h_cfg.hidden_size}d  talker {n_tkr}L', flush=True)

    cap = {}
    def _mk(key):
        def _h(_m, _i, o):
            x = o[0] if isinstance(o, tuple) else o
            cap.setdefault(key, x.detach()[0].float())
        return _h
    for i, ly in enumerate(talker.model.layers):
        ly.register_forward_hook(_mk(f'tkr{i}'))

    rows = load_rows(args.parquet)[args.shard::args.nshards]
    feats = {'thinker_aud': {f'L{i}': [] for i in range(1, n_thk + 1)},
             'thinker_ans': {f'L{i}': [] for i in range(1, n_thk + 1)},
             'talker': {f'L{i}': [] for i in range(1, n_tkr + 1)},
             'talker_aud': {f'L{i}': [] for i in range(1, n_tkr + 1)}}
    labels, speakers, ids = [], [], []
    emb_fn = thinker.get_input_embeddings()
    teos = emb_fn(torch.tensor([[talker.text_eos_token]], device=args.device))
    tpad = emb_fn(torch.tensor([[talker.text_pad_token]], device=args.device))
    tbos = emb_fn(torch.tensor([[talker.text_bos_token]], device=args.device))
    with torch.no_grad():
        for n, r in enumerate(rows, 1):
            try:
                wav = wav16_of(r)
                fb = proc.feature_extractor(wav, sampling_rate=16000,
                                             padding='longest', return_attention_mask=True)
                af = torch.from_numpy(fb['input_features']).to(args.device, torch.bfloat16)
                am = torch.from_numpy(fb['attention_mask']).to(args.device)
                k = min(af.shape[-1], am.shape[-1])
                af, am = af[..., :k], am[..., :k]
                # 音频 token 数 = CNN 下采样后帧数（_preprocess_speech L459-460）；
                # 用原始 mel 帧数会多约 2×，masked_scatter 触发 size 断言
                n_sp_token = max(1, int(_get_feat_extract_output_lengths(am.sum(dim=1))[0].item()))
                bundle = build_prompt_bundle(
                    task_type=resolve_tts_task_type(is_paired=True, ref_lang=REF_LANG,
                                                    tgt_lang=TGT_LANG),
                    text=FIXED_TEXT, target_text=FIXED_TEXT,
                    src_lang=REF_LANG, tgt_lang=TGT_LANG, n_audio_tokens=n_sp_token)
                prefix_prompt, full_prompt = build_prompt_texts(bundle)
                pfx = tok(prefix_prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
                full = tok(full_prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
                prefix_len, reply_len = pfx.shape[1], full.shape[1] - pfx.shape[1]
                assert reply_len > 0, 'reply 段为空'
                full = full.to(args.device)
                # thinker 单次整段前向（同 prefill_only L784-796）
                emb = emb_fn(full)
                af_feat = thinker.get_audio_features(af, feature_attention_mask=am.to(torch.bool))
                aud_mask = (full[0] == a_idx)
                emb = emb.masked_scatter(aud_mask.unsqueeze(0).unsqueeze(-1).expand_as(emb),
                                          af_feat.to(emb.dtype))
                tk_out = thinker.model(inputs_embeds=emb, use_cache=False,
                                       output_hidden_states=True, return_dict=True)
                ans_mask = torch.zeros_like(aud_mask)
                ans_mask[prefix_len:prefix_len + reply_len] = True      # reply = 译文段
                for i in range(1, n_thk + 1):
                    h = tk_out.hidden_states[i][0].float()
                    feats['thinker_aud'][f'L{i}'].append(_mean_masked(h, aud_mask))
                    feats['thinker_ans'][f'L{i}'].append(_mean_masked(h, ans_mask))
                # talker 单次前向（同 prefill L800-847）
                cond = tk_out.last_hidden_state + emb          # 残差式注入，保持 bf16
                prefill_cond = cond[:, :prefix_len]
                reply_cond = cond[:, prefix_len:prefix_len + reply_len]
                t_inputs = torch.cat([prefill_cond, tbos, reply_cond[:, :1]], dim=1)
                reply_part = torch.cat(
                    [reply_cond[:, 1:] if reply_cond.shape[1] > 1 else teos, teos, tpad], dim=1)
                t_ids = torch.cat([torch.full((1, prefix_len), talker.codec_mask_token,
                                               dtype=torch.long, device=args.device),
                                   torch.tensor([[talker.codec_pad_token,
                                                  talker.codec_bos_token]],
                                                device=args.device)], dim=1)
                t_text_ids = torch.cat([full[:, :prefix_len],
                                        torch.tensor([[talker.text_bos_token,
                                                       full[0, prefix_len].item()]],
                                                     device=args.device)], dim=1)
                cap.clear()
                talker(input_ids=t_ids, input_text_ids=t_text_ids,
                       thinker_reply_part=reply_part, inputs_embeds=t_inputs,
                       attention_mask=torch.ones_like(t_ids))
                # talker 位置 0..prefix_len-1 与 thinker 前缀 1:1 对应（含音频位置）
                pfx_mask = torch.zeros(t_inputs.size(1), dtype=torch.bool, device=args.device)
                pfx_mask[:prefix_len] = True
                aud_in_pfx = torch.zeros_like(pfx_mask)
                aud_in_pfx[:prefix_len] = aud_mask[:prefix_len]
                for i in range(1, n_tkr + 1):
                    h = cap[f'tkr{i-1}']
                    feats['talker'][f'L{i}'].append(_mean_masked(h, pfx_mask))
                    feats['talker_aud'][f'L{i}'].append(_mean_masked(h, aud_in_pfx))
                labels.append(r['style']); speakers.append(r['speaker_id']); ids.append(r['id'])
            except Exception as exc:
                if os.environ.get('PROBE_DEBUG'):
                    import traceback
                    traceback.print_exc()
                    raise
                print(f'  [FAIL] {r.get("id")}: {type(exc).__name__}: {exc}', flush=True)
            if n % 100 == 0 or n == len(rows):
                print(f'[omni shard {args.shard}] {n}/{len(rows)} ok={len(labels)}', flush=True)
    _save(args, feats, labels, speakers, ids)


def _save(args, feats, labels, speakers, ids):
    for part, d in feats.items():
        os.makedirs(f'{args.out}/{part}', exist_ok=True)
        np.savez_compressed(f'{args.out}/{part}/feat_shard{args.shard}.npz',
                            **{k: np.asarray(v) for k, v in d.items()},
                            labels=np.asarray(labels), speakers=np.asarray(speakers),
                            ids=np.asarray(ids))
    print(f'-> shard {args.shard} 写出（{len(labels)} 条，{len(feats)} 个位置变体）', flush=True)


def merge(args):
    parts = [p for p in sorted(os.listdir(args.out))
             if os.path.isdir(os.path.join(args.out, p))]
    for part in parts:
        shards = sorted(glob.glob(f'{args.out}/{part}/feat_shard*.npz'))
        if not shards:
            continue
        agg, labels, speakers, ids = None, [], [], []
        for p in shards:
            z = np.load(p, allow_pickle=True)
            keys = [k for k in z.files if k not in ('labels', 'speakers', 'ids')]
            if agg is None:
                agg = {k: [z[k]] for k in keys}
            else:
                for k in keys:
                    agg[k].append(z[k])
            labels += z['labels'].tolist(); speakers += z['speakers'].tolist()
            ids += z['ids'].tolist()
        merged = {k: np.concatenate(v) for k, v in agg.items()}
        np.savez_compressed(f'{args.out}/{part}/feats.npz', **merged,
                            labels=np.asarray(labels), speakers=np.asarray(speakers),
                            ids=np.asarray(ids))
        print(f'{part}: {len(shards)} 分片 -> {len(labels)} 条, {len(merged)} 层', flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('extract-si', 'extract-omni'):
        p = sub.add_parser(name)
        p.add_argument('--parquet', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--shard', type=int, required=True)
        p.add_argument('--nshards', type=int, default=8)
        p.add_argument('--device', default='cuda:0')
        if name == 'extract-si':
            p.add_argument('--ck', required=True)
        else:
            p.add_argument('--lora', default=f'{_OMNI}/simuls2st_adapter/thinker_lora')
    p = sub.add_parser('merge')
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'extract-si':
        extract_si(a)
    elif a.cmd == 'extract-omni':
        extract_omni(a)
    else:
        merge(a)


if __name__ == '__main__':
    main()
