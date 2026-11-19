#!/usr/bin/env python3
"""给「官方 Omni 仓库的本地副本」打运行期补丁（只改 /tmp 里的副本，不碰资产原目录）。

目前两组补丁，都由环境变量打开（不设则完全不动，行为与 README_self 一致）：

  SIMULS2ST_FM_STEPS        VoiceBox flow matching 的 Euler 步数（默认 16）
  SIMULS2ST_FM_CFG          CFG 强度（默认 2.0；0 表示不做 CFG，可省一半前向）
  SIMULS2ST_FM_RESCALE_CFG  默认 0.75
  SIMULS2ST_VOICEBOX_DTYPE  bf16|fp16：把 VoiceBox 与 Vocos 转成半精度（默认 fp32）

用法：patch_omni_repo.py <staged_repo_dir>
"""
import os
import sys
from pathlib import Path


def _sub(text: str, old: str, new: str, path: Path) -> str:
    if new in text:
        return text                      # 已打过
    if old not in text:
        raise SystemExit(f'[patch] 在 {path} 里找不到锚点：{old[:60]!r}')
    return text.replace(old, new, 1)


def main() -> None:
    repo = Path(sys.argv[1])
    steps = os.environ.get('SIMULS2ST_FM_STEPS', '').strip()
    cfg = os.environ.get('SIMULS2ST_FM_CFG', '').strip()
    rescale = os.environ.get('SIMULS2ST_FM_RESCALE_CFG', '').strip()
    dtype = os.environ.get('SIMULS2ST_VOICEBOX_DTYPE', '').strip().lower()

    if os.environ.get('SIMULS2ST_TALKER_ACCEL') or os.environ.get('SIMULS2ST_TRACE_CODES'):
        p = repo / 'src' / 'inference' / 'infer_omni_talker_tts.py'
        t = p.read_text(encoding='utf-8')
        for name, end, cache_only in [
            ('_talker_forward_prefill', '_talker_forward_step', True),
            ('_talker_forward_step', '_tail_cond_for_current_chunk', False),
        ]:
            start = t.index(f'        def {name}(')
            stop = t.index(f'        def {end}(', start)
            block = t[start:stop]
            if 'omni_talker_accel import forward' not in block:
                hook = (
                    '            if os.environ.get("SIMULS2ST_TALKER_ACCEL") in {"static", "graph"}:\n'
                    '                from src.inference.omni_talker_accel import forward as _accelerated\n'
                    '                _out = _accelerated(self.talker, input_ids=input_ids,\n'
                    '                    input_text_ids=input_text_ids, inputs_embeds=inputs_embeds,\n'
                    '                    past_key_values=past_key_values, position_ids=position_ids,\n'
                    '                    use_cache=True, return_dict=True)\n'
                    + ('                return _out.past_key_values\n' if cache_only else '                return _out\n')
                )
                block = block.replace('            seq_len = int(input_ids.shape[1])\n',
                                      hook + '            seq_len = int(input_ids.shape[1])\n', 1)
                t = t[:start] + block + t[stop:]
        trace = '            return generated_codes_tensor, next_past_key_values\n'
        if '[talker-codes]' not in t:
            t = _sub(t, trace,
                '            if os.environ.get("SIMULS2ST_TRACE_CODES") == "1":\n'
                '                print("[talker-codes] " + str(generated_code_ids), flush=True)\n' + trace, p)
        p.write_text(t, encoding='utf-8')
        helper = Path(__file__).with_name('omni_talker_accel.py')
        (p.parent / helper.name).write_text(helper.read_text(), encoding='utf-8')
        print('[patch] opt-in static cache / CUDA Graph + code tracing')

    if steps or cfg or rescale:
        p = repo / 'src' / 'inference' / 'generate_tts_offline.py'
        t = p.read_text(encoding='utf-8')
        if steps:
            t = _sub(t, '        n_timesteps: int = 16,',
                     '        n_timesteps: int = int(os.environ.get("SIMULS2ST_FM_STEPS", "16")),', p)
        if cfg:
            t = _sub(t, '        cfg_scale: float = 2.0,',
                     '        cfg_scale: float = float(os.environ.get("SIMULS2ST_FM_CFG", "2.0")),', p)
        if rescale:
            t = _sub(t, '        rescale_cfg: float = 0.75,',
                     '        rescale_cfg: float = float(os.environ.get("SIMULS2ST_FM_RESCALE_CFG", "0.75")),', p)
        p.write_text(t, encoding='utf-8')
        print(f'[patch] codes_to_mel 默认值：steps={steps or "16"} cfg={cfg or "2.0"} rescale={rescale or "0.75"}')

    if dtype in {'bf16', 'bfloat16', 'fp16', 'float16'}:
        p = repo / 'src' / 'inference' / 'infer_omni_talker_tts.py'
        t = p.read_text(encoding='utf-8')
        anchor = '        accelerate.load_checkpoint_and_dispatch(vocoder, vocos_path)\n'
        add = anchor + (
            '        _dt_name = os.environ.get("SIMULS2ST_VOICEBOX_DTYPE", "").strip().lower()\n'
            '        if _dt_name in {"bf16", "bfloat16", "fp16", "float16"}:\n'
            '            _dt = torch.bfloat16 if _dt_name.startswith("bf") else torch.float16\n'
            '            voicebox.to(device=device, dtype=_dt)\n'
            '            vocoder.to(device=device, dtype=_dt)\n'
            '            logger.info("VoiceBox / Vocos cast to %s (SIMULS2ST_VOICEBOX_DTYPE)", _dt)\n')
        t = _sub(t, anchor, add, p)
        p.write_text(t, encoding='utf-8')
        print(f'[patch] VoiceBox / Vocos -> {dtype}')

        # mel 转音频时跟随 vocoder 的 dtype
        p2 = repo / 'src' / 'inference' / 'generate_tts_offline.py'
        t2 = p2.read_text(encoding='utf-8')
        old = ('        audio = self.vocoder(mel.transpose(1, 2)).detach().cpu().numpy()[0][0]\n')
        new = ('        _w = next(self.vocoder.parameters(), None)\n'
               '        if _w is not None and mel.dtype != _w.dtype:\n'
               '            mel = mel.to(dtype=_w.dtype)\n'
               '        audio = self.vocoder(mel.transpose(1, 2)).detach().cpu().numpy()[0][0]\n')
        if '_w = next(self.vocoder.parameters()' not in t2:
            t2 = _sub(t2, old, new, p2)
            p2.write_text(t2, encoding='utf-8')
        # prompt_mel 也要跟随 cond_feature 的 dtype
        old2 = ('        if prompt_codes is not None:\n'
                '            prompt_codes = prompt_codes.clamp(0, SEMANTIC_CODE_VOCAB_SIZE - 1)\n')
        if 'prompt_mel = prompt_mel.to(dtype=cond_feature.dtype)' not in t2:
            t2 = t2.replace(
                '        # Generate mel using reverse diffusion\n',
                '        if prompt_mel is not None and prompt_mel.dtype != cond_feature.dtype:\n'
                '            prompt_mel = prompt_mel.to(dtype=cond_feature.dtype)\n'
                '        # Generate mel using reverse diffusion\n', 1)
            p2.write_text(t2, encoding='utf-8')
        print('[patch] mel/prompt_mel dtype 跟随模块')

        # DiT 内部 diffusion step 的正弦嵌入是纯 float32 计算，转半精度后要跟上
        p3 = repo / 'src' / 'model' / 'llama_nar.py'
        t3 = p3.read_text(encoding='utf-8')
        old3 = '        diffusion_step = self.diff_step_embedding(diffusion_step).to(x.device)\n'
        new3 = '        diffusion_step = self.diff_step_embedding(diffusion_step).to(device=x.device, dtype=x.dtype)\n'
        if new3 not in t3:
            t3 = _sub(t3, old3, new3, p3)
            p3.write_text(t3, encoding='utf-8')
            print('[patch] DiT 的 diffusion_step 嵌入跟随 x.dtype')

    if os.environ.get('SIMULS2ST_FAST_TALKER', '').strip() in {'1', 'true', 'yes'}:
        # talker 的逐 token 解码里，repetition penalty 与 no-repeat-ngram 是「把整条序列拉回 Python、
        # 对词表做标量循环」的写法：每生成一个 code 要发上百次 kernel，实测约 32ms/token。
        # 这里改成等价的向量化写法（语义不变）：unique/index_copy_ 与 unfold 匹配。
        p5 = repo / 'src' / 'inference' / 'infer_omni_talker_tts.py'
        t5 = p5.read_text(encoding='utf-8')
        old_rp = '''                if repetition_penalty_talker != 1.0:
                    for prev_token in set(int(x) for x in manual_sequence[0].tolist()):
                        if scores[0, prev_token] < 0:
                            scores[0, prev_token] *= repetition_penalty_talker
                        else:
                            scores[0, prev_token] /= repetition_penalty_talker
'''
        new_rp = '''                if repetition_penalty_talker != 1.0:
                    _prev = torch.unique(manual_sequence[0])
                    _vals = scores[0].index_select(0, _prev)
                    _vals = torch.where(
                        _vals < 0,
                        _vals * repetition_penalty_talker,
                        _vals / repetition_penalty_talker,
                    )
                    scores[0].index_copy_(0, _prev, _vals)
'''
        old_ng = '''                if no_repeat_ngram_talker > 0 and manual_sequence.shape[1] + 1 >= no_repeat_ngram_talker:
                    prefix = tuple(manual_sequence[0, -(no_repeat_ngram_talker - 1) :].tolist())
                    seq = manual_sequence[0].tolist()
                    banned = []
                    for i in range(len(seq) - no_repeat_ngram_talker + 1):
                        if tuple(seq[i : i + no_repeat_ngram_talker - 1]) == prefix:
                            banned.append(seq[i + no_repeat_ngram_talker - 1])
                    if banned:
                        scores[0, banned] = float("-inf")
'''
        new_ng = '''                if no_repeat_ngram_talker > 1 and manual_sequence.shape[1] + 1 >= no_repeat_ngram_talker:
                    _n = no_repeat_ngram_talker
                    _seq = manual_sequence[0]
                    _prefix = _seq[-(_n - 1) :]
                    _win = _seq.unfold(0, _n - 1, 1)[: _seq.numel() - _n + 1]
                    _hit = (_win == _prefix).all(dim=1).nonzero(as_tuple=True)[0]
                    if _hit.numel():
                        scores[0].index_fill_(0, _seq.index_select(0, _hit + (_n - 1)), float("-inf"))
'''
        if 'torch.unique(manual_sequence[0])' in t5:
            print('[patch] fast talker 已打过')
        else:
            t5 = _sub(t5, old_rp, new_rp, p5)
            t5 = _sub(t5, old_ng, new_ng, p5)
            p5.write_text(t5, encoding='utf-8')
            print('[patch] fast talker：repetition penalty / n-gram 向量化')

    if os.environ.get('SIMULS2ST_PROFILE_TALKER', '').strip() in {'1', 'true', 'yes'}:
        # 用 torch.profiler 抓 talker 单步前向里的热点 op（只抓前 6 步）
        p6 = repo / 'src' / 'inference' / 'infer_omni_talker_tts.py'
        t6 = p6.read_text(encoding='utf-8')
        old_call = '''            return self.talker(
                input_ids=input_ids,
                input_text_ids=input_text_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
'''
        new_call = '''            if os.environ.get("SIMULS2ST_PROFILE_TALKER") and getattr(self, "_prof_n", 0) < 6:
                self._prof_n = getattr(self, "_prof_n", 0) + 1
                import torch.profiler as _tp
                with _tp.profile(activities=[_tp.ProfilerActivity.CUDA, _tp.ProfilerActivity.CPU]) as _prof:
                    _out = self.talker(
                        input_ids=input_ids,
                        input_text_ids=input_text_ids,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        use_cache=True,
                        return_dict=True,
                    )
                    torch.cuda.synchronize()
                print(f"[prof-talker-step {self._prof_n}] seq={past_seen}", file=sys.stderr, flush=True)
                print(_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12), file=sys.stderr, flush=True)
                return _out
            return self.talker(
                input_ids=input_ids,
                input_text_ids=input_text_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
'''
        if '[prof-talker-step' in t6:
            print('[patch] talker profiler 已打过')
        else:
            t6 = _sub(t6, old_call, new_call, p6)
            p6.write_text(t6, encoding='utf-8')
            print('[patch] talker 单步 profiler')

    if os.environ.get('SIMULS2ST_PROFILE', '').strip() in {'1', 'true', 'yes'}:
        p = repo / 'src' / 'agents' / 'simuleval_omni_talker_s2st_agent.py'
        t = p.read_text(encoding='utf-8')
        if '[prof-synth]' not in t:
            if '\nimport time\n' not in t:
                t = t.replace('import sys\n', 'import sys\nimport time\n', 1)
            t = _sub(t, '        ref_codes = self.inference.audio_tokenizer(ref_audio_16k.numpy())',
                     '        torch.cuda.synchronize(); _tp = time.perf_counter()\n'
                     '        ref_codes = self.inference.audio_tokenizer(ref_audio_16k.numpy())', p)
            t = _sub(t, '        mel = self.inference.codes_to_mel(\n',
                     '        torch.cuda.synchronize(); _tf = time.perf_counter()\n'
                     '        mel = self.inference.codes_to_mel(\n', p)
            t = _sub(t, '        return self.inference.mel_to_audio(mel)\n',
                     '        torch.cuda.synchronize(); _tv = time.perf_counter()\n'
                     '        _audio = self.inference.mel_to_audio(mel)\n'
                     '        torch.cuda.synchronize(); _te = time.perf_counter()\n'
                     '        print(f"[prof-synth] prompt={(_tf-_tp)*1000:.1f} fm={(_tv-_tf)*1000:.1f} "\n'
                     '              f"vocos={(_te-_tv)*1000:.1f} codes={generated_codes.shape[1]}",\n'
                     '              file=sys.stderr, flush=True)\n'
                     '        return _audio\n', p)
            p.write_text(t, encoding='utf-8')
        print('[patch] 打开 SIMULS2ST_PROFILE：_synthesize_codes 分阶段计时')
        # 在 policy 入口记一个起点：gen ≈ 特征提取 + thinker + talker（到 synth 开始为止）
        t = p.read_text(encoding='utf-8')
        anchor2 = '        if states is None:\n            states = self.states\n'
        add2 = anchor2 + ('        if os.environ.get("SIMULS2ST_PROFILE"):\n'
                          '            self._tp_policy = time.perf_counter()\n')
        if '_tp_policy' not in t:
            t = _sub(t, anchor2, add2, p)
            t = t.replace(
                'f"vocos={(_te-_tv)*1000:.1f} codes={generated_codes.shape[1]}",',
                'f"vocos={(_te-_tv)*1000:.1f} codes={generated_codes.shape[1]} "\n'
                '              f"gen={(_tp-getattr(self, \'_tp_policy\', _tp))*1000:.1f}",', 1)
            p.write_text(t, encoding='utf-8')
            print('[patch] policy 入口计时（gen = 特征+thinker+talker）')

        # talker / thinker 各自耗时（同一次 generate 内部分开计时）
        p4 = repo / 'src' / 'inference' / 'infer_omni_talker_tts.py'
        t4 = p4.read_text(encoding='utf-8')
        if '[prof-gen]' not in t4:
            if '\nimport time\n' not in t4:
                t4 = t4.replace('import sys\n', 'import sys\nimport time\n', 1)
            t4 = _sub(t4, '        talker_result = self.talker.generate(\n',
                     '        torch.cuda.synchronize(); _tt0 = time.perf_counter()\n'
                     '        talker_result = self.talker.generate(\n', p4)
            t4 = _sub(t4, '        talker_sequences = talker_result.sequences if hasattr(talker_result, "sequences") else talker_result\n',
                     '        torch.cuda.synchronize(); _tt1 = time.perf_counter()\n'
                     '        print(f"[prof-gen] talker={(_tt1-_tt0)*1000:.1f}ms "\n'
                     '              f"new_tokens={talker_result.sequences.shape[-1]-talker_input_ids.shape[-1]}",\n'
                     '              file=sys.stderr, flush=True)\n'
                     '        talker_sequences = talker_result.sequences if hasattr(talker_result, "sequences") else talker_result\n', p4)
            t4 = _sub(t4, '        thinker_result = self.thinker.generate(**thinker_kwargs)\n',
                     '        torch.cuda.synchronize(); _tth0 = time.perf_counter()\n'
                     '        thinker_result = self.thinker.generate(**thinker_kwargs)\n'
                     '        torch.cuda.synchronize()\n'
                     '        print(f"[prof-gen] thinker={(time.perf_counter()-_tth0)*1000:.1f}ms", file=sys.stderr, flush=True)\n', p4)
            p4.write_text(t4, encoding='utf-8')
            print('[patch] talker / thinker 计时')


if __name__ == '__main__':
    main()
