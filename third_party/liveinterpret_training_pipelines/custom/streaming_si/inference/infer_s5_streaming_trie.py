#!/usr/bin/env python3
"""S5（流式同传 LoRA）+ 术语约束解码。

用法与 infer_s5_streaming.py 一致，只多一个 --terms：

    python infer_s5_streaming_trie.py --model_dir <S4 底座> --adapter_dir <S5 adapter> \
        --audio a.wav --tgt_lang zh --out_dir out --terms terms.json

术语表 JSON：

    {"错误术语": "正确术语", ...}
    或 {"terms": {...}}

实现要点：只包装 sample_delta 的采样函数（译文 token 采样点）。
决策步（wait / W / eos）走的是原始 pick，不受约束影响；
其余流式逻辑（chunk 推进、滑窗、talker 条件、时间线）全部复用原脚本。

约束发生在"选 token"时，因此不需要回退 KV cache ——
这对 Qwen3.5 的线性注意力增量状态尤其重要（它无法回退）。
"""
import json as _json
import os
import sys

from streaming_si.inference import infer_s5_streaming as base
from streaming_si.inference.term_constraint import TermConstraint, load_rules
from streaming_si.data.s4_pipeline import add_s4_stream_tokens


def _pop_terms(argv):
    """从 argv 摘出 --terms（原脚本不认识该参数），返回 (argv, terms_path)。"""
    argv = list(argv)
    if '--terms' not in argv:
        return argv, None
    i = argv.index('--terms')
    if i + 1 >= len(argv):
        sys.exit('--terms 需要一个 JSON 路径')
    terms = argv[i + 1]
    del argv[i:i + 2]
    return argv, terms


def _vocab_size(model_dir, tokenizer):
    """与 base.load_model 一致：容量取 config.vocab_size（4B 变体 248k）。"""
    with open(os.path.join(model_dir, 'config.json'), encoding='utf-8') as f:
        cfg = _json.load(f)
    v = cfg.get('vocab_size')
    if v is None:
        v = (cfg.get('text_config') or {}).get('vocab_size')
    if v is None:
        v = getattr(tokenizer, 'vocab_size', 151936)
    return int(v)


def main(argv=None):
    argv, terms = _pop_terms(sys.argv[1:] if argv is None else argv)
    if terms is None:
        sys.exit('本脚本需要 --terms <术语表.json>；不用术语约束请直接用 infer_s5_streaming.py')

    args0 = base.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args0.model_dir)
    # 与 base.load_model 同序：先补流式特殊 token，再编码术语
    add_s4_stream_tokens(tok, capacity=_vocab_size(args0.model_dir, tok))

    constraint = TermConstraint(tok, load_rules(terms), verbose=True)

    # 只包装 sample_delta 的采样函数
    orig_sd = base.sample_delta

    def sample_delta(embeds, run, tok_emb, pick, chunk_id, eos_id, wait_id, **kw):
        def cpick(logits, temperature, top_p, ban_above=None, allow=()):
            if constraint.forced:
                return constraint.forced.pop(0)
            t = pick(logits, temperature, top_p, ban_above=ban_above, allow=allow)
            return constraint.apply(t)
        return orig_sd(embeds, run, tok_emb, cpick, chunk_id, eos_id, wait_id, **kw)

    base.sample_delta = sample_delta

    # 每条音频（每次 stream_translate）前清空强制队列
    orig_st = base.stream_translate

    def stream_translate(*a, **kw):
        constraint.reset()
        return orig_st(*a, **kw)

    base.stream_translate = stream_translate

    return base.main(argv)


if __name__ == '__main__':
    main()
