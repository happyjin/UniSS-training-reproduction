#!/usr/bin/env python3
"""S4（离线 S2ST joint）+ 术语约束解码。

用法与 infer_s2st.py 完全一致，只多一个 --terms：

    python infer_s2st_trie.py --model_dir <S4 ckpt> --audio a.wav --tgt_lang zh \
        --out_dir out --terms terms.json

术语表 JSON：

    {"错误术语": "正确术语", ...}
    或 {"terms": {...}}

实现要点：只替换 infer_s2st._pick_token（译文采样的唯一点），
其余（prompt 构造、talker 条件、声码器、批量、BLEU）全部复用原脚本。
约束发生在"选 token"时，错误术语不会进入序列，因此无需回退 KV cache，
且 text 路与 cond 路的 hidden 天然基于替换后的序列，语音与文本一致。
"""
import sys

from streaming_si.inference import infer_s2st as base
from streaming_si.inference.term_constraint import TermConstraint, load_rules


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


def main(argv=None):
    argv, terms = _pop_terms(sys.argv[1:] if argv is None else argv)
    if terms is None:
        sys.exit('本脚本需要 --terms <术语表.json>；不用术语约束请直接用 infer_s2st.py')

    # 只为拿 model_dir 加载 tokenizer（模型本体由 base.main 后续加载）
    args0 = base.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args0.model_dir)
    constraint = TermConstraint(tok, load_rules(terms), verbose=True)

    # 只替换译文采样的唯一点
    orig_pick = base._pick_token

    def constrained_pick(logits, temperature, top_p):
        return constraint.apply(orig_pick(logits, temperature, top_p))

    base._pick_token = constrained_pick

    # 每条音频前清空未完成的强制队列
    orig_translate = base.S2STPipeline.translate

    def translate(self, *a, **kw):
        constraint.reset()
        return orig_translate(self, *a, **kw)

    base.S2STPipeline.translate = translate

    return base.main(argv)


if __name__ == '__main__':
    main()
