"""术语约束解码（错词表 / trie）。

核心思路：在 Thinker **选 token** 的阶段干预，而不是生成完再替换文本。

当采样出的 token 命中错词表某个术语的首 token，就替换成正确术语的首 token，
并把正确术语的剩余 token 压入强制队列，后续逐步直接输出（不再采样）。

相比"生成后替换文本"的优点：

* 错误术语一个 token 都没进序列，**不需要回退 KV cache** —— 流式（S5）与
  离线（S4）都能用，Qwen3.5 的线性注意力增量状态也无需回退；
* text 路与 cond 路的 hidden 都基于**替换后**的序列计算，
  语音与文本天然一致（不会出现 cond 语义陈旧）；
* 长度自动对齐，不会踩 `gather_answer_hidden` 的 answer_mask 错位。

局限：

* 判定发生在"错误术语的首 token"。若某个非术语词与术语同首 token，会被误替换。
  选择术语时尽量让首 token 独特（例如带前导空格的英文词 " GPT"）。
* 只做 token 级替换，不做句法/形态适配（如单复数、时态），
  正确术语应当写成直接可拼接的形式。
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List


def load_rules(path: str) -> Dict[str, str]:
    """加载术语表 JSON：{"错误术语": "正确术语"} 或 {"terms": {...}}。"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get('terms'), dict):
        data = data['terms']
    if not isinstance(data, dict):
        raise SystemExit(f'{path}: 术语表需为对象 {{"错误术语": "正确术语"}}')
    return {str(k): str(v) for k, v in data.items() if str(k) != str(v)}


class TermConstraint:
    """错词表约束：bad -> good 的 token 级替换。"""

    def __init__(self, tokenizer, rules: Dict[str, str], verbose: bool = False):
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.forced: List[int] = []
        self.rules: List[tuple] = []
        self.first: Dict[int, int] = {}
        # 首 token 冲突记录：(先登记的 bad, 被顶掉的 bad, 首 token id)
        self.conflicts: List[tuple] = []
        for bad, good in rules.items():
            b = tokenizer.encode(str(bad), add_special_tokens=False)
            g = tokenizer.encode(str(good), add_special_tokens=False)
            if not b or not g:
                print(f'[term][WARN] 跳过空编码规则 {bad!r} -> {good!r}', file=sys.stderr)
                continue
            idx = len(self.rules)
            self.rules.append((b, g, str(bad), str(good)))
            ft = int(b[0])
            if ft in self.first:
                # 判定发生在 bad 的首 token，故同一首 token 只能绑定一条规则。
                # 先登记者优先，后来者永远不会通过 first 触发 —— 显式告警而非静默失效。
                owner = self.rules[self.first[ft]][2]
                self.conflicts.append((owner, str(bad), ft))
                print(f'[term][WARN] 首 token 冲突：{str(bad)!r} 与已登记的 {owner!r} '
                      f'首 token 相同（id={ft}，字符 {self._tok_str(ft)!r}）。'
                      f'该首 token 已绑定 {owner!r}，因此 {str(bad)!r} 的替换**不会生效**。'
                      f'请让两条术语的首 token 不同，或只保留其中一条。', file=sys.stderr)
            else:
                self.first[ft] = idx
        if verbose:
            print(f'[term] 载入 {len(self.rules)} 条术语规则', file=sys.stderr)
            for b, g, bt, gt in self.rules:
                dead = any(c[1] == bt for c in self.conflicts)
                print(f'  {bt!r} ({len(b)} tok) -> {gt!r} ({len(g)} tok)'
                      f'{"   [失效：首 token 冲突]" if dead else ""}', file=sys.stderr)

    def _tok_str(self, tok: int) -> str:
        """把 token id 还原成字符，让冲突提示可读。"""
        try:
            return self.tokenizer.decode([int(tok)])
        except Exception:
            return f'<id {tok}>'

    def reset(self) -> None:
        """每条音频 / 每个会话开始前调用，清空未完成的强制队列。"""
        self.forced = []

    def apply(self, tok: int) -> int:
        """对采样出的 token 做替换，返回真正要输出的 token。"""
        if self.forced:
            return int(self.forced.pop(0))
        t = int(tok)
        idx = self.first.get(t)
        if idx is None:
            return t
        _b, g, bad, good = self.rules[idx]
        self.forced = [int(x) for x in g[1:]]
        if self.verbose:
            print(f'[term] 命中 {bad!r} -> {good!r}', file=sys.stderr)
        return int(g[0])
