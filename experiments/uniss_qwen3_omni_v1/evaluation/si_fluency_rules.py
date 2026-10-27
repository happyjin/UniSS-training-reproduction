"""Fluency rules between the Thinker's commit and the Talker (docs/m2_fluency_plan_v1.zh-CN.md, steps 1 and 2).

  hygiene (step 1, on what is committed): a block that is not the sentence's last
    * does not end dangling -- on an English function word ("Choose to", "But"), on a single-character Chinese word or
      a Chinese function word ("鲁本斯的画", "叶", "这是为"); those trailing words wait for the next read;
    * is not shorter than --hyg-min units while the Talker is idle (nothing queued), unless it ends a clause;
    * is not punctuation alone.
    What is held back is not lost: the next read continues from the shorter commit and says it again.
  clean (step 2, on what the Talker says; the Thinker keeps its own words as context):
    * fillers go (uh / um / er / erm / ah / hmm / mm; 嗯 / 呃 / 额 / 唔, and 啊 opening a clause);
    * an immediately repeated word or phrase of up to three words (Chinese: separated by a comma) is said once,
      also across the seam with the block before ("Their, | their stock price").

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_rules   # self-test on the analysis's cases
"""

from __future__ import annotations

import re

CJK = re.compile(r"[一-鿿]")
CLAUSE_END = tuple(",.;:!?，。；：！？、…")
EN_FUNCTION = set(
    "a an the of to in on at for with by from and or but nor that which who whom whose is are was were be been being am "
    "this these those its as than so if then not no our your their his her my its very more most less can will would "
    "could should shall may might must do does did have has had into onto about because when while where whether what how "
    "just also really we you they i he she it there here some any each every such like both either neither all one "
    "um uh er".split())
ZH_FUNCTION = {"但是", "因为", "所以", "如果", "而且", "然后", "以及", "或者", "并且", "虽然", "就是", "还是", "不过", "那么",
               "可是", "而是", "由于", "对于", "关于", "通过", "根据", "为了", "一些", "这个", "那个", "这些", "那些", "一个",
               "这种", "那种", "我们", "他们", "你们", "它们", "可以", "应该", "需要", "已经", "正在", "非常", "比较", "其中"}
ZH_FINAL_OK = set("了吗呢吧啊呀嘛哦哈")
EN_FILLER = {"uh", "um", "er", "erm", "ah", "hmm", "mm", "mhm", "uhm"}
ZH_FILLER = re.compile(r"(?:(?<=^)|(?<=[，。,、！？；：\s]))(?:嗯|呃|额|唔)+[，,、。]?|(?:(?<=^)|(?<=[，。,、！？；：]))啊[，,、]")


def _core(w: str) -> str:
    return re.sub(r"[^\w'-]", "", w).lower()


def content_units(units: list[str], zh: bool) -> int:
    """Units that are spoken (not punctuation alone)."""
    return sum(1 for u in units if (CJK.search(u) or re.search(r"[A-Za-z0-9]", u)))


def _zh_tail_tokens(prefix: str, new: str) -> list[str]:
    """jieba tokens of prefix+new that lie inside new, in order."""
    import jieba
    toks, pos, out = jieba.lcut(prefix + new), 0, []
    for t in toks:
        if pos + len(t) > len(prefix):
            out.append(t)
        pos += len(t)
    return out


def trim_dangling(committed: list[str], new: list[str], zh: bool) -> list[str]:
    """new without the trailing words a block should not end on."""
    if not new or new[-1].endswith(CLAUSE_END):
        return new
    if not zh:
        new = list(new)
        while new and not new[-1].endswith(CLAUSE_END) and _core(new[-1]) in EN_FUNCTION:
            new.pop()
        return new
    prefix, text = "".join(committed), "".join(new)
    toks = _zh_tail_tokens(prefix, text)
    cut = len(text)
    while toks:
        t = toks[-1]
        if t[-1] in CLAUSE_END:
            break
        single = len(t) == 1 and CJK.match(t) and t not in ZH_FINAL_OK
        if not (single or t in ZH_FUNCTION):
            break
        cut -= len(t)
        toks.pop()
    cut = max(0, cut)
    return list(text[:cut])   # Chinese units are characters


def hygiene(committed: list[str], new: list[str], zh: bool, idle: bool, min_units: int) -> list[str]:
    """What of a non-final read's new units to commit now."""
    if content_units(new, zh) == 0:
        return []
    new = trim_dangling(committed, new, zh)
    if not new:
        return []
    if idle and content_units(new, zh) < min_units and not new[-1].endswith(CLAUSE_END):
        return []
    return new


def _collapse_en(words: list[str], keep_from: int) -> list[str]:
    """Drop the earlier copy of an immediately repeated 1-3 word phrase; returns the words from keep_from on."""
    w = list(words)
    i = 0
    changed = True
    while changed:
        changed = False
        for n in (3, 2, 1):
            i = 0
            while i + 2 * n <= len(w):
                a = [_core(x) for x in w[i: i + n]]
                b = [_core(x) for x in w[i + n: i + 2 * n]]
                if a == b and w[i + n - 1].endswith((".", "?", "!")):   # "...copyright. Copyright is": a new sentence
                    i += 1
                elif a == b and all(a) and i + 2 * n <= keep_from:   # both copies already spoken
                    i += 1
                elif a == b and all(a) and i < keep_from:   # the first copy was spoken: drop the new one
                    if i + n < keep_from:
                        i += 1
                        continue
                    del w[i + n: i + 2 * n]
                    changed = True
                elif a == b and all(a):
                    cap = w[i][:1].isupper()
                    del w[i: i + n]
                    if i < keep_from:
                        keep_from = max(i, keep_from - n)
                    if cap and w[i][:1].islower():
                        w[i] = w[i][:1].upper() + w[i][1:]
                    changed = True
                else:
                    i += 1
    return w[keep_from:]


def clean_en(spoken_tail: str, text: str) -> str:
    lead = " " if text[:1].isspace() else ""
    ctx = spoken_tail.split()[-6:]
    words = text.split()
    kept, cap_next = [], False
    for j, x in enumerate(words):
        if _core(x) in EN_FILLER:
            p = re.sub(r"^[\w'-]+", "", x)   # a filler's own punctuation: keep a sentence end on the word before
            if p and p[-1] in ".?!" and kept:
                kept[-1] = kept[-1].rstrip(",;:") + p[-1]
            elif p.endswith(",") and kept and kept[-1].endswith(","):   # "very, uh, interpretive" -> "very interpretive"
                kept[-1] = kept[-1][:-1]
            prev = kept[-1] if kept else (ctx[-1] if ctx else "")
            if x[:1].isupper() and (not prev or prev.endswith((".", "?", "!"))):
                cap_next = True
            continue
        if cap_next and x[:1].islower():
            x = x[:1].upper() + x[1:]
        cap_next = False
        kept.append(x)
    out = _collapse_en(ctx + kept, len(ctx))
    if not out:
        return ""
    if kept and out and kept[0][:1].isupper() and out[0][:1].islower():
        out[0] = out[0][:1].upper() + out[0][1:]
    if not ctx or ctx[-1].endswith((".", "?", "!", ",")):
        out[0] = out[0].lstrip(",;:") or out[0]
    return lead + " ".join(out) if spoken_tail else " ".join(out)


ZH_PUNCT = "，,、。！？；：?!"


def clean_zh(spoken_tail: str, text: str) -> str:
    """Only text is edited; what was spoken is context (its last character decides where a clause starts)."""
    c = spoken_tail[-1:]
    s = ZH_FILLER.sub("", c + text)
    s = s[1:] if c and s.startswith(c) else s   # c consumed: it was a filler whose punctuation opened this block
    # 是的,是的。-> 是的。; 复制,复制,对吧 -> 复制,对吧: a repeat standing alone before punctuation (not 谢谢, which has no
    # comma, nor 电流,电流通过, where the second copy starts a clause)
    s = re.sub(r"([一-鿿]{1,4})[，,、]\1(?=[，,、。！？；：?!]|$)", r"\1", s)
    said = spoken_tail.rstrip("，,、")
    if said != spoken_tail:   # the same across the seam: "复制," | "复制,对吧"
        for n in (4, 3, 2, 1):
            x = s[:n]
            if len(x) == n and all(CJK.match(ch) for ch in x) and said.endswith(x) and (len(s) == n or s[n] in ZH_PUNCT):
                s = s[n:]
                break
    if not spoken_tail or spoken_tail[-1] in ZH_PUNCT:
        s = s.lstrip("，,、；：")
    return s


def clean(spoken: str, text: str, zh: bool) -> str:
    """The Talker's text for a new block, given everything spoken before it."""
    return clean_zh(spoken, text) if zh else clean_en(spoken, text)


def _selftest() -> None:
    zh = lambda s: list(s)
    en = lambda s: s.split()
    cases = [
        (trim_dangling(zh(""), zh("轮"), True), []),
        (trim_dangling(zh(""), zh("它不是鲁本斯的画"), True), zh("它不是鲁本斯")),
        (trim_dangling(zh("那看起来像"), zh("叶"), True), []),
        (trim_dangling(zh("我们可以说,"), zh("这是为那些"), True), []),   # 这 / 是 / 为 / 那些: all dangling
        (trim_dangling(zh(""), zh("轮转负载均衡"), True), zh("轮转负载均衡")),
        (trim_dangling(zh(""), zh("方案。"), True), zh("方案。")),
        (trim_dangling(en(""), en("Choose to"), False), en("Choose")),
        (trim_dangling(en(""), en("But"), False), []),
        (trim_dangling(en("Yes, I"), en("think"), False), en("think")),
        (hygiene(zh(""), zh(","), True, True, 3), []),
        (hygiene(en(""), en("Choose to"), False, True, 2), []),
        (hygiene(en(""), en("Choose to"), False, False, 2), en("Choose")),
        (hygiene(zh(""), zh("所以我不确定"), True, True, 3), zh("所以我不确定")),
        (clean("", "Then um, he, uh, has trade to manipulate.", False), "Then he has trade to manipulate."),
        (clean("", "Uh, mainly because, um, that's uh, some manipulation.", False), "Mainly because that's some manipulation."),
        (clean("", "It is called copyright. Copyright is actually what you have.", False),
         "It is called copyright. Copyright is actually what you have."),
        (clean("", "它们用来产生电流,电流通过海绵内部的管子流动。", True), "它们用来产生电流,电流通过海绵内部的管子流动。"),
        (clean("", "你的收入不重要,重要的是你内心的故事。", True), "你的收入不重要,重要的是你内心的故事。"),
        (clean("", "也许三倍复制,复制,对吧?", True), "也许三倍复制,对吧?"),
        (clean("嗯", ",我想说", True), "我想说"),
        (clean("话虽如此,截至一个月前,", "当我参加", True), "当我参加"),
        (clean("也许三倍复制,", "复制,对吧?", True), "对吧?"),
        (clean("产生电流,", "电流通过管子。", True), "电流通过管子。"),
        (clean("Then he, has trade to manipulate.", " Their, their, uh, their stock price.", False), " Their stock price."),
        (clean("Ah, just", " as we are as a whole, uh, uh, uh, department, the whole", False),
         " as we are as a whole department, the whole"),
        (clean("Their,", " their stock price.", False), " stock price."),
        (clean("", "是的,是的,我想这回答了我的问题。", True), "是的,我想这回答了我的问题。"),
        (clean("", "嗯,嗯。给我,给我一点时间。我,我", True), "给我,给我一点时间。我"),   # 给我,给我一点 is 电流,电流通过
        (clean("", "谢谢大家", True), "谢谢大家"),
        (clean("", "Yes, yes, I think I did.", False), "Yes, I think I did."),
    ]
    bad = [(i, got, want) for i, (got, want) in enumerate(cases) if got != want]
    for i, got, want in bad:
        print(f"case {i}: got {got!r} want {want!r}")
    print("selftest", "OK" if not bad else f"{len(bad)} failed", f"({len(cases)} cases)")


def clean_blocks_file(src: str, dst: str, zh: bool) -> None:
    """An si_thinker_stream blocks file with every block's text cleaned as --clean would (commits unchanged)."""
    import json
    from pathlib import Path

    if Path(dst).exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    with open(src, encoding="utf-8") as fi, open(dst, "x", encoding="utf-8") as fo:
        for line in fi:
            r = json.loads(line)
            spoken, blocks = "", []
            for b in r["blocks"]:
                t = clean(spoken, b["text"], zh)
                spoken += t
                if content_units(list(t) if zh else t.split(), zh) or b["final"]:
                    blocks.append({**b, "text": t, "raw": b["text"]})
            if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:
                blocks = blocks[:-1]
                blocks[-1]["final"] = True
            fo.write(json.dumps({**r, "blocks": blocks, "spoken": spoken}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4 and sys.argv[1] in ("--clean-blocks-zh", "--clean-blocks-en"):
        clean_blocks_file(sys.argv[2], sys.argv[3], sys.argv[1].endswith("zh"))
    else:
        _selftest()
