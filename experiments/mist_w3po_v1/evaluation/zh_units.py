"""Latency / lattice units of Chinese output text (en2zh; reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §5.3).

The zh2en tools count English words with str.split(). For Chinese a unit is one CJK character, and a Latin / digit
run (an English word or number inside the Chinese text) is one unit, as sacrebleu's zh tokenisation treats them;
punctuation is not a unit: it is attached to the preceding unit (to the following one at the start of a text).

  units(s)    the units of s (punctuation dropped)
  n_units(s)  len(units(s))
  charify(s)  s rewritten so that str.split() yields exactly one token per unit (punctuation kept, attached), i.e.
              len(charify(s).split()) == n_units(s) whenever s has a unit; idempotent; a text with no unit
              becomes its non-space characters (one token, or "" when empty)
"""
from __future__ import annotations

import re

UNIT = re.compile(r"[㐀-鿿豈-﫿]|[A-Za-z0-9]+(?:['’.\-][A-Za-z0-9]+)*")


def units(s: str) -> list[str]:
    return UNIT.findall(s or "")


def n_units(s: str) -> int:
    return len(units(s))


def charify(s: str) -> str:
    s = s or ""
    toks, last, lead = [], 0, ""
    for m in UNIT.finditer(s):
        g = "".join(s[last: m.start()].split())
        if toks:
            toks[-1] += g
        else:
            lead += g
        toks.append(m.group())
        last = m.end()
    if not toks:
        return "".join(s.split())
    toks[0] = lead + toks[0]
    toks[-1] += "".join(s[last:].split())
    return " ".join(toks)
