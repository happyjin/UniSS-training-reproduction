"""Character front end and streaming units.

Characters (not espeak phonemes) are used on purpose: a phonemizer runs on the whole
sentence, which would leak future context into a prefix (handbook G2). Cleaning is a
per-character map, so cleaning a prefix equals the prefix of the cleaned sentence.

A sentence is split into arrival units: whitespace words with their attached punctuation
(the word-level incremental protocol of Ma et al. 2020 and Stephenson et al. 2021 [R40]).
Unit u owns its characters plus the single space that precedes it, as in LLM token streams (" word"):
when unit j has arrived the text is cleaned[:end_j] and nothing reveals whether the sentence goes on.
"""
import re
import unicodedata

_pad = "_"
_punctuation = ";:,.!?-\"'() "
_letters = "abcdefghijklmnopqrstuvwxyz"
symbols = [_pad] + list(_punctuation) + list(_letters)
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
SPACE_ID = _symbol_to_id[" "]
PUNCT_IDS = {_symbol_to_id[c] for c in ";:,.!?"}

_abbreviations = [(re.compile(r"\b%s\." % a, re.IGNORECASE), b) for a, b in [
    ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"),
    ("jr", "junior"), ("maj", "major"), ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"),
    ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"),
    ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort")]]


def clean_text(text):
  text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
  for regex, replacement in _abbreviations:
    text = regex.sub(replacement, text)
  text = text.lower()
  text = "".join(c for c in text if c in _symbol_to_id and c != _pad)
  text = re.sub(r"\s+", " ", text).strip()
  return text


def text_to_ids(cleaned):
  return [_symbol_to_id[c] for c in cleaned]


def intersperse(ids, item=0):
  result = [item] * (len(ids) * 2 + 1)
  result[1::2] = ids
  return result


def split_units(cleaned):
  """Return [(start, end)] character spans; each unit includes its leading space."""
  spans = []
  for m in re.finditer(r"\S+", cleaned):
    spans.append((m.start() - 1 if m.start() > 0 else 0, m.end()))
  return spans


def char_span_to_token_span(start, end, n_chars):
  """Map a character span to the interspersed token span.

  Token 2*i+1 is char i and token 2*i+2 is the blank after it; a unit owns each of its chars with the blank
  that follows (so a word owns its final blank, where word-final lengthening lands). The first unit also owns
  the leading blank, token 0. The spans of all units tile the 2*n_chars+1 tokens.
  """
  return (0 if start == 0 else 2 * start + 1), 2 * end + 1
