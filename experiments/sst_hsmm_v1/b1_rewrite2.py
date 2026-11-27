"""change 57 B1, rewrite rule v2 (v1 = b1_rewrite.py failed the correctness check: ~22 % of 60 sampled rewrites broken). Same goal:
rare English proper names stay in Latin script in the en2zh training references (open cvdev convention). Changes vs v1:
  * spans = maximal runs of capitalised tokens that Stanza (en, pretokenised on the UniST source words) tags PROPN; tokens with a
    demonym / -ism suffix are excluded; sentence-initial PROPN allowed.
  * keep-in-English decision on the span's LAST token (the name head): lower-cased count < RARE in the English UniST transcriptions;
    the WHOLE span is then replaced (no half-replaced "罗纳德·Reagan").
  * target side works on OmniAlign target words: the linked run (confidence >= 0.5) must be contiguous and linked only to the span;
    it is extended over ADJACENT target words that have no link outside the span and consist only of transliteration characters
    (+ '·'); the final run must consist only of transliteration characters (+ '·'), so meaning translations (皇后路, 清教主义,
    环球旅行者) and partial runs are never replaced. Transliteration characters = data-derived (b1prep/translit_chars_v2.json,
    share in aligned capitalised-name runs >= 0.30 over 263k English UniST rows) minus a hand blacklist of meaning characters plus a
    hand whitelist of standard transliteration characters below the cut.
    python -m experiments.sst_hsmm_v1.b1_rewrite2 --out eval_outputs/sst_hsmm_v1/b1prep/v2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
ROWS = REPO / "data/unist_simul_v1/rows"
PREP = REPO / "eval_outputs/sst_hsmm_v1/b1prep"
QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
STANZA_LIB = "/opt/dlami/nvme/jasonleeeli/pylibs/stanza_v1"
STANZA_RES = "/opt/dlami/nvme/jasonleeeli/cache/stanza_resources"
RARE = 15
BLACK = set("国英州帝洲院联星月党先皇圣徒委县郡港叔爸姐嗯夕陛诞勋盛京湾宪侠晤锦凤凰雀犬虎墓馆岛河海军日南北语省协洋峡裔姑姨妈哦拌盟邦督爵印欧牙华澳荷俄犹腊瑞匈挪葡萄韩刘廖侯魁萧肖氏")
WHITE = set("加达安查拿密古佛索皮康劳戴登典伍夏麦耳泽扎堪叙惠昆舍捷甘翁陀嫩契嘉蕾伐唐朱汤杰")
SUFFIX = ("ism", "isms", "ist", "ists", "ian", "ians", "ese", "ans", "ish", "ic", "ics")
PRON = {"i", "i'm", "i'll", "i've", "i'd"}


def clean_tok(w):
    w = re.sub(r"(['’]s|['’])$", "", w.strip())
    return w.strip(".,;:!?\"“”()[]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    for x in ("train_en2zh.jsonl", "changes.jsonl"):
        if (out / x).exists():
            sys.exit(f"exists: {out / x} (refuse to overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    freq = json.load(open(PREP / "en_freq.json"))["count"]
    TR = (set(json.load(open(PREP / "translit_chars_v2.json"))["chars"]) - BLACK) | WHITE
    rows = [json.loads(l) for l in open(DATA / "train_en2zh.jsonl", encoding="utf-8")]
    want = {r["id"] for r in rows}
    U = {}
    for f in sorted(ROWS.glob("part-*.parquet")):
        for r in pq.read_table(f, columns=["id", "source_words", "target_words", "word_links"]).to_pylist():
            if r["id"] in want and r["id"] not in U and r["word_links"]:
                U[r["id"]] = r
    sys.path.insert(0, STANZA_LIB)
    import stanza
    nlp = stanza.Pipeline("en", dir=STANZA_RES, processors="tokenize,pos", tokenize_pretokenized=True, use_gpu=False,
                          download_method=None, verbose=False)
    order = [r["id"] for r in rows if r["id"] in U]
    SW = {i: [w["text"] for w in json.loads(U[i]["source_words"])] for i in order}
    POS = {}
    B = 256
    for k in range(0, len(order), B):
        ids = [i for i in order[k:k + B] if SW[i]]
        doc = nlp([SW[i] for i in ids])
        for i, s in zip(ids, doc.sentences):
            POS[i] = [w.upos for w in s.words]
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(QWEN3))
    n_ch = n_sp = 0
    rej = {"pos_len": 0, "not_contig": 0, "shared_link": 0, "non_translit": 0, "not_unique": 0, "no_link": 0}
    with open(out / "train_en2zh.jsonl", "x", encoding="utf-8") as fo, open(out / "changes.jsonl", "x", encoding="utf-8") as fc:
        for r in rows:
            i = r["id"]
            new, done = r["ref"], []
            if i in POS and len(POS[i]) == len(SW[i]):
                sw, pos = SW[i], POS[i]
                tw = [w["text"] for w in json.loads(U[i]["target_words"])]
                L = [(l["source_index"], l["target_index"]) for l in json.loads(U[i]["word_links"]) if l.get("confidence", 0) >= 0.5]
                ok = [bool(pos[k] == "PROPN" and clean_tok(sw[k]) and clean_tok(sw[k])[0].isupper()
                           and re.match(r"^[A-Za-z][A-Za-z\-]*$", clean_tok(sw[k])) and clean_tok(sw[k]).lower() not in PRON
                           and not clean_tok(sw[k]).lower().endswith(SUFFIX)) for k in range(len(sw))]
                k = 0
                while k < len(sw):
                    if not ok[k]:
                        k += 1; continue
                    j = k
                    while j + 1 < len(sw) and ok[j + 1]:
                        j += 1
                    span = (k, j); k = j + 1
                    if freq.get(clean_tok(sw[span[1]]).lower(), 0) >= RARE:
                        continue
                    a_, b_ = span
                    T = sorted({t for s, t in L if a_ <= s <= b_})
                    if not T:
                        rej["no_link"] += 1; continue
                    if T != list(range(T[0], T[-1] + 1)):
                        rej["not_contig"] += 1; continue
                    if any(not (a_ <= s <= b_) for s, t in L if t in T):
                        rej["shared_link"] += 1; continue
                    linked_out = lambda t: any(not (a_ <= s <= b_) for s, t2 in L if t2 == t)
                    is_tr = lambda w: all(c in TR or c == "·" for c in w) and any(c in TR for c in w)
                    lo, hi = T[0], T[-1]
                    while lo - 1 >= 0 and not linked_out(lo - 1) and is_tr(tw[lo - 1]):
                        lo -= 1
                    while hi + 1 < len(tw) and not linked_out(hi + 1) and is_tr(tw[hi + 1]):
                        hi += 1
                    zh = "".join(tw[lo:hi + 1]).strip("·")
                    if not is_tr(zh):
                        rej["non_translit"] += 1; continue
                    if new.count(zh) != 1:
                        rej["not_unique"] += 1; continue
                    en = " ".join(clean_tok(sw[q]) for q in range(a_, b_ + 1))
                    p = new.index(zh)
                    pre = " " if p > 0 and re.match(r"[一-鿿]", new[p - 1]) else ""
                    post = " " if p + len(zh) < len(new) and re.match(r"[一-鿿]", new[p + len(zh)]) else ""
                    new = new[:p] + pre + en + post + new[p + len(zh):]
                    done.append({"en": en, "zh": zh})
            elif i in U:
                rej["pos_len"] += 1
            if done:
                n_ch += 1; n_sp += len(done)
                fc.write(json.dumps({"id": i, "src": r["src"], "old": r["ref"], "new": new, "spans": done}, ensure_ascii=False) + "\n")
                r = dict(r, ref=new, tok=tk(new, add_special_tokens=False)["input_ids"])
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "rows_changed": n_ch, "spans": n_sp, "rejected": rej, "translit_chars": len(TR)}))


if __name__ == "__main__":
    main()
