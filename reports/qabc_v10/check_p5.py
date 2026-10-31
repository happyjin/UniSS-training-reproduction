"""Does si_qabc_rules' p5 replay commit what si_thinker_stream --policy selfwait committed? Compares, sentence by sentence,
the translations and block texts of a p5 file with the selfwait file of the same Thinker, window, bias and rate (on the
sentences both have), and prints the share identical; and for the other rules the share identical to p5.

    python reports/qabc_v10/check_p5.py <p5 blocks file> <selfwait blocks file> [<other rule file> ...]
"""
import json
import sys


def load(f):
    return {r["id"]: r for r in map(json.loads, open(f, encoding="utf-8"))}


a, b = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(a) & set(b))
same_t = sum(a[i]["translation"] == b[i]["translation"] for i in ids)
same_b = sum([x["text"] for x in a[i]["blocks"]] == [x["text"] for x in b[i]["blocks"]] for i in ids)
print(f"p5 vs selfwait: {len(ids)} sentences, translation identical {same_t / len(ids):.3f}, blocks identical {same_b / len(ids):.3f}")
for i in ids[:3]:
    if [x["text"] for x in a[i]["blocks"]] != [x["text"] for x in b[i]["blocks"]]:
        print("  p5      ", [x["text"] for x in a[i]["blocks"]])
        print("  selfwait", [x["text"] for x in b[i]["blocks"]])
for f in sys.argv[3:]:
    c = load(f)
    ids2 = sorted(set(a) & set(c))
    print(f"{f.rsplit('/', 1)[-1]} vs p5: blocks identical {sum([x['text'] for x in a[i]['blocks']] == [x['text'] for x in c[i]['blocks']] for i in ids2) / len(ids2):.3f}, "
          f"blocks per sentence {sum(len(c[i]['blocks']) for i in ids2) / len(ids2):.2f} vs {sum(len(a[i]['blocks']) for i in ids2) / len(ids2):.2f}")
