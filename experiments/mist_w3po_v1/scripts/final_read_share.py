"""Final-read word share of a decode (T1-quick's 49.7% / 84.3%): over the windows of <windows.jsonl>, the words written
at each window's last read_log entry / all words written; plus, among windows >= 4 s with any word, those written only
at the final read.  python final_read_share.py <hyp.jsonl> <windows.jsonl>"""
import json
import re
import sys

TAG = re.compile(r"<\|[^|]*\|>")


def nw(s: str) -> int:
    return len(TAG.sub(" ", s).split())


def main() -> None:
    hyp, win = sys.argv[1], sys.argv[2]
    dur = {}
    for l in open(win, encoding="utf-8"):
        w = json.loads(l)
        dur[w["id"]] = w["dur_ms"]
    tot = fin = n = n4 = only = 0
    for l in open(hyp, encoding="utf-8"):
        r = json.loads(l)
        if r["id"] not in dur:
            continue
        w = [nw(x["new"]) for x in r["read_log"]]
        n += 1
        tot += sum(w)
        fin += w[-1] if w else 0
        if dur[r["id"]] >= 4000 and sum(w) > 0:
            n4 += 1
            only += sum(w[:-1]) == 0
    print(json.dumps({"windows": n, "of": len(dur), "words": tot, "final_share": round(fin / max(tot, 1), 4),
                      "ge4s_windows": n4, "ge4s_only_final": only}))


if __name__ == "__main__":
    main()
