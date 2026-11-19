"""S2 文本评测打分：Text-BLEU / Text-chrF / Text-COMET。

口径与 score_eval.py 一致：BLEU/chrF 先轻归一化（NFKC、去标点、en 小写、zh 去空白），
zh 用 tokenize='zh'，en 用 13a；COMET 用未归一化的 src/mt/ref。
"""
import argparse, glob, json, os, re, sys, unicodedata

PUNCT = set('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~«»“”‘’„…–—！￥%（）*+，-。/：；《<=>？@【】^_`{|}~')

def norm(text, lang):
    t = unicodedata.normalize('NFKC', text or '')
    keep = "'" if lang == 'en' else ''
    t = ''.join(' ' if (c in PUNCT and c not in keep) else c for c in t)
    if lang == 'en':
        return re.sub(r'\s+', ' ', t.lower()).strip()
    return re.sub(r'\s+', '', t).strip()


def bleu_chrf(hyps, refs, lang):
    import sacrebleu
    tok = 'zh' if lang == 'zh' else '13a'
    b = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)
    c = sacrebleu.corpus_chrf(hyps, [refs])
    return round(b.score, 2), round(c.score, 2)


def comet_score(model, srcs, mts, refs):
    from comet import CSRCMetadata  # noqa: F401  (确认版本可 import)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='含 <set>/<tag>/results.jsonl 的根目录')
    ap.add_argument('--comet_model', default='Unbabel/wmt22-comet-da')
    ap.add_argument('--no_comet', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    comet_model = None
    if not args.no_comet:
        from comet import load_from_checkpoint, download_model
        ck = os.environ.get('COMET_CKPT') or download_model(args.comet_model)
        comet_model = load_from_checkpoint(ck)
        comet_model.to(args.device)

    print(f"{'set':<16}{'mode':<7}{'N':>5}{'BLEU':>8}{'chrF':>8}{'COMET':>8}{'empty':>7}")
    for rf in sorted(glob.glob(f'{args.root}/*/*/results.jsonl')):
        set_, tag = rf.split('/')[-3], rf.split('/')[-2]
        rows = [json.loads(l) for l in open(rf, encoding='utf-8')]
        lang = 'zh' if rows[0].get('tgt_lang') == 'zh' else 'en'
        hyps = [norm(r.get('text') or '', lang) for r in rows]
        refs = [norm(r.get('reference') or '', lang) for r in rows]
        bleu, chrf = bleu_chrf(hyps, refs, lang)
        empty = sum(1 for h in hyps if not h)
        comet = ''
        if comet_model is not None:
            data = [dict(src=r.get('src_text') or '', mt=r.get('text') or '',
                         ref=r.get('reference') or '') for r in rows]
            out = comet_model.predict(data, batch_size=32, gpus=1)
            comet = f"{out.system_score:8.4f}"
        print(f"{set_:<16}{tag:<7}{len(rows):>5}{bleu:>8.2f}{chrf:>8.2f}{comet:>8}{empty:>7}", flush=True)


if __name__ == '__main__':
    main()
