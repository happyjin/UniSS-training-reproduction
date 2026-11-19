"""统一打分：ASR-BLEU / ASR-chrF / ASR-COMET、Text-BLEU / Text-chrF / Text-COMET、RTF、时长比。

输入是各系统的 `results.jsonl`（需含 `id`、`text`（系统自己的译文）、`wav`、`wall_sec`、
`output_sec`、`source_sec`）与对应的 ASR 转写 `asr.jsonl`（`id`、`asr`），参考与源文取自评测清单。

口径（三个系统完全一致）：
* **BLEU / chrF**：SacreBLEU。中文 `tokenize='zh'`，英文 `13a`。先做轻归一化（去标点、英文小写、
  压空白），假设与参考同样处理 —— ASR 转写不带或少带标点，不归一化会系统性压低分数。
* **COMET**：`Unbabel/wmt22-comet-da`，输入 `src`（源文）/`mt`/`ref`，用**未归一化**原文
  （COMET 在自然文本上训练）。`asr_comet` 的 mt 是 ASR 转写，`text_comet` 的 mt 是系统译文。
* **RTF**：`wall_sec / source_sec`，报语料级（Σwall / Σsource）与逐句中位。各系统的 `wall_sec`
  都只覆盖「模型前向 + 声码器」，不含读文件；多卡并行会抬高 RTF，单卡干净值另测。
* **空输出**：译文或语音为空的条数；它们仍计入语料级 BLEU（算作空假设），与论文口径一致。
* **UTMOS**（可选）：`run_utmos.py` 的逐句分数取均值（空语音不计）。
* **SIM-o / ESim / AutoPCP**（可选）：`run_para_metrics.py` 的逐句值取均值，并给出随机配对基线与
  「配对 − 随机」的差（`*_rand` / `*_delta`）。
* **LAAL / StartOffset / EndOffset**（可选，流式系统）：`latency_eval.py` 的逐句值取均值（秒），
  NCA = computation-unaware、CA = computation-aware，BOW / EOW = 按词起点 / 终点对齐。

用法：python score_eval.py --manifest m.jsonl \
          --sys 名称=results.jsonl:asr.jsonl[:utmos.jsonl[:lat.jsonl[:para.jsonl]]] [--sys ...] \
          --out report.json [--comet_model <ckpt>] [--no_comet] [--source_utmos src_utmos.jsonl]
"""
import argparse
import json
import math
import os
import re
import string
import sys
import unicodedata

ZH_PUNCT = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏·'
PUNCT = set(string.punctuation) | set(ZH_PUNCT)


def norm(text, lang):
    """BLEU / chrF 前的轻归一化：NFKC、去标点（英文保留撇号）、英文小写、压空白。"""
    t = unicodedata.normalize('NFKC', text or '')
    keep = "'" if lang == 'en' else ''
    t = ''.join(' ' if (c in PUNCT and c not in keep) else c for c in t)
    if lang == 'en':
        t = t.lower()
    else:
        t = re.sub(r'\s+', '', t)          # 中文去掉所有空白，由 zh tokenizer 按字切
        return t.strip()
    return re.sub(r'\s+', ' ', t).strip()


def bleu_chrf(hyps, refs, lang):
    import sacrebleu
    tok = 'zh' if lang == 'zh' else '13a'
    b = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)
    c = sacrebleu.corpus_chrf(hyps, [refs])
    return round(b.score, 2), round(c.score, 2)


def _read_jsonl(paths):
    out = {}
    for p in (paths.split(',') if paths else []):
        if p and os.path.isfile(p):
            for l in open(p, encoding='utf-8'):
                if l.strip():
                    r = json.loads(l)
                    out[r['id']] = r
    return out


def load_sys(spec):
    """名称=results[,..]:asr[,..][:utmos[,..][:latency[,..][:para[,..]]]]"""
    name, paths = spec.split('=', 1)
    parts = (paths.split(':') + ['', '', '', ''])[:5]
    res = _read_jsonl(parts[0])
    asr = {k: v.get('asr', '') for k, v in _read_jsonl(parts[1]).items()}
    utmos = {k: v.get('utmos') for k, v in _read_jsonl(parts[2]).items()}
    lat = _read_jsonl(parts[3])
    para = _read_jsonl(parts[4])
    return name, res, asr, utmos, lat, para


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (round(sum(xs) / len(xs), 4), len(xs)) if xs else (None, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--sys', action='append', required=True,
                    help='名称=results.jsonl[,更多]:asr.jsonl[,更多]')
    ap.add_argument('--out', required=True)
    ap.add_argument('--comet_model', default=os.environ.get('COMET_CKPT'))
    ap.add_argument('--comet_batch', type=int, default=32)
    ap.add_argument('--no_comet', action='store_true')
    ap.add_argument('--source_utmos', default=None, help='源音频的 UTMOS（作参照，写进报告 source_utmos）')
    args = ap.parse_args()

    man = [json.loads(l) for l in open(args.manifest, encoding='utf-8') if l.strip()]
    ids = [r['id'] for r in man]
    ref = {r['id']: r['reference'] for r in man}
    src = {r['id']: r['src_text'] for r in man}
    src_sec = {r['id']: r['source_sec'] for r in man}
    lang = man[0]['tgt_lang']
    assert all(r['tgt_lang'] == lang for r in man), '一个清单内目标语言必须一致'

    systems, comet_jobs = {}, {}
    for spec in args.sys:
        name, res, asr, utmos, lat, para = load_sys(spec)
        miss = [i for i in ids if i not in res]
        rows = [res.get(i, {}) for i in ids]
        hyp_text = [(r.get('text') or '').strip() for r in rows]
        hyp_asr = [(asr.get(i) or '').strip() for i in ids]
        refs = [ref[i] for i in ids]
        srcs = [src[i] for i in ids]
        walls = [r.get('wall_sec') for r in rows if r.get('wall_sec')]
        out_secs = [r.get('output_sec') or 0.0 for r in rows]
        sum_src = sum(src_sec[i] for i in ids)
        tb, tc = bleu_chrf([norm(h, lang) for h in hyp_text], [norm(x, lang) for x in refs], lang)
        ab, ac = bleu_chrf([norm(h, lang) for h in hyp_asr], [norm(x, lang) for x in refs], lang)
        systems[name] = dict(
            n=len(ids), missing=len(miss), text_bleu=tb, text_chrf=tc, asr_bleu=ab, asr_chrf=ac,
            empty_text=sum(1 for h in hyp_text if not h), empty_asr=sum(1 for h in hyp_asr if not h),
            empty_wav=sum(1 for s in out_secs if s <= 0.05),
            failed=sum(1 for r in rows if r.get('status') not in (None, 'ok')),
            rtf_corpus=round(sum(walls) / sum_src, 3) if walls else None,
            rtf_median=round(sorted(w / src_sec[i] for i, r, w in
                                    ((i, r, r.get('wall_sec')) for i, r in zip(ids, rows))
                                    if w)[len(walls) // 2], 3) if walls else None,
            len_ratio=round(sum(out_secs) / sum_src, 3),
            mean_wall_sec=round(sum(walls) / len(walls), 2) if walls else None,
        )
        if utmos:
            systems[name]['utmos'], systems[name]['utmos_n'] = _mean(utmos.get(i) for i in ids)
        if lat:
            for k in ('laal_nca_bow', 'laal_nca_eow', 'laal_ca_bow', 'laal_ca_eow',
                      'start_offset_nca', 'start_offset_ca', 'end_offset_nca', 'end_offset_ca'):
                v, n = _mean(lat.get(i, {}).get(k) for i in ids)
                systems[name][k] = None if v is None else round(v, 3)
            systems[name]['latency_n'] = n
        if para:
            for k in ('sim_o', 'esim', 'autopcp'):
                v, n = _mean(para.get(i, {}).get(k) for i in ids)
                vr, _ = _mean(para.get(i, {}).get(f'{k}_rand') for i in ids)
                systems[name][k] = v
                systems[name][f'{k}_rand'] = vr
                # 这三项在随机配对下本身就不低，「配对 − 随机」才说明信息真的被迁移
                systems[name][f'{k}_delta'] = None if (v is None or vr is None) else round(v - vr, 4)
            systems[name]['para_n'] = n
        if miss:
            systems[name]['missing_ids'] = miss[:10]
        comet_jobs[name] = dict(srcs=srcs, refs=refs, text=hyp_text, asr=hyp_asr)

    if not args.no_comet and args.comet_model:
        from comet import load_from_checkpoint
        model = load_from_checkpoint(args.comet_model)
        for name, job in comet_jobs.items():
            for key in ('asr', 'text'):
                data = [dict(src=s, mt=m, ref=r) for s, m, r in zip(job['srcs'], job[key], job['refs'])]
                out = model.predict(data, batch_size=args.comet_batch, gpus=1, progress_bar=False)
                systems[name][f'{key}_comet'] = round(float(out.system_score), 4)
                print(f'{name} {key}_comet = {systems[name][f"{key}_comet"]:.4f}', flush=True)

    report = dict(manifest=os.path.abspath(args.manifest), n_utt=len(ids), tgt_lang=lang,
                  total_source_sec=round(sum(src_sec.values()), 1), systems=systems)
    if args.source_utmos:
        su = {k: v.get('utmos') for k, v in _read_jsonl(args.source_utmos).items()}
        report['source_utmos'], _ = _mean(su.get(i) for i in ids)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    cols = ['asr_bleu', 'asr_comet', 'text_bleu', 'utmos', 'sim_o', 'esim_delta', 'autopcp_delta',
            'laal_nca_bow', 'laal_ca_bow', 'rtf_corpus', 'empty_wav', 'failed']
    print(f'\n{os.path.basename(args.manifest)}  ({len(ids)} 句, {lang})')
    print('system'.ljust(22) + ''.join(c.rjust(12) for c in cols))
    for name, s in systems.items():
        print(name.ljust(22) + ''.join(('-' if s.get(c) is None else f'{s[c]}').rjust(12) for c in cols))


if __name__ == '__main__':
    main()
