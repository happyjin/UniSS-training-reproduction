import json, sys
sys.path.insert(0,'.')
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
from experiments.mist_w3po_v1.voicecond.test_score import EXCL
from evaluation.text_metrics import normalize_for_bleu
T='eval_outputs/mist_w3po_v1/test_robust'
jl=lambda p:[json.loads(l) for l in open(p) if l.strip()]
LANG={'en2zh':'cmn','zh2en':'eng'}
toks=lambda x,l:[c for c in normalize_for_bleu(x or '',l) if not c.isspace()] if l=='cmn' else normalize_for_bleu(x or '',l).split()
SYS={'B_b':('B','B/b'),'UE_b':('UE','UE/b'),'UE_v1':('UE','UE/v1'),'UE_abs':('UE','UE_abs/abs')}
CELLS={'rsc':'RealSI 干净','rsn':'RealSI 带噪','obclean':'OBSI clean','obid2':'OBSI id2','obood1':'OBSI ood1'}
for d,l in LANG.items():
    bl=Bleu(l); meta={m['id']:m for m in jl(f'{T}/data/meta_{d}.jsonl')}
    ref={r['id']:r['translation'] for r in jl(f'{T}/data/items_{d}.jsonl')}
    th={k:{r['id']:r['translation'] for r in jl(f'{T}/{k}/thinker_{d}/blocks_{d}_w1.jsonl')} for k in ('B','UE')}
    asr={s:{r['id']:r.get('asr_text','') for r in jl(f'{T}/{p}/talker_{d}/asr_speech.jsonl')} for s,(t,p) in SYS.items()}
    bad={}
    for s,(t,p) in SYS.items():
        b=set()
        for i in asr['UE_abs']:
            rt=toks(ref[i],l)
            if len(toks(th[t][i],l))>3*len(rt)+10: b.add(i)            # Thinker degenerate
            if len(toks(asr[s][i],l))>3*len(toks(th[t][i],l))+10: b.add(i)   # Talker runaway
        bad[s]=b
    U=set().union(*bad.values())
    print(d,'degenerate/runaway items per system',{s:len(b) for s,b in bad.items()},'union',len(U))
    for c,name in CELLS.items():
        ids=[i for i in asr['UE_abs'] if i.startswith(c+'__') and not (meta[i]['set']=='obsi' and meta[i]['pair'] in EXCL[d])]
        keep=[i for i in ids if i not in U]
        sc=lambda s,I: bl.score(bl.stats([asr[s][i] for i in I],[ref[i] for i in I]))
        a={s:sc(s,ids) for s in SYS}; k={s:sc(s,keep) for s in SYS}
        print(f"  {name:12s} n={len(ids)} drop={len(ids)-len(keep)} | all " + ' '.join(f"{s} {a[s]:.2f}" for s in SYS) +
              " | clean " + ' '.join(f"{s} {k[s]:.2f}({k[s]-k['B_b']:+.2f})" for s in SYS))
