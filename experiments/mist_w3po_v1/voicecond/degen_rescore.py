import json, sys, glob
sys.path.insert(0,'.')
import numpy as np
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
from evaluation.text_metrics import normalize_for_bleu
E='eval_outputs/mist_w3po_v1'
LANG={'en2zh':'cmn','zh2en':'eng'}
S=json.load(open(sys.argv[1]))
toks=lambda x,l:[c for c in normalize_for_bleu(x or '',l) if not c.isspace()] if l=='cmn' else normalize_for_bleu(x or '',l).split()
refs={}
for p in glob.glob(f'{E}/**/items_*.jsonl',recursive=True)+glob.glob('data/**/items_*.jsonl',recursive=True):
    d=p.split('items_')[-1].split('.')[0]
    if d not in LANG: continue
    for l in open(p):
        if l.strip():
            r=json.loads(l)
            if 'id' in r and r.get('translation') is not None: refs.setdefault((d,r['id']),r['translation'])
runs={}
for p,d,n,miss,deg in S:
    if miss or 'superseded' in p or '/parts/' in p: continue
    hyp={json.loads(l)['id']:json.loads(l).get('translation','') for l in open(f'{E}/{p}') if l.strip()}
    runs[p]=(d,hyp,{x[0] for x in deg})
BASEPAT=('wave2_S1/thinker','/eval_base/','base_rerun1','/ref_base/','eval_same_base')
bl={d:Bleu(l) for d,l in LANG.items()}
def cellbleu(d,hyp,ids):
    return bl[d].score(bl[d].stats([hyp[i] for i in ids],[refs[(d,i)] for i in ids]))
rows=[]
for p,(d,hyp,deg) in runs.items():
    if any(b in p for b in BASEPAT): continue
    bases=[q for q,(dd,h,_) in runs.items() if dd==d and set(h)==set(hyp) and any(b in q for b in BASEPAT)]
    if not bases: continue
    bases.sort(key=lambda q:(0 if 'wave2_S1' in q else 1 if 'eval_base' in q else 2))
    q=bases[0]; _,hb,degb=runs[q]
    U=deg|degb
    if not U: continue
    cells=sorted({i.split('__')[0] for i in hyp})
    for c in cells:
        ids=[i for i in hyp if i.split('__')[0]==c]
        if not (set(ids)&U): continue
        keep=[i for i in ids if i not in U]
        a=cellbleu(d,hyp,ids)-cellbleu(d,hb,ids); b=cellbleu(d,hyp,keep)-cellbleu(d,hb,keep)
        rows.append((p.split('/thinker')[0],d,c,len(ids),len(set(ids)&deg),len(set(ids)&degb),round(a,2),round(b,2),round(b-a,2),q.split('/thinker')[0]))
rows.sort(key=lambda r:-abs(r[8]))
print(f"{'run':40s} dir   cell      n  deg_m deg_b  Δall  Δclean shift  base")
for r in rows: print(f"{r[0]:40s} {r[1]} {r[2]:9s} {r[3]:4d} {r[4]:3d} {r[5]:3d} {r[6]:+6.2f} {r[7]:+6.2f} {r[8]:+5.2f}  {r[9]}")
