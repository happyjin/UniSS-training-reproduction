import json, sys, glob, collections
sys.path.insert(0,'.')
from evaluation.text_metrics import normalize_for_bleu
E='eval_outputs/mist_w3po_v1'
LANG={'en2zh':'cmn','zh2en':'eng'}
def toks(x,l):
    x=normalize_for_bleu(x or '',l); return [c for c in x if not c.isspace()] if l=='cmn' else x.split()
refs={}; clash=0
for p in glob.glob(f'{E}/**/items_*.jsonl',recursive=True)+glob.glob('data/**/items_*.jsonl',recursive=True):
    d=p.split('items_')[-1].split('.')[0]
    if d not in LANG: continue
    for l in open(p):
        if not l.strip(): continue
        r=json.loads(l); t=r.get('translation')
        if t is None or 'id' not in r: continue
        k=(d,r['id'])
        if k in refs and refs[k]!=t: clash+=1
        refs.setdefault(k,t)
print('refs',len(refs),'clash',clash)
out=[]
for p in sorted(glob.glob(f'{E}/**/blocks_*_w1.jsonl',recursive=True)):
    if 'test_robust' in p: continue
    d=p.split('blocks_')[-1].split('_w1')[0]
    if d not in LANG: continue
    n=0;deg=[];miss=0
    for l in open(p):
        if not l.strip(): continue
        r=json.loads(l); n+=1
        ref=refs.get((d,r['id']))
        if ref is None: miss+=1; continue
        h=toks(r.get('translation',''),LANG[d]); t=toks(ref,LANG[d])
        if len(h)>3*len(t)+10: deg.append((r['id'],len(h),len(t)))
    out.append((p[len(E)+1:],d,n,miss,deg))
json.dump(out,open(sys.argv[1],'w'))
tot=sum(1 for o in out if o[4])
print('runs',len(out),'runs with degenerate',tot)
for p,d,n,miss,deg in out:
    if deg or miss: print(f'{p:75s} n={n} miss={miss} deg={len(deg)}', [(i[0][:30],i[1],i[2]) for i in deg][:3])
