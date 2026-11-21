import json, sys
sys.argv=['x']
exec(open(__file__.replace('_tail.py', '.py')).read().split("out = {}")[0])
for d, cell in (("zh2en","a_noisy"),("en2zh","b_noisy")):
    lang=LANG[d]; R={}
    for (arm,s),p in runs(d).items():
        recs={r["id"]:r for r in jl(p/"results.jsonl")}; asr={r["id"]:r.get("asr_text","") for r in jl(p/"asr_speech.jsonl")}
        R[(arm,s)]=(recs,asr)
    th=R[("V1",1)][0]; ids=[i for i in th if i.startswith(cell+"__")]
    E_={}
    for i in ids:
        t=toks(th[i]["translated_text"],lang)
        E_[i]={k:edit(toks(R[k][1].get(i,""),lang),t) for k in R}; E_[i]["n"]=len(t)
    tot=lambda k: sum(E_[i][k] for i in ids)
    exc={s: tot(("AbS",s))-tot(("V1",s)) for s in (1,2)}
    print(d,cell,"n",len(ids),"total ref tokens",sum(E_[i]["n"] for i in ids),"excess edits AbS-V1 per seed",exc,
          "oracle-V1 s1",tot(("oracle",1))-tot(("V1",1)))
    diff={i:[E_[i][("AbS",s)]-E_[i][("V1",s)] for s in (1,2)] for i in ids}
    top=sorted(ids,key=lambda i:-sum(diff[i]))[:8]
    for i in top: print("  ",i[:45],"n",E_[i]["n"],"AbS s1/s2",E_[i][("AbS",1)],E_[i][("AbS",2)],"V1",E_[i][("V1",1)],E_[i][("V1",2)],"B",E_[i][("B",1)],E_[i][("B",2)],"orc",E_[i][("oracle",1)])
    both=sum(1 for i in ids if diff[i][0]>=5 and diff[i][1]>=5); one=sum(1 for i in ids if (diff[i][0]>=5)!=(diff[i][1]>=5))
    vv=sum(1 for i in ids if abs(E_[i][("V1",1)]-E_[i][("V1",2)])>=5)
    print("   items AbS worse by>=5 edits in both seeds",both,"| in one seed only",one,"| V1 own seed swing>=5:",vv)
    top5=sorted(ids,key=lambda i:-sum(diff[i]))[:5]
    print("   excess share of top5 items:", round(sum(sum(diff[i]) for i in top5)/max(1,exc[1]+exc[2]),2))
