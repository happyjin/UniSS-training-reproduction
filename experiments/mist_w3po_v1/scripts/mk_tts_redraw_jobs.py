# One-off: TTS re-draw (talker --seed 20261109) of existing guarded dev decodes, from EA50's pipelined speech jobs.
import json, glob, os, sys, shutil
Q='logs/mist_w3po_v1/queue_v2/jobs'; D='eval_outputs/mist_w3po_v1/decode'
SRC='dev_pilot50_eagrpoctrl631sc_ctx0_guard_s8'
systems=sys.argv[2:]; dry=sys.argv[1]=='dry'
tmpl=[json.load(open(f)) for f in sorted(glob.glob(f'{Q}/076_{SRC}_*.json'))]
tmpl=[t for t in tmpl if t['id'].split('__')[-1] not in [f'dec{k}' for k in range(8)]+['score']]
assert len(tmpl)==36, len(tmpl)
for s in systems:
    new=s+'_tts2'
    assert not os.path.exists(f'{D}/{new}') and not os.path.exists(f'eval_outputs/mist_w3po_v1/speech/{new}')
    for i,t in enumerate(tmpl):
        j=json.loads(json.dumps(t).replace(SRC,new))
        if j['id'].endswith('__sprep'): j['deps']=[]
        if '__talk' in j['id']:
            assert '--num-shards 8 >' in j['cmd']; j['cmd']=j['cmd'].replace('--num-shards 8 >','--num-shards 8 --seed 20261109 >')
        out=f'{Q}/084_tts2_{s}_{i:02d}.json'
        assert not os.path.exists(out)
        if dry: 
            if i in (0,1,35): print(j['id'], j['deps'][:1], j['cmd'][-160:])
        else:
            with open(out,'x') as fh: json.dump(j,fh)
    if not dry:
        os.makedirs(f'{D}/{new}')
        for f in ('hyp.jsonl','score_text.json'): shutil.copy2(f'{D}/{s}/{f}', f'{D}/{new}/{f}')
    print(s,'ok')
