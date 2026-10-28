"""S3 diagnosis on 12 test utterances: UTMOS of the recording, AudioVAE reconstruction (codec ceiling), the S3
prior at temperature 0 / 0.667 with full and chunked streaming decoding, and the streamed render (tts_schemes env)."""
import sys, json, numpy as np, torch, soundfile as sf, torchaudio
R="/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/llm_vits_route_v1"
sys.path[:0]=[R+"/src",R+"/src/vits",R+"/scripts"]
from audiovae import load_audiovae, encode_stats, decode_22k
from codec_priors import S3Prior
from acoustic_common import overlap_matrix
from stream_common import load_backbone
from text import intersperse, text_to_ids
dev="cuda"
utmos=torch.hub.load("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/tarepan_SpeechMOS_v1.2.0","utmos22_strong",source="local").cuda().eval()
mos=lambda w: float(utmos(torch.from_numpy(np.asarray(w,dtype=np.float32))[None].cuda(),22050).item())
vae=load_audiovae(dev)
prior=S3Prior().to(dev).eval(); prior.load_state_dict(torch.load(R+"/runs/all_schemes/s3/s3_40000.pth",map_location="cpu")["prior"])
net=load_backbone(R+"/configs/ljs_char_base.json",R+"/runs/e0_backbone_v2/G_60000.pth").to(dev).eval()
durs=dict(np.load(R+"/runs/final/align/durations.npz"))
rows=[r for r in map(json.loads,open(R+"/data/manifest_v2.jsonl")) if r["split"]=="test"][:12]
res={k:[] for k in ["gt","recon_mu","prior_t0","prior_t0.667","prior_t0.667_stream","rendered"]}
with torch.no_grad():
  for r in rows:
    w,_=sf.read(r["audio_path"],dtype="float32"); res["gt"].append(mos(w))
    mu,lv=encode_stats(vae,torch.from_numpy(w)[None].to(dev))
    res["recon_mu"].append(mos(decode_22k(vae,mu)[0].cpu().numpy()))
    ids=torch.LongTensor(intersperse(text_to_ids(r["clean"])))[None].to(dev)
    h,m_p,_,_=net.enc_p(ids,torch.LongTensor([ids.size(1)]).to(dev))
    d=durs[r["utt_id"]]; o=torch.from_numpy(overlap_matrix(d,25.0)).to(dev)
    c=(o@torch.cat([h,m_p],1)[0].T).T[None]; mask=torch.ones(1,1,c.size(-1),device=dev)
    for t in [0.0,0.667]:
      eps=torch.randn(1,64,c.size(-1),device=dev)
      u=prior.sample(c,mask,eps,t)
      res[f"prior_t{t:g}" if t else "prior_t0"].append(mos(decode_22k(vae,u)[0].cpu().numpy()))
      if t>0:
        with vae.streaming_decode() as sd:
          outs=[sd.decode_chunk(u[:,:,s:s+10])[0,0] for s in range(0,u.size(-1),10)]
        y=torchaudio.functional.resample(torch.cat(outs).float(),48000,22050)
        res["prior_t0.667_stream"].append(mos(y.cpu().numpy()))
    rw,_=sf.read(R+"/runs/all_schemes/render_test/s3/wavs/"+r["utt_id"]+".wav"); res["rendered"].append(mos(rw))
    print("mu stats: prior-sample u std",float(u.std()),"posterior mu std",float(mu.std()), flush=True) if r is rows[0] else None
print({k:round(float(np.mean(v)),3) for k,v in res.items()})
