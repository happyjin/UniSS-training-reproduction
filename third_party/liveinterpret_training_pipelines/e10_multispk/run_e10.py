"""E10 / E10-0 decoding of the online test windows (each window = one fresh session, as for E7/E8/E9).

The step loop is S5's `stream_translate` (infer_s5_streaming_dynchunk.py: 640 ms chunks, argmax WAIT/WRITE, greedy
deltas, Thinker KV cache + CUDA graphs, Talker StaticCache + CUDA graph, one Talker state per window) with:
  * E10 (`--mode e10`, adapter from train_e10.sh): the multi-speaker instruction; audio frame j gets
    e10_sd_proj(post[j - 13]) from the window's Nemotron lowlat posteriors (the training rule); the speaker tags may be
    sampled inside a delta; each delta is split at its tags and every piece is a separate Talker call (tags never
    enter the Talker condition) attributed to its tag (text before any tag: the last tag; none yet: unknown);
  * E10-0 (`--mode e10_0`, the stock S5 adapter): S5's instruction, no posteriors, no tags (speakers are assigned
    later by build_e10.py with the causal relabelling used for E7/E8).
Latency is computation-excluded: a piece is written at the source time consumed by its step.

Voices (decoded after the window; the timing does not depend on it):
  * E10: per speaker, baseline B's causal voice rule (gstar_talker_v6 prep --causal): the slot's solo runs
    (p > 0.5, others < 0.5, >= 0.4 s) in the meeting before its first piece's write time, the most recent 8 s;
    fallback the window audio up to that time. CAM++ embedding + speech-token flow prompt of that audio; all of the
    speaker's tokens are decoded together, then cut back into pieces by token count.
  * E10-0: S5's own rolling reference: the last 2 s of the window audio before each piece's write time.

Raw output (build_e10.py turns it into the scorer formats): <out>/part_<shard>.jsonl, one row per window
  {id, segs:[{spk, text, write_ms, n_tok, dur_ms, wav}], stop, n_write, n_wait, n_text, error}, wavs 16 kHz mono.

    CUDA_VISIBLE_DEVICES=0 $SIMULS2S/bin/python e10_multispk/run_e10.py --mode e10 --set ami_zh2en \
        --windows <online windows jsonl> --adapter <E10 adapter> --out <raw dir> --shard 0 --num-shards 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LI = HERE.parent
sys.path[:0] = [str(LI / "custom"), str(HERE)]
UNISS = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")

SR, OUT_SR = 16000, 24000
SETS = {   # B's talker-prep settings per test set (final_test_orch.py / en2zh_orch.py)
    "ami_zh2en": dict(tgt="en", data="data/ami_multispk_si_v1", sess="zh_mix_v8", diar="nemotron_lowlat_zh8"),
    "ali": dict(tgt="en", data="data/alimeeting_si_v1", sess="near_mix", diar="nemotron_lowlat"),
    "ami_en2zh": dict(tgt="zh", data="data/ami_multispk_si_v1", sess="ihm_mix", diar="nemotron_lowlat"),
}


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------------------------------- voices
class Sessions:
    """One meeting's posteriors (post80) and audio at a time (gstar_talker_v6.session_of)."""

    def __init__(self, cfg):
        self.cfg, self.m, self.val = cfg, None, None

    def get(self, m):
        if m != self.m:
            import soundfile as sf
            d = UNISS / self.cfg["data"]
            post = torch.load(d / "diar" / self.cfg["diar"] / f"{m}.post.pt", weights_only=False)["post80"]
            name = f"{m}.Mix-Headset.wav" if self.cfg["sess"] == "ihm_mix" else f"{m}.wav"
            x, sr = sf.read(str(d / "session_audio" / self.cfg["sess"] / name), dtype="float32", always_2d=True)
            self.m, self.val = m, (post.float().numpy(), x.mean(1), sr)
        return self.val


def resample(x, sr_in, sr_out):
    import torchaudio
    if sr_in == sr_out:
        return np.ascontiguousarray(x, dtype=np.float32)
    return torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                                          sr_in, sr_out).numpy()


def causal_voice(sessions, w, k, write_ms, wav16):
    """B's --causal voice: slot k's solo runs before the cut, most recent first, up to 8 s (16 kHz)."""
    from experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker_v6 import solo_runs
    post, x, sr = sessions.get(w["meeting"])
    cut = w["start_s"] + write_ms / 1000
    pieces, tot = [], 0.0
    for a, b in reversed(solo_runs(post, k - 1, int(cut / 0.08))):
        pieces.insert(0, x[int(a * 0.08 * sr): int(b * 0.08 * sr)])
        tot += (b - a) * 0.08
        if tot >= 8.0:
            break
    if tot >= 1.0:
        return resample(np.concatenate(pieces), sr, SR)[-int(8.0 * SR):], "causal"
    return wav16[: max(SR, int(write_ms * SR / 1000))], "fallback"


# ---------------------------------------------------------------------------------------------------- decoding
def sample_delta(embeds, run, tok_emb, pick, stops, extra_allow, *, ban_above, max_tokens):
    """S5 sample_delta with the speaker tags allowed inside a delta."""
    delta, out, term = [], None, None
    for _ in range(max(0, int(max_tokens))):
        logits, out = run()
        t = pick(logits, 1.0, 0.0, ban_above=ban_above, allow=tuple(stops) + tuple(extra_allow))
        if t in stops:
            term = t
            break
        delta.append(t)
        embeds.append(tok_emb(t))
    if delta and term is None:
        _, out = run()
    return delta, out, term


@torch.inference_mode()
def translate_window(model, tok, ids, spk_ids, mel_fn, wav16, tgt, post, a):
    from streaming_si.inference.infer_s5_streaming_dynchunk import MAX_TAIL_STEPS, ThinkerRunner, _pick
    from streaming_si.model.model_si import TalkerStreamState
    from e10.common import delayed_post, instruction
    from e10.patch import add_sd
    wait_id, write_id, chunk_id, src_end_id = ids
    dev = a.device
    audio_pad = int(model.config.audio_ids[0])
    eos_id, ban_above = int(tok.eos_token_id), int(tok.vocab_size)
    emb = model.thinker.embed_tokens
    dtype = emb.weight.dtype
    spk_of = {t: k + 1 for k, t in enumerate(spk_ids)}

    def tok_emb(i):
        return emb(torch.tensor([[int(i)]], device=dev))

    mel = mel_fn(torch.from_numpy(wav16), sr=SR).unsqueeze(0).to(dev)
    t_mel = mel.size(-1)
    sp = torch.from_numpy(delayed_post(post, t_mel // 8)) if a.mode == "e10" else None
    embeds = [emb(torch.tensor([tok.encode(instruction(tgt, a.mode == "e10"), add_special_tokens=False)],
                               device=dev))]
    run = ThinkerRunner(model, embeds, use_cache=True, graph=not a.no_graph)
    chunk_mel = a.chunk_ms // 10
    state = TalkerStreamState(model.talker, cuda_graph=not a.no_graph)
    torch.manual_seed(a.seed)
    pos, fpos, vcache, src_done, tail, n_text, i = 0, 0, None, False, 0, 0, 0
    segs, cur, stop, n_write, n_wait = [], None, "max_steps", 0, 0
    n_steps = int(np.ceil(t_mel / chunk_mel))
    while True:
        b = min(t_mel, pos + chunk_mel)
        n_frames = (b // 8 - pos // 8) if not src_done else 0
        is_last = b >= t_mel
        embeds.append(tok_emb(chunk_id))
        if n_frames > 0:
            feats, vcache = model.encode_audio_chunk(mel[..., pos:b], cache=vcache)
            if feats is not None and feats.size(1) > 0:
                f = feats[0]
                if sp is not None:
                    f = add_sd(model, f, sp[fpos: fpos + f.size(0)])
                fpos += f.size(0)
                pad = torch.full((1, f.size(0)), audio_pad, device=dev)
                embeds.append(model.inject_audio_features(pad, emb(pad), [f]).to(dtype))
        if is_last and not src_done:
            embeds.append(tok_emb(src_end_id))
            src_done = True
        logits, out = run()
        allow = (wait_id, write_id, eos_id) if src_done else (wait_id, write_id)
        dec = _pick(logits, 1.0, 0.0, ban_above=ban_above, allow=allow)
        if dec == eos_id:
            stop = "eos"
            break
        embeds.append(tok_emb(dec))
        pos = b
        term = None
        if dec == write_id:
            n_write += 1
            delta, out, term = sample_delta(embeds, run, tok_emb, _pick, (chunk_id, eos_id, wait_id),
                                            spk_ids if a.mode == "e10" else (), ban_above=ban_above,
                                            max_tokens=a.max_text_tokens - n_text)
            n_text += len(delta)
            if delta:
                hid = model.pick_bridge_hidden(out)
                L = hid.size(1)
                base = L - len(delta)
                groups, gk = [], cur
                for j, t in enumerate(delta):          # split at speaker tags
                    if t in spk_of:
                        gk = spk_of[t]
                        groups.append([gk, []])
                    else:
                        if not groups:
                            groups.append([gk, []])
                        groups[-1][1].append(j)
                cur = gk
                for k, js in groups:
                    if not js:
                        continue
                    am = torch.zeros((1, L), dtype=torch.long, device=dev)
                    am[0, [base + j for j in js]] = 1
                    ch, lens = model.gather_answer_hidden(hid, am)
                    tids = [delta[j] for j in js]
                    cond = model.build_stream_talker_condition(ch[:, :int(lens[0])], torch.tensor([tids], device=dev))
                    seg = list(model.talker.stream_generate_s4(state, cond_chunk=cond,
                                                               max_speech_tokens=a.max_speech_tokens,
                                                               top_k=a.talker_top_k, top_p=a.talker_top_p))
                    segs.append({"spk": k, "text": tok.decode(tids, skip_special_tokens=True).strip(),
                                 "write_ms": int(b * 10), "tokens": [int(x) for x in seg]})
        else:
            n_wait += 1
        i += 1
        if term == eos_id or (src_done and dec == wait_id):
            stop = "eos"
            break
        if src_done:
            tail += 1
            if tail > MAX_TAIL_STEPS:
                stop = "tail_cap"
                break
        if n_text >= a.max_text_tokens:
            stop = "max_text_tokens"
            break
        if i > n_steps + MAX_TAIL_STEPS + 1:
            break
    rest = list(model.talker.stream_generate_s4(state, cond_chunk=None, finalize=True,
                                                max_speech_tokens=a.max_speech_tokens,
                                                top_k=a.talker_top_k, top_p=a.talker_top_p))
    if segs:
        segs[-1]["tokens"] += [int(x) for x in rest]
    return {"segs": segs, "stop": stop, "n_write": n_write, "n_wait": n_wait, "n_text": n_text}


def vocode(model, speaker, prompter, tokens, ref16, a):
    """tokens -> 16 kHz waveform with the reference's CAM++ embedding and speech-token flow prompt."""
    emb = speaker(ref16) if ref16 is not None and ref16.size >= SR // 10 else None
    p_tok = p_feat = None
    if prompter is not None and ref16 is not None and ref16.size >= 6400:
        p_tok, p_feat = prompter(ref16, max_sec=a.prompt_sec, position="tail", wav24=resample(ref16, SR, OUT_SR))
    wav = model.decode_speech(tokens, spk_emb=emb, ode_steps=a.ode_steps, prompt_speech_tokens=p_tok,
                              prompt_feat=p_feat)
    return resample(wav.detach().float().cpu().numpy().reshape(-1), OUT_SR, SR)


def speak(model, speaker, prompter, w, res, wav16, sessions, out_dir, a, stats):
    import soundfile as sf
    segs = res["segs"]
    wav_dir = out_dir / "wav" / w["id"]
    pieces = [None] * len(segs)
    if a.mode == "e10":   # one decode per speaker with B's causal voice
        by = defaultdict(list)
        for n, s in enumerate(segs):
            if s["tokens"]:
                by[s["spk"]].append(n)
        for k, ns in by.items():
            if k is None:
                ref, how = wav16[: max(SR, int(segs[ns[0]]["write_ms"] * SR / 1000))], "unknown_speaker"
            else:
                ref, how = causal_voice(sessions, w, k, segs[ns[0]]["write_ms"], wav16)
            stats[f"voice_{how}"] += 1
            toks = [t for n in ns for t in segs[n]["tokens"]]
            x = vocode(model, speaker, prompter, toks, ref, a)
            cum = np.cumsum([len(segs[n]["tokens"]) for n in ns])
            cuts = [0] + [int(round(c / cum[-1] * len(x))) for c in cum]
            for j, n in enumerate(ns):
                pieces[n] = x[cuts[j]: cuts[j + 1]]
    else:                 # E10-0: S5's rolling 2 s reference, per piece
        for n, s in enumerate(segs):
            if s["tokens"]:
                end = int(s["write_ms"] * SR / 1000)
                ref = wav16[max(0, end - 2 * SR): max(end, int(0.4 * SR))]
                pieces[n] = vocode(model, speaker, prompter, s["tokens"], ref, a)
                stats["voice_rolling"] += 1
    out = []
    for n, s in enumerate(segs):
        row = {"spk": s["spk"], "text": s["text"], "write_ms": s["write_ms"], "n_tok": len(s["tokens"]),
               "dur_ms": 0, "wav": None}
        if pieces[n] is not None and len(pieces[n]):
            wav_dir.mkdir(parents=True, exist_ok=True)
            p = wav_dir / f"{n:03d}.wav"
            sf.write(str(p), pieces[n], SR)
            row.update(dur_ms=int(round(len(pieces[n]) / 16)), wav=str(p))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["e10", "e10_0"], required=True)
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model_dir", default=str(LI / "checkpoints/S4_1031v2_ckpt93767"))
    ap.add_argument("--cosyvoice_dir", default=str(LI / "checkpoints/speech_decoder"))
    ap.add_argument("--cosyvoice_repo", default=str(LI / "external/CosyVoice"))
    ap.add_argument("--prompt_tokenizer", default=os.environ.get(
        "E10_SPEECH_TOKENIZER", str(LI / "checkpoints/cosyvoice3/speech_tokenizer_v3.batch.onnx")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated window ids (smoke tests)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk_ms", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0, help="Talker sampling seed (TTS seed)")
    ap.add_argument("--max_text_tokens", type=int, default=256)
    ap.add_argument("--max_speech_tokens", type=int, default=750)
    ap.add_argument("--talker_top_k", type=int, default=25)
    ap.add_argument("--talker_top_p", type=float, default=0.8)
    ap.add_argument("--ode_steps", type=int, default=5)
    ap.add_argument("--prompt_sec", type=float, default=8.0)
    ap.add_argument("--no_graph", action="store_true")
    ap.add_argument("--no_vocoder", action="store_true")
    a = ap.parse_args()
    # torch defaults to one intra-op thread per core (192 here): 16 decoders = 236 threads each, load ~600, other
    # users' jobs starved (2026-11-19). Cap the CPU threads per decoder.
    torch.set_num_threads(int(os.environ.get("E10_THREADS", "4")))
    if str(UNISS) not in sys.path:
        sys.path.append(str(UNISS))
    cfg = SETS[a.set]
    out = Path(a.out).resolve()   # wav paths in the raw rows must be absolute
    out.mkdir(parents=True, exist_ok=True)
    part = out / f"part_{a.shard:02d}.jsonl"
    if part.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {part}")

    W = load_jsonl(a.windows)
    if a.ids:
        keep = set(a.ids.split(","))
        W = [w for w in W if w["id"] in keep]
    # contiguous shards of the (meeting, start) order: more shards than meetings, few meeting reloads per shard
    W = sorted(W, key=lambda w: (w["meeting"], w["start_s"]))
    W = W[a.shard * len(W) // a.num_shards: (a.shard + 1) * len(W) // a.num_shards]
    if a.limit:
        W = W[: a.limit]

    from e10 import patch
    patch.install()
    from e10.common import spk_token_ids
    from e10.load import load_e10_model, load_s5
    from streaming_si.inference.infer_s2st import CampplusSpeaker, load_wav16
    from streaming_si.inference.source_prompt import SourcePrompt
    from streaming_si.model.base_model import VoxtralAudioProcessor
    from streaming_si.model.speech_decoder import resolve_speech_decoder
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
    t0 = time.time()
    if a.mode == "e10":
        model, tok, ids = load_e10_model(a.model_dir, a.adapter, device=a.device)
        spk_ids = spk_token_ids(tok)
    else:
        model, tok, ids = load_s5(a.model_dir, a.adapter, device=a.device)
        spk_ids = []
    speaker = prompter = None
    if not a.no_vocoder:
        dec_dir, yaml = resolve_speech_decoder(a.model_dir, a.cosyvoice_dir, None)
        model.config.talker_config_yaml = yaml
        model.attach_speech_decoder(dec_dir, device=a.device, cosyvoice_repo=a.cosyvoice_repo, strict=True)
        speaker = CampplusSpeaker(os.path.join(dec_dir, "campplus.onnx"))
        prompter = SourcePrompt(a.prompt_tokenizer)
    mel_fn = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(a.model_dir))
    sessions = Sessions(cfg)
    stats = defaultdict(int)
    print(json.dumps({"mode": a.mode, "set": a.set, "windows": len(W), "load_s": round(time.time() - t0, 1),
                      "adapter": a.adapter, "prompt_tokenizer": a.prompt_tokenizer}), flush=True)
    with open(part, "x", encoding="utf-8") as fh:
        for n, w in enumerate(W):
            t1 = time.time()
            row = {"id": w["id"], "mode": a.mode, "set": a.set, "seed": a.seed}
            try:
                wav16 = load_wav16(w["audio"])
                pp = (w.get("post") or {}).get("lowlat")
                post = np.load(pp) if (a.mode == "e10" and pp) else None
                if a.mode == "e10" and post is None:
                    stats["windows_without_post"] += 1
                res = translate_window(model, tok, ids, spk_ids, mel_fn, wav16, cfg["tgt"], post, a)
                segs = (speak(model, speaker, prompter, w, res, wav16, sessions, out, a, stats)
                        if not a.no_vocoder else
                        [{"spk": s["spk"], "text": s["text"], "write_ms": s["write_ms"], "n_tok": len(s["tokens"]),
                          "dur_ms": len(s["tokens"]) * 40, "wav": None} for s in res["segs"]])
                row.update({k: res[k] for k in ("stop", "n_write", "n_wait", "n_text")}, segs=segs, error=None)
                toks = [t for s in res["segs"] for t in s["tokens"]]
                src_s, sp_s = wav16.size / SR, len(toks) / 25.0
                # S5 collapse_ok (infer_s5_streaming_dynchunk): few distinct codes AND speech much longer than source
                row.update(source_sec=round(src_s, 2), speech_sec=round(sp_s, 2), speech_tokens=len(toks),
                           speech_uniq=round(len(set(toks)) / len(toks), 3) if toks else None,
                           collapse=bool(toks) and len(set(toks)) / len(toks) < 0.3 and sp_s > 1.6 * src_s + 1.0)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row.update(segs=[], error=f"{type(e).__name__}: {e}")
                stats["errors"] += 1
            row["wall_s"] = round(time.time() - t1, 2)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if n < 3 or n % 50 == 0:
                print(f"[e10] {n + 1}/{len(W)} {w['id']} {row.get('stop')} {row['wall_s']}s | "
                      + " ".join(f"S{s['spk']}:{s['text']}" for s in row["segs"])[:300], flush=True)
    (out / f"part_{a.shard:02d}.stats.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps({"done": len(W), **stats}), flush=True)



if __name__ == "__main__":
    main()
