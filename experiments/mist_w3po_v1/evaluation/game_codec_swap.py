"""Where does the prompt's background act? 2x2 decomposition for reports/mist_w3po_v1/game_muffled_speech_analysis.zh-CN.md.
For the paired prompt-swap streams (speech/game_swap_<d>_{O,V}: same item order, same seeds, only the voice prompt
differs) decode with UniSSTokenizer.decode exactly as si_infer does ([32 BiCodec global tokens] + semantic codes):
Semantic codes = BiCodec re-encoding of each arm's generated speech (results keep only code counts), so every cell is
one encode/decode round trip and OO / VV approximate the arm outputs.
  OO  codes of arm O (talker LM conditioned on the raw prompt)    + global tokens of the raw prompt    (~ arm O output)
  OV  codes of arm O                                              + global tokens of the vocals prompt (decoder swap)
  VO  codes of arm V (LM conditioned on the vocals prompt)        + global tokens of the raw prompt    (LM swap)
  VV  codes of arm V                                              + global tokens of the vocals prompt (= arm V output)
Global tokens = flat(front.bicodec_tokenizer.encode_wav_to_tokens(prompt))[:32], as si_infer. Writes
eval_outputs/mist_w3po_v1/game_muffle_v1/codec_swap/<d>/<id>__{OO,OV,VO,VV}.wav and index.jsonl. Refuses existing.
Run in venvs/vllm_omni_030 (uniss package), one GPU.

    python -m experiments.mist_w3po_v1.evaluation.game_codec_swap --dir zh2en|en2zh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
SPD = REPO / "eval_outputs/mist_w3po_v1/speech"


def flat(x):
    x = x.flatten().tolist() if hasattr(x, "flatten") else x
    out = []
    for v in x:
        out += flat(v) if isinstance(v, (list, tuple)) or hasattr(v, "flatten") else [int(v)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, choices=["zh2en", "en2zh"])
    a = ap.parse_args()
    out = REPO / "eval_outputs/mist_w3po_v1/game_muffle_v1/codec_swap" / a.dir
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    from uniss import UniSSTokenizer
    dev = torch.device("cuda:0")
    front = UniSSTokenizer.from_pretrained(str(REPO / "pretrained_models/UniSS"), device=dev)
    arms = {}
    for arm in ("O", "V"):
        S = SPD / f"game_swap_{a.dir}_{arm}"
        items = {r["id"]: r for r in map(json.loads, open(S / "talker_in/items.jsonl", encoding="utf-8"))}
        res = {r["id"]: r for r in map(json.loads, open(S / "talker/part_0.jsonl", encoding="utf-8"))}
        arms[arm] = (items, res)
    ids = sorted(set(arms["O"][1]) & set(arms["V"][1]))
    with open(out / "index.jsonl", "x", encoding="utf-8") as fh:
        for i in ids:
            g = {arm: flat(front.bicodec_tokenizer.encode_wav_to_tokens(arms[arm][0][i]["voice_audio"]))[:32]
                 for arm in ("O", "V")}
            # the talker's semantic codes are not kept in the results (blocks hold only their count): re-encode each
            # arm's generated speech with BiCodec and take its semantic tokens; all four cells share this round trip
            c = {arm: flat(front.bicodec_tokenizer.encode_wav_to_tokens(arms[arm][1][i]["speech_path"]))[32:]
                 for arm in ("O", "V")}
            rec = {"id": i, "same_global": g["O"] == g["V"], "n_codes": {k: len(v) for k, v in c.items()}}
            for ca in ("O", "V"):
                for ga in ("O", "V"):
                    if not c[ca]:
                        continue
                    y = np.asarray(front.decode(torch.tensor([*g[ga], *c[ca]], dtype=torch.long, device=dev)),
                                   dtype=np.float32).reshape(-1)
                    p = out / f"{i}__{ca}{ga}.wav"
                    sf.write(str(p), y, 16000)
                    rec[ca + ga] = str(p)
            fh.write(json.dumps(rec) + "\n")
    print("done", len(ids), out)


if __name__ == "__main__":
    main()
