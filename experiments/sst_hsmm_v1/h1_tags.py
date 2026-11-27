"""plan v3 (change 56): drafter heads for H1 (horizon-limited anticipation) and H4 (natural-monotone tag). Training (train_h1 /
train_h4) and inference (hsc22 --draft-tag --tag-set h1|mono) build the heads with this one function, so the strings are identical."""
from experiments.sst_hsmm_v1.p2ft_tags import TAG_C, TGT

TAG_H = "\n(The audio stops before the speaker has finished. Translate what has been said; anticipate at most the next second.)"
TAG_M = "\n(Keep the speaker's word order where it stays natural.)"


def heads(proc, kind):
    """-> (heads for non-final reads, heads for the final read), per direction. kind: h1 | mono | plain"""
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    ins = inference_instruction()
    tags = {"h1": (TAG_H, TAG_C), "mono": (TAG_M, TAG_M), "plain": ("", "")}[kind]
    out = ({}, {})
    for d in ("en2zh", "zh2en"):
        for tag, h in zip(tags, out):
            conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins[TGT[d]] + tag}]}]
            h[d] = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    return out
