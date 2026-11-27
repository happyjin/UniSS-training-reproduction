"""portfolio 2026-11-27 (method E1): observation-horizon tags appended to the translation instruction. Training (train_p2ft) and
inference (hsc22 --draft-tag) build the tagged heads with this one function, so the strings are identical."""
TAG_P = "\n(The audio stops before the speaker has finished.)"
TAG_C = "\n(The audio is complete.)"
TGT = {"en2zh": "cmn", "zh2en": "eng"}


def tagged_heads(proc):
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    ins = inference_instruction()
    hp, hc = {}, {}
    for d in ("en2zh", "zh2en"):
        for tag, h in ((TAG_P, hp), (TAG_C, hc)):
            conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins[TGT[d]] + tag}]}]
            h[d] = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    return hp, hc
