"""E10 data-pipeline check (CPU, real S4 tokenizer + mel front end): records -> pointer rows -> patched S4Encoder ->
patched collate. Asserts, for every encoded sample:
  * the speaker tags are single tokens and equal chunk+1..chunk+8 (e10.tuner.spk_ids_from_stream);
  * the instruction is the record's (multi-speaker for G-STAR rows, the S5 one for sd rows);
  * labels contain one speaker tag per chunk of the record, none for sd rows;
  * spk_post has one row per <|audio_pad|>, its first 13 rows are zero and (unperturbed val rows) row j = post[j-13];
  * collate gives `spk_posts` aligned with the batch.

    $SIMULS2S/bin/python e10_multispk/tests/check_data_pipeline.py --records e10_multispk/runs/smoke_records_v1 \
        --model checkpoints/S4_1031v2_ckpt93767
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E10 = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.dirname(E10), "custom"), E10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    R = os.path.abspath(a.records)
    os.environ.update({
        "SI_S4_DATA": f"{R}/records/en2zh/train/records,{R}/records/zh2en/train/records",
        "SI_S4_DATA_ROOT": R, "SI_S4_VAL_DIR": f"{R}/val", "SI_S4_INDEX_DIR": "", "SI_S4_STRATIFIED": "0",
        "SI_MAX_LENGTH": "1024", "SI_MAX_AUDIO_SEC": "30", "SI_MAX_SPEECH_LEN": "750", "SI_S4_USE_SRC_END": "1",
        "SI_S4_LATENCY_MULTIPLIER": "1", "SI_S4_SEED": "20261026"})
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from streaming_si.data.s4_pipeline import add_s4_stream_tokens, s4_stream_token_ids, s4_env, build_train_rows, \
        build_val_rows
    from streaming_si.model.base_model import VoxtralAudioProcessor
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
    from e10 import patch
    from e10.common import SD_DELAY_FRAMES, SPK_TOKENS, spk_token_ids, instruction
    from e10.tuner import spk_ids_from_stream
    patch.install()
    from streaming_si.data import s4_pipeline as P

    cfgm = json.load(open(os.path.join(a.model, "config.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    add_s4_stream_tokens(tok, capacity=int(cfgm["vocab_size"]))
    sids = s4_stream_token_ids(tok)
    spk = spk_token_ids(tok)
    assert spk == spk_ids_from_stream(sids), (spk, sids)
    print("[check] stream ids", sids, "speaker ids", spk)
    mel = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(a.model))
    enc = P.S4Encoder(tok, mel)
    cfg = s4_env()
    rows = build_train_rows(cfg)
    val = build_val_rows(cfg)
    audio_id = tok.convert_tokens_to_ids("<|audio_pad|>")
    batch, n_ok, kinds = [], 0, {}

    def rows_iter():
        for i in range(min(a.n, len(rows["s4_index"]))):
            yield {k: v[i] for k, v in rows.items()}
        for i in range(min(a.n, len(val["s4_index"]))):
            yield {k: v[i] for k, v in val.items()}

    for row in rows_iter():
        item = enc.encode(row)
        ds = enc.train_ds() if row["s4_split"] == "train" else enc.val_ds(row["s4_split"][4:])
        rid = item["record_id"]
        rec = next(r for r in (ds.indexes[fi].read(li) for fi, li in ds.items) if r["record_id"] == rid)
        e10 = rec["e10"]
        ids, labels = item["input_ids"].tolist(), item["labels"].tolist()
        gstar = e10["inst"] == "gstar"
        instr = tok.encode(instruction(rec["target_lang"], gstar), add_special_tokens=False)
        assert ids[: len(instr)] == instr, (rid, tok.decode(ids[:40]))
        n_tags = sum(l in spk for l in labels)
        n_chunks = sum(s["text_delta"].count("<|dur:") for s in rec["trajectory"]["steps"])
        assert n_tags == n_chunks, (rid, n_tags, n_chunks)
        assert gstar or n_tags == 0, rid
        n_pad = ids.count(audio_id)
        sp = item["spk_post"]
        assert tuple(sp.shape) == (n_pad, 8), (rid, tuple(sp.shape), n_pad)
        assert float(sp[:SD_DELAY_FRAMES].abs().sum()) == 0.0, rid
        if e10["post"] and row["s4_split"] != "train" and n_pad > SD_DELAY_FRAMES:
            raw = np.load(e10["post"]).astype(np.float32)
            m = min(n_pad - SD_DELAY_FRAMES, len(raw))
            assert np.allclose(sp[SD_DELAY_FRAMES: SD_DELAY_FRAMES + m].numpy(), raw[:m]), rid
        if not e10["post"]:
            assert float(sp.abs().sum()) == 0.0, rid
        kinds[e10["kind"]] = kinds.get(e10["kind"], 0) + 1
        if n_ok < 3 or (e10["kind"] != "stream" and kinds[e10["kind"]] == 1):
            txt = tok.decode([t for t in labels if t != -100], skip_special_tokens=False)
            print(f"[check] {rid} ({e10['dir']}, {e10['kind']}) len={len(ids)} pads={n_pad} tags={n_tags}\n"
                  f"        labels: {txt[:300]}")
        n_ok += 1
        if len(batch) < 4:
            batch.append(item)
    out = P.collate_s4(batch)
    assert len(out["spk_posts"]) == len(batch) and out["input_ids"].size(0) == len(batch)
    print(json.dumps({"ok": n_ok, "kinds": kinds, "collate_keys": sorted(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
