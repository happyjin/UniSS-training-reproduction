"""E10 v3 data-pipeline check (CPU, real S4 tokenizer + mel front end): records -> pointer rows -> v3-patched
S4Encoder -> collate. Asserts for every encoded sample:
  * instruction is the record's; one speaker tag per written chunk, none for sd rows;
  * decision labels (wait / W) are supervised only on streaming records, masked on offline / sd records,
    while text labels and EOS stay supervised everywhere;
  * audio_pad count == encoder frame count; speaker slot count == e10.v3.n_slots(frames, layout);
  * every slot sits right after the audio frame given by the layout rule (stream: frame 5j+5+13, offline: 5j+5 or
    the last frame), counted over the audio_pad positions; sd rows have no slot and no posterior;
  * spk_post has >= 5 * slots rows and (unperturbed val rows) equals the raw posterior;
  * collate gives spk_posts / spk_ns.

    $SIMULS2S/bin/python e10_multispk/tests/check_data_v3.py --records e10_multispk/data/li_v3_joint \
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
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    R = os.path.abspath(a.records)
    os.environ.update({
        "SI_S4_DATA": f"{R}/records/en2zh/train/records,{R}/records/zh2en/train/records",
        "SI_S4_DATA_ROOT": R, "SI_S4_VAL_DIR": f"{R}/val", "SI_S4_INDEX_DIR": "", "SI_S4_STRATIFIED": "0",
        "SI_MAX_LENGTH": "1280", "SI_MAX_AUDIO_SEC": "30", "SI_MAX_SPEECH_LEN": "750", "SI_S4_USE_SRC_END": "1",
        "SI_S4_LATENCY_MULTIPLIER": "1", "SI_S4_SEED": "20261026"})
    import numpy as np
    from transformers import AutoTokenizer
    from streaming_si.data.s4_pipeline import add_s4_stream_tokens, s4_stream_token_ids, s4_env, build_train_rows, \
        build_val_rows
    from streaming_si.model.base_model import VoxtralAudioProcessor
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
    from e10 import v3
    from e10.common import spk_token_ids, instruction
    v3.install()
    from streaming_si.data import s4_pipeline as P

    cfgm = json.load(open(os.path.join(a.model, "config.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    add_s4_stream_tokens(tok, capacity=int(cfgm["vocab_size"]))
    wait_id, write_id = s4_stream_token_ids(tok)[:2]
    spk = spk_token_ids(tok)
    slot = v3.slot_id(tok)
    assert slot not in spk and slot not in s4_stream_token_ids(tok), slot
    audio_id = tok.convert_tokens_to_ids("<|audio_pad|>")
    print("[check] slot id", slot, "speaker ids", spk)
    mel = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(a.model))
    enc = P.S4Encoder(tok, mel)
    cfg = s4_env()
    rows, val = build_train_rows(cfg), build_val_rows(cfg)
    by_id = {}
    for ds in (enc.train_ds(), enc.val_ds("en2zh"), enc.val_ds("zh2en")):
        for fi, li in ds.items:
            r = ds.indexes[fi].read(li)
            by_id[r["record_id"]] = r

    def rows_iter():
        n = len(rows["s4_index"])
        step = max(1, n // a.n)
        for i in range(0, n, step):
            yield {k: v[i] for k, v in rows.items()}
        for i in range(min(a.n // 3, len(val["s4_index"]))):
            yield {k: v[i] for k, v in val.items()}

    batch, kinds, n_ok = [], {}, 0
    for row in rows_iter():
        item = enc.encode(row)
        rec = by_id[item["record_id"]]
        e10, lay = rec["e10"], rec["e10"]["v3"]
        ids = item["input_ids"].tolist()        # S4Encoder.encode already un-shifts (unshift_s4_item)
        lab = item["labels"].tolist()
        instr = tok.encode(instruction(rec["target_lang"], e10["inst"] == "gstar"), add_special_tokens=False)
        assert ids[: len(instr)] == instr, item["record_id"]
        # decisions
        dec_pos = [i for i, t in enumerate(ids) if t in (wait_id, write_id)]
        sup = [lab[i] != -100 for i in dec_pos]
        if lay["mask_decision"]:
            assert not any(sup), (item["record_id"], "decision supervised on a masked record")
        else:
            assert all(sup), (item["record_id"], "decision masked on a streaming record")
        assert lab[-1] == tok.eos_token_id
        n_text = sum(1 for i, t in enumerate(ids) if lab[i] != -100 and t not in (wait_id, write_id))
        assert n_text > 0, item["record_id"]
        n_tags = sum(l in spk for l in lab)
        n_chunks = sum(s["text_delta"].count("<|dur:") for s in rec["trajectory"]["steps"])
        assert n_tags == n_chunks, (item["record_id"], n_tags, n_chunks)
        # audio / slots
        n_pad = ids.count(audio_id)
        assert n_pad == int(item["audio_seg_lens"].sum()), item["record_id"]
        offline = lay["layout"] == "offline"
        exp = 0 if lay["layout"] == "none" else v3.n_slots(n_pad, 0 if offline else v3.STREAM_DELAY, offline)
        assert item["spk_n"] == exp == ids.count(slot), (item["record_id"], item["spk_n"], exp, ids.count(slot))
        g, j = 0, 0
        for t in ids:
            if t == audio_id:
                g += 1
            elif t == slot:
                due = (n_pad if (offline and j == n_pad // 5 and n_pad % 5) else
                       v3.block_due(j, 0 if offline else v3.STREAM_DELAY))
                assert g == due, (item["record_id"], j, g, due)
                j += 1
        assert all(lab[i] == -100 for i, t in enumerate(ids) if t in (slot, audio_id))
        if exp:
            sp = item["spk_post"]
            assert sp.shape[0] >= 5 * exp and sp.shape[1] == 8
            if row["s4_split"] != "train":
                raw = np.load(lay["post"]).astype(np.float32)[:, :8]
                m = min(len(raw), sp.shape[0])
                assert np.allclose(sp[:m].numpy(), raw[:m]), item["record_id"]
        else:
            assert item["spk_post"] is None
        k = e10["kind"] + ("" if row["s4_split"] == "train" else "/val")
        kinds[k] = kinds.get(k, 0) + 1
        if kinds[k] == 1:
            print(f"[check] {item['record_id']} {k} len={len(ids)} pads={n_pad} slots={exp} "
                  f"dec_supervised={sum(sup)}/{len(sup)}\n   labels: "
                  f"{tok.decode([t for t in lab if t != -100])[:220]}")
        n_ok += 1
        if len(batch) < 6:
            batch.append(item)
    out = P.collate_s4(batch)
    assert len(out["spk_posts"]) == len(out["spk_ns"]) == out["input_ids"].size(0)
    print(json.dumps({"ok": n_ok, "kinds": kinds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
