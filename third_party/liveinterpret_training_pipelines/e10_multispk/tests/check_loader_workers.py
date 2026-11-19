"""Multi-worker DataLoader stress test of the E10 data path (CPU): the main process encodes one row first (as swift
does before training), then a DataLoader with --workers processes encodes --n pointer rows. Without the per-process
JsonlIndex reopen (e10.patch._patch_jsonl_index) the workers share one file offset and hit torn lines.

    $SIMULS2S/bin/python e10_multispk/tests/check_loader_workers.py --records e10_multispk/data/li_s5_records_v1 \
        --model checkpoints/S4_1031v2_ckpt93767 [--no-fix]
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
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-fix", action="store_true", help="reproduce the failure without the reopen patch")
    a = ap.parse_args()
    R = os.path.abspath(a.records)
    os.environ.update({"SI_S4_DATA": f"{R}/records/en2zh/train/records,{R}/records/zh2en/train/records",
                       "SI_S4_DATA_ROOT": R, "SI_S4_VAL_DIR": f"{R}/val", "SI_S4_STRATIFIED": "0",
                       "SI_MAX_LENGTH": "1024", "SI_MAX_AUDIO_SEC": "30", "SI_S4_SEED": "20261026"})
    import torch
    from transformers import AutoTokenizer
    from e10 import patch
    if a.no_fix:
        patch._patch_jsonl_index = lambda: None
    patch.install()
    from streaming_si.data import s4_pipeline as P
    from streaming_si.model.base_model import VoxtralAudioProcessor
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
    tok = AutoTokenizer.from_pretrained(a.model)
    P.add_s4_stream_tokens(tok, capacity=int(json.load(open(os.path.join(a.model, "config.json")))["vocab_size"]))
    enc = P.S4Encoder(tok, VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(a.model)))
    rows = P.build_train_rows(enc.cfg)
    row = lambda i: {k: v[i] for k, v in rows.items()}  # noqa: E731
    enc.encode(row(0))                                 # main process reads first (opens the file)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return a.n

        def __getitem__(self, i):
            return enc.encode(row(i))

    dl = torch.utils.data.DataLoader(DS(), batch_size=4, num_workers=a.workers, collate_fn=P.collate_s4)
    n = 0
    for b in dl:
        n += len(b["record_ids"])
    print(json.dumps({"ok": n, "workers": a.workers, "fix": not a.no_fix}))


if __name__ == "__main__":
    main()
