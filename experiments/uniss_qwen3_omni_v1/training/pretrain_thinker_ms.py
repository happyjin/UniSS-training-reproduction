"""Megatron entry point for E2E-v0 (docs/multi_speaker_si_plan_v2.zh-CN.md): Plan 4A's selfwait training
(pretrain_thinker_sd.py, its model, batch layout and loss) on the multi-speaker rows of
data_clean/build_ami_ms_rows.py. A row with "inst": "ms" is prompted with MS_INSTRUCTION -- the E1 instruction
plus how to label speakers -- and a row without it (the Plan 4A rows mixed in) with the E1 instruction alone, so
the single-speaker behaviour is trained under the prompt it is served with.
"""

from __future__ import annotations

from experiments.uniss_qwen3_omni_v1.training import pretrain_thinker_sd as sd
from experiments.uniss_qwen3_omni_v1.training.ms_prompt import ms_instruction


def head(key: str) -> str:
    """key = "<direction>|<inst>"; the chat-template head the reply continues."""
    if key not in sd._HEADS:
        direction, _, inst = key.partition("|")
        from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

        ins = ms_instruction(direction) if inst == "ms" else inference_instruction()[
            "cmn" if direction.endswith("zh") else "eng"]
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins}]}]
        sd._HEADS[key] = sd._PROCESSOR.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    return sd._HEADS[key]


class MSRows(sd.SDRows):
    def __getitem__(self, i):
        item = super().__getitem__(i)
        r = self.rows[int(self.order[int(i)])]
        item["dir"] = f"{r['dir']}|{r.get('inst', 'sd')}"
        return item


sd.head = head
sd.SDRows = MSRows

if __name__ == "__main__":
    sd.main()
