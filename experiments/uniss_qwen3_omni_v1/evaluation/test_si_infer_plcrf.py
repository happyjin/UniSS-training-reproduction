"""Unit tests for si_infer's talker-scheme helpers (build_seq, backlog_bin); CPU, no vLLM."""
from experiments.uniss_qwen3_omni_v1.evaluation import si_infer as S
from experiments.uniss_qwen3_omni_v1.training.e2_batch import (
    BALANCE, END_CONTENT, END_GLOBAL, GLM_0, GLOBAL_0, LANG_TOKEN, START_CONTENT, START_GLOBAL, START_SEMANTIC,
    TASK_S2S, TASK_S2T, TASK_TTS, WRITE)
from experiments.uniss_qwen3_omni_v1.training.si_batch import LOOK_CLOSE, LOOK_OPEN


def original(it, r, lang):
    """si_infer.main's inline sequence (the default path), copied verbatim."""
    glm = it["blocks"][r]["glm"]
    seq = [TASK_S2S, BALANCE, lang, START_GLOBAL] + [GLOBAL_0 + g for g in it["voice"]] + [END_GLOBAL]
    g_prev = 0
    for k, b in enumerate(it["blocks"][: r + 1]):
        seq += [GLM_0 + t for t in glm[g_prev: b["glm_end"]]]
        seq += [WRITE, TASK_S2T, lang, b["speed"], START_CONTENT] + b["ids"] + [END_CONTENT]
        if b["look_ids"]:
            seq += [LOOK_OPEN] + b["look_ids"] + [LOOK_CLOSE]
        seq += [WRITE, TASK_TTS, lang, b["speed"]]
        seq += ([] if b["final"] else [S.STREAMING_MODE]) + [START_SEMANTIC]
        if k < r:
            seq += [S.SEMANTIC_0 + c for c in b["codes"]] + [b["closing"]]
        g_prev = b["glm_end"]
    return seq


def item():
    sp = 180337
    return {"voice": list(range(32)), "blocks": [
        {"glm": [1, 2, 3], "glm_end": 2, "ids": [10, 11], "look_ids": [], "final": False, "speed": sp + 12,
         "speed_base": sp + 11, "codes": [5] * 30, "closing": S.WAIT_READ, "write_ms": 1000, "backlog_bin": 0},
        {"glm": [1, 2, 3, 4, 5], "glm_end": 5, "ids": [12], "look_ids": [7], "final": True, "speed": sp + 15,
         "speed_base": sp + 13, "codes": [6] * 10, "closing": S.END_SEMANTIC, "write_ms": 1200, "backlog_bin": 1}]}


def test_default_equals_original():
    lang = LANG_TOKEN["eng"]
    it = item()
    for r in (0, 1):
        assert S.build_seq(it, r, lang, False) == original(it, r, lang)


def test_backlog_and_speed():
    lang = LANG_TOKEN["eng"]
    it = item()
    s = S.build_seq(it, 1, lang, True, cur_speed=180337 + 20)
    # block 0: S2T speed = speed_base then backlog token 0; TTS speed = its own speed
    i = s.index(TASK_S2T)
    assert s[i + 2] == 180337 + 11 and s[i + 3] == S.BACKLOG_IDS[0] and s[i + 4] == START_CONTENT
    j = s.index(TASK_TTS)
    assert s[j + 2] == 180337 + 12
    # block 1 (current): S2T speed_base 13, backlog 1, TTS = cur_speed
    i2 = len(s) - 1 - s[::-1].index(TASK_S2T)
    assert s[i2 + 2] == 180337 + 13 and s[i2 + 3] == S.BACKLOG_IDS[1]
    j2 = len(s) - 1 - s[::-1].index(TASK_TTS)
    assert s[j2 + 2] == 180337 + 20 and s[-1] == START_SEMANTIC
    u = S.build_seq(it, 1, lang, True, upto_speed=True)
    assert u[-3:] == [WRITE, TASK_TTS, lang] and u == s[: j2 + 2]
    # default with cur_speed changes only block r's two speed tokens
    d = S.build_seq(it, 1, lang, False, cur_speed=180337 + 20)
    o = original(it, 1, lang)
    diff = [k for k, (a, b) in enumerate(zip(d, o)) if a != b]
    assert len(d) == len(o) and len(diff) == 2 and all(d[k] == 180337 + 20 for k in diff)


def test_backlog_bin():
    bl = [{"write_ms": 1000, "codes": [0] * 100}, {"write_ms": 1500, "codes": [0] * 50}, {"write_ms": 9000, "codes": []}]
    assert S.backlog_bin(bl, 0) == 0
    assert S.backlog_bin(bl, 1) == 3          # block 0 ends at 3000 ms -> 1.5 s queued at 1500
    assert S.backlog_bin(bl, 2) == 0          # all spoken by 9000
    bl2 = [{"write_ms": 0, "codes": [0] * 1000}, {"write_ms": 100, "codes": []}]
    assert S.backlog_bin(bl2, 1) == 7


if __name__ == "__main__":
    test_default_equals_original()
    test_backlog_and_speed()
    test_backlog_bin()
    print("ok")
