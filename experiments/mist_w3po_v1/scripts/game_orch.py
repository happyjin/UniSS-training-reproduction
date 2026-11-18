#!/usr/bin/env python
"""Game (HOK esports commentary) listening demo (reports/mist_w3po_v1/game_multispk_demo_plan.zh-CN.md §3):
joint_orch.py on the game excerpts, exactly as alieval_orch.py does for AliMeeting Eval -- only the set definition is
swapped, in this process:
  --dir zh2en --set ali   final_test_orch.SETS["ali"] = game zh2en windows / SILVER refs, data data/game_esports_si_v1/
                          zh2en (session "mix", diar nemotron_lowlat), scorer set "ali" (English tokenisation);
                          joint_orch.ZH_WIN["ali"] = the same windows
  --dir en2zh --set test  en2zh_orch.SETS["test"] = game en2zh windows / SILVER refs; every command's
                          GSTAR_DATA=data/ami_multispk_si_v1 -> data/game_esports_si_v1/en2zh (session ihm_mix,
                          diar nemotron_lowlat, as en2zh_orch writes them)
Everything else (decode flags, talkers, placement, --cap135, ledger, scoring, latency) is the wrapped orchestrator's.
Silver refs = the data pipeline's GPT translations of its own ASR turns: scores are indicative only.

    python game_orch.py --dir zh2en --set ali|--dir en2zh --set test --base <merged> ... <joint_orch.py arguments>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import en2zh_orch as E  # noqa: E402
import final_test_orch as F  # noqa: E402
import joint_orch as J  # noqa: E402

WD = "eval_outputs/mist_w3po_v1/data"
GD = "data/game_esports_si_v1"
ZWIN = f"{WD}/windows_game_zh2en_c2g06_v1/windows_test_online_zh2en.jsonl"
EWIN = f"{WD}/windows_game_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl"
F.SETS["ali"] = {"win": ZWIN, "ref": f"{GD}/zh2en/refs_windows.jsonl", "sset": "ali",
                 "data": f"{GD}/zh2en", "sess": "mix", "diar": "nemotron_lowlat"}
J.ZH_WIN["ali"] = ZWIN
E.SETS["test"] = {"win": EWIN, "ref": f"{GD}/en2zh/refs_windows.jsonl"}
_efix = E.fix


def efix(cmd: str, S: dict, test: bool = False) -> str:
    cmd = cmd.replace("GSTAR_DATA=data/ami_multispk_si_v1 ", f"GSTAR_DATA={GD}/en2zh ")
    cmd = _efix(cmd, S, test)
    assert "ami_multispk_si_v1" not in cmd and "windows_test_en2zh_c2g06" not in cmd, cmd
    return cmd


E.fix = efix


def main() -> None:
    argv = sys.argv[1:]
    d = argv[argv.index("--dir") + 1]
    s = argv[argv.index("--set") + 1]
    if (d, s) not in (("zh2en", "ali"), ("en2zh", "test")):
        raise SystemExit("--dir zh2en --set ali | --dir en2zh --set test (the game swaps)")
    J.main(argv)


if __name__ == "__main__":
    main()
