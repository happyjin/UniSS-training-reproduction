#!/usr/bin/env python
"""v11 P0-2 (reports/mist_w3po_v1/v11_p02_alieval_prereg.zh-CN.md): joint_orch.py (or joint_orch_wb.py with
--wait-bias) on the AliMeeting Eval clean set. Only the AliMeeting set definition is swapped, in this process:
  final_test_orch.SETS["ali"]  win  eval_outputs/mist_w3po_v1/data/windows_alieval_c2g06_v1/windows_test_online_ali.jsonl
                               ref  data/diarist_alimeeting_refs_v1/windows_eval_ref_human.jsonl (DiariST dev, human)
                               data data/alimeeting_eval_si_v1 (near_mix, nemotron_lowlat), sset "ali"
  joint_orch.ZH_WIN["ali"]     the same windows
Everything else (decode flags, talkers, placement, --cap135, ledger, scoring, latency) is the wrapped orchestrator's.
--set ali is required. Additive: no wrapped file is changed.

    python alieval_orch.py [--wait-bias <b>] --dir zh2en --set ali --base <merged> ... <joint_orch.py arguments>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_test_orch as F  # noqa: E402
import joint_orch as J  # noqa: E402

WIN = "eval_outputs/mist_w3po_v1/data/windows_alieval_c2g06_v1/windows_test_online_ali.jsonl"
F.SETS["ali"] = {"win": WIN, "ref": "data/diarist_alimeeting_refs_v1/windows_eval_ref_human.jsonl", "sset": "ali",
                 "data": "data/alimeeting_eval_si_v1", "sess": "near_mix", "diar": "nemotron_lowlat"}
J.ZH_WIN["ali"] = WIN


def main() -> None:
    argv = sys.argv[1:]
    if "--set" not in argv or argv[argv.index("--set") + 1] != "ali":
        raise SystemExit("--set ali required (the AliMeeting Eval swap)")
    if "--wait-bias" in argv:
        import joint_orch_wb as WB
        WB.main(argv)
    else:
        J.main(argv)


if __name__ == "__main__":
    main()
