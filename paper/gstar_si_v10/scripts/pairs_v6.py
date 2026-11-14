"""The comparisons of paper v6 (bootstrap_v6.py). Pre-registered family (prereg_v6, Holm): E-on1..3 (ONLINE, MIST-Simul
vs MIST + LA2 on whole meetings), E-seed (scripts/seeds_v6.py), E-zh (zh2en S2dlchar_d1 vs O2_LA2_d1). Everything else
is exploratory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SC = ROOT / "eval_outputs/uniss_qwen3_omni_v1/gstar_v6_20261031/scores"

PAIRS = []
for dr in ("en2zh", "zh2en"):
    t = lambda k: f"{dr}:{k}"   # noqa: E731
    PAIRS += [(t("O2"), t("C-ft2")), (t("O2"), t("O2-nosd-relabel")), (t("O2"), t("OC")), (t("O2"), t("CC")),
              (t("O2"), t("C-turn")), (t("O2-nosd-relabel"), t("C-ft2")),
              (t("S2dl_d1"), t("O2_LA2_d1")), (t("S2dl_d1"), t("S2dl_LA2_d1")), (t("S2dl_d1"), t("S2dl_lag2")),
              (t("S2dl_d1"), t("SC")), (t("S2dlabc_d1"), t("S2_b0_d1")),
              ("sp:" + t("O2"), "sp:" + t("C-ft2")), ("sp:" + t("S2dl_d1"), "sp:" + t("O2_LA2_d1"))]
PAIRS += [("en2zh:S2dl_seed1_d1", "en2zh:O2_LA2_d1"), ("en2zh:S2dl_seed1_d1", "en2zh:S2dl_d1"),
          ("en2zh:S2dlm_d1", "en2zh:S2dlm_LA2_d1"), ("en2zh:S2dl_d1", "en2zh:S2dl_lag1"), ("en2zh:S2dl_d1", "en2zh:S2dl_lag3"),
          ("zh2en:S2dlchar_d1", "zh2en:O2_LA2_d1"), ("zh2en:S2dlchar_d1", "zh2en:S2dl_d1")]
PAIRS += [("ali:O2", "ali:C-ft2"), ("ali:O2", "ali:O2-nosd-relabel"), ("ali:O2-nosd-relabel", "ali:C-ft2"),
          ("ali:S2dl_d1", "ali:O2_LA2_d1"), ("sp:ali:O2", "sp:ali:C-ft2"), ("sp:ali:S2dl_d1", "sp:ali:O2_LA2_d1"),
          ("ali:S2dlchar_d1", "ali:O2_LA2_d1")]
REMATCH = {("en2zh:O2", "en2zh:C-ft2"), ("zh2en:O2", "zh2en:C-ft2"), ("ali:O2", "ali:C-ft2")}

# whole-meeting online runs: (dir, a, b, meeting_score json of a, of b)
ONLINE = [(d, "on_S2dl", "on_O2_LA2", str(SC / f"on_{s}_S2dl.json"), str(SC / f"on_{s}_O2_LA2.json"))
          for d, s in (("en2zh", "en2zh"), ("zh2en", "zh2en"), ("ali", "ali"))]
ONLINE += [("en2zh", "on_S2dlm", "on_O2_LA2", str(SC / "on_en2zh_S2dlm.json"), str(SC / "on_en2zh_O2_LA2.json"))]

# the pre-registered family (prereg_v6, Holm over these five): key in bootstrap_v6.json -> label
REGISTERED = {"on:en2zh:on_S2dl_vs_on_O2_LA2": "E-on1", "on:zh2en:on_S2dl_vs_on_O2_LA2": "E-on2",
              "on:ali:on_S2dl_vs_on_O2_LA2": "E-on3", "en2zh:S2dl_seed1_d1_vs_en2zh:O2_LA2_d1": "E-seed",
              "zh2en:S2dlchar_d1_vs_zh2en:O2_LA2_d1": "E-zh"}
