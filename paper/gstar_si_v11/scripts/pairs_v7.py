"""The comparisons of paper v7 (bootstrap_v7.py). Registered family (prereg_v7, Holm over eight): E-on1..3 (whole
meetings, MIST-Simul vs MIST + LA2, tcpBLEU), E-cas1..3 (offline MIST vs the turn-trained cascade, cpBLEU_sess),
E-seed-en / E-seed-zh (seed-averaged MIST-Simul vs MIST + LA2, cpBLEU_sess). Everything else is exploratory."""
from systems_v7 import SC7

PAIRS = []
for dr in ("en2zh", "zh2en", "ali"):
    t = lambda k: f"{dr}:{k}"   # noqa: E731
    PAIRS += [(t("O2"), t("C-turn")), (t("O2"), t("C-ft2")), (t("C-turn"), t("C-ft2")), (t("O2"), t("O2-nosd-relabel")),
              (t("S2dl_d1"), t("O2_LA2_d1"))]
    if dr != "ali":
        PAIRS += [(t("O2"), t("OC")), (t("O2"), t("CC")), (t("S2dl_d1"), t("SC")),
                  ("sp:" + t("O2"), "sp:" + t("C-ft2")), ("sp:" + t("S2dl_d1"), "sp:" + t("O2_LA2_d1"))]
PAIRS += [("en2zh:S2dl_d1", "en2zh:S2dl_LA2_full"), ("en2zh:S2dl_LA2_full", "en2zh:O2_LA2_d1"),
          ("zh2en:S2dl_d1", "zh2en:S2dlZ_LA2_d1"), ("zh2en:S2dlZ_LA2_d1", "zh2en:O2_LA2_d1"),
          ("ali:S2dl_d1", "ali:S2dlZ_LA2_d1"), ("ali:S2dlZ_LA2_d1", "ali:O2_LA2_d1"),
          ("en2zh:S2dl_d1", "en2zh:S2dl_lag1_sub"), ("en2zh:S2dl_d1", "en2zh:S2dl_lag2_sub"),
          ("en2zh:S2dl_lag2_sub", "en2zh:O2_LA2_d1"),
          ("sp:ali:O2", "sp:ali:C-ft2"), ("sp:ali:S2dl_d1", "sp:ali:O2_LA2_d1")]

N_ON = {"en2zh": 3387, "zh2en": 3365, "ali": 2837}
ONLINE = [(s, "on_S2dl", "on_O2_LA2", str(SC7 / f"on_{s}_S2dl_full.json"), str(SC7 / f"on_{s}_O2_LA2_full.json"), N_ON[s])
          for s in ("en2zh", "zh2en", "ali")]
ONLINE += [(s, "sp2_on_S2dl", "sp2_on_O2_LA2", str(SC7 / f"sp_on_{s}_S2dl_full2.json"), str(SC7 / f"sp_on_{s}_O2_LA2_full2.json"),
            N_ON[s]) for s in ("en2zh", "zh2en", "ali")]

SEEDS = [("en2zh", ["S2dl_d1", "S2dl_seed2_full", "S2dl_seed3_d1"], "O2_LA2_d1"),
         ("zh2en", ["S2dl_d1", "S2dlZ_seed2_d1"], "O2_LA2_d1"),
         ("ali", ["S2dl_d1", "S2dlZ_seed2_d1"], "O2_LA2_d1")]

REGISTERED = {"on:en2zh:on_S2dl_vs_on_O2_LA2": "E-on1", "on:zh2en:on_S2dl_vs_on_O2_LA2": "E-on2",
              "on:ali:on_S2dl_vs_on_O2_LA2": "E-on3",
              "en2zh:O2_vs_en2zh:C-turn": "E-cas1", "zh2en:O2_vs_zh2en:C-turn": "E-cas2", "ali:O2_vs_ali:C-turn": "E-cas3",
              "seeds:en2zh:S2dl_d1+S2dl_seed2_full+S2dl_seed3_d1_vs_O2_LA2_d1": "E-seed-en",
              "seeds:zh2en:S2dl_d1+S2dlZ_seed2_d1_vs_O2_LA2_d1": "E-seed-zh"}
