"""Figure 1 of reports/mist_w3po_v1/game_muffled_speech_analysis.zh-CN.md (venvs/tts_schemes: matplotlib):
(a) prompt background-to-vocals ratio vs translated-speech SQUIM PESQ, game vs meeting sets; (b) paired prompt swaps;
(c) the 2x2 codes x global-token decomposition. Writes reports/mist_w3po_v1/game_muffled_speech_fig1.png (refuses existing).
    python -m experiments.mist_w3po_v1.evaluation.game_muffle_figure
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
O = REPO / "eval_outputs/mist_w3po_v1/game_muffle_v1"
OUT = REPO / "reports/mist_w3po_v1/game_muffled_speech_fig1.png"


def load(n):
    return [json.loads(l) for l in open(O / f"{n}.jsonl")]


def paired(a, b, k):
    A = {r["id"]: r for r in load(a)}
    B = {r["id"]: r for r in load(b)}
    ids = [i for i in A if i in B and k in A[i]["speech"] and k in B[i]["speech"]]
    return np.array([A[i]["speech"][k] for i in ids]), np.array([B[i]["speech"][k] for i in ids])


def main():
    if OUT.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {OUT}")
    sets = [("game_zh2en", "Game Zh->En", "#c0392b"), ("game_en2zh", "Game En->Zh", "#e67e22"),
            ("ami_zh2en", "AMI Zh->En", "#2980b9"), ("ami_en2zh", "AMI En->Zh", "#16a085"),
            ("ali_zh2en", "AliMeeting Zh->En", "#8e44ad")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for s, lab, c in sets:
        L = [r for r in load(s) if "pesq" in r["speech"] and "bvr_db" in r["prompt"]]
        ax[0].scatter([max(-45, r["prompt"]["bvr_db"]) for r in L], [r["speech"]["pesq"] for r in L],
                      s=9, alpha=.45, c=c, label=f"{lab} (n={len(L)})")
    ax[0].set_xlabel("voice prompt background / vocals (dB, HDemucs; clipped at -45)")
    ax[0].set_ylabel("translated speech PESQ (SQUIM estimate)")
    ax[0].set_title("(a) noisier prompt -> worse translated voice")
    ax[0].legend(fontsize=8)
    groups = [("swap_zh2en_O", "swap_zh2en_V", "Game Zh->En\nraw -> vocals-only"),
              ("swap_en2zh_O", "swap_en2zh_V", "Game En->Zh\nraw -> vocals-only"),
              ("swap_amien_O", "swap_amien_N", "AMI En->Zh\nclean -> +game bg")]
    for j, (a, b, _) in enumerate(groups):
        x, y = paired(a, b, "pesq")
        for xi, yi in zip(x, y):
            ax[1].plot([j * 3, j * 3 + 1], [xi, yi], c="grey", alpha=.15, lw=.6)
        ax[1].boxplot([x, y], positions=[j * 3, j * 3 + 1], widths=.7, showfliers=False)
    ax[1].set_xticks([0.5, 3.5, 6.5])
    ax[1].set_xticklabels([g[2] for g in groups], fontsize=8)
    ax[1].set_ylabel("translated speech PESQ (SQUIM)")
    ax[1].set_title("(b) paired: same 80 streams + seeds, only the prompt differs")
    cells = ["OO", "OV", "VO", "VV"]
    for d, c in [("zh2en", "#c0392b"), ("en2zh", "#e67e22")]:
        L = [json.loads(l) for l in open(O / f"codec_swap/{d}/measure.jsonl")]
        L = [r for r in L if all(k in r and "pesq" in r[k] for k in cells)]
        ax[2].plot(range(4), [np.mean([r[k]["pesq"] for r in L]) for k in cells], "o-", c=c, label=f"Game {d} (n={len(L)})")
    ax[2].set_xticks(range(4))
    ax[2].set_xticklabels(["codes raw\nglobal raw", "codes raw\nglobal clean", "codes clean\nglobal raw",
                           "codes clean\nglobal clean"], fontsize=8)
    ax[2].set_ylabel("mean PESQ (SQUIM)")
    ax[2].set_title("(c) both paths: LM codes and BiCodec global tokens")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(OUT)


if __name__ == "__main__":
    main()
