"""Route 2: Route 1's posterior adapted to each talk online, from labels the system gives itself.

When a sentence ends the whole of it has been heard, so every read of it can be labelled the way the prior was
trained: the Thinker continues that read's committed prefix with the whole sentence heard (the oracle), and unit u
of the read's draft survived iff draft[:u+1] == oracle[:u+1]. No reference is used; the labels only reach the
sentences that start after (si_thinker_stream --adapt runs a talk's sentences one after another).

  bayes  a talk-level offset on the prior's log-odds, b_s ~ N(0, sigma^2): after each sentence the MAP of b_s given
         all labelled tokens of the talk so far (Newton on the logistic likelihood), with its Laplace variance v_s;
         the commit probability is the posterior predictive, sigmoid((z + b_s) / sqrt(1 + pi v_s / 8)) (MacKay 1992)
  aci    adaptive conformal inference (Gibbs & Candes 2021) on the threshold: after each labelled read,
         tau <- tau + gamma (err - eps), err = 1 when the read committed a unit the oracle did not say; over any
         sequence of reads the long-run rate of such reads is eps within O(1 / (gamma T))
  both   the two together: the offset follows the talk, the threshold keeps the risk
"""

from __future__ import annotations

import math

import numpy as np


class TalkAdapt:
    def __init__(self, mode: str, sigma: float, gamma: float, eps: float, tau0: float):
        self.mode, self.sigma2, self.gamma, self.eps, self.tau0 = mode, sigma * sigma, gamma, eps, tau0
        self.state = {}

    def get(self, talk: str) -> dict:
        return self.state.setdefault(talk, {"b": 0.0, "v": self.sigma2, "tau": self.tau0, "z": [], "y": [],
                                            "reads": 0, "errs": 0, "trace": []})

    def probs(self, talk: str, z: np.ndarray) -> np.ndarray:
        s = self.get(talk)
        if self.mode in ("bayes", "both"):
            z = (z + s["b"]) / math.sqrt(1.0 + math.pi * s["v"] / 8.0)
        return 1.0 / (1.0 + np.exp(-z))

    def tau(self, talk: str) -> float:
        return self.get(talk)["tau"] if self.mode in ("aci", "both") else self.tau0

    def update(self, talk: str, reads: list[tuple[np.ndarray, np.ndarray, bool]]) -> None:
        """reads: (logits without the offset, token labels, committed past the oracle) of one finished sentence."""
        s = self.get(talk)
        for z, y, err in reads:
            s["reads"] += 1
            s["errs"] += int(err)
            if self.mode in ("aci", "both"):
                s["tau"] = float(min(0.99, max(0.01, s["tau"] + self.gamma * (float(err) - self.eps))))
            if len(z):
                s["z"].append(z)
                s["y"].append(y)
        if self.mode in ("bayes", "both") and s["z"]:
            z, y = np.concatenate(s["z"]), np.concatenate(s["y"])
            b = s["b"]
            for _ in range(8):   # MAP of b under N(0, sigma^2), all of the talk's labels so far
                p = 1.0 / (1.0 + np.exp(-(z + b)))
                h = float((p * (1 - p)).sum()) + 1.0 / self.sigma2
                b += (float((y - p).sum()) - b / self.sigma2) / h
            s["b"], s["v"] = b, 1.0 / h
        s["trace"].append({"b": round(s["b"], 4), "v": round(s["v"], 5), "tau": round(s["tau"], 4),
                           "reads": s["reads"], "errs": s["errs"]})

    def summary(self) -> dict:
        return {t: {"reads": s["reads"], "err_rate": round(s["errs"] / max(s["reads"], 1), 4), "b": round(s["b"], 4),
                    "tau": round(s["tau"], 4), "trace": s["trace"]} for t, s in self.state.items()}
