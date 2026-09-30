"""
Reciprocal Peer Review: concentration of accepted topics around the
establishment's tastes under varying strengths of reciprocity (rho).

Produces a single plot: mean Euclidean distance from each accepted paper's
topic to the prolific clique's taste centroid, over submission rounds,
averaged across 25 replicates, for rho in {0, 3, 15}.
Stronger reciprocity => accepted topics more concentrated around
prolific scientists' tastes.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- parameters ----------
N            = 100       # number of scientists
D            = 2         # topic dimensions
T            = 100       # submission rounds
L            = 3         # reviewers per paper
PI           = 0.25      # target acceptance rate
DELTA        = 0.5       # reviewer-assignment matching strength
GAMMA        = 0.2       # quality penalty: distance from author's identity
BETA         = 1.5       # taste penalty: distance from reviewer's taste
SIGMA_NOISE  = 0.3       # review score noise
SIGMA_X      = 0.10      # within-author topic variability
S_BAR        = 5.0       # baseline score (cancels out; kept for interpretability)
ALPHA        = 0.1       # rejected-paper weight in identity update
LAMBDA       = 0.25      # theta plasticity
ETA          = 0.25      # tau plasticity
N_PROLIFIC   = 5         # number of prolific scientists
MU_PROLIFIC  = 20.0      # submission rate for prolific scientists
MU_REGULAR   = 2.0       # submission rate for regular scientists
PROLIFIC_SCALE = 0.2     # tight clique of establishment tastes near origin
REGULAR_SCALE  = 1.5     # regulars spread out
REPLICATES   = 25
RHOS         = [0.0, 3.0, 15.0]


def sample_reviewers_without_replacement(weights: np.ndarray,
                                         L: int,
                                         rng: np.random.Generator) -> np.ndarray:
    """Sequential weighted sampling without replacement."""
    w = weights.copy()
    n = len(w)
    L_eff = min(L, int(np.sum(w > 0)))
    chosen = np.empty(L_eff, dtype=np.int64)
    for k in range(L_eff):
        total = w.sum()
        if total <= 0:
            return chosen[:k]
        p = w / total
        idx = rng.choice(n, p=p)
        chosen[k] = idx
        w[idx] = 0.0
    return chosen


def run_once(rho: float, seed: int) -> np.ndarray:
    """One replicate. Returns array of length T with per-round mean distance
    from accepted-paper topics to the prolific clique's taste centroid."""
    rng = np.random.default_rng(seed)

    # regulars are spread out; prolific clique is tightly clustered near origin
    theta = rng.normal(0.0, REGULAR_SCALE, size=(N, D))
    tau   = rng.normal(0.0, REGULAR_SCALE, size=(N, D))
    prolific = rng.choice(N, size=N_PROLIFIC, replace=False)
    theta[prolific] = rng.normal(0.0, PROLIFIC_SCALE, size=(N_PROLIFIC, D))
    tau[prolific]   = theta[prolific].copy()

    mu = np.full(N, MU_REGULAR)
    mu[prolific] = MU_PROLIFIC

    concentration = np.full(T, np.nan)

    for t in range(T):
        # 1. submissions
        k = rng.poisson(mu)
        authors, topics = [], []
        for i in range(N):
            if k[i] == 0:
                continue
            xs = rng.normal(theta[i], SIGMA_X, size=(k[i], D))
            authors.extend([i] * k[i])
            topics.append(xs)
        if not topics:
            continue
        authors = np.array(authors, dtype=np.int64)
        topics  = np.concatenate(topics, axis=0)
        P = topics.shape[0]

        # 2. reviewer availability
        w_avail = 1.0 + rho * k

        # 3. reviewer assignment + scoring
        d2_taste   = np.sum((topics[:, None, :] - tau[None, :, :]) ** 2, axis=-1)
        base_wts   = np.exp(-DELTA * d2_taste) * w_avail[None, :]
        d2_quality = np.sum((topics - theta[authors]) ** 2, axis=-1)

        scores = np.zeros(P)
        for p in range(P):
            wp = base_wts[p].copy()
            wp[authors[p]] = 0.0
            reviewers = sample_reviewers_without_replacement(wp, L, rng)
            if len(reviewers) == 0:
                scores[p] = S_BAR
                continue
            noise = rng.normal(0.0, SIGMA_NOISE, size=len(reviewers))
            s = (S_BAR
                 - GAMMA * d2_quality[p]
                 - BETA  * d2_taste[p, reviewers]
                 + noise)
            scores[p] = s.mean()

        # 4. acceptance
        n_accept = max(1, int(round(PI * P)))
        jitter = rng.uniform(0, 1e-9, size=P)
        order = np.argsort(-(scores + jitter))
        accepted = np.zeros(P, dtype=bool)
        accepted[order[:n_accept]] = True

        # 5. record concentration of accepted topics around prolific taste centroid
        clique_center = tau[prolific].mean(axis=0)
        accepted_topics = topics[accepted]
        concentration[t] = np.linalg.norm(
            accepted_topics - clique_center, axis=1
        ).mean()

        # 6. identity updates
        weights_p = np.where(accepted, 1.0, ALPHA)
        new_theta = theta.copy()
        for i in range(N):
            mask = (authors == i)
            if not mask.any():
                continue
            wpi = weights_p[mask]
            xpi = topics[mask]
            centroid = (wpi[:, None] * xpi).sum(axis=0) / wpi.sum()
            new_theta[i] = (1 - LAMBDA) * theta[i] + LAMBDA * centroid
        theta = new_theta
        tau = (1 - ETA) * tau + ETA * theta

    return concentration


def main():
    results = {}
    for rho in RHOS:
        runs = np.stack([run_once(rho, seed=1000 * int(rho * 10) + r)
                         for r in range(REPLICATES)])
        results[rho] = runs
        print(f"rho={rho:>5.1f}  round-1 concentration = {np.nanmean(runs[:, 0]):.3f}  "
              f"final = {np.nanmean(runs[:, -1]):.3f}")

    plt.figure(figsize=(7.5, 5))
    colors = {0.0: "#1f77b4", 3.0: "#ff7f0e", 15.0: "#d62728"}
    t_axis = np.arange(1, T + 1)
    for rho in RHOS:
        runs = results[rho]
        mean = np.nanmean(runs, axis=0)
        se   = np.nanstd(runs, axis=0, ddof=1) / np.sqrt(REPLICATES)
        plt.plot(t_axis, mean, color=colors[rho], lw=2,
                 label=fr"$\rho = {rho:g}$")
        plt.fill_between(t_axis, mean - se, mean + se,
                         color=colors[rho], alpha=0.20, linewidth=0)

    plt.xlabel("Submission round $t$")
    plt.ylabel("Mean distance of accepted topics\nfrom establishment taste centroid")
    plt.title("Stronger reciprocal review concentrates accepted topics\n"
              "around prolific scientists' tastes")
    plt.legend(title="Reciprocity strength", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reciprocal_review_concentration.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
