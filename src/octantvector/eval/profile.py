"""Taste clusters from a set of liked maps (personal.view_score scores against them), and the difficulty-fit term."""
import numpy as np


def spherical_kmeans(X, k=3, weights=None, seed=0, iters=25, restarts=4):
    """Weighted spherical k-means (cosine) with k-means++ seeding; X rows must be unit-norm. -> (k, d) centroids."""
    X = np.asarray(X, np.float64)
    w = np.ones(len(X)) if weights is None else np.asarray(weights, np.float64)
    k = min(k, len(X))
    best, best_obj = None, -np.inf
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        cent = [X[rng.choice(len(X), p=w / w.sum())]]
        for _ in range(k - 1):
            d = np.clip(1 - np.max(X @ np.array(cent).T, axis=1), 0, None) * w  # clip: float error can go < 0
            cent.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
        cent = np.array(cent)
        for _ in range(iters):
            lab = np.argmax(X @ cent.T, axis=1)
            new = np.array([(w[lab == j, None] * X[lab == j]).sum(0) if np.any(lab == j) else cent[j] for j in range(k)])
            new /= np.linalg.norm(new, axis=1, keepdims=True)
            if np.allclose(new, cent):
                break
            cent = new
        obj = (w * np.max(X @ cent.T, axis=1)).sum()
        if obj > best_obj:
            best, best_obj = cent, obj
    return best


def difficulty_band(points, lo_pct=10, hi_pct=90):
    """points: (n, 2) array of (nps, njs) of the difficulty actually chosen on finished plays.
    -> (lo, hi, sd) per dimension; the comfortable range is [lo, hi]."""
    a = np.asarray(points, np.float64)
    return np.percentile(a, lo_pct, axis=0), np.percentile(a, hi_pct, axis=0), a.std(0)


def fit_penalty(difficulties, lo, hi, sd):
    """difficulties: per map, an (n_diff, 2) array of (nps, njs). -> per map the best (least negative) penalty
    -0.5 * sum((distance outside the band / sd)^2) over its difficulties; 0 means some difficulty is in range."""
    out = np.empty(len(difficulties))
    for i, D in enumerate(difficulties):
        if len(D) == 0:
            out[i] = -2.0
            continue
        z = np.maximum(0, np.maximum(lo - D, D - hi)) / sd
        out[i] = (-0.5 * (z ** 2).sum(1)).max()
    return out
