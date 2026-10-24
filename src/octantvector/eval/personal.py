"""Leave-one-out evaluation against a player's own liked maps (protocol: OctantSpecs octantvector/design.md).

Candidates = maps the player interacted with that are in the index; liked maps by OctantSpecs law 4.1 (runtime/likes.py,
the rule recommendations use); everything else unlabeled. For each held-out liked map p, the profile, clusters and difficulty band are fit WITHOUT p, and p is
ranked among the candidates minus the other liked maps. Reported: hit@10/20, median rank, MRR.
"""
import numpy as np

from ..runtime import likes
from .profile import difficulty_band, fit_penalty, spherical_kmeans


def view_score(Xc, liked_local, kind="km3", r_hub=None):
    """z-scored profile score over candidate rows for one view. liked_local: indices into Xc."""
    X = Xc[liked_local]
    if kind == "mean":
        q = X.mean(0)
        s = Xc @ (q / np.linalg.norm(q))
    else:
        s = (Xc @ spherical_kmeans(X, int(kind[2:])).T).max(1)
    if r_hub is not None:  # CSLS: 2cos - r_q - r_c; r_q is constant per query, so rank by cos - r_c / 2
        s = s - 0.5 * r_hub
    return (s - s.mean()) / s.std()


def loo_ranks(score_fn, liked_local, n_cand):
    """score_fn(p, others) -> scores over candidates. -> 1-based rank of each held-out p."""
    liked_local = np.asarray(liked_local)
    ranks = []
    for p in liked_local:
        others = liked_local[liked_local != p]
        s = np.array(score_fn(p, others), np.float64)
        s[others] = -np.inf
        ranks.append(1 + int((s > s[p]).sum()))
    return np.array(ranks)


def summarise(ranks, n_rankable):
    r = np.asarray(ranks)
    return {"n_liked": len(r), "rank_base": n_rankable, "hit@10": 100 * float(np.mean(r <= 10)),
            "hit@20": 100 * float(np.mean(r <= 20)), "median_rank": float(np.median(r)), "mrr": float(np.mean(1 / r)),
            "random_hit@10": 100 * 10 / n_rankable, "random_median_rank": n_rankable / 2}


def labels(merged):
    """merged: {pool hash: signal Counter} -> {pool hash: 1 (liked) | 0 (unlabeled)} for interacted maps. The signals
    carry no dates, so "played again on a later day" never counts here."""
    from .playdata import INTERACTION
    return {h: int(likes.classify(s) == "liked") for h, s in merged.items() if any(s.get(k) for k in INTERACTION)}


def run(index, cand_hashes, liked_hashes, kind="km3", csls=True, beta=0.0,
        difficulties=None, finished_points=None, csls_density="view", key_of=None, keep_ranks=False):
    """Late fusion of per-view z-scores (weighted by the index weights) + beta * difficulty fit.
    difficulties: {hash: (n_diff, 2) nps/njs}; finished_points: [(BeatSaver key of the played map, nps, njs)] of
    finished plays (band source); key_of: {candidate hash: key}. The band for a held-out map excludes every play of
    that key, whichever map version was played.
    csls_density: "view" = hub density measured inside each view (matches the per-view score it corrects);
    "combined" = one density from the weighted concatenation of all index views, applied to every view.
    keep_ranks: also return "ranks", {held-out hash: rank} (personal data: results.py writes it only where asked)."""
    cand = [index.pos[h] for h in cand_hashes]
    pos = {h: j for j, h in enumerate(cand_hashes)}
    liked = np.array([pos[h] for h in liked_hashes if h in pos])
    weights = {v: w for v, (_, w) in index.views.items()}
    tot = sum(weights.values())
    Xs = {v: index.views[v][0][cand] for v in weights}
    r_hub = {v: None for v in weights}
    if csls:
        if csls_density == "combined":
            r_all = index.local_density(10)[cand]
            r_hub = {v: r_all for v in weights}
        elif csls_density == "view":
            r_hub = {v: index.view_density(v, 10)[cand] for v in weights}
        else:
            raise ValueError(f"unknown csls_density {csls_density!r}")
    pen_cache = {}

    def penalty(p):
        if beta == 0 or difficulties is None:
            return 0.0
        h = cand_hashes[p]
        if h not in pen_cache:
            key = key_of[h] if key_of else h
            pts = np.array([(n, j) for k, n, j in finished_points if k != key])
            lo, hi, sd = difficulty_band(pts)
            pen_cache[h] = fit_penalty([difficulties.get(c, np.zeros((0, 2))) for c in cand_hashes], lo, hi, sd)
        return pen_cache[h]

    def score(p, others):
        s = sum(weights[v] / tot * view_score(Xs[v], others, kind, r_hub[v]) for v in weights)
        return s + beta * penalty(p)

    ranks = loo_ranks(score, liked, len(cand))
    res = summarise(ranks, len(cand) - len(liked) + 1)
    if keep_ranks:
        res["ranks"] = {cand_hashes[p]: int(r) for p, r in zip(liked, ranks)}
    return res
