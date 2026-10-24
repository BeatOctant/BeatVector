"""Catalogue-only neighbour quality (no personal data): agreement of k-NN pairs on tags, artist, song, mapper,
max NPS and BPM versus random pairs, plus hubness of the k-NN graph.

The same agreement is also given per group of queries, so a weakness confined to some songs is not hidden by the
mean: by upvotes (from the catalogue; for evaluation only, never a feature), and by isolation, the mean score of a
query's k nearest maps in one reference view. An isolated song has few maps that sound like it, and "似た曲" from it
is weaker (OctantSpecs open.md: マイナーな曲ほど似ていない). Groups are quantiles of the queries evaluated, so two runs
over the same maps with the same reference view compare the same queries."""
import re

import numpy as np

GENRE_TAGS = {"pop", "rock", "metal", "electronic", "hip-hop-rap", "dance", "jazz", "classical-orchestral", "funk-disco",
              "ambient", "alternative", "dubstep", "drum-and-bass", "vocaloid", "k-pop", "j-pop", "j-rock", "anime",
              "video-game-soundtrack", "tv-movie-soundtrack", "swing", "folk-acoustic", "house", "trance", "nightcore",
              "instrumental", "rb", "punk", "indie", "soul", "techno", "hardcore", "speedcore"}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower().split(" feat")[0].split(" ft.")[0])


def pair_metrics(pairs, meta):
    """pairs: [(hash_a, hash_b)]; meta: catalogue.load_meta output. Pairs with a map the catalogue no longer has (a
    store keeps the vectors of deleted versions) are left out."""
    pairs = [(x, y) for x, y in pairs if x in meta and y in meta]
    a = [meta[x] for x, _ in pairs]; b = [meta[y] for _, y in pairs]
    ga = [set(m["tags"]) & GENRE_TAGS for m in a]; gb = [set(m["tags"]) & GENRE_TAGS for m in b]
    jac = [len(p & q) / len(p | q) for p, q in zip(ga, gb) if p and q]
    art = [_norm(p["artist"]) == _norm(q["artist"]) for p, q in zip(a, b) if _norm(p["artist"])]
    song = [_norm(p["artist"]) == _norm(q["artist"]) and _norm(p["song"].split("(")[0]) == _norm(q["song"].split("(")[0])
            for p, q in zip(a, b) if _norm(p["song"])]
    nps = [abs(p["maxnps"] - q["maxnps"]) for p, q in zip(a, b) if p["maxnps"] is not None and q["maxnps"] is not None]
    return {"tag_jaccard": float(np.mean(jac)) if jac else float("nan"),
            "same_artist_pct": 100 * float(np.mean(art)), "same_song_pct": 100 * float(np.mean(song)),
            "same_mapper_pct": 100 * float(np.mean([p["mapper"] == q["mapper"] for p, q in zip(a, b)])),
            "max_nps_gap": float(np.mean(nps)), "bpm_gap": float(np.mean([abs(p["bpm"] - q["bpm"]) for p, q in zip(a, b)]))}


def hubness(idx, n):
    """idx: (n, k) neighbour indices. k-occurrence skew, max, share never retrieved, share taken by the top 1%."""
    nk = np.bincount(np.asarray(idx).ravel(), minlength=n).astype(float)
    top = np.sort(nk)[::-1]
    return {"nk_skew": float(((nk - nk.mean()) ** 3).mean() / (nk.std() ** 3 + 1e-12)), "nk_max": int(top[0]),
            "never_retrieved_pct": 100 * float(np.mean(nk == 0)),
            "top1pct_share": 100 * float(top[: max(1, n // 100)].sum() / nk.sum())}


def isolation(index, k=10):
    """{hash: mean score of its k nearest maps} in `index` (one reference view): low = isolated."""
    _, sc = index.knn(k)
    return dict(zip(index.hashes, sc.mean(1).tolist()))


def _quantile_groups(values, n):
    """Group number 0..n-1 of each value (NaN: -1) by quantiles of the finite values; ties go to the lower group."""
    v = np.asarray(values, np.float64)
    ok = np.isfinite(v)
    out = np.full(len(v), -1)
    if ok.any():
        edges = np.quantile(v[ok], np.linspace(0, 1, n + 1)[1:-1])
        out[ok] = np.searchsorted(edges, v[ok], side="left")
    return out


def _by_group(groups, n, idx, sc, hs, meta, low, high):
    out = {}
    for g in range(n):
        q = np.flatnonzero(groups == g)
        if not len(q):
            continue
        name = f"q{g + 1}" + (f" ({low})" if g == 0 else f" ({high})" if g == n - 1 else "")
        out[name] = {"queries": int(len(q)), "mean_score": float(sc[q].mean()),
                     **pair_metrics([(hs[i], hs[j]) for i in q for j in idx[i]], meta)}
    return out


def evaluate(index, meta, k=10, csls=False, n_random=40000, seed=0, rows=None, isolation_of=None, groups=4):
    """rows: optional list that receives one {"query", "results": [[hash, score], ...]} per item (results.py).
    isolation_of: {hash: isolation} (isolation() over a reference view) to also break the agreement down by it;
    queries missing there are left out of that breakdown."""
    idx, sc = index.knn(k, csls=csls)
    hs = index.hashes
    if rows is not None:
        rows.extend({"query": hs[i], "results": [[hs[j], float(s)] for j, s in zip(idx[i], sc[i])]} for i in range(len(hs)))
    pairs = [(hs[i], hs[j]) for i in range(len(hs)) for j in idx[i]]
    rng = np.random.default_rng(seed)
    rnd = [(hs[i], hs[j]) for i, j in rng.integers(0, len(hs), (n_random, 2)) if i != j]
    out = {"knn": {**pair_metrics(pairs, meta), **hubness(idx, len(hs))}, "random": pair_metrics(rnd, meta)}
    up = _quantile_groups([meta.get(h, {}).get("up", np.nan) for h in hs], groups)
    out["by_upvotes"] = _by_group(up, groups, idx, sc, hs, meta, "fewest upvotes", "most upvotes")
    if isolation_of is not None:
        iso = _quantile_groups([isolation_of.get(h, np.nan) for h in hs], groups)
        out["by_isolation"] = _by_group(iso, groups, idx, sc, hs, meta, "most isolated", "least isolated")
    return out
