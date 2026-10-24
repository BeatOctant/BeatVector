"""Training/evaluation rows: one per Standard difficulty of every pool map, with swing features and the labels the
catalogue already has (BeatSaver style tags, ScoreSaber/BeatLeader stars for ranked charts, BeatSaver's own parity
summary). The rows are derived from public maps only; where they are written is up to the caller.

`train` fits the style model (style.py); `evaluate` reports how well the simulation tracks those labels (OctantSpecs octantvector/swing.md, "評価"):
  difficulty  Spearman of pass/tech/NPS with stars, cross-validated ridge on all features (grouped by mapper)
  order       share of maps whose pass rating rises from easier to harder difficulty
  style       AUC of a cross-validated logistic model per BeatSaver style tag, against an NPS/NJS baseline
  fine_tags   share of maps with each fine tag, its lift on the BeatSaver style tags, and the AUC of its score
              against description/label keywords (KEYWORDS)
  keyword_features  the single features that best separate maps carrying each keyword
  parity      Spearman of simulated resets (bomb / other) with BeatSaver's parity summary (resets / errors / warns)
"""
import gzip
import re
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

from .. import archive, catalogue, pools
from ...common.charts import difficulty_rank
from ..errors import GenerationError
from . import params, style
from .features import NAMES, chart_features

STYLE_TAGS = style.LEARNED_TAGS
MOD_FLAGS = ("ne", "me", "vivify")  # per-difficulty catalogue flags that make a chart "modded"
# Weak labels for checking fine tags only: words in the map description or difficulty label (never features;
# "no resets" style negations make them noisy, so agreement is a lower bound)
KEYWORDS = {
    "angle_tech": r"\btech", "acc": r"\bacc(uracy)?\b", "stream": r"\bstreams?\b", "jumps": r"\bjumps?\b",
    "speed": r"\bspeed\b", "vibro": r"\bvibro", "sliders": r"\bsliders?\b", "chains": r"\bchains?\b",
    "arcs": r"\barcs?\b", "bombs": r"\bbombs?\b", "bomb_resets": r"\bbomb ?resets?\b",
    "resets": r"\bresets?\b",  # usually meant as intended (bomb) resets: checked against features only, no tag
    "wrist_roll": r"\bwrist|\brolls?\b", "dodge_walls": r"\bwalls?\b", "flowy": r"\bflow(y|ing)?\b",
    "crossover": r"\bcross ?overs?\b", "triplets": r"\btriplets?\b", "inverted": r"\binvert|\bupside", "dots": r"\bdots?\b",
    "windows": r"\bwindows?\b", "stamina": r"\bstamina\b|\bendurance\b",
    "art_walls": r"\bwall ?art|\bwall ?lighting", "workout_walls": r"\bwork ?out|\bexercis|\bcardio",
    "crouch_walls": r"\bcrouch|\bduck(ing)?\b|\bsquats?\b", "vision_walls": r"\bvision ?block",
    "art_bombs": r"\bbomb ?art", "dodge_bombs": r"\bbomb ?(dodg|avoid|weav)", "unsnapped": r"\bunsnapped|\boff ?beat|\bswing ?rhythm",
}
_KW = {t: re.compile(p, re.I) for t, p in KEYWORDS.items()}
_W = {}


def _init(archive_root, P):
    _W["archive"], _W["P"] = Path(archive_root), P


def _work(h):
    from . import standard_charts
    rows = []
    try:
        with archive.MapZip(_W["archive"], h) as mz:
            for diff, ch in standard_charts(mz):
                v, both, _ = chart_features(ch, _W["P"])
                if v is not None:
                    rows.append({"diff": diff, "f": [round(float(x), 5) for x in v], "swings": len(both)})
        return h, rows, None
    except Exception as ex:
        return h, rows, repr(ex)


def _labels(m, v):
    by = {d["difficulty"]: d for d in catalogue.standard_diffs(v)}
    return {"key": m["id"], "mapper": str((m.get("uploader") or {}).get("id") or m["metadata"].get("levelAuthorName")),
            "tags": [t for t in m.get("tags", []) if t in STYLE_TAGS], "upvotes": m["stats"].get("upvotes", 0),
            "year": int((m.get("uploaded") or "2018")[:4]),
            "auto": bool(m.get("automapper")) or m.get("declaredAi", "None") not in ("None", None)}, by


def _diff_labels(lab, d, description):
    tags = lab["tags"] + (["modded"] if any(d.get(f) for f in MOD_FLAGS) else [])
    text = f"{description} {d.get('label') or ''}"
    return {**lab, "tags": tags, "kw": [t for t, rx in _KW.items() if rx.search(text)]}


def _pool(cat, available, min_upvotes, n, seed):
    """Published versions in `available` (hash -> commit_seq) with a Standard chart and >= min_upvotes, optionally a
    random subset of n, in commit_seq order as [{key, hash, seq}]."""
    d = {"min_upvotes": min_upvotes, "standard": True}
    out = []
    for m in catalogue.read_jsonl(cat):
        v = catalogue.published_version(m)
        if v and v["hash"] in available and pools.matches(d, m, v):
            out.append({"key": m["id"], "hash": v["hash"], "seq": available[v["hash"]]})
    if n is not None:
        random.Random(seed).shuffle(out)
        out = out[:n]
    return sorted(out, key=lambda r: r["seq"])


def build(cfg, out, min_upvotes, n=None, seed=2, log=sys.stderr):
    arch, cat = cfg.require("archive"), cfg.require("catalogue")
    P = params.load()
    pool = _pool(cat, archive.commit_order(arch), min_upvotes, n, seed)
    recs = catalogue.records_for(cat, [r["hash"] for r in pool])
    print(f"pool: {len(pool)} maps", file=log)
    t0, k, nrows = time.time(), 0, 0
    with gzip.open(out, "wt") as fo, mp.get_context("spawn").Pool(cfg.workers, initializer=_init, initargs=(str(arch), P)) as p:
        fo.write(json.dumps({"header": True, "features": NAMES, "swing": P}) + "\n")
        for k, (h, rows, err) in enumerate(p.imap(_work, [r["hash"] for r in pool], chunksize=4), 1):
            if err:
                print(f"{h}: {err}", file=log)
            if h in recs:
                lab, by = _labels(*recs[h])
                desc = recs[h][0].get("description") or ""
                for r in rows:
                    d = by.get(r["diff"], {})
                    ps = d.get("paritySummary") or {}
                    fo.write(json.dumps({"hash": h, **_diff_labels(lab, d, desc), **r, "stars": d.get("stars") or 0, "bl_stars": d.get("blStars") or 0,
                                         "bs_nps": d.get("nps"), "bs_resets": ps.get("resets"), "bs_errors": ps.get("errors"),
                                         "bs_warns": ps.get("warns")}) + "\n")
                    nrows += 1
            if k % 500 == 0:
                rate = k / (time.time() - t0)
                print(f"{k}/{len(pool)}  {rate:.1f} maps/s  eta {(len(pool) - k) / rate / 60:.1f} min", file=log, flush=True)
    print(f"wrote {out}: {nrows} rows", file=log)


def load(path):
    rows, header = [], None
    with gzip.open(path, "rt") as f:
        for line in f:
            j = json.loads(line)
            if j.get("header"):
                header = j
            else:
                rows.append(j)
    if header and header["features"] != NAMES:
        raise GenerationError(f"{path} was built with a different feature list; rebuild it")
    return rows


# ---------------------------------------------------------------- statistics (numpy only)
def rank(a):
    a = np.asarray(a, float)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); r[o] = np.arange(len(a))
    s = a[o]
    i = 0
    while i < len(s):  # average ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if len(a) < 3 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score, y):
    y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rank(score)
    return float((r[y].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def _std(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return np.clip((Xtr - mu) / sd, -6, 6), np.clip((Xte - mu) / sd, -6, 6)


def ridge(Xtr, ytr, Xte, lam):
    Xtr, Xte = _std(Xtr, Xte)
    A = np.c_[Xtr, np.ones(len(Xtr))]
    reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
    w = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
    return np.c_[Xte, np.ones(len(Xte))] @ w


def logistic_weights(A, y, lam, iters=50):
    """L2-regularised logistic regression by Newton steps on standardised A; -> weights with the bias last."""
    A = np.c_[A, np.ones(len(A))]
    w = np.zeros(A.shape[1])
    reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y) + reg @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def logistic(Xtr, ytr, Xte, lam):
    Xtr, Xte = _std(Xtr, Xte)
    w = logistic_weights(Xtr, ytr, lam)
    return Xte @ w[:-1] + w[-1]


def group_folds(groups, k, seed=0):
    """Fold id per row such that a group (mapper) never spans folds."""
    u = sorted(set(map(str, groups)))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(u, rng.permutation(len(u)) % k))
    return np.array([fold_of[str(g)] for g in groups])


def cv_predict(fit, X, y, folds, **kw):
    out = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f
        out[te] = fit(X[~te], y[~te], X[te], **kw)
    return out


# ---------------------------------------------------------------- training and evaluation
def hardest_rows(rows):
    """Index of each map's hardest Standard difficulty (where BeatSaver's map-level tags apply)."""
    hardest = {}
    for i, r in enumerate(rows):
        d = difficulty_rank(r["diff"])
        if r["hash"] not in hardest or d > hardest[r["hash"]][0]:
            hardest[r["hash"]] = (d, i)
    return np.array(sorted(i for _, i in hardest.values()), int)


def train(rows, lam=10.0):
    """Style model (quantiles over every row + learned tag weights on each map's hardest difficulty)."""
    X = np.array([r["f"] for r in rows], float)
    idx = np.array([i for i in hardest_rows(rows) if not rows[i].get("auto")], int)
    return style.fit(X, X[idx], [rows[i]["tags"] for i in idx], lam)


def evaluate(rows, k=5, lam=10.0, seed=0):
    X = np.array([r["f"] for r in rows], float)
    col = {n: i for i, n in enumerate(NAMES)}
    res = {"rows": len(rows), "maps": len({r["hash"] for r in rows})}

    # difficulty vs stars (ranked charts only)
    diff = {}
    for star in ("stars", "bl_stars"):
        m = np.array([r[star] > 0 for r in rows])
        if m.sum() < 30:
            continue
        y = np.array([r[star] for r in rows])[m]
        Xm = X[m]
        folds = group_folds([rows[i]["mapper"] for i in np.where(m)[0]], k, seed)
        pred = cv_predict(ridge, style.signed_log(Xm), y, folds, lam=lam)
        base = cv_predict(ridge, style.signed_log(Xm[:, [col["nps"], col["njs"]]]), y, folds, lam=lam)
        diff[star] = {"n": int(m.sum()),
                      "spearman": {"pass": spearman(Xm[:, col["pass"]], y), "tech": spearman(Xm[:, col["tech"]], y),
                                   "nps": spearman(Xm[:, col["nps"]], y),
                                   "ridge_all_features_cv": spearman(pred, y), "ridge_nps_njs_cv": spearman(base, y)},
                      "mae": {"ridge_all_features_cv": float(np.abs(pred - y).mean()),
                              "ridge_nps_njs_cv": float(np.abs(base - y).mean())}}
    res["difficulty"] = diff

    # ordering of difficulties within a map
    by_map = {}
    for r, x in zip(rows, X):
        by_map.setdefault(r["hash"], []).append((difficulty_rank(r["diff"]), x))
    ok = {"pass": [], "nps": []}
    for lst in by_map.values():
        lst.sort(key=lambda t: t[0])
        for a, b in zip(lst, lst[1:]):
            for name in ok:
                ok[name].append(b[1][col[name]] > a[1][col[name]])
    res["order"] = {f"{n}_rises_pct": 100 * float(np.mean(v)) for n, v in ok.items() if v}

    # learned style tags (hardest difficulty per map, automapped/AI maps left out), cross-validated by mapper
    idx = np.array([i for i in hardest_rows(rows) if not rows[i].get("auto")], int)
    Xh = X[idx]
    folds = group_folds([rows[i]["mapper"] for i in idx], k, seed)
    learned = {}
    for tag in STYLE_TAGS:
        y = np.array([tag in rows[i]["tags"] for i in idx], float)
        if y.sum() < 20:
            continue
        p_all = cv_predict(logistic, style.signed_log(Xh), y, folds, lam=lam)
        p_base = cv_predict(logistic, style.signed_log(Xh[:, [col["nps"], col["njs"]]]), y, folds, lam=lam)
        learned[tag] = {"n_pos": int(y.sum()), "auc_logistic_cv": auc(p_all, y), "auc_nps_njs_cv": auc(p_base, y)}
    res["style"] = learned

    # fine tags: share of maps, and how much each raises the odds of each BeatSaver style tag (lift)
    model = style.fit(X, Xh, [[]] * len(idx), lam)  # quantiles only
    rules = style.load_rules()
    S, on = style.fine_tags(model, Xh, rules)
    T = {t: np.array([t in rows[i]["tags"] for i in idx]) for t in STYLE_TAGS}
    fine = {}
    for j, name in enumerate(rules):
        m = on[:, j]
        lift = {t: round(float(y[m].mean() / max(y.mean(), 1e-9)), 2) for t, y in T.items() if m.sum() >= 10 and y.sum()}
        fine[name] = {"maps_pct": round(100 * float(m.mean()), 2),
                      "lift": dict(sorted(lift.items(), key=lambda t: -t[1])[:3])}
        kw = np.array([name in rows[i].get("kw", []) for i in idx])
        if kw.sum() >= 20:
            fine[name]["keyword"] = {"n": int(kw.sum()), "auc_score": round(auc(S[:, j], kw), 3)}
    res["fine_tags"] = fine

    # which single features (either direction) best separate maps carrying each keyword: guides the fine-tag rules
    kwf = {}
    Lh = style.signed_log(Xh)
    for name in KEYWORDS:
        kw = np.array([name in rows[i].get("kw", []) for i in idx])
        if kw.sum() >= 20:
            a = np.array([auc(Lh[:, j], kw) for j in range(len(NAMES))])
            top = np.argsort(-np.abs(np.nan_to_num(a, nan=0.5) - 0.5))[:5]
            kwf[name] = {"n": int(kw.sum()), "top": {NAMES[j]: round(float(a[j]), 3) for j in top}}
    res["keyword_features"] = kwf

    # simulated resets vs BeatSaver's parity summary, by kind: BeatSaver's checker (the bs-parity lineage) counts
    # bomb-forced parity flips as resets and same-parity hits without bombs as errors (borderline ones as warns)
    m = [i for i, r in enumerate(rows) if r.get("bs_resets") is not None and r["swings"]]
    if len(m) > 30:
        sim = {k: [X[i, col[f"{k}_frac"]] * rows[i]["swings"] for i in m] for k in ("reset", "bomb_reset", "rebound_reset")}
        bs = {k: [rows[i].get(f"bs_{k}") or 0 for i in m] for k in ("resets", "errors", "warns")}
        bs["errors+warns"] = [a + b for a, b in zip(bs["errors"], bs["warns"])]
        res["parity"] = {"n": len(m), "spearman": {f"{a}_vs_{b}": spearman(x, y) for a, x in sim.items() for b, y in bs.items()}}
        has_bombs = [j for j, i in enumerate(m) if X[i, col["bomb_ps"]] > 0]
        if len(has_bombs) > 30:
            res["parity"]["with_bombs"] = {"n": len(has_bombs), "bomb_reset_vs_resets": spearman(
                [sim["bomb_reset"][j] for j in has_bombs], [bs["resets"][j] for j in has_bombs])}
    return res
