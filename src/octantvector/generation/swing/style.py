"""Play-style tags from the swing features (OctantSpecs octantvector/swing.md, "スタイルタグ").

Two kinds:
  learned  one logistic model per BeatSaver style tag (accuracy, balanced, ...), trained on the mapper-chosen tags
           by `octantvector swing-train`; outputs a probability.
  fine     finer tags (stream, vibro, inverted, bomb_reset, ...) defined as rules over the features. A rule
           compares a feature with a population quantile ("q>=" / "q<=", 0..1) or an absolute value (">=" / "<=").
           A tag is on when every condition holds; its score (0..1) is its weakest condition, so it also orders maps
           that miss the tag. A quantile condition scores the quantile on the tag's side; an absolute one scores
           half the share of the threshold reached below it and 0.5 + half the quantile above it, so maps past the
           threshold stay ordered.
The quantiles and the learned weights live in one model file (JSON) written by `swing-train`. Rules can be changed
or added in the `[swing_tags]` table of the config file, e.g. `stream = [["alternation", "q>=", 0.7]]`.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import config
from ..errors import GenerationError
from .features import HAZARD_NAMES, NAMES

# BeatSaver style tags, the "poodle" tag, and "modded" (the chart needs Noodle Extensions, Mapping Extensions or Vivify)
LEARNED_TAGS = ["accuracy", "balanced", "challenge", "dance-style", "fitness", "speed", "tech", "poodle", "modded"]

FINE_TAGS = {
    # speed
    "speed":        [["ebpm_p90", "q>=", 0.85]],
    "stream":       [["alternation", "q>=", 0.6], ["gap_lt_1", "q>=", 0.8]],
    "vibro":        [["gap_lt_0", "q>=", 0.95], ["gap_lt_0", ">=", 0.05]],
    "burst":        [["sustain_ratio", "q>=", 0.85], ["pass_short", "q>=", 0.6]],
    "stamina":      [["pass_long", "q>=", 0.8], ["length_s", "q>=", 0.6]],
    "midspeed":     [["ebpm_median", "q>=", 0.35], ["ebpm_median", "q<=", 0.75], ["strain_mean", "q<=", 0.6]],
    "jumps":        [["hit_distance_p90", "q>=", 0.8], ["wide_jump_frac", "q>=", 0.8]],
    # tech
    "angle_tech":   [["strain_p90", "q>=", 0.85]],
    "inverted":     [["inverted_frac", "q>=", 0.9], ["inverted_frac", ">=", 0.03]],
    "wrist_roll":   [["roll_change_mean", "q>=", 0.85], ["roll_speed_p90", "q>=", 0.6]],
    "rotation":     [["rotation_mean", "q>=", 0.85]],
    "parity_breaks": [["rebound_reset_frac", "q>=", 0.9], ["rebound_reset_frac", ">=", 0.03]],
    "bomb_resets":  [["bomb_reset_frac", "q>=", 0.9], ["bomb_reset_frac", ">=", 0.02]],
    "rhythm_tech":  [["rhythm_entropy", "q>=", 0.8], ["rhythm_change", "q>=", 0.8]],
    "syncopation":  [["offbeat_frac", "q>=", 0.9], ["offbeat_frac", ">=", 0.1]],
    "triplets":     [["triplet_frac", "q>=", 0.9], ["triplet_frac", ">=", 0.1]],
    "unsnapped":    [["unsnapped_frac", "q>=", 0.9], ["unsnapped_frac", ">=", 0.1]],
    "crossover":    [["crossover_frac", "q>=", 0.9]],
    "wide_reach":   [["outer_lane_frac", "q>=", 0.8], ["hand_x_std", "q>=", 0.8]],
    "dots":         [["dot_frac", "q>=", 0.9], ["dot_frac", ">=", 0.2]],
    "free_angles":  [["offgrid_angle_frac", ">=", 0.05]],
    # flow / acc
    "flowy":        [["linear_frac", "q>=", 0.75], ["rotation_mean", "q<=", 0.4], ["ebpm_median", "q<=", 0.7]],
    "acc":          [["full_swing_frac", "q>=", 0.65], ["ebpm_median", "q<=", 0.4], ["strain_mean", "q<=", 0.5]],
    # note patterns
    "windows":      [["window_frac", "q>=", 0.9], ["window_frac", ">=", 0.02]],
    "stacks":       [["stack_frac", "q>=", 0.9], ["stack_frac", ">=", 0.03]],
    "sliders":      [["slider_frac", "q>=", 0.9], ["slider_frac", ">=", 0.03]],
    "chains":       [["chain_frac", ">=", 0.02]],
    "arcs":         [["arc_ps", ">=", 0.05]],
    "doubles":      [["double_frac", "q>=", 0.85]],
    # walls
    "art_walls":    [["art_wall_ps", ">=", 0.5], ["art_wall_frac", ">=", 0.8]],
    "workout_walls": [["head_move_mps", "q>=", 0.9], ["head_move_mps", ">=", 0.05]],
    "dodge_walls":  [["dodge_wall_ps", ">=", 0.05]],
    "crouch_walls": [["crouch_wall_ps", ">=", 0.02]],
    "rhythm_walls": [["wall_rhythm_frac", ">=", 0.5], ["dodge_wall_ps", ">=", 0.03]],
    "dance_walls":  [["wall_rhythm_frac", ">=", 0.5], ["head_move_mps", "q>=", 0.8]],
    "vision_walls": [["vision_wall_frac", ">=", 0.05]],
    # bombs
    "bombs":        [["bomb_ps", "q>=", 0.9], ["bomb_ps", ">=", 0.1]],
    "art_bombs":    [["bomb_art_frac", ">=", 0.3], ["bomb_ps", ">=", 0.2]],
    "dodge_bombs":  [["bomb_dodge_ps", "q>=", 0.9], ["bomb_dodge_ps", ">=", 0.1]],
    "rhythm_bombs": [["bomb_rhythm_frac", ">=", 0.5], ["bomb_dodge_ps", ">=", 0.05]],
    "bomb_walls":   [["bomb_cluster_frac", ">=", 0.3], ["bomb_dodge_ps", ">=", 0.05]],
    # body
    "one_hand":     [["hands_ratio", "<=", 0.15]],
    # reading
    "dense_low_njs": [["nps", "q>=", 0.7], ["njs", "q<=", 0.3]],
    "high_njs":     [["njs", "q>=", 0.9]],
}
OPS = ("q>=", "q<=", ">=", "<=")
_IDX = {n: i for i, n in enumerate(NAMES)}
N_QUANTILES = 101


def is_hazard(conds):
    """Wall and bomb tags (every condition on a wall/bomb feature) go to the `swing_hazards` view, the rest to
    `swing_style`."""
    return all(f in HAZARD_NAMES for f, _, _ in conds)


def load_rules(group=None):
    """FINE_TAGS with tags replaced or added from the `[swing_tags]` table of the config file; group "style" or
    "hazards" keeps only that view's tags."""
    rules = {k: [list(c) for c in v] for k, v in FINE_TAGS.items()}
    for name, conds in config.section("swing_tags").items():
        for c in conds:
            if len(c) != 3 or c[0] not in _IDX or c[1] not in OPS:
                raise GenerationError(f"[swing_tags] {name}: each condition is [feature, one of {OPS}, number]; "
                                      f"got {c}")
        rules[name] = [[f, op, float(v)] for f, op, v in conds]
    if group:
        rules = {k: v for k, v in rules.items() if is_hazard(v) == (group == "hazards")}
    return rules


# ---------------------------------------------------------------- model file
def signed_log(X):
    """sign(x) * log(1 + |x|): heavy-tailed features on a scale linear models can use."""
    return np.sign(X) * np.log1p(np.abs(X))


def fit(X_all, X_tagged, tags, lam=10.0):
    """X_all: features of every row (quantiles); X_tagged + tags: one row per map (hardest difficulty) and its
    BeatSaver style tags. -> model dict (JSON-serialisable)."""
    from .dataset import logistic_weights
    q = np.quantile(X_all, np.linspace(0, 1, N_QUANTILES), axis=0).T  # (features, N_QUANTILES)
    L = signed_log(np.asarray(X_tagged, float))
    mu, sd = L.mean(0), L.std(0) + 1e-6
    learned = {}
    for t in LEARNED_TAGS:
        y = np.array([t in ts for ts in tags], float)
        if y.sum() >= 20:
            w = logistic_weights(np.clip((L - mu) / sd, -6, 6), y, lam)
            learned[t] = {"w": w[:-1].round(6).tolist(), "b": float(w[-1]), "n_pos": int(y.sum())}
    return {"features": NAMES, "quantiles": q.round(6).tolist(), "mu": mu.round(6).tolist(), "sd": sd.round(6).tolist(),
            "learned": learned, "n_rows": int(len(X_all)), "n_maps": int(len(X_tagged))}


def save(model, path):
    Path(path).write_text(json.dumps(model))


def load_model(path):
    m = json.loads(Path(path).read_text())
    if m["features"] != NAMES:
        raise GenerationError(f"{path} was trained on a different feature list; run `octantvector swing-train` again")
    return m


def model_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- applying
def quantile_of(model, X):
    """Population quantile (0..1) of every feature value."""
    q = np.asarray(model["quantiles"])
    X = np.atleast_2d(X)
    # mid-rank, so a value shared by many maps (e.g. 0 for "no bomb resets") sits in the middle of its run, not the top
    return np.stack([(np.searchsorted(q[j], X[:, j], "left") + np.searchsorted(q[j], X[:, j], "right")) / (2 * q.shape[1])
                     for j in range(X.shape[1])], 1)


def learned_probs(model, X):
    """-> (n, len(LEARNED_TAGS)); NaN for tags the model has no weights for."""
    Z = np.clip((signed_log(np.atleast_2d(X)) - np.asarray(model["mu"])) / np.asarray(model["sd"]), -6, 6)
    out = np.full((Z.shape[0], len(LEARNED_TAGS)), np.nan)
    for j, t in enumerate(LEARNED_TAGS):
        if t in model["learned"]:
            m = model["learned"][t]
            out[:, j] = 1 / (1 + np.exp(-np.clip(Z @ np.asarray(m["w"]) + m["b"], -30, 30)))
    return out


def fine_tags(model, X, rules=None):
    """-> (scores (n, len(rules)) in 0..1, on (n, len(rules)) bool), columns in rules order."""
    rules = rules or FINE_TAGS
    X = np.atleast_2d(X)
    Q = quantile_of(model, X)
    S = np.zeros((X.shape[0], len(rules))); on = np.zeros_like(S, bool)
    for j, conds in enumerate(rules.values()):
        ok = np.ones(X.shape[0], bool); sc = np.ones(X.shape[0])
        for f, op, v in conds:
            i = _IDX[f]
            if op == "q>=":
                ok &= Q[:, i] >= v; sc = np.minimum(sc, Q[:, i])
            elif op == "q<=":
                ok &= Q[:, i] <= v; sc = np.minimum(sc, 1 - Q[:, i])
            elif op == ">=":
                met = X[:, i] >= v; ok &= met
                below = 0.5 * np.clip(X[:, i] / v, 0, 1) if v > 0 else 0.0
                sc = np.minimum(sc, np.where(met, 0.5 + 0.5 * Q[:, i], below))
            else:
                met = X[:, i] <= v; ok &= met
                above = 0.5 * np.clip(v / np.maximum(X[:, i], 1e-9), 0, 1) if v > 0 else 0.0
                sc = np.minimum(sc, np.where(met, 1 - 0.5 * Q[:, i], above))
        S[:, j], on[:, j] = sc, ok
    return S, on


def describe(model, x, rules=None):
    """One chart's features -> (learned {tag: prob}, fine tags that are on, best first)."""
    rules = rules or load_rules()
    p = learned_probs(model, x)[0]
    S, on = fine_tags(model, x, rules)
    names = list(rules)
    fine = sorted((names[j] for j in np.where(on[0])[0]), key=lambda n: -S[0, names.index(n)])
    return {t: round(float(v), 3) for t, v in zip(LEARNED_TAGS, p) if not np.isnan(v)}, fine
