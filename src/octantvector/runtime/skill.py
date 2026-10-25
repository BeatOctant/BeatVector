"""The player's skill as distance from their strong charts (OctantSpecs plans/difficulty-fit.md; law 3.3).

Per play style (charts of different play styles are never compared, constitution principle 2):
  features   chart features from charts.parquet, standardised over the DB's charts of that play style
  strong set the most recent plain clears (speed 1.0, no modifiers; quits never count), then, while there is room,
             the charts PlayerData says were cleared before OctantPath (imported_play with `cleared`), most played
             first: PlayerData has no dates, and its clears may have had modifiers. Each new own clear takes the
             place of the least played of them once the set is full. They count towards minClears, so the skill is
             there from the first play, and only place the skill: the curve and the weak directions come from the
             player's own plays (exp-early-fit, 2026-10-25: closest to the full history both for a player whose
             skill is stable and for one whose PlayerData is easier than their skill now; dropping them sooner did
             not do better)
  distance   of a chart: mean distance to its k nearest strong charts (the chart itself excluded)
  weak directions  a feature higher, or lower, than the strong charts, where the player fails more than the
             configured weights say: learned from the recent plain clears and fails (since the strong set's oldest
             clear) when there are enough of those fails, and weighed more in the distance (Style.learn)
  curve      clear chance against distance, a logistic fit on the plain clears and fails, when there are enough fails
Aims become target distances here, never in OctantCompass (contract 1 (OctantCompass to OctantVector)): normal = the median distance of the strong
charts to each other, warmup = a share of it, challenge = where the curve gives the challenge clear chance.
"""
import numpy as np

RIDGE = 1.0  # L2 penalty of the weak-direction fits: keeps them finite when one direction alone separates clears from
             # fails; how far to trust them is the job of their standard errors and directionShrinkPlays (Style.learn)


class Style:
    """Skill in one play style. rows: chart rows of the style; X: their standardised features; w: feature weights;
    strong: chart rows of the strong set. A chart's difference from a strong chart is weighed by `up` in the features
    where the chart is higher and by `down` where it is lower; both are `w` until weak directions are learned."""

    def __init__(self, style, rows, X, w, strong, p):
        self.style, self.X, self.w, self.p = style, X, w, p
        self.up, self.down = w, w
        self.weak = []  # learned weak directions: (feature index, "high" or "low", weight over the configured one)
        self.rows = np.asarray(rows)
        self.pos = {int(r): n for n, r in enumerate(rows)}
        self.strong = [int(r) for r in strong]
        self.curve = None
        self._measure()

    def _measure(self):
        """The strong charts' distances to each other and their spread (the units of the targets), from scratch."""
        self._all = None
        self.self_d = np.array([self._dist(self.X[self.pos[r]], exclude=r) for r in self.strong])
        self.sd = float(self.self_d.std()) or 1.0

    def _to_strong(self, x, exclude=None):
        """-> (the strong chart rows but `exclude`, the distance of features x to each of them)."""
        S = [s for s in self.strong if s != exclude]
        e = x - self.X[[self.pos[s] for s in S]]
        return S, np.sqrt(((e ** 2) * np.where(e > 0, self.up, self.down)).sum(1))

    def _dist(self, x, exclude=None):
        S, d = self._to_strong(x, exclude)
        if not S:
            return np.nan
        k = min(self.p["k"], len(d))
        return float(np.sort(d)[:k].mean())

    def distance(self, row):
        n = self.pos.get(row)
        if n is None or not np.isfinite(self.X[n]).all():
            return None
        return self._dist(self.X[n], exclude=row)

    def distances(self):
        """-> (distance, nearest strong chart row) of every chart of the style, in `rows` order, as `distance` and
        `nearest_strong` give them one by one (distance NaN where `distance` is None). Computed once, as a matrix of
        every chart against the strong set: a recommendation needs them for every candidate."""
        if self._all is None:
            S = np.array([self.pos[s] for s in self.strong])
            Xs, k = self.X[S], min(self.p["k"], len(S))
            dist, nearest = np.empty(len(self.X)), np.empty(len(self.X), np.int64)
            for a in range(0, len(self.X), 8192):  # in blocks, so the matrix stays small
                X = self.X[a:a + 8192]
                E = X[:, :1] - Xs[:, 0]
                D = (E ** 2) * self._weights(E, 0)  # summed feature by feature, in _dist's order
                for f in range(1, X.shape[1]):
                    E = X[:, f:f + 1] - Xs[:, f]
                    D += (E ** 2) * self._weights(E, f)
                D = np.sqrt(D)
                dist[a:a + 8192] = np.sort(np.partition(D, k - 1, 1)[:, :k], 1).mean(1)
                own = np.flatnonzero((S >= a) & (S < a + 8192))
                D[S[own] - a, own] = np.inf  # a strong chart is not its own nearest
                nearest[a:a + 8192] = np.array(self.strong)[np.argmin(D, 1)]
            dist[~np.isfinite(self.X).all(1)] = np.nan
            for r in self.strong:  # its distance leaves itself out of the strong set
                dist[self.pos[r]] = self._dist(self.X[self.pos[r]], exclude=r)
            self._all = dist, nearest
        return self._all

    def _weights(self, E, f):
        """Weights of the differences E in feature f (a chart minus strong charts), by their sign when they differ."""
        return self.up[f] if self.up[f] == self.down[f] else np.where(E > 0, self.up[f], self.down[f])

    def nearest_strong(self, row):
        return int(self.distances()[1][self.pos[row]])

    def learn(self, rows, cleared):
        """Weak directions from plays (their chart rows, and whether each was a clear; difficulty-fit.md, 2). A play's
        chart minus its nearest strong chart gives, per feature, how far higher and how far lower it is (scaled like
        the distance, by the square root of the configured weight). A logistic regression of the clears on these
        2 x features columns gives each direction its drop: how fast the clear chance falls along it (none where it
        rises). The configured weights count every direction alike, as if each dropped by the mean drop of all
        2 x features directions. A direction whose drop, less its standard error (the noise of the fit), is still
        faster than that mean is weak: its weight is multiplied by (that drop / the mean) squared, so a step along it
        counts as that much more distance. The others keep the configured weight: a chart easier than the strong ones
        still does not fit (difficulty-fit.md, 2). Learned from n plays, a multiplier m is shrunk towards 1 as
        (n m + n0) / (n + n0), n0 = directionShrinkPlays (personal-data-overfitting.md, 個人化の強さ). The distances
        are then measured again with the new weights. Only the played charts are measured here, not the whole style
        (`distances`)."""
        at = [self.pos[r] for r in rows]
        near = (self._to_strong(self.X[n], exclude=r) for n, r in zip(at, rows))
        E = self.X[at] - self.X[[self.pos[S[int(np.argmin(d))]] for S, d in near]]
        U = np.sqrt(np.r_[self.w, self.w]) * np.concatenate([np.maximum(E, 0), np.maximum(-E, 0)], 1)
        y = np.asarray(cleared, float)
        out = _logistic(U, y, RIDGE, noise=True)
        if out is None:
            return
        fit, se = out
        drop = np.maximum(0.0, -np.array(fit[1:]))
        if not (drop - se > drop.mean()).any():
            return  # no direction drops faster than the others beyond the noise: nothing to learn
        m = np.maximum(1.0, (drop - se) / drop.mean()) ** 2
        n0 = float(self.p["directionShrinkPlays"])
        m = (len(y) * m + n0) / (len(y) + n0)
        F = len(self.w)
        self.up, self.down = self.w * m[:F], self.w * m[F:]
        weak = [(f, side, float(m[s * F + f])) for s, side in enumerate(("high", "low")) for f in range(F)]
        self.weak = sorted((x for x in weak if x[2] > 1), key=lambda x: -x[2])
        self._measure()

    def chance(self, d):
        if self.curve is None or d is None:
            return None
        a, b = self.curve
        return float(1 / (1 + np.exp(-(a + b * d))))

    def target(self, difficulty):
        """-> (aim actually used, target distance)."""
        normal = float(np.median(self.self_d))
        if isinstance(difficulty, (int, float)) and not isinstance(difficulty, bool):
            if self.curve is not None and 0 < difficulty < 1 and self.curve[1] < 0:
                a, b = self.curve
                return "clearChance", max(0.0, (np.log(difficulty / (1 - difficulty)) - a) / b)
            return "normal", normal
        if difficulty == "warmup":
            return "warmup", self.p["warmupScale"] * normal
        if difficulty == "challenge":
            c = self.p["challengeClearChance"]
            if self.curve is not None and self.curve[1] < 0:
                a, b = self.curve
                return "challenge", max(normal, (np.log(c / (1 - c)) - a) / b)
            return "challenge", float(np.percentile(self.self_d, self.p["normalPercentile"])) + self.sd
        return "normal", normal


def _logistic(x, y, ridge=0.0, iters=50, noise=False):
    """Logistic regression by Newton's method, P(y) = 1 / (1 + exp(-(a + x b))), on one column x or several, with an L2
    penalty `ridge` on b. -> (a, *b) or None if it does not converge; with `noise`, ((a, *b), standard errors of b
    from the penalised Hessian at the fit)."""
    X = np.column_stack([np.ones(len(x)), x])
    pen = np.r_[0.0, np.full(X.shape[1] - 1, float(ridge))]
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        H = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen) + 1e-6 * np.eye(len(beta))
        step = np.linalg.solve(H, X.T @ (y - p) - pen * beta)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    if not np.isfinite(beta).all():
        return None
    fit = tuple(float(v) for v in beta)
    return (fit, np.sqrt(np.diag(np.linalg.inv(H)))[1:]) if noise else fit


def estimate(db, hist, p):
    """-> {play style: Style} for the play styles with enough plain clears."""
    feats = [f for f, w in p["features"].items() if w and f in db.chart_metrics]
    if not feats:
        return {}
    w = np.array([float(p["features"][f]) for f in feats])
    raw = np.stack([db.chart_metrics[f] for f in feats], 1)
    styles = np.array(db.play_styles, dtype=object)
    out = {}
    plain = [pl for pl in hist.plays if pl.plain and pl.chart is not None and pl.result in ("clear", "fail")]
    imported = [db.chart_pos[k] for k, _ in sorted(hist.imported_clears.items(), key=lambda kv: (-kv[1], kv[0]))]
    for style in sorted(set(db.play_styles)):
        rows = np.flatnonzero(styles == style)
        X = raw[rows]
        ok = np.isfinite(X).all(1)
        if ok.sum() < 2:
            continue
        mu, sd = X[ok].mean(0), X[ok].std(0)
        X = (X - mu) / np.where(sd > 0, sd, 1)
        mine = [pl for pl in plain if db.play_styles[db.chart_pos[pl.chart]] == style]
        # the recent plays (difficulty-fit.md, 2 「最近」) start with the strong set's oldest clear once the set is
        # full; until then every play is recent
        clears, strong, since = [pl for pl in mine if pl.result == "clear"], [], ""
        for pl in reversed(clears):  # most recent first, one entry per chart
            r = db.chart_pos[pl.chart]
            if r not in strong and np.isfinite(raw[r]).all():
                strong.append(r)
            if len(strong) >= p["strongRecentClears"]:
                since = pl.at
                break
        for r in imported:  # PlayerData's clears fill what is left
            if len(strong) >= p["strongRecentClears"]:
                break
            if db.play_styles[r] == style and r not in strong and np.isfinite(raw[r]).all():
                strong.append(r)
        if len(strong) < max(2, p["minClears"]):
            continue
        st = Style(style, rows, X, w, strong, p)
        fails = [pl for pl in mine if pl.result == "fail"]
        if len(fails) >= p["minFailsForCurve"]:
            pts = [(db.chart_pos[pl.chart], pl.result == "clear", pl.at >= since) for pl in mine]
            pts = [(r, c, new) for r, c, new in pts if st.distance(r) is not None]
            if pts:
                if sum(pl.at >= since for pl in fails) >= p["minFailsForCurve"]:  # recent enough fails
                    st.learn([r for r, _, new in pts if new], [c for _, c, new in pts if new])
                charts, cleared, _ = zip(*pts)
                st.curve = _logistic(np.array([st.distance(r) for r in charts]), np.array(cleared, float))
        out[style] = st
    return out
