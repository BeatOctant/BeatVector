"""recommend and describe (OctantSpecs contracts/, contract 1 (OctantCompass to OctantVector); plans/recommendation-definition.md).

recommend(db, history, request, params) ranks charts for one player:
  1. the history is read as it is (history.py): liked maps by the 好みの判定ルール, plays per chart, old versions
     moved to the current one
  2. a reference is chosen by the entry point: the seed chart when there is one; with a genre word, the liked maps
     the player played recently (music views only, inside the maps closest to the word); otherwise all liked maps.
     Chart views take the charts the player played in those maps (imported_play charts for a map with no play).
     With a seed and a genre word, the seed stays the reference and the word only narrows the candidates to the maps
     closest to it. With `genreFrom` (a list of genre words) and neither a seed nor a word, the word is picked for
     the player: the one of the list whose value is highest on average over the liked maps; the reference is all
     liked maps and the candidates are that word's maps ("your genre", measured by listening in OctantVector
     spikes/2026-10-29-genre-view).
  3. every candidate chart gets a weighted mean of its views' cosines to the reference. A view the chart (or its
     map) has no vector in counts 0 and the weights are not renormalised; a chart whose views carry less than
     `minViewWeight` of the weight is dropped (OctantSpecs law 3.1). A view no reference has a vector in is not used.
     Chart views compare charts of one play style only. A relation view (the artists, contract 2) gives a map its
     closeness to each reference map that has members and takes their mean; a map without members does not have it.
  4. difficulty (skill.py): in each play style the player has enough clears in, charts further from the aim's target
     distance than the tolerance are dropped (widened up to the maximum when too few are left) and the rest get a
     penalty growing with the distance from the target. Play styles without enough clears are not filtered.
     Maps the player already knows are left out unless `includeKnown`; with the challenge aim, goals (law 4.1) stay
     in, in the play styles with a skill only, so a goal comes back once the skill has caught up with it.
     With `ceiling` (switched on by the player), charts above the player's ceiling in any of `ceilingFeatures` or
     below their floor in any of `floorFeatures` (skill.ceilings, from their recent clears) are dropped in every hand
     and never widened back; the challenge aim then takes the hardest charts left under it when too few are near
     its target.
  5. the best `chartsPerMap` charts of each map, the best `mapsPerSong` maps of each song and the best
     `mapsPerArtist` maps of each artist, then the best `count` overall. Fewer are returned when fewer pass. Two maps
     are the same song when their music views' mean cosine is at least `sameSongCosine` (other mappers' maps of one
     song have nearly the same audio and would otherwise fill the list) and, when both have artists, their artist
     sets are the same (a cover by other artists stays apart from the original); maps with no music view in common
     are never the same song. Two maps are the same artist when their artist sets are the same; a map without
     artists is not capped. Names are not used (OctantSpecs law 3.2 and 3.4).
With `narrow` (views with weights, like `weights`), the candidates are first narrowed to the
`narrowTopMaps` maps closest to the same reference by those views (maps already excluded do not take a place), and
the rest is ranked by `weights` as above; when fewer than `count` are left after the difficulty tolerance is widened,
the narrowed maps are multiplied by `narrowWiden` until enough are left or every map is in (a rare play style has few
charts among the closest maps): "close in sound, then ranked by the chart" (the narrow_then_rank handling,
OctantSpecs plans/presets.md §8). Names the DB or the reference cannot serve are ignored and returned like weights.
The result is the same for the same DB, history, request and params. There is no score in it: the reason is for
display only (constitution principles 4 and 6).
"""
import numpy as np

from ..common import charts as chartids
from ..common.hands import view_keys
from . import history as hist_mod
from . import skill as skill_mod
from ..common.params import AIMS, resolve, why_invalid

TASTE_ROLES = ("music", "chart", "meta")


class RequestError(ValueError):
    """The request does not fit the DB (unknown genre word or aim)."""


class SeedNotInDBError(RequestError):
    """The seed map or chart is not in the DB (not even as an old version of a map in it)."""


class SeedNotEmbeddedError(SeedNotInDBError):
    """The seed chart is in the DB's catalogue but neither it nor its map has a vector in any view the request could
    use, so there is nothing to compare candidates with. A SeedNotInDBError, so a caller that knows only that error
    still tells the player the seed cannot be used; the message says which seed and why."""


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _keys(db, name):
    return view_keys(name, db.roles())


def _split(db, wanted):
    """({view: weight}, names the DB lacks): each name's weight split equally over the views it stands for
    (common.hands.view_keys: a bare id over the versions the DB has)."""
    out, missing = {}, []
    for name, w in wanted.items():
        views = _keys(db, name)
        if not views:
            missing.append(name)
        for v in views:
            out[v] = out.get(v, 0.0) + float(w) / len(views)
    return {v: w for v, w in out.items() if w > 0}, missing


def _even(db, roles):
    """Every view of `roles` the DB has, each role the same weight split over its views."""
    out = {}
    for r in roles:
        views = db.views_of_role(r)
        out.update((v, 1.0 / len(views)) for v in views)
    return out


def _weights(db, wanted):
    """({view: weight}, ignored names). Names the DB lacks are ignored and returned, so a DB swap does not stop
    recommendations (contract 1). By default, or when every name is ignored, every taste role present gets the same
    weight, split over its views."""
    out, ignored = _split(db, wanted or {})
    if not wanted or len(ignored) == len(wanted):
        out = _even(db, TASTE_ROLES)
    return out, ignored


def _unserved(db, wanted, weights, refs, ignored):
    """Add to `ignored` each requested name none of whose views got a reference (the history, or the seed, has no
    vector in them): it is returned like a name the DB lacks (contract 1, 2026-10-08)."""
    for name in wanted:
        if name not in ignored and not any(v in refs and v in weights for v in _keys(db, name)):
            ignored.append(name)


def _settle(db, wanted, weights, refs, ignored, default, make_refs):
    """Weights the reference can serve (_unserved); when no weighted view has a reference and weights were
    requested, `default` is used as if none had been."""
    if not wanted:
        return weights, refs
    _unserved(db, wanted, weights, refs, ignored)
    if any(v in refs for v in weights):
        return weights, refs
    return default, make_refs(default)


def _map_view_ref(db, view, hashes):
    x, _ = db.views[view]
    vecs = [x[n] for n in (db.row_in_view(view, db.pos[h]) for h in hashes) if n is not None]
    return _unit(np.mean(vecs, 0)) if vecs else None


def _chart_view_refs(db, view, chart_rows):
    """{play style: reference vector} from chart rows."""
    x, _ = db.views[view]
    by_style = {}
    for r in chart_rows:
        n = db.row_in_view(view, r)
        if n is not None:
            by_style.setdefault(db.play_styles[r], []).append(x[n])
    return {s: u for s, v in by_style.items() if (u := _unit(np.mean(v, 0))) is not None}


def _view_scores(db, view, ref):
    """-> (cosine, has) per chart row: a map view's cosine of the chart's map, a chart view's of the chart itself, a
    relation view's closeness of the chart's map; `has` is False where the row has none."""
    n = len(db.chart_keys)
    if db.rows[view] == "charts":
        x, rows = db.views[view]
        out, has = np.zeros(n), np.zeros(n, bool)
        styles = db.styles_in_view(view)
        for style, q in ref.items():
            m = styles == style
            out[rows[m]] = x[m] @ q
            has[rows[m]] = True
        return out, has
    per_map, has_map = np.zeros(len(db.hashes)), np.zeros(len(db.hashes), bool)
    if view in db.relations:
        rows, s = db.relations[view].rows, ref
    else:
        x, rows = db.views[view]
        s = x @ ref
    per_map[rows], has_map[rows] = s, True
    m = db.chart_map_array()
    ok = m >= 0
    return np.where(ok, per_map[m], 0.0), ok & has_map[m]


def _references(db, views, ref_maps, ref_charts):
    refs = {}
    for v in views:
        if v in db.relations:
            r = db.relations[v].reference([db.pos[h] for h in ref_maps])
        elif db.rows[v] == "charts":
            r = _chart_view_refs(db, v, ref_charts)
        else:
            r = _map_view_ref(db, v, ref_maps)
        if r is not None and (not isinstance(r, dict) or r):
            refs[v] = r
    return refs


class _Taste:
    """Scores of every chart row: `score` (the weighted mean), `ok` (enough of the weight present, the chart's map in
    the DB) and per view its contribution and where it is present."""

    def __init__(self, n):
        self.score, self.ok, self.parts = np.zeros(n), np.zeros(n, bool), {}

    def parts_of(self, c):
        return {v: float(s[c]) for v, (s, has) in self.parts.items() if has[c]}


def _taste(db, weights, refs, min_share):
    """-> _Taste. A view with no reference (none of the reference maps or charts has a vector in it) is left out of the
    weights: that is the player's history lacking it, not the candidate."""
    weights = {v: w for v, w in weights.items() if v in refs}
    total = sum(weights.values())
    out = _Taste(len(db.chart_keys))
    if not total:
        return out
    present = 0.0
    for v, w in weights.items():
        s, has = _view_scores(db, v, refs[v])
        part = np.where(has, w * s / total, 0.0)
        out.parts[v] = part, has
        out.score = out.score + part
        present = present + np.where(has, w, 0.0)
    out.ok = (db.chart_map_array() >= 0) & (present / total >= min_share)
    return out


def _seed_chart(db, seed, replaced):
    if not isinstance(seed, dict):
        raise RequestError(f"seed must be an object with hash, characteristic and difficulty, got {seed!r}")
    h, c, d = seed.get("hash"), seed.get("characteristic"), seed.get("difficulty")
    cur = db.current(h)
    if cur is None:
        raise SeedNotInDBError(f"seed {h} is not in DB {db.version}")
    key = chartids.key(cur, c, d)
    if key not in db.chart_pos:
        raise SeedNotInDBError(f"seed chart {chartids.key(h, c, d)} is not in DB {db.version}")
    if cur != h:
        replaced.append({"from": chartids.key(h, c, d), "to": key})
    return cur, db.chart_pos[key]


def _exclusions(db, exclude):
    maps, chart_rows = set(), set()
    for e in exclude or ():
        if not isinstance(e, dict):
            raise RequestError(f"an exclude entry must be an object with a hash, got {e!r}")
        h = e.get("hash")
        h = db.current(h) or h
        if e.get("characteristic") and e.get("difficulty"):
            r = db.chart_pos.get(chartids.key(h, e["characteristic"], e["difficulty"]))
            if r is not None:
                chart_rows.add(r)
        else:
            maps.add(h)
    return maps, chart_rows


def _taste_word(db, words, liked):
    """The word of `words` whose value is highest on average over the liked maps that have it; None without any."""
    g = db.genres()
    missing = [w for w in words if w not in g]
    if missing:
        raise RequestError(f"genre words {missing!r} are not in DB {db.version}; words: {sorted(g)}")
    best = None
    for w in words:
        view, dim = g[w]
        x, rows = db.views[view]
        on = np.isin(rows, [db.pos[h] for h in liked])
        if on.any():
            mean = float(x[on, dim].mean())
            if best is None or mean > best[0]:
                best = (mean, w)
    return None if best is None else best[1]


def _genre_maps(db, word, share):
    g = db.genres()
    if word not in g:
        raise RequestError(f"genre word {word!r} is not in DB {db.version}; words: {sorted(g)}")
    view, dim = g[word]
    x, rows = db.views[view]
    s = x[:, dim]
    keep = max(1, int(round(len(s) * share)))
    top = np.argsort(-s, kind="stable")[:keep]
    return {db.hashes[int(rows[n])] for n in top}


def recommend(db, history, request, params=None):
    why = why_invalid(params)
    if why:  # a value that could run away (an unbounded tolerance widening) or break numpy stops here
        raise RequestError(f"params: {why}")
    p = resolve(params)
    request = request or {}
    hist = hist_mod.parse(db, history)
    replaced = list(hist.replaced)
    liked = hist.liked()
    wanted = request.get("weights")
    weights, ignored = _weights(db, wanted)
    default = _weights(db, None)[0]
    ex_maps, ex_charts = _exclusions(db, request.get("exclude"))
    # goals (law 4.1) stay candidates of the challenge aim, to come back once the skill has caught up with them
    goals = hist.goals() if request.get("difficulty") == "challenge" and not p["includeKnown"] else set()
    if not p["includeKnown"]:
        ex_maps |= hist.known() - goals
    allowed_maps = None

    def played_charts(maps):
        """Chart rows the player played in `maps`: from plays, or from imported_play for a map with no play (the
        same rule as for liked maps), so chart views have a reference before OctantPath has observed any play."""
        rows = {db.chart_pos[pl.chart] for pl in hist.plays if pl.chart and pl.hash in maps}
        with_plays = {pl.hash for pl in hist.plays}
        rows |= {db.chart_pos[k] for h in maps if h not in with_plays for k in hist.imported.get(h, ())}
        return sorted(rows)

    if request.get("seed") is not None:
        if request.get("genre") is not None:  # with a seed, the word only narrows: "near this one, in that genre"
            allowed_maps = _genre_maps(db, request["genre"], float(p["genreTopShare"]))
        seed_map, seed_row = _seed_chart(db, request["seed"], replaced)
        ex_maps.add(seed_map)
        ref_maps, ref_charts = [seed_map], [seed_row]
        weights, refs = _settle(db, wanted, weights, _references(db, weights, [seed_map], [seed_row]), ignored,
                                default, lambda w: _references(db, w, [seed_map], [seed_row]))
        if not refs:
            raise SeedNotEmbeddedError(f"seed chart {db.chart_keys[seed_row]} has no vector in DB {db.version}")
        mix = float(p["seedTasteWeight"])
        if mix > 0 and liked:
            taste_refs = _references(db, weights, sorted(liked), played_charts(liked))
            scored = _taste(db, weights, refs, p["minViewWeight"])
            b = _taste(db, weights, taste_refs, 0.0)
            scored.score = (1 - mix) * scored.score + mix * np.where(b.ok, b.score, 0.0)
        else:
            scored = _taste(db, weights, refs, p["minViewWeight"])
    else:
        if request.get("genre") is not None:
            allowed_maps = _genre_maps(db, request["genre"], float(p["genreTopShare"]))
            recent = [pl.hash for pl in hist.plays[-int(p["recentPlays"]):]]
            ref_maps = sorted(set(recent) & liked)
            default = _even(db, ("music",))
            weights = {v: w for v, w in weights.items() if db.manifest["views"][v].get("role") == "music"} or default
        elif request.get("genreFrom") is not None:
            word = _taste_word(db, request["genreFrom"], liked)
            if word is not None:
                allowed_maps = _genre_maps(db, word, float(p["genreTopShare"]))
            ref_maps = sorted(liked)
        else:
            ref_maps = sorted(liked)
        if not ref_maps or not weights:
            return _result(db, [], replaced, {}, hist, ignored)
        ref_charts = played_charts(set(ref_maps))
        weights, refs = _settle(db, wanted, weights, _references(db, weights, ref_maps, ref_charts), ignored, default,
                                lambda w: _references(db, w, ref_maps, ref_charts))
        scored = _taste(db, weights, refs, p["minViewWeight"])

    narrowed = _narrow(db, request["narrow"], ref_maps, ref_charts, ex_maps, ignored, p) if request.get("narrow") else None

    difficulty = request.get("difficulty", "normal")
    if difficulty in (None, "auto"):
        difficulty = "normal"
    if not (difficulty in AIMS or isinstance(difficulty, (int, float))):
        raise RequestError(f"difficulty must be one of {AIMS} or a clear chance, got {difficulty!r}")
    skills = skill_mod.estimate(db, hist, p)
    targets = {s: st.target(difficulty) for s, st in skills.items()}

    chart_map = db.chart_map_array()
    cand = scored.ok & ~_map_mask(db, ex_maps)[chart_map]
    cand[sorted(ex_charts)] = False
    if allowed_maps is not None:
        cand &= _map_mask(db, allowed_maps)[chart_map]
    in_narrow = None
    if narrowed is not None:
        rank = np.full(len(db.hashes), len(narrowed))
        rank[[db.pos[h] for h in narrowed]] = np.arange(len(narrowed))
        in_narrow = rank[chart_map]
    skilled, dist, z = np.zeros(len(cand), bool), np.full(len(cand), np.nan), np.zeros(len(cand))
    for style, st in skills.items():
        d = st.distances()[0]
        skilled[st.rows], dist[st.rows] = True, d
        z[st.rows] = np.abs(d - targets[style][1]) / st.sd
    cand &= ~skilled | np.isfinite(dist)
    cand &= skilled | ~_map_mask(db, goals)[chart_map]  # a goal only in play styles whose skill can tell
    caps = skill_mod.ceilings(db, hist, p) if p["ceiling"] else {}
    if caps:
        styles = np.array(db.play_styles, dtype=object)
        for style, cap in caps.items():
            rows = np.flatnonzero(styles == style)
            for f, (low, top) in cap.items():
                v = db.chart_metrics[f][rows]
                cand[rows[(v < low) | (v > top)]] = False  # a chart without the value stays

    def diff(c):
        if not skilled[c]:
            return None
        st, d = skills[db.play_styles[c]], float(dist[c])
        aim, target = targets[db.play_styles[c]]
        return {"aim": aim, "distance": d, "target": target, "z": float(z[c]), "clearChance": st.chance(d),
                "nearestStrong": db.chart_keys[st.nearest_strong(c)]}

    count = int(request.get("count", 10))
    per_map = int(request.get("chartsPerMap", 1))
    tol, tol_max = float(p["toleranceSd"]), float(p["toleranceMaxSd"])
    songs = _Songs(db, float(p["sameSongCosine"]), int(p["mapsPerSong"]), int(p["mapsPerArtist"]))
    pen = np.where(skilled, p["penaltyWeight"] * 0.05 * z ** 2, 0.0)
    top = int(p["narrowTopMaps"])
    while True:
        within = cand if in_narrow is None else cand & (in_narrow < top)
        tol = float(p["toleranceSd"])
        picked = _pick(db, within, skilled, z, scored.score - pen, tol, count, per_map, songs)
        while len(picked) < count and tol < tol_max:
            tol = min(tol_max, tol + 0.5)
            picked = _pick(db, within, skilled, z, scored.score - pen, tol, count, per_map, songs)
        if len(picked) < count and caps and difficulty == "challenge":
            # under the ceiling the challenge target may lie above it: take the rest by the score, whose penalty puts
            # the charts nearest the target (the hardest left under the ceiling) first
            picked = _pick(db, within, skilled, z, scored.score - pen, np.inf, count, per_map, songs)
        if len(picked) >= count or in_narrow is None or top >= len(narrowed):
            break
        top *= max(2, int(p["narrowWiden"]))  # too few in the narrowed maps (a rare play style): take more of them
    picked = [(c, db.hashes[chart_map[c]], scored.parts_of(c), diff(c)) for c in picked]
    return _result(db, picked, replaced, {s: t[0] for s, t in targets.items()}, hist, ignored)


def _map_mask(db, hashes):
    """Map rows of `hashes` (those in the DB) as a mask."""
    out = np.zeros(len(db.hashes), bool)
    out[[db.pos[h] for h in hashes if h in db.pos]] = True
    return out


def _narrow(db, wanted, ref_maps, ref_charts, ex_maps, ignored, p):
    """Every map not excluded, closest first to the reference by the `narrow` views (recommend keeps the first
    `narrowTopMaps`, more when too few charts are left); None when no name in it can be served (those names go to
    `ignored`, and nothing is narrowed)."""
    weights = _split(db, wanted)[0]
    refs = _references(db, weights, ref_maps, ref_charts)
    _unserved(db, wanted, weights, refs, ignored)
    weights = {v: w for v, w in weights.items() if v in refs}
    if not weights:
        return None
    taste = _taste(db, weights, refs, p["minViewWeight"])
    chart_map = db.chart_map_array()
    keep = taste.ok & ~_map_mask(db, ex_maps)[chart_map]
    best = np.full(len(db.hashes), -np.inf)
    np.maximum.at(best, chart_map[keep], taste.score[keep])
    maps = np.flatnonzero(np.isfinite(best))
    ranked = maps[np.lexsort((db.hash_array()[maps], -best[maps]))]
    return [db.hashes[m] for m in ranked]


class _Songs:
    """Which picked maps are the same song, or the same artist, as a candidate map (OctantSpecs law 3.4): the music views
    (map rows), and the artist sets of the relation view of role `artist` when the DB has one."""

    def __init__(self, db, threshold, per_song, per_artist=0):
        self.db, self.threshold, self.per_song, self.per_artist = db, threshold, per_song, per_artist
        self.views = db.views_of_role("music")
        self.rel = db.artists()

    def artists(self, h):
        return frozenset(self.rel.members_of(self.db.pos[h])) if self.rel is not None else frozenset()

    def vectors(self, h):
        out = {}
        for v in self.views:
            n = self.db.row_in_view(v, self.db.pos[h])
            if n is not None:
                out[v] = self.db.views[v][0][n]
        return out

    def same(self, a, b, artists_a=frozenset(), artists_b=frozenset()):
        if artists_a and artists_b and artists_a != artists_b:
            return False
        shared = [v for v in a if v in b]
        return bool(shared) and float(np.mean([a[v] @ b[v] for v in shared])) >= self.threshold

    def full(self, h, picked):
        """True when `mapsPerSong` maps of h's song, or `mapsPerArtist` maps of h's artists, are already in `picked`
        (map hashes, in order)."""
        artists = self.artists(h)
        if self.per_artist > 0 and artists and sum(self.artists(o) == artists for o in picked) >= self.per_artist:
            return True
        if self.per_song <= 0 or not self.views:
            return False
        a = self.vectors(h)
        if not a:
            return False
        return sum(self.same(a, self.vectors(o), artists, self.artists(o)) for o in picked) >= self.per_song


def _pick(db, cand, skilled, z, score, tol, count, per_map, songs):
    """-> chart rows, best first by score then row: within `tol` of the target, `per_map` per map, and the song and
    artist caps (_Songs)."""
    rows = np.flatnonzero(cand & ~(skilled & (z > tol)))
    rows = rows[np.lexsort((rows, -score[rows]))]
    chart_map = db.chart_map_array()
    out, per, maps = [], {}, []
    for c in rows.tolist():
        h = db.hashes[chart_map[c]]
        if per.get(h, 0) >= per_map:
            continue
        if h not in per:
            if songs.full(h, maps):
                continue
            maps.append(h)
        per[h] = per.get(h, 0) + 1
        out.append(c)
        if len(out) >= count:
            break
    return out


def _result(db, picked, replaced, aims, hist, ignored):
    """aims: {play style: aim used}; a play style missing there was not filtered by difficulty (too few clears)."""
    items = []
    for rank, (c, h, parts, diff) in enumerate(picked, 1):
        _, ch, d = chartids.split(db.chart_keys[c])
        reason = {"views": {v: round(s, 6) for v, s in parts.items()}, "difficulty": None}
        if diff is not None:
            nh, nc, nd = chartids.split(diff["nearestStrong"])
            reason["difficulty"] = {"aim": diff["aim"], "distance": round(diff["distance"], 6),
                                    "target": round(diff["target"], 6),
                                    "clearChance": None if diff["clearChance"] is None else round(diff["clearChance"], 4),
                                    "nearestStrong": {"hash": nh, "characteristic": nc, "difficulty": nd}}
        items.append({"hash": h, "characteristic": ch, "difficulty": d, "rank": rank, "reason": reason})
    return {"dbVersion": db.version, "items": items, "aims": aims, "replaced": replaced, "ignoredWeights": ignored,
            "unknownEvents": hist.unknown}


def describe(db, history, params=None):
    """A display summary of the player (contract 1 `describe`, shown by /v1/profile)."""
    why = why_invalid(params)
    if why:
        raise RequestError(f"params: {why}")
    p = resolve(params)
    hist = hist_mod.parse(db, history)
    skills = skill_mod.estimate(db, hist, p)
    feats = [f for f, w in p["features"].items() if w and f in db.chart_metrics]
    styles = {}
    for style, st in skills.items():
        strong = st.strong
        styles[style] = {
            "strongCharts": len(strong),
            "features": {f: float(np.nanmedian(db.chart_metrics[f][strong])) for f in feats},
            "typicalDistance": float(np.median(st.self_d)),
            "clearCurve": st.curve is not None,
            # heaviest first; weight: how many times the configured weight a difference that way counts (skill.py)
            "weakDirections": [{"feature": feats[f], "direction": side, "weight": m} for f, side, m in st.weak],
        }
    return {"dbVersion": db.version, "likedMaps": len(hist.liked()), "plays": len(hist.plays),
            "playStyles": styles, "replaced": hist.replaced, "unknownEvents": hist.unknown}
