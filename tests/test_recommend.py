"""recommend and describe (runtime/recommend.py, contract 1 (OctantCompass/OctantHarvest to OctantVector)) on a small synthetic DB written with common/dbformat.py."""
import json

import numpy as np
import pytest

from octantvector.common import dbformat
from octantvector.runtime import RequestError, SeedNotEmbeddedError, SeedNotInDBError, describe, open_db, recommend

MUSIC, CHART, GENRE = "music_x@1", "chart_x@1", "genre_x@1"
N = 40


def H(i):
    return f"{i:040x}"


OLD = "f" * 40  # an earlier version of map 0


BARE = "b" * 40


def seed(h, characteristic, difficulty):
    return {"hash": h, "characteristic": characteristic, "difficulty": difficulty}


def unit(v):
    v = np.asarray(v, np.float64)
    return (v / np.linalg.norm(v)).astype(np.float16)


@pytest.fixture
def db(tmp_path):
    rng = np.random.default_rng(0)
    c = dbformat.empty()
    c["calibration"] = {"id": "cal1"}
    c["superseded"] = {OLD: H(0)}
    music, chart, genre = {}, {}, {}
    for i in range(N):
        h = H(i)
        c["maps"][h] = ("artist", "mapper", 120.0, "2025-01-01")
        rows = [("Standard", "Hard", 3 + 0.05 * i, 12 + 0.1 * i, None, None, "Standard"),
                ("Standard", "Expert", 6 + 0.05 * i, 16 + 0.1 * i, None, None, "Standard")]
        if i % 5 == 0:
            rows.append(("OneSaber", "Expert", 5 + 0.05 * i, 15, None, None, "OneSaber"))
        c["charts"][h] = rows
        side = 0 if i < N // 2 else 1  # two clusters of songs
        if i != 7:  # map 7 has no music vector
            music[h] = unit(np.eye(4)[side] + 0.2 * rng.normal(size=4))
        genre[h] = unit([1.0, 0.1] if i % 2 == 0 else [0.1, 1.0])  # even maps: "jpop"
        for ch, d, *_ in rows:
            chart[f"{h}:{ch}:{d}"] = unit(np.eye(3)[side] + 0.2 * rng.normal(size=3))
    # A map in the catalogue with no vector in any view (the encoders have not reached it yet)
    c["maps"][BARE] = ("artist", "mapper", 120.0, "2025-01-01")
    c["charts"][BARE] = [("Standard", "Expert", 6.0, 16.0, None, None, "Standard")]
    c["views"] = {
        MUSIC: {"meta": {"role": "music", "rows": "maps", "dim": 4}, "vecs": music},
        CHART: {"meta": {"role": "chart", "rows": "charts", "dim": 3}, "vecs": chart},
        GENRE: {"meta": {"role": "genre", "rows": "maps", "dim": 2, "genres": ["jpop", "rock"]}, "vecs": genre},
    }
    return open_db(dbformat.write_full(tmp_path / "db", c, "2026.10.04", "2026-10-04T00:00:00Z"))


# The fixture's maps in one cluster have music vectors close enough to count as one song; the tests that are not about
# songs treat every map as its own song.
ONE_SONG_EACH = {"mapsPerSong": 0}


def like(i, at="2026-10-01T00:00:00Z"):
    return {"type": "like", "hash": H(i) if isinstance(i, int) else i, "at": at}


def play(i, diff="Hard", result="clear", at="2026-10-01T00:00:00Z", ch="Standard", **kw):
    return {"type": "play", "hash": H(i), "characteristic": ch, "difficulty": diff, "result": result, "at": at, **kw}


def liked(i):
    """A favourite on a cleared map (OctantSpecs law 4.1). The clear is on a chart the DB does not have, so the map is
    liked without giving the chart views a reference."""
    return [like(i), play(i, "ExpertPlus")]


def maps_of(out):
    return [int(it["hash"], 16) for it in out["items"]]


def test_taste_recommends_near_the_liked_maps_without_a_score(db):
    hist = liked(1) + liked(2) + liked(3)
    out = recommend(db, hist, {"count": 8}, ONE_SONG_EACH)
    assert len(out["items"]) == 8 and [it["rank"] for it in out["items"]] == list(range(1, 9))
    assert all(m < N // 2 for m in maps_of(out))  # the liked maps' cluster
    assert not {1, 2, 3} & set(maps_of(out))  # already known
    assert len(set(maps_of(out))) == 8  # one chart per map by default
    assert all("score" not in it for it in out["items"])
    assert set(out["items"][0]["reason"]["views"]) <= {MUSIC, CHART}  # the genre view is not a taste view
    assert out["dbVersion"] == "2026.10.04" and out["aims"] == {}  # too few clears: no difficulty filter
    assert recommend(db, list(reversed(hist)), {"count": 8}, ONE_SONG_EACH) == out  # same input, same list


def test_no_history_gives_an_empty_list(db):
    assert recommend(db, [], {"count": 5})["items"] == []


def test_a_missing_view_counts_zero_and_is_not_renormalised(db):
    hist = [like(1), like(2), play(1), play(2)]  # plays give the chart view a reference
    out = recommend(db, hist, {"count": N, "chartsPerMap": 3})
    seven = [it for it in out["items"] if it["hash"] == H(7)]
    assert seven and all(MUSIC not in it["reason"]["views"] for it in seven)
    other = next(it for it in out["items"] if it["hash"] != H(7))
    assert set(other["reason"]["views"]) == {MUSIC, CHART}
    # with only half of the weight present, a stricter minViewWeight drops it
    out = recommend(db, hist, {"count": N, "chartsPerMap": 3}, {"minViewWeight": 0.6})
    assert H(7) not in {it["hash"] for it in out["items"]}


def test_chart_views_compare_one_play_style_only(db):
    out = recommend(db, [], {"seed": seed(H(5), "OneSaber", "Expert"), "count": 20, "weights": {"chart_x": 1}})
    assert out["items"] and all(it["characteristic"] == "OneSaber" for it in out["items"])
    assert H(5) not in {it["hash"] for it in out["items"]}  # the seed's own map is left out


def test_seed_and_old_versions_are_replaced(db):
    out = recommend(db, [like(OLD)], {"seed": seed(OLD, "Standard", "Hard"), "count": 3})
    assert {"from": OLD, "to": H(0)} in out["replaced"]
    assert {"from": f"{OLD}:Standard:Hard", "to": f"{H(0)}:Standard:Hard"} in out["replaced"]
    assert H(0) not in {it["hash"] for it in out["items"]}
    with pytest.raises(SeedNotInDBError):
        recommend(db, [], {"seed": seed("0" * 39 + "z", "Standard", "Hard")})
    with pytest.raises(SeedNotInDBError):
        recommend(db, [], {"seed": seed(H(1), "Lawless", "Expert")})
    with pytest.raises(RequestError, match="must be an object"):  # contract 1: the seed is an object
        recommend(db, [], {"seed": f"{H(1)}:Standard:Hard"})


def test_seed_with_no_vector_is_reported_not_returned_empty(db):
    with pytest.raises(SeedNotEmbeddedError, match="has no vector") as ex:
        recommend(db, [], {"seed": seed(BARE, "Standard", "Expert"), "count": 5})
    assert isinstance(ex.value, SeedNotInDBError)
    with pytest.raises(SeedNotEmbeddedError):
        recommend(db, [like(1)], {"seed": seed(BARE, "Standard", "Expert"), "weights": {"chart_x": 1}})
    assert BARE not in {it["hash"] for it in recommend(db, [like(1)], {"count": N + 1})["items"]}


def test_genre_filters_by_the_word_and_ranks_by_recent_liked_plays(db):
    assert db.info()["genres"] == ["jpop", "rock"]
    hist = [like(1), like(2), like(22), play(22, at="2026-10-02T00:00:00Z")] + [play(22, at=f"2026-10-02T0{n}:00:00Z")
                                                                              for n in range(1, 3)]
    out = recommend(db, hist, {"genre": "jpop", "count": 4}, {"genreTopShare": 0.5, **ONE_SONG_EACH})
    assert out["items"] and all(m % 2 == 0 for m in maps_of(out))
    assert all(set(it["reason"]["views"]) == {MUSIC} for it in out["items"])  # music views only
    assert all(m >= N // 2 for m in maps_of(out))  # near map 22 (played recently), not maps 1 and 2 (never played)
    with pytest.raises(RequestError, match="genre word") as ex:
        recommend(db, hist, {"genre": "polka"})
    assert not isinstance(ex.value, SeedNotInDBError)


def test_exclusions_and_charts_per_map(db):
    hist = liked(1)
    out = recommend(db, hist, {"count": 10, "chartsPerMap": 2})
    per = {}
    for it in out["items"]:
        per[it["hash"]] = per.get(it["hash"], 0) + 1
    assert max(per.values()) == 2
    first = out["items"][0]
    ex = [{"hash": H(4), "reason": "anything"}, {k: first[k] for k in ("hash", "characteristic", "difficulty")}]
    out2 = recommend(db, hist, {"count": 10, "chartsPerMap": 2, "exclude": ex})
    keys = {(it["hash"], it["characteristic"], it["difficulty"]) for it in out2["items"]}
    assert H(4) not in {k[0] for k in keys}
    assert (first["hash"], first["characteristic"], first["difficulty"]) not in keys
    assert H(1) in {it["hash"] for it in recommend(db, hist, {"count": N}, {"includeKnown": True})["items"]}


def strong_history(n=12, fails=0):
    """Clears of the easier Hard charts of maps 20..; fails on Expert charts."""
    ev = [like(30)]
    for k in range(n):
        ev.append(play(20 + k, "Hard", at=f"2026-09-{k + 1:02d}T00:00:00Z"))
    for k in range(fails):
        ev.append(play(20 + k % 20, "Expert", "fail", at=f"2026-09-{k % 28 + 1:02d}T01:00:00Z"))
    return ev


def test_difficulty_filters_by_distance_from_the_strong_charts(db):
    params = {"minClears": 10, "includeKnown": True}
    normal = recommend(db, strong_history(), {"count": 10, "chartsPerMap": 2}, params)
    assert normal["aims"] == {"Standard": "normal"}
    for it in normal["items"]:
        if it["characteristic"] == "Standard":
            d = it["reason"]["difficulty"]
            assert d["aim"] == "normal" and d["clearChance"] is None
            assert d["nearestStrong"]["difficulty"] == "Hard"
        else:
            assert it["reason"]["difficulty"] is None  # OneSaber: no clears there, not filtered
    hard = [it for it in normal["items"] if it["characteristic"] == "Standard"]
    assert sum(it["difficulty"] == "Hard" for it in hard) > len(hard) / 2
    ch = recommend(db, strong_history(), {"count": 10, "chartsPerMap": 2, "difficulty": "challenge"}, params)
    dist = [it["reason"]["difficulty"]["distance"] for it in ch["items"] if it["reason"]["difficulty"]]
    assert ch["aims"] == {"Standard": "challenge"}
    assert np.mean(dist) > np.mean([it["reason"]["difficulty"]["distance"] for it in hard])
    with pytest.raises(RequestError):
        recommend(db, strong_history(), {"difficulty": "brutal"}, params)


def test_quits_and_modified_plays_are_not_strong(db):
    ev = [like(30)] + [play(20 + k, "Hard", "quit") for k in range(12)]
    assert recommend(db, ev, {"count": 3}, {"minClears": 10})["aims"] == {}
    for mods in ({"speed": 1.2}, {"speed": 1.0, "noFail": True}, {"fasterSong": True}):
        ev = [like(30)] + [play(20 + k, "Hard", modifiers=mods) for k in range(12)]
        assert recommend(db, ev, {"count": 3}, {"minClears": 10})["aims"] == {}
    ev = [like(30)] + [play(20 + k, "Hard", modifiers={"speed": 1.0, "noFail": False}) for k in range(12)]
    assert recommend(db, ev, {"count": 3}, {"minClears": 10})["aims"] == {"Standard": "normal"}


def test_recommendation_events_are_not_taste(db):
    ev = [{"type": "recommendation", "hash": H(1), "at": "2026-10-01T00:00:00Z", "stage": "shown"}]
    assert recommend(db, ev, {"count": 5})["items"] == []


def test_describe_summarises_without_scores(db):
    s = describe(db, strong_history(), {"minClears": 10})
    assert s["likedMaps"] == 1 and s["plays"] == 12
    st = s["playStyles"]["Standard"]
    assert st["strongCharts"] == 12 and set(st["features"]) == {"nps", "njs"} and st["clearCurve"] is False
    json.dumps(s)  # plain JSON for /v1/profile
    assert describe(db, [], None)["playStyles"] == {}


def test_imported_play_counts_only_without_plays_and_never_as_skill(db):
    imp = [{"type": "imported_play", "hash": H(3), "characteristic": "Standard", "difficulty": "Hard", "count": 3,
            "at": "2026-09-01T00:00:00Z"}]
    assert recommend(db, imp, {"count": 3})["items"] == []  # the count alone is not a liked map
    out = recommend(db, imp + [like(3)], {"count": 3}, ONE_SONG_EACH)
    assert out["items"] and all(m < N // 2 for m in maps_of(out))  # a favourite, clear unknown: liked, its cluster
    # a play replaces the import: not cleared, so the favourite is a goal
    assert recommend(db, imp + [like(3), play(3, result="quit")], {"count": 3})["items"] == []
    many = [dict(imp[0], hash=H(20 + k)) for k in range(12)]
    assert recommend(db, many, {"count": 3}, {"minClears": 10})["aims"] == {}  # not clears


def test_like_removed_is_back_to_no_favorite_not_a_dislike(db):
    removed = {"type": "like_removed", "hash": H(1), "at": "2026-10-01T00:01:00Z"}
    assert describe(db, [like(1), play(1), removed])["likedMaps"] == 0
    relike = like(1, at="2026-10-01T00:02:00Z")
    assert describe(db, [like(1), play(1), removed, relike])["likedMaps"] == 1  # the latest one wins
    again = play(1, at="2026-10-02T00:00:00Z")
    assert describe(db, [like(1), play(1), removed, again])["likedMaps"] == 1  # back on a later day


def test_liked_maps_goals_and_challenges(db):
    """OctantSpecs law 4.1: liked needs a clear; a mark on an uncleared map is a goal; uncleared repeats are neither."""
    def n(ev):
        return describe(db, ev)["likedMaps"]
    up = {"type": "upvote", "hash": H(1), "at": "2026-10-01T00:00:00Z"}
    assert n([up, play(1)]) == 1 and n([like(1), up, play(1)]) == 1  # either mark, or both
    assert n([play(1)]) == 0  # one clear only
    assert n([play(1), play(1, at="2026-10-01T05:00:00Z")]) == 0  # again, but on the same day
    assert n([play(1), play(1, "Expert", "fail", at="2026-10-03T00:00:00Z")]) == 1  # back on a later day
    assert n([play(1), play(1, result="quit", at="2026-10-03T00:00:00Z")]) == 0  # a quit is not a play
    assert n([play(1, result="fail"), play(1, at="2026-10-03T00:00:00Z")]) == 0  # the first clear starts the count
    assert n([play(1, result="fail", at=f"2026-10-0{d}T00:00:00Z") for d in range(1, 6)]) == 0  # challenging
    assert n([like(1), play(1, result="fail")]) == 0  # a goal
    goal = [like(1), play(1, result="fail")]
    assert recommend(db, goal, {"count": 3})["items"] == []  # a goal is not a taste reference
    assert n(goal + [play(1, at="2026-10-02T00:00:00Z")]) == 1  # cleared later: liked


def test_goals_come_back_in_the_challenge_aim_only(db):
    """OctantSpecs law 4.1: a goal (a mark on a map not cleared yet) is used for challenges. Map 32's Hard chart is within
    the challenge tolerance of strong_history; map 35 has a OneSaber chart, a play style with no skill."""
    params = {"minClears": 10, **ONE_SONG_EACH}

    def maps(ev, aim, more=None):
        out = recommend(db, strong_history() + ev, {"count": N, "chartsPerMap": 3, "difficulty": aim},
                        {**params, **(more or {})})
        return {(int(it["hash"], 16), it["characteristic"]) for it in out["items"]}
    fail = play(32, "Expert", "fail")
    assert (32, "Standard") in maps([], "challenge")
    assert (32, "Standard") not in maps([fail], "challenge")  # known, not a goal
    assert (32, "Standard") in maps([like(32), fail], "challenge")
    assert (32, "Standard") in maps([{"type": "upvote", "hash": H(32), "at": "2026-10-01T00:00:00Z"}], "challenge")
    assert (32, "Standard") not in maps([like(32), fail], "normal")  # other aims still leave it out
    deleted = {"type": "user_delete", "hash": H(32), "at": "2026-10-02T00:00:00Z"}
    assert (32, "Standard") not in maps([like(32), fail, deleted], "challenge")
    assert (35, "OneSaber") in maps([], "challenge")
    assert 35 not in {m for m, _ in maps([like(35), play(35, "Expert", "fail")], "challenge")}  # no skill there
    assert (35, "OneSaber") in maps([like(35), play(35, "Expert", "fail")], "challenge", {"includeKnown": True})


def test_weights_for_views_the_db_lacks_are_ignored_and_returned(db):
    out = recommend(db, liked(1), {"count": 3, "weights": {"music_x": 1, "gone@9": 2}})
    assert out["ignoredWeights"] == ["gone@9"]
    assert out["items"] and all(set(it["reason"]["views"]) <= {MUSIC} for it in out["items"])
    only_gone = recommend(db, liked(1), {"count": 3, "weights": {"gone@9": 1}})
    assert only_gone["ignoredWeights"] == ["gone@9"]
    assert maps_of(only_gone) == maps_of(recommend(db, liked(1), {"count": 3}))  # same as no weights
    assert recommend(db, liked(1), {"count": 3})["ignoredWeights"] == []


def test_chart_views_take_imported_play_charts_as_the_reference(db):
    imp = [{"type": "imported_play", "hash": H(3), "characteristic": "Standard", "difficulty": "Hard", "count": 3,
            "at": "2026-09-01T00:00:00Z"}]
    out = recommend(db, imp + [like(3)], {"count": 3, "weights": {"chart_x": 1}})
    assert out["items"] and all(m < N // 2 for m in maps_of(out))  # map 3's cluster, through its imported chart
    assert all(set(it["reason"]["views"]) == {CHART} for it in out["items"])
    assert out["ignoredWeights"] == []


def test_weights_without_a_reference_are_ignored_and_returned(db):
    # the liked map's only clear is on a chart the DB lacks: no chart reference, so the chart weight is ignored like a
    # missing view
    liked_only = recommend(db, liked(3), {"count": 3, "weights": {"chart_x": 1}})
    assert liked_only["ignoredWeights"] == ["chart_x"]
    assert maps_of(liked_only) == maps_of(recommend(db, liked(3), {"count": 3}))  # same as no weights
    both = recommend(db, liked(3), {"count": 3, "weights": {"chart_x": 1, "music_x": 1}})
    assert both["ignoredWeights"] == ["chart_x"]
    assert both["items"] and all(set(it["reason"]["views"]) == {MUSIC} for it in both["items"])
    genre = recommend(db, [like(2), play(2)], {"count": 3, "genre": "jpop", "weights": {"chart_x": 1}})
    assert genre["ignoredWeights"] == ["chart_x"]  # the genre entry ranks by music only


def test_other_mappers_maps_of_one_song_take_one_place(tmp_path):
    """A famous song mapped by many mappers: the maps have nearly the same audio and would fill the list."""
    rng = np.random.default_rng(1)
    c = dbformat.empty()
    c["calibration"] = {"id": "cal1"}
    music, chart = {}, {}
    for i in range(30):
        h = H(i)
        song = i < 8  # maps 0..7: one song, eight mappers
        c["maps"][h] = ("artist", f"mapper{i}", 120.0, "2025-01-01")
        c["charts"][h] = [("Standard", "Expert", 6, 16, None, None, "Standard")]
        base = np.eye(6)[0] if song else np.eye(6)[0] * 0.6 + np.eye(6)[1] * 0.8
        music[h] = unit(base + (0.02 if song else 0.4) * rng.normal(size=6))
        chart[f"{h}:Standard:Expert"] = unit(rng.normal(size=3))
    c["views"] = {MUSIC: {"meta": {"role": "music", "rows": "maps", "dim": 6}, "vecs": music},
                  CHART: {"meta": {"role": "chart", "rows": "charts", "dim": 3}, "vecs": chart}}
    db = open_db(dbformat.write_full(tmp_path / "db", c, "2026.10.08", "2026-10-08T00:00:00Z"))
    hist = liked(0)
    out = recommend(db, hist, {"count": 10})
    assert len(out["items"]) == 10 and len([m for m in maps_of(out) if m < 8]) == 1
    out = recommend(db, hist, {"count": 10}, {"mapsPerSong": 3})
    assert len([m for m in maps_of(out) if m < 8]) == 3
    out = recommend(db, hist, {"count": 10}, {"mapsPerSong": 0})  # no limit: the song fills the top
    assert maps_of(out)[:7] and len([m for m in maps_of(out) if m < 8]) == 7


def test_narrow_keeps_the_maps_closest_by_one_view_and_ranks_them_by_another(db):
    seed = {"hash": H(1), "characteristic": "Standard", "difficulty": "Hard"}
    params = {**ONE_SONG_EACH, "narrowTopMaps": 5}
    narrowed = recommend(db, [], {"seed": seed, "count": 5, "narrow": {"music_x": 1}, "weights": {"chart_x": 1}}, params)
    by_music = recommend(db, [], {"seed": seed, "count": 5, "weights": {"music_x": 1}}, params)
    assert len(narrowed["items"]) == 5 and set(maps_of(narrowed)) == set(maps_of(by_music))
    by_chart = [m for m in maps_of(recommend(db, [], {"seed": seed, "count": 40, "weights": {"chart_x": 1}}, params))
                if m in set(maps_of(narrowed))]
    assert maps_of(narrowed) == by_chart  # inside the narrowed maps, the order is the chart order
    out = recommend(db, [], {"seed": seed, "count": 3, "narrow": {"meta_x": 1}, "weights": {"chart_x": 1}}, params)
    assert out["ignoredWeights"] == ["meta_x"] and len(out["items"]) == 3  # nothing to narrow by: not narrowed


def test_narrow_widens_when_too_few_are_left(db):
    seed = {"hash": H(1), "characteristic": "Standard", "difficulty": "Hard"}
    params = {**ONE_SONG_EACH, "narrowTopMaps": 5, "narrowWiden": 3}
    wide = recommend(db, [], {"seed": seed, "count": 10, "narrow": {"music_x": 1}, "weights": {"chart_x": 1}}, params)
    by_music = recommend(db, [], {"seed": seed, "count": 15, "weights": {"music_x": 1}}, params)
    assert len(wide["items"]) == 10 and set(maps_of(wide)) <= set(maps_of(by_music))  # 5 maps, then 15


def test_a_seed_with_a_genre_word_stays_the_reference_and_the_word_narrows(db):
    seed = {"hash": H(2), "characteristic": "Standard", "difficulty": "Hard"}
    params = {"genreTopShare": 0.5, **ONE_SONG_EACH}
    out = recommend(db, [], {"seed": seed, "genre": "rock", "count": 6, "weights": {"music_x": 1}}, params)
    assert out["items"] and all(m % 2 == 1 for m in maps_of(out))  # rock: the odd maps
    assert all(m < N // 2 for m in maps_of(out))  # still near the seed's cluster


def test_a_view_id_without_its_version_names_that_view(db):
    bare = MUSIC.split("@")[0]
    out = recommend(db, liked(1), {"count": 3, "weights": {bare: 1}})
    assert out["ignoredWeights"] == []
    assert maps_of(out) == maps_of(recommend(db, liked(1), {"count": 3, "weights": {MUSIC: 1}}))


def test_skill_distances_match_one_by_one():
    """Style.distances (every chart at once, in blocks) gives what distance and nearest_strong give one by one, with
    the configured weights and with learned weak directions (a difference weighed by its sign)."""
    from octantvector.common.params import DEFAULTS
    from octantvector.runtime.skill import Style
    rng = np.random.default_rng(1)
    rows = np.arange(100, 20100)
    X = rng.normal(size=(len(rows), 3))
    X[[5, 9000]] = np.nan  # charts without features
    strong = [int(r) for r in rng.choice(rows[10:], 60, replace=False)]
    st = Style("Standard", rows, X, np.array([1.0, 0.5, 2.0]), strong, DEFAULTS)
    for up, down in ((st.w, st.w), (st.w * [1, 3, 1], st.w * [2, 1, 1.5])):
        st.up, st.down = up, down
        st._measure()
        dist, nearest = st.distances()
        for n in list(range(0, 20)) + [9000] + [st.pos[r] for r in strong[:20]]:
            r = int(rows[n])
            one = st.distance(r)
            assert (np.isnan(dist[n]) if one is None else dist[n] == one)
            if one is not None:
                S = [s for s in strong if s != r]
                e = X[n] - X[[st.pos[s] for s in S]]
                d = np.sqrt(((e ** 2) * np.where(e > 0, up, down)).sum(1))
                assert nearest[n] == st.nearest_strong(r) == S[int(np.argmin(d))]


def weak_player(path, logit, seed=2):
    """One Standard chart per map, NPS and NJS drawn apart, and two more at middle NPS that nobody plays, below and
    above every other in NJS. Recently the player cleared the charts in the middle, then played each other chart once,
    clearing it with the chance logit(nps, njs) gives; the strong set is every chart cleared, so every play is recent.
    -> (db, events, params, map of the chart below, map of the chart above)."""
    rng = np.random.default_rng(seed)
    c = dbformat.empty()
    c["calibration"] = {"id": "cal1"}
    nps, njs, music = rng.uniform(3, 9, 300), rng.uniform(10, 22, 300), {}
    for i, a, b in [(i, nps[i], njs[i]) for i in range(300)] + [(300, 5.5, 6.0), (301, 5.5, 26.0)]:
        c["maps"][H(i)] = ("artist", "mapper", 120.0, "2025-01-01")
        c["charts"][H(i)] = [("Standard", "Expert", float(a), float(b), None, None, "Standard")]
        music[H(i)] = unit(rng.normal(size=4))
    c["views"] = {MUSIC: {"meta": {"role": "music", "rows": "maps", "dim": 4}, "vecs": music}}
    db = open_db(dbformat.write_full(path / "db", c, "2026.10.15", "2026-10-15T00:00:00Z"))
    middle = (nps >= 4.5) & (nps <= 6.5) & (njs >= 14) & (njs <= 18)
    ev = []
    for k, i in enumerate(list(np.flatnonzero(middle)) + list(np.flatnonzero(~middle))):
        cleared = middle[i] or rng.random() < 1 / (1 + np.exp(-logit(nps[i], njs[i])))
        ev.append(play(i, "Expert", "clear" if cleared else "fail", at=f"2026-09-01T00:{k // 60:02d}:{k % 60:02d}Z"))
    params = {"minClears": 20, "strongRecentClears": sum(e["result"] == "clear" for e in ev)}
    return db, ev, params, 300, 301


@pytest.fixture
def weak(tmp_path):
    """The clear chance falls fast as NJS goes below the middle, slower as NPS goes above it, and not at all the other
    two ways (the logit in standard deviations of the features)."""
    return weak_player(tmp_path, lambda nps, njs: 2.5 - 15 * max(0, 14 - njs) / 3.46 - 4.5 * max(0, nps - 6.5) / 1.73)


def test_weak_directions_are_learned_from_enough_fails(weak):
    """OctantSpecs difficulty-fit.md 2: with enough fails, the direction that tells fails from clears is learned and shown
    by describe; its weight is shrunk towards the configured one by the number of plays."""
    db, ev, params, _, _ = weak
    fails = sum(e["result"] == "fail" for e in ev)
    assert fails >= 30
    st = describe(db, ev, params)["playStyles"]["Standard"]
    assert st["clearCurve"] is True
    assert st["weakDirections"][0]["feature"] == "njs" and st["weakDirections"][0]["direction"] == "low"
    assert all(d["weight"] > 1 for d in st["weakDirections"])
    assert ("nps", "low") not in {(d["feature"], d["direction"]) for d in st["weakDirections"]}  # easier: not weak
    json.dumps(st)

    def njs_low(extra):
        return describe(db, ev, {**params, **extra})["playStyles"]["Standard"]["weakDirections"][0]["weight"]
    assert njs_low({"directionShrinkPlays": 0}) > njs_low({}) > njs_low({"directionShrinkPlays": 1e9}) > 1
    assert njs_low({"directionShrinkPlays": 1e9}) < 1.001  # few plays against a strong pull: about the configured
    st = describe(db, ev, {**params, "minFailsForCurve": fails + 1})["playStyles"]["Standard"]
    assert st["weakDirections"] == [] and st["clearCurve"] is False


def test_weak_directions_weigh_more_in_the_distance(weak):
    """Once NJS lower is learned, the chart below every other in NJS moves further from the strong charts; the chart
    above them stays where it was."""
    from octantvector.common.params import resolve
    from octantvector.runtime import history, skill
    db, ev, params, below, above = weak
    hist = history.parse(db, ev)

    def far(extra):
        st = skill.estimate(db, hist, resolve({**params, **extra}))["Standard"]
        return [st.distance(db.chart_pos[f"{H(i)}:Standard:Expert"]) for i in (below, above)]
    before, after = far({"minFailsForCurve": len(ev) + 1}), far({})
    assert after[0] > before[0] * 1.4 and after[1] == pytest.approx(before[1])


def test_weak_directions_are_learned_from_the_recent_plays(weak):
    """OctantSpecs difficulty-fit.md 2 「最近」: weak directions are learned from the plays since the strong set's oldest
    clear, and need minFailsForCurve fails among them. Fails from before it (the player may have grown since) change
    nothing; the same fails made recently do."""
    db, ev, params, _, above = weak

    def learned(at, extra=None):
        fails = [play(above, "Expert", "fail", at=f"{at}T00:00:{n:02d}Z") for n in range(30)]
        return describe(db, ev + fails, {**params, **(extra or {})})["playStyles"]["Standard"]
    recent = describe(db, ev, params)["playStyles"]["Standard"]["weakDirections"]
    assert recent and learned("2026-06-01")["weakDirections"] == recent
    assert learned("2026-09-30")["weakDirections"] != recent
    fails = sum(e["result"] == "fail" for e in ev)
    st = learned("2026-06-01", {"minFailsForCurve": fails + 1})  # enough fails for the curve, too few recent ones
    assert st["clearCurve"] is True and st["weakDirections"] == []


def test_learning_measures_the_played_charts_only(weak, monkeypatch):
    """Learning looks up the nearest strong chart of the played charts one by one: describe never measures every chart
    of the play style (Style.distances), and a recommendation does it once, with the learned weights."""
    from octantvector.runtime import skill
    db, ev, params, _, _ = weak
    measured, real = [], skill.Style.distances

    def distances(self):
        measured.append(self._all is None)
        return real(self)
    monkeypatch.setattr(skill.Style, "distances", distances)
    assert describe(db, ev, params)["playStyles"]["Standard"]["weakDirections"] and measured == []
    out = recommend(db, ev + [like(0)], {"count": 5, "difficulty": "challenge"}, {**params, "includeKnown": True})
    assert out["items"] and sum(measured) == 1


def test_a_direction_that_alone_lowers_the_clear_chance_is_learned(tmp_path):
    """When the clear chance falls one way only, that way is learned for every player drawn alike, however the ways
    that do not matter happen to fit (a little falling or a little rising)."""
    for extra, logit, way in (({"features": {"njs": 1.0}}, lambda nps, njs: 2.5 - 15 * max(0, njs - 18) / 3.46,
                               ("njs", "high")),
                              ({}, lambda nps, njs: 2.5 - 15 * max(0, 14 - njs) / 3.46, ("njs", "low"))):
        for n in range(2, 8):
            db, ev, params, _, _ = weak_player(tmp_path / f"{way[1]}{n}", logit, n)
            st = describe(db, ev, {**params, **extra})["playStyles"]["Standard"]
            assert [(d["feature"], d["direction"]) for d in st["weakDirections"]] == [way]
            assert st["weakDirections"][0]["weight"] > 2


def test_no_weak_direction_when_the_configured_distance_is_right(tmp_path):
    """When the clear chance falls with the configured distance alike every way, the learned weights stay about the
    configured ones for every player drawn alike: what one direction's fit adds by chance stays within its noise."""
    def logit(nps, njs):  # falling with the distance from the middle, in standard deviations
        return 3 - 1.2 * np.hypot((nps - 5.5) / 1.73, (njs - 16) / 3.46)
    for n in range(2, 10):
        db, ev, params, _, _ = weak_player(tmp_path / str(n), logit, n)
        assert all(d["weight"] < 1.5 for d in describe(db, ev, params)["playStyles"]["Standard"]["weakDirections"])


def test_weak_direction_weights_do_not_hang_on_the_sign_of_noise(monkeypatch):
    """A direction's drop is weighed against the mean drop of every direction (one where the clear chance rises counts
    as no drop), so a direction whose fit is about 0 counts the same on either side of 0; and a drop counts only beyond
    its standard error, so a drop within the noise learns nothing however the others fit."""
    from octantvector.common.params import DEFAULTS
    from octantvector.runtime import skill
    X = np.random.default_rng(3).normal(size=(200, 2))

    def weak(coefficients, se=(0, 0, 0, 0)):  # of (NPS higher, NJS higher, NPS lower, NJS lower)
        monkeypatch.setattr(skill, "_logistic", lambda *a, **k: ((1.0, *coefficients), np.array(se, float)))
        st = skill.Style("Standard", np.arange(200), X, np.ones(2), range(50), {**DEFAULTS, "directionShrinkPlays": 0})
        st.learn(range(50, 200), [True] * 150)
        return [(f, side, round(m, 6)) for f, side, m in st.weak]
    assert weak((1e-9, 0, 0, -3)) == weak((-1e-9, 0, 0, -3)) == weak((0, 0, 0.2, -3)) == [(1, "low", 16.0)]
    assert weak((0, -2, 0, -2)) == [(1, "high", 4.0), (1, "low", 4.0)]
    assert weak((-1, -1, -1, -1)) == weak((0, 0, 0, 0)) == []  # as the configured weights say; nothing drops
    assert weak((0, 0, 0, -3), (1, 1, 1, 1)) == [(1, "low", round((2 / 0.75) ** 2, 6))]
    assert weak((0, -0.2, 0, 0), (0.9, 0.9, 0.9, 0.9)) == []  # alone, but within the noise


def test_params_values_outside_the_bounds_are_refused(db):
    """A value that could run the host away (the tolerance widening steps 0.5 up to toleranceMaxSd) or break numpy is
    refused as a RequestError before anything runs (notes/security-audit-2026-10-21.md, 8)."""
    for bad in ({"toleranceMaxSd": 1e308}, {"toleranceMaxSd": float("inf")}, {"normalPercentile": 200},
                {"includeKnown": "yes"}, {"k": 0}, {"narrowTopMaps": 2.5}, {"features": {"nps": float("nan")}},
                {"challengeClearChance": 1.0}):
        with pytest.raises(RequestError, match="params"):
            recommend(db, [], {"count": 5}, bad)
        with pytest.raises(RequestError, match="params"):
            describe(db, [], bad)
    # a key this version does not know is still ignored, and whole numbers written as floats still pass
    recommend(db, [], {"count": 5}, {"futureKnob": 1e308, "narrowTopMaps": 300.0})
