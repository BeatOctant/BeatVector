"""The relation view of role `artist` (OctantSpecs contracts/, contract 2 ビューの種類; law 3.4): written and read with
common/dbformat.py, built from the crawl's two files (generation/artists.py), and used by recommend."""
import gzip
import json

import numpy as np

from octantvector.common import dbformat
from octantvector.generation import artists
from octantvector.runtime import open_db, presets, read_info, recommend
from octantvector.runtime.db import apply_diff

MUSIC, ART = "music_x@1", "listenbrainz_artists@20261012"


def H(i):
    return f"{i:040x}"


def unit(v):
    v = np.asarray(v, np.float64)
    return (v / np.linalg.norm(v)).astype(np.float16)


def content(n=30, members=None, pairs=None, song=()):
    """n maps with nearly random music; maps in `song` share one tune. members: {map: (artist, ...)}."""
    rng = np.random.default_rng(3)
    c = dbformat.empty()
    c["calibration"] = {"id": "cal1", "views": {}}
    music = {}
    tune = unit(rng.normal(size=8))
    for i in range(n):
        h = H(i)
        c["maps"][h] = ("artist", "mapper", 120.0, "2025-01-01")
        c["charts"][h] = [("Standard", "Expert", 6, 16, None, None, "Standard")]
        music[h] = unit(tune + 0.01 * rng.normal(size=8)) if i in song else unit(rng.normal(size=8))
    c["views"] = {MUSIC: {"meta": {"role": "music", "rows": "maps", "dim": 8, "fingerprint": "f1"}, "vecs": music}}
    if members is not None:
        c["relations"] = {ART: {"meta": {"role": "artist", "rows": "maps", "source": "listenbrainz:test",
                                         "fetchedAt": "2026-10-12T00:00:00Z", "perMember": 30,
                                         "fingerprint": "listenbrainz:test", "sources": []},
                                "members": {H(i): tuple(m) for i, m in members.items()},
                                "pairs": sorted(pairs or [])}}
    return c


def write(tmp_path, c, name="db"):
    return dbformat.write_full(tmp_path / name, c, "2026.10.12", "2026-10-12T00:00:00Z")


def liked(i):
    return [{"type": "like", "hash": H(i), "at": "2026-10-01T00:00:00Z"},
            {"type": "play", "hash": H(i), "characteristic": "Standard", "difficulty": "ExpertPlus",
             "result": "clear", "at": "2026-10-01T00:00:00Z"}]


def seed(i):
    return {"hash": H(i), "characteristic": "Standard", "difficulty": "Expert"}


def maps_of(out):
    return [int(it["hash"], 16) for it in out["items"]]


def test_a_relation_view_is_written_read_and_listed(tmp_path):
    c = content(members={0: ["a"], 1: ["a", "b"], 2: ["c"]}, pairs=[("a", "c", 0.5)])
    m = json.loads((write(tmp_path, c) / "manifest.json").read_text())
    v = m["views"][ART]
    assert v["kind"] == "relation" and v["role"] == "artist" and v["rows"] == "maps"
    assert v["membersFile"] in m["files"] and v["pairsFile"] in m["files"]
    back = dbformat.read_content(tmp_path / "db")
    assert back["relations"][ART]["members"] == {H(0): ("a",), H(1): ("a", "b"), H(2): ("c",)}
    assert back["relations"][ART]["pairs"] == [("a", "c", 0.5)]
    assert back["relations"][ART]["meta"]["source"] == "listenbrainz:test" and not back["unread"]
    db = open_db(tmp_path / "db")
    info = db.info()["views"]
    assert info[ART] == {"role": "artist", "kind": "relation", "rowKind": "maps", "rows": 3, "dim": 0, "credits": []}
    assert info[MUSIC]["kind"] == "vector"
    # read without opening: the same, less the rows, in the same order
    assert list(read_info(tmp_path / "db")["views"]) == list(info)
    assert read_info(tmp_path / "db")["views"][ART] == {k: v for k, v in info[ART].items() if k != "rows"}


def test_closeness_is_one_for_a_shared_artist_else_the_best_pair_either_way(tmp_path):
    c = content(members={0: ["a"], 1: ["a", "b"], 2: ["c"], 3: ["d"], 4: ["e"]},
                pairs=[("a", "c", 0.5), ("d", "a", 0.8), ("c", "a", 0.3)])
    db = open_db(write(tmp_path, c))
    rel = db.relations[ART]
    close = dict(zip(rel.rows.tolist(), rel.closeness(db.pos[H(0)]).tolist()))
    assert close[db.pos[H(1)]] == 1.0 and close[db.pos[H(0)]] == 1.0
    assert abs(close[db.pos[H(2)]] - 0.5) < 1e-6  # the larger of a->c 0.5 and c->a 0.3
    assert abs(close[db.pos[H(3)]] - 0.8) < 1e-6  # only d->a: used both ways
    assert close[db.pos[H(4)]] == 0.0
    assert rel.closeness(db.pos[H(9)]) is None  # no artists
    two = rel.reference([db.pos[H(2)], db.pos[H(4)], db.pos[H(9)]])  # mean over the references with artists
    assert abs(dict(zip(rel.rows.tolist(), two.tolist()))[db.pos[H(0)]] - (0.5 + 0.0) / 2) < 1e-6


def test_the_artist_weight_lifts_nearby_artists_and_never_drops_maps_without_artists(tmp_path):
    members = {i: [f"x{i}"] for i in range(20)}
    members[0], members[5] = ["star"], ["friend"]
    c = content(members=members, pairs=[("star", "friend", 1.0)])
    db = open_db(write(tmp_path, c))
    plain = recommend(db, [], {"seed": seed(0), "count": 29, "weights": {"music_x": 1}})
    lifted = recommend(db, [], {"seed": seed(0), "count": 29, "weights": {"music_x": 1, "listenbrainz_artists": 0.25}})
    assert maps_of(lifted)[0] == 5 and maps_of(plain)[0] != 5
    assert ART in lifted["items"][0]["reason"]["views"] and not lifted["ignoredWeights"]
    assert set(maps_of(lifted)) == set(maps_of(plain))  # maps 20..29 have no artists and stay candidates
    # several references: the mean closeness of the liked maps
    out = recommend(db, liked(0) + liked(7), {"count": 5, "weights": {"music_x": 1, "listenbrainz_artists": 2}})
    assert maps_of(out)[0] == 5


def test_maps_per_artist_caps_one_artist_set_and_a_duet_is_another_artist(tmp_path):
    members = {i: ["solo"] for i in range(1, 8)}
    members[0], members[8] = ["solo"], ["solo", "guest"]
    c = content(members=members)
    db = open_db(write(tmp_path, c))
    w = {"music_x": 1, "listenbrainz_artists": 1.0}
    out = recommend(db, [], {"seed": seed(0), "count": 10, "weights": w})
    assert len([m for m in maps_of(out) if 1 <= m <= 7]) == 3 and 8 in maps_of(out)  # the default, 3
    out = recommend(db, [], {"seed": seed(0), "count": 10, "weights": w}, {"mapsPerArtist": 5})
    assert len([m for m in maps_of(out) if 1 <= m <= 7]) == 5
    out = recommend(db, [], {"seed": seed(0), "count": 10, "weights": w}, {"mapsPerArtist": 0})
    assert len([m for m in maps_of(out) if 1 <= m <= 7]) == 7  # no limit


def test_a_cover_by_other_artists_is_not_the_same_song(tmp_path):
    """Maps 1..4 have one tune: 1 and 2 by the same artist, 3 by that artist with a guest (one artist set holds the
    other: the same song, as "A feat. B" mostly is), 4 a cover by another artist."""
    c = content(members={1: ["orig"], 2: ["orig"], 3: ["orig", "guest"], 4: ["other"]}, song=(1, 2, 3, 4))
    db = open_db(write(tmp_path, c))
    out = recommend(db, [], {"seed": seed(0), "count": 29, "weights": {"music_x": 1}})
    tune = [m for m in maps_of(out) if m in (1, 2, 3, 4)]
    assert len(tune) == 2 and 4 in tune  # one of 1, 2 and 3 (the same song), and the cover
    without = open_db(write(tmp_path, content(song=(1, 2, 3, 4)), "plain"))
    out = recommend(without, [], {"seed": seed(0), "count": 29, "weights": {"music_x": 1}})
    assert len([m for m in maps_of(out) if m in (1, 2, 3, 4)]) == 1  # without artists: audio only


def test_the_seeds_song_counts_as_already_in_the_list(tmp_path):
    """Maps 0..2 have one tune by one artist: seeded with map 0, other mappers' maps of its song stay out."""
    c = content(members={0: ["orig"], 1: ["orig"], 2: ["orig"]}, song=(0, 1, 2))
    db = open_db(write(tmp_path, c))
    out = recommend(db, [], {"seed": seed(0), "count": 29, "weights": {"music_x": 1}})
    assert len(out["items"]) == 27 and not {1, 2} & set(maps_of(out))
    out = recommend(db, [], {"seed": seed(0), "count": 29, "weights": {"music_x": 1}}, {"mapsPerSong": 2})
    assert len([m for m in maps_of(out) if m in (1, 2)]) == 1  # the seed takes one of the two places


def test_a_hand_naming_the_artist_view_is_served_only_with_it(tmp_path):
    hand = {"id": "song", "label": "Similar songs", "inputs": [{"kind": "seed"}], "handling": "near",
            "knobs": {"weights": {"music_x": 1, "listenbrainz_artists": 0.25}}}
    c = content(members={0: ["a"]})
    c["presets"] = [hand]
    assert [p["id"] for p in presets(open_db(write(tmp_path, c)))["presets"]] == ["song"]
    c = content()
    c["presets"] = [hand]
    assert presets(open_db(write(tmp_path, c, "plain")))["presets"] == []


def test_relation_views_travel_whole_in_a_diff_and_full_plus_diff_is_the_new_full(tmp_path):
    base = content(n=10, members={0: ["a"], 1: ["b"]}, pairs=[("a", "b", 1.0)])
    b = write(tmp_path, base, "base")
    new = content(n=12, members={0: ["a"], 1: ["b"], 11: ["a"]}, pairs=[("a", "b", 0.9)])
    diff = dbformat.write_diff(tmp_path / "d", dbformat.read_content(b), json.loads((b / "manifest.json").read_text()),
                               new, "2026.10.13", "2026-10-13T00:00:00Z")
    dm = json.loads((diff / "manifest.json").read_text())
    assert dm["views"][ART]["change"] == "added" and dm["views"][MUSIC]["change"] == "rows"
    applied = dbformat.read_content(apply_diff(b, diff, tmp_path / "applied"))
    full = dbformat.read_content(dbformat.write_full(tmp_path / "full", new, "2026.10.13", "2026-10-13T00:00:00Z"))
    assert applied["relations"] == full["relations"]
    assert applied["relations"][ART]["members"][H(11)] == ("a",)
    # a new fetch replaces the view by its new version
    newer = content(n=12, members={0: ["a"]})
    newer["relations"]["listenbrainz_artists@20261020"] = newer["relations"].pop(ART)
    f = tmp_path / "full"
    d2 = dbformat.write_diff(tmp_path / "d2", full, json.loads((f / "manifest.json").read_text()), newer,
                             "2026.10.20", "2026-10-20T00:00:00Z")
    m2 = json.loads((d2 / "manifest.json").read_text())
    assert m2["views"]["listenbrainz_artists@20261020"]["replaces"] == ART and ART not in m2["removedViews"]
    applied = dbformat.read_content(apply_diff(f, d2, tmp_path / "applied2"))
    assert list(applied["relations"]) == ["listenbrainz_artists@20261020"]


def _gz(p, rows):
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_the_relation_is_built_from_the_crawls_two_files(tmp_path):
    _gz(tmp_path / artists.MAP_FILE, [{"hash": H(0).upper(), "mbids": ["a"]}, {"hash": H(1), "mbids": ["b", "a"]},
                                      {"hash": H(2), "mbids": ["c"]}, {"hash": H(9), "mbids": ["z"]},
                                      {"hash": H(3), "mbids": []}])
    _gz(tmp_path / artists.SIMILAR_FILE, [{"algorithm": "algo1", "fetchedAt": "2026-10-12T12:00:00Z"},
                                          {"mbid": "a", "similar": [["q", 10], ["c", 5], ["b", 2.5]]},
                                          {"mbid": "c", "similar": []}, {"mbid": "z", "similar": [["a", 1]]}])
    key, rel = artists.relation(tmp_path, [H(0), H(1), H(2), H(3)], per_member=1)
    assert key == "listenbrainz_artists@20261012"
    assert rel["members"] == {H(0): ("a",), H(1): ("a", "b"), H(2): ("c",)}  # z's map is not in the DB
    assert rel["pairs"] == [("a", "c", 0.5)]  # divided by the list's best (q, not a member); one per member
    assert rel["meta"]["source"] == "listenbrainz:algo1" and rel["meta"]["role"] == "artist"
    assert rel["meta"]["credits"] == list(artists.CREDITS)
    assert artists.relation(tmp_path / "none", [H(0)]) is None
