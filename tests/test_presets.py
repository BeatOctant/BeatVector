"""Presets (手札): listed for the DB, expanded into contract 1 (OctantCompass/OctantHarvest to OctantVector) conditions (OctantSpecs plans/presets.md §8)."""
import pytest

from octantvector.common.hands import check_hands
from octantvector.runtime import RequestError, UnknownPresetError, expand_preset, presets
from test_recommend import BARE, H, db  # noqa: F401  (fixture)


HANDS = [
    {"id": "song", "label": "似た曲", "inputs": [{"kind": "seed"}], "handling": "near",
     "knobs": {"weights": {"music_x": 1}}},
    {"id": "genre", "label": "ジャンル", "inputs": [{"kind": "choice", "name": "genre", "label": "ジャンル"}],
     "handling": "near", "knobs": {}},
    {"id": "chart_genre", "label": "曲調も近いマップ", "inputs": [{"kind": "seed"}], "handling": "narrow_then_rank",
     "knobs": {"narrow": {"music_x": 1}, "rank": {"chart_x": 1}}},
    {"id": "again", "label": "久しぶり", "inputs": [], "handling": "near", "knobs": {"params": {"includeKnown": True}}},
]


@pytest.fixture
def hands(db):
    db.manifest["presets"] = HANDS
    yield db
    del db.manifest["presets"]


def test_the_default_is_only_all_since_other_hands_name_views(db):
    from octantvector.runtime.presets import DEFAULT_PRESETS
    assert [h["id"] for h in DEFAULT_PRESETS] == ["all"] and DEFAULT_PRESETS[0]["knobs"] == {}
    listing = presets(db)
    assert [p["id"] for p in listing["presets"]] == ["all"]
    assert all(set(p) <= {"id", "group", "label", "hint", "inputs"} for p in listing["presets"])
    assert presets(db)["revision"] == listing["revision"]


def test_a_genre_choice_without_options_takes_the_db_words(hands):
    listing = presets(hands)["presets"]
    assert [p["id"] for p in listing] == ["song", "genre", "chart_genre", "again"]
    genre = next(p for p in listing if p["id"] == "genre")
    assert [o["value"] for o in genre["inputs"][0]["options"]] == ["jpop", "rock"]


def test_hands_from_the_manifest_replace_the_defaults_and_what_they_need_is_read_from_the_knobs(db):
    db.manifest["presets"] = [
        {"id": "loud", "label": "大きい", "inputs": [], "handling": "near", "knobs": {"weights": {"music_x": 1}}},
        {"id": "future", "label": "未来", "inputs": [], "handling": "narrow-then-rank", "knobs": {}},
        {"id": "nometa", "label": "タグ", "inputs": [], "handling": "near", "knobs": {"weights": {"meta_x": 1}}},
        {"id": "role", "label": "音", "inputs": [], "handling": "near", "knobs": {"weights": {"music": 1}}},
        {"id": "oddknob", "label": "謎", "inputs": [], "handling": "near", "knobs": {"stages": 2}},
        {"id": "oddparam", "label": "謎", "inputs": [], "handling": "near", "knobs": {"params": {"warp": 1}}},
        {"id": "enka", "label": "演歌", "inputs": [], "handling": "near", "knobs": {"genre": "enka"}},
        {"id": "aim", "label": "狙い", "handling": "near", "knobs": {},
         "inputs": [{"kind": "choice", "name": "difficulty", "label": "狙い",
                     "options": [{"value": "warmup", "label": "軽め"}, {"value": "insane", "label": "無茶"}]}]},
        {"id": "slider", "label": "つまみ", "inputs": [{"kind": "slider"}], "handling": "near", "knobs": {}},
        {"id": "hint", "label": "任意", "inputs": [{"kind": "slider", "optional": True}], "handling": "near",
         "knobs": {}},
        {"label": "名前なし", "inputs": [], "handling": "near", "knobs": {}},
        {"id": "loud", "label": "二つ目", "inputs": [], "handling": "near", "knobs": {}},
    ]
    try:
        listing = presets(db)["presets"]
        assert [p["id"] for p in listing] == ["loud", "aim", "hint"]
        aim = next(p for p in listing if p["id"] == "aim")
        assert [o["value"] for o in aim["inputs"][0]["options"]] == ["warmup"]  # an option the runtime cannot serve
        assert next(p for p in listing if p["id"] == "hint")["inputs"] == []
        assert expand_preset(db, {"preset": "aim", "choices": {"difficulty": "warmup"}})["difficulty"] == "warmup"
    finally:
        del db.manifest["presets"]


def test_check_hands_names_why_each_hand_is_left_out():
    views = {"music_x@1": "music", "chart_x@1": "chart"}
    bad = check_hands([
        {"id": "ok", "label": "良い", "inputs": [], "handling": "narrow_then_rank",
         "knobs": {"narrow": {"music_x": 1}, "rank": {"chart_x@1": 2}}},
        {"id": "half", "label": "半分", "inputs": [], "handling": "narrow_then_rank", "knobs": {"narrow": {"music_x": 1}}},
        {"id": "tags", "label": "タグ", "inputs": [], "handling": "near", "knobs": {"weights": {"meta_x": 1}}},
        {"id": "ok", "label": "同じ id", "inputs": [], "handling": "near", "knobs": {}},
        {"id": "yours", "label": "好み", "inputs": [], "handling": "near", "knobs": {"genreFrom": ["jpop", "rock"]}},
        {"id": "enka", "label": "演歌", "inputs": [], "handling": "near", "knobs": {"genreFrom": ["jpop", "enka"]}},
        {"id": "empty", "label": "空", "inputs": [], "handling": "near", "knobs": {"genreFrom": []}},
    ], views, {"jpop", "rock"})
    assert set(bad) == {"half", "tags", "ok", "enka", "empty"}
    assert "rank" in bad["half"] and "meta_x" in bad["tags"] and "earlier" in bad["ok"]


def test_narrow_then_rank_expands_into_narrow_and_weights(hands):
    seed = {"hash": H(1), "characteristic": "Standard", "difficulty": "Hard"}
    assert expand_preset(hands, {"preset": "chart_genre", "seed": seed}) == {
        "seed": seed, "narrow": {"music_x": 1}, "weights": {"chart_x": 1}, "difficulty": "auto"}


def test_expand_turns_a_hand_into_conditions(hands):
    seed = {"hash": H(1), "characteristic": "Standard", "difficulty": "Hard"}
    assert expand_preset(hands, {"preset": "song", "seed": seed, "count": 5}) == {
        "seed": seed, "count": 5, "weights": {"music_x": 1}, "difficulty": "auto"}
    assert expand_preset(hands, {"preset": "genre", "choices": {"genre": "rock"}})["genre"] == "rock"
    assert expand_preset(hands, {"preset": "again"})["params"] == {"includeKnown": True}
    hands.manifest["presets"] = HANDS + [{"id": "yours", "label": "好み", "inputs": [], "handling": "near",
                                          "knobs": {"genreFrom": ["jpop"], "weights": {"music_x": 1}}}]
    assert expand_preset(hands, {"preset": "yours"})["genreFrom"] == ["jpop"]
    hands.manifest["presets"] = HANDS
    assert expand_preset(hands, {"preset": "song", "seed": seed, "weights": {"chart_x": 1}})["weights"] == {"chart_x": 1}
    with pytest.raises(UnknownPresetError):
        expand_preset(hands, {"preset": "pumpup"})
    with pytest.raises(RequestError):
        expand_preset(hands, {"preset": "song"})
    with pytest.raises(RequestError):
        expand_preset(hands, {"preset": "genre", "choices": {"genre": "enka"}})


def test_a_request_without_a_preset_is_returned_as_it_is(db):
    assert expand_preset(db, {"count": 3}) == {"count": 3}
    request = {"count": 3, "choices": {"genre": "rock"}}
    assert expand_preset(db, request) == request  # choices kept (contract 1, 2026-10-14)
    assert expand_preset(db, None) == {}


def test_a_weight_names_a_view_id_or_a_view_key_never_a_role():
    from octantvector.common.hands import view_keys
    views = {"clap@1": "music", "mert@2": "music", "chart_x@1": "chart"}
    assert view_keys("mert", views) == ["mert@2"]  # any version of the id, so a new version keeps the hand
    assert view_keys("mert@2", views) == ["mert@2"]
    assert view_keys("mert@1", views) == [] and view_keys("meta", views) == []
    assert view_keys("music", views) == []  # a role is a label only
    bad = check_hands([
        {"id": "mix", "label": "音 7:3", "inputs": [{"kind": "seed"}], "handling": "near",
         "knobs": {"weights": {"mert": 0.7, "clap": 0.3}}},
        {"id": "solo", "label": "MERT だけ", "inputs": [{"kind": "seed"}], "handling": "near",
         "knobs": {"weights": {"mert": 1}}},
        {"id": "gone", "label": "ない", "inputs": [{"kind": "seed"}], "handling": "near",
         "knobs": {"weights": {"mert_v1_330m": 1}}},
        {"id": "role", "label": "音", "inputs": [{"kind": "seed"}], "handling": "near",
         "knobs": {"weights": {"music": 1}}},
        {"id": "narrowrole", "label": "音で絞る", "inputs": [{"kind": "seed"}], "handling": "narrow_then_rank",
         "knobs": {"narrow": {"music": 1}, "rank": {"chart_x": 1}}},
        {"id": "rankrole", "label": "譜面で並べる", "inputs": [{"kind": "seed"}], "handling": "narrow_then_rank",
         "knobs": {"narrow": {"clap": 1}, "rank": {"chart": 1}}},
    ], views, set())
    assert set(bad) == {"gone", "role", "narrowrole", "rankrole"}
    assert "'music'" in bad["role"] and "'chart'" in bad["rankrole"]


def test_a_shown_text_may_be_one_string_or_one_per_language(db):
    db.manifest["presets"] = [
        {"id": "both", "group": {"en": "Song", "ja": "曲"}, "label": {"en": "Similar songs", "ja": "似た曲"},
         "hint": {"en": "Sounds alike"}, "handling": "near", "knobs": {},
         "inputs": [{"kind": "choice", "name": "genre", "label": {"en": "Genre", "ja": "ジャンル"},
                     "options": [{"value": "rock", "label": {"en": "Rock", "ja": "ロック"}}]}]},
        {"id": "plain", "label": "似た曲", "inputs": [], "handling": "near", "knobs": {}},
        {"id": "emptymap", "label": {}, "inputs": [], "handling": "near", "knobs": {}},
        {"id": "blank", "label": {"en": ""}, "inputs": [], "handling": "near", "knobs": {}},
        {"id": "badgroup", "group": {"en": 1}, "label": "見出しが壊れている", "inputs": [], "handling": "near",
         "knobs": {}},
    ]
    listing = presets(db)["presets"]
    assert [p["id"] for p in listing] == ["both", "plain", "badgroup"]
    both = listing[0]
    assert both["label"] == {"en": "Similar songs", "ja": "似た曲"} and both["group"] == {"en": "Song", "ja": "曲"}
    assert both["inputs"][0]["options"] == [{"value": "rock", "label": {"en": "Rock", "ja": "ロック"}}]
    assert "group" not in listing[2]
    assert set(check_hands(db.manifest["presets"], *_facts(db))) == {"emptymap", "blank"}


def test_the_default_hands_carry_english_and_japanese(db):
    for p in presets(db)["presets"]:
        for key in ("label", "group", "hint"):
            assert set(p[key]) == {"en", "ja"}, (p["id"], key)


def _facts(db):
    return {k: db.manifest["views"][k].get("role") for k in db.views}, set(db.genres())


def test_a_hand_with_params_values_outside_the_bounds_is_left_out(db):
    """A DB carries hands, so a hand whose params could run the host away is left out of the list, like one naming a
    view the DB lacks (notes/security-audit-2026-10-21.md, 8)."""
    db.manifest["presets"] = HANDS + [
        {"id": "spin", "label": "回る", "inputs": [], "handling": "near", "knobs": {"params": {"toleranceMaxSd": 1e308}}}]
    try:
        listed = [p["id"] for p in presets(db)["presets"]]
        assert "spin" not in listed and "again" in listed
        assert check_hands(db.manifest["presets"], {"music_x@1": "music"}, set())["spin"].startswith("toleranceMaxSd")
    finally:
        del db.manifest["presets"]


MIXED = [
    {"id": "song", "label": "似た曲", "inputs": [{"kind": "seed"}], "handling": "near",
     "knobs": {"weights": {"music_x": 1}}},
    {"id": "chart", "label": "似た譜面", "inputs": [{"kind": "seed"}], "handling": "near",
     "knobs": {"weights": {"chart_x": 1}, "difficulty": "challenge", "params": {"mapsPerSong": 2}}},
    {"id": "yours", "label": "好み", "inputs": [], "handling": "near", "knobs": {}},
    {"id": "genre", "label": "ジャンル", "inputs": [{"kind": "choice", "name": "genre"}], "handling": "near",
     "knobs": {}},
    {"id": "mixed", "label": "おまかせ", "inputs": [{"kind": "seed"}], "handling": "mixed",
     "knobs": {"hands": ["song", "chart"], "params": {"mapsPerSong": 0, "mixPriorDrafts": 5}}},
]


def test_a_mixed_hand_names_hands_of_the_list_and_takes_only_what_they_all_can(db):
    db.manifest["presets"] = MIXED + [
        {"id": "self", "label": "自分", "inputs": [{"kind": "seed"}], "handling": "mixed", "knobs": {"hands": ["self", "song"]}},
        {"id": "gone", "label": "ない", "inputs": [{"kind": "seed"}], "handling": "mixed", "knobs": {"hands": ["song", "pumpup"]}},
        {"id": "deep", "label": "二重", "inputs": [{"kind": "seed"}], "handling": "mixed", "knobs": {"hands": ["song", "mixed"]}},
        {"id": "one", "label": "一つ", "inputs": [{"kind": "seed"}], "handling": "mixed", "knobs": {"hands": ["song"]}},
        {"id": "twice", "label": "二度", "inputs": [{"kind": "seed"}], "handling": "mixed", "knobs": {"hands": ["song", "song"]}},
        {"id": "noseed", "label": "起点なし", "inputs": [], "handling": "mixed", "knobs": {"hands": ["song", "yours"]}},
        {"id": "asks", "label": "選ばせる", "inputs": [{"kind": "seed"}], "handling": "mixed", "knobs": {"hands": ["song", "genre"]}},
        {"id": "free", "label": "起点なしでよい", "inputs": [], "handling": "mixed", "knobs": {"hands": ["yours", "chart"]}},
        {"id": "late", "label": "後ろの手札", "inputs": [{"kind": "seed"}], "handling": "mixed", "knobs": {"hands": ["song", "tail"]}},
        {"id": "tail", "label": "最後", "inputs": [{"kind": "seed"}], "handling": "near", "knobs": {}},
        {"id": "bare", "label": "つまみなし", "inputs": [{"kind": "seed"}], "handling": "mixed", "knobs": {}},
    ]
    try:
        listed = presets(db)["presets"]
        assert [p["id"] for p in listed] == ["song", "chart", "yours", "genre", "mixed", "late", "tail"]
        assert next(p for p in listed if p["id"] == "mixed")["inputs"] == [{"kind": "seed"}]
        bad = check_hands(db.manifest["presets"], *_facts(db))
        assert set(bad) == {"self", "gone", "deep", "one", "twice", "noseed", "asks", "free", "bare"}
        assert "'pumpup'" in bad["gone"] and "mixed itself" in bad["deep"] and "'song'" in bad["noseed"]
        assert "choice" in bad["asks"] and "'chart'" in bad["free"] and "hands" in bad["bare"]
    finally:
        del db.manifest["presets"]


def test_a_mixed_hand_expands_into_one_request_per_hand(db):
    db.manifest["presets"] = MIXED
    try:
        seed = {"hash": H(1), "characteristic": "Standard", "difficulty": "Hard"}
        out = expand_preset(db, {"preset": "mixed", "seed": seed, "count": 6, "params": {"mapsPerSong": 1}})
        assert out["seed"] == seed and out["count"] == 6 and out["difficulty"] == "auto"
        assert out["params"] == {"mapsPerSong": 1, "mixPriorDrafts": 5}  # the caller's value over the hand's
        assert [m["hand"] for m in out["mix"]] == ["song", "chart"]
        song, chart = (m["request"] for m in out["mix"])
        assert song == {"seed": seed, "count": 6, "params": {"mapsPerSong": 1, "mixPriorDrafts": 5},
                        "weights": {"music_x": 1}}  # no difficulty of its own: recommend takes the request's
        assert chart == {"seed": seed, "count": 6, "params": {"mapsPerSong": 1, "mixPriorDrafts": 5},
                         "weights": {"chart_x": 1}, "difficulty": "challenge"}  # its own aim stays
        with pytest.raises(RequestError):
            expand_preset(db, {"preset": "mixed"})
    finally:
        del db.manifest["presets"]


def test_a_chained_hand_is_a_near_hand_with_chain_params(db):
    db.manifest["presets"] = HANDS + [
        {"id": "chain", "label": "連想", "inputs": [{"kind": "seed"}], "handling": "near",
         "knobs": {"params": {"chainEvery": 10}}},
        {"id": "chain_bad", "label": "連想", "inputs": [{"kind": "seed"}], "handling": "near",
         "knobs": {"params": {"chainEvery": 2.5}}}]
    try:
        listed = [p["id"] for p in presets(db)["presets"]]
        assert "chain" in listed and "chain_bad" not in listed
    finally:
        del db.manifest["presets"]


def test_a_hands_genre_labels_are_listed_for_the_words_the_db_has(hands):
    yours = {"id": "yours", "label": "好み", "inputs": [], "handling": "near", "knobs": {"genreFrom": ["jpop", "rock"]},
             "genreLabels": {"jpop": {"en": "J-pop", "ja": "J ポップ"}, "rock": "", "enka": "演歌"}}
    hands.manifest["presets"] = HANDS + [yours]
    listed = next(p for p in presets(hands)["presets"] if p["id"] == "yours")
    assert listed["genreLabels"] == {"jpop": {"en": "J-pop", "ja": "J ポップ"}}  # not a text, not in the DB: out
    hands.manifest["presets"] = HANDS + [{**yours, "genreLabels": "jpop"}]
    assert "genreLabels" not in next(p for p in presets(hands)["presets"] if p["id"] == "yours")


def test_all_is_a_near_knob_of_two_or_more_view_groups_and_stands_in_for_weights(db):
    views = {"music_x@1": "music", "chart_x@1": "chart"}
    both = {"id": "both", "label": "両方", "inputs": [{"kind": "seed"}], "handling": "near",
            "knobs": {"all": [{"music_x": 1}, {"chart_x": 1}]}}
    bad = check_hands([
        both,
        {"id": "one", "label": "一つ", "inputs": [], "handling": "near", "knobs": {"all": [{"music_x": 1}]}},
        {"id": "gone", "label": "無い", "inputs": [], "handling": "near",
         "knobs": {"all": [{"music_x": 1}, {"meta_x": 1}]}},
        {"id": "twice", "label": "二重", "inputs": [], "handling": "near",
         "knobs": {"weights": {"music_x": 1}, "all": [{"music_x": 1}, {"chart_x": 1}]}},
    ], views, set())
    assert set(bad) == {"one", "gone", "twice"} and "meta_x" in bad["gone"] and "one or the other" in bad["twice"]
    db.manifest["presets"] = [both]
    try:
        seed = {"hash": H(1), "characteristic": "Standard", "difficulty": "Hard"}
        assert expand_preset(db, {"preset": "both", "seed": seed})["all"] == both["knobs"]["all"]
        own = expand_preset(db, {"preset": "both", "seed": seed, "weights": {"music_x": 1}})
        assert own["weights"] == {"music_x": 1} and "all" not in own  # the caller's weights stand for the hand's all
    finally:
        del db.manifest["presets"]
