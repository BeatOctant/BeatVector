import json
import zipfile

import numpy as np
import pytest

from octantvector.generation import GenerationError
from octantvector.generation.archive import MapZip
from octantvector.generation.encoders import get
from octantvector.generation.store import Store
from octantvector.generation.swing import NAMES, chart_features, dataset, params, standard_charts, style
from octantvector.generation.swing.chart import parse
from octantvector.generation.swing.simulate import simulate_all

P = params.load()


def v3(seq, c=1, step=1.0, bombs=(), walls=()):
    """seq: [(x, y, cut)] one note per `step` beats for colour c."""
    return {"version": "3.2.0", "colorNotes": [{"b": i * step, "x": x, "y": y, "c": c, "d": d} for i, (x, y, d) in enumerate(seq)],
            "bombNotes": [{"b": b, "x": x, "y": y} for b, x, y in bombs], "obstacles": list(walls)}


def sim(dat, bpm=120, njs=16):
    return simulate_all(parse(dat, bpm, njs), P)[:2]


@pytest.mark.parametrize("seq,resets", [
    ([(2, 0, 1), (2, 1, 0)] * 8, 0),       # down/up flow alternates forehand/backhand
    ([(2, 1, 1)] * 16, 15),                # repeated downs need a reset each time
    ([(2, 1, 1), (2, 1, 6)] * 8, 15),      # down, down-left: same parity for the right hand
    ([(2, 1, 8)] * 16, 0),                 # dots continue the flow
    ([(2, 0, 8), (2, 2, 8)] * 8, 0),
])
def test_parity_resets(seq, resets):
    both, _ = sim(v3(seq))
    assert sum(s.reset for s in both) == resets
    assert all(abs(s.roll) <= 180 for s in both)


def test_bomb_reset_only_when_bombs_sit_where_the_saber_rests():
    downs = [(2, 1, 1)] * 16
    both, _ = sim(v3(downs, bombs=[(i + 0.5, 2, 0) for i in range(16)]))
    assert sum(s.bomb_reset for s in both) == 15
    both, _ = sim(v3([(2, 0, 1), (2, 1, 0)] * 8, bombs=[(i + 0.5, 0, 2) for i in range(16)]))
    assert sum(s.reset for s in both) == 0  # decorative bombs elsewhere change nothing


def test_dots_follow_the_flow_and_wrist_roll_sign():
    both, _ = sim(v3([(2, 0, 8), (2, 2, 8)] * 4))
    assert [round(s.dir) for s in both[:4]] == [270, 90, 270, 90]
    right_down = sim(v3([(2, 1, 1)] * 2))[0][0]
    left_down = sim(v3([(1, 1, 1)] * 2, c=0))[0][0]
    assert right_down.roll == pytest.approx(-left_down.roll)  # mirrored hands roll the opposite way


def test_swing_kinds():
    stack = {"version": "3.2.0", "colorNotes": [{"b": 0, "x": 2, "y": 0, "c": 1, "d": 1}, {"b": 0, "x": 2, "y": 1, "c": 1, "d": 1},
                                                 {"b": 1, "x": 0, "y": 0, "c": 1, "d": 0}, {"b": 1, "x": 3, "y": 2, "c": 1, "d": 0},
                                                 {"b": 2, "x": 2, "y": 2, "c": 1, "d": 1}, {"b": 2.0625, "x": 2, "y": 1, "c": 1, "d": 8},
                                                 {"b": 2.125, "x": 2, "y": 0, "c": 1, "d": 8}],
             "burstSliders": [{"b": 3, "x": 2, "y": 0, "c": 1, "d": 0, "tb": 3.25, "tx": 2, "ty": 2}]}
    stack["colorNotes"].append({"b": 3, "x": 2, "y": 0, "c": 1, "d": 0})
    both, _ = sim(stack)
    assert [s.kind for s in both] == ["stack", "window", "slider", "chain"]
    assert both[0].notes[0].y == 1  # a down swing meets the upper note of the stack first


def test_faster_is_harder_and_formats_agree():
    flow = [(2, 0, 1), (2, 1, 0)] * 32
    passes = [sim(v3(flow, step=0.5), bpm=b)[1]["pass"] for b in (100, 200, 300)]
    assert passes[0] < passes[1] < passes[2]
    v2 = {"_version": "2.2.0", "_notes": [{"_time": n["b"], "_lineIndex": n["x"], "_lineLayer": n["y"], "_type": n["c"],
                                           "_cutDirection": n["d"]} for n in v3(flow)["colorNotes"]], "_obstacles": []}
    a, b = chart_features(parse(v3(flow), 120, 16), P)[0], chart_features(parse(v2, 120, 16), P)[0]
    assert a.shape == (len(NAMES),) and np.allclose(a, b)


def test_bpm_changes_and_mapping_extensions():
    dat = v3([(2, 0, 1), (2, 1, 0)] * 4)
    dat["bpmEvents"] = [{"b": 4, "m": 240}]
    ch = parse(dat, 120, 16)
    assert ch.notes[4].t == pytest.approx(2.0) and ch.notes[5].t == pytest.approx(2.25)
    assert parse({"_notes": [], "_BPMChanges": 0}, 120, 16).notes == []  # malformed, seen in the archive
    v2 = {"_notes": [{"_time": i, "_lineIndex": 2, "_lineLayer": 0, "_type": 1, "_cutDirection": 1} for i in range(4)],
          "_customData": {"_BPMChanges": [{"_time": 0, "_BPM": 240}]}, "_BPMChanges": [{"_time": 2, "_BPM": 480}]}
    assert [n.t for n in parse(v2, 120, 16).notes] == [0.0, 0.5, 1.0, 1.5]  # editor-only data: Info BPM rules
    nd = {"colorNotes": [{"b": 0, "x": 0, "y": 0, "c": 1, "d": 1, "customData": {"coordinates": [-0.5, 1.5]}}],
          "bombNotes": [{"b": 1, "x": 0, "y": 0, "customData": {"coordinates": [1, 0]}}]}
    ch = parse(nd, 120, 16)
    assert (ch.notes[0].x, ch.notes[0].y) == (1.5, 1.5) and ch.bombs[0][1:3] == (3.0, 0.0)  # Noodle placement
    me = parse({"colorNotes": [{"b": 0, "x": 2500, "y": 1000, "c": 1, "d": 1045}]}, 120, 16).notes[0]
    assert (me.x, me.y, me.ang) == (pytest.approx(1.5), pytest.approx(0.0), 315.0)


def test_params_from_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "octantvector.toml").write_text('[swing]\nreset_cost = 5.0\n')
    assert params.load()["reset_cost"] == 5.0
    many = parse(v3([(2, 1, 1), (2, 1, 6)] * 8), 120, 16)
    assert sum(s.reset for s in simulate_all(many, params.load())[0]) < 15  # resets became expensive
    (tmp_path / "octantvector.toml").write_text('[swing]\nno_such_knob = 1\n')
    with pytest.raises(GenerationError):
        params.load()


def _zip(tmp_path, h, diffs):
    info = {"_beatsPerMinute": 120, "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
        {"_difficulty": d, "_beatmapFilename": f"{d}.dat", "_noteJumpMovementSpeed": 16} for d in diffs]}]}
    (tmp_path / "maps" / h[:2]).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_path / "maps" / h[:2] / f"{h}.zip", "w") as z:
        z.writestr("Info.dat", json.dumps(info))
        for i, (d, dat) in enumerate(diffs.items()):
            z.writestr(f"{d}.dat", json.dumps(dat))


def test_encoders_store_and_axes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    hashes = []
    for k in range(12):
        h = f"{k:02x}" + "0" * 38
        seq = [(int(rng.integers(0, 4)), int(rng.integers(0, 3)), int(rng.integers(0, 9))) for _ in range(40)]
        _zip(tmp_path, h, {"Hard": v3(seq[:20] * 2, step=1.0), "Expert": v3(seq, step=0.5 / (1 + k % 3))})
        hashes.append(h)
    enc = get("swing_flow")
    assert enc.row_kind() == "charts" and get("swing_style").row_kind() == "charts"  # derived views follow the source
    st = Store(enc, tmp_path / "store")
    for h in hashes:
        with MapZip(tmp_path, h) as mz:
            assert [d for d, _ in standard_charts(mz)] == ["Hard", "Expert"]
            out = enc.embed_zip(mz)
            assert sorted(out) == [f"{h}:Standard:Expert", f"{h}:Standard:Hard"]  # every Standard chart
            st.add_many(out)
    st.flush()
    with pytest.raises(GenerationError):  # no style model configured yet
        get("swing_style").derive(tmp_path / "store")
    keys, X = Store(enc, tmp_path / "store").load()
    style.save(style.fit(X, X, [["tech"] if i % 2 else [] for i in range(24)], lam=1.0), tmp_path / "m.json")
    (tmp_path / "octantvector.toml").write_text('swing_model = "m.json"\n')
    assert get("swing_style").derive(tmp_path / "store") == 24
    tag_keys, A = Store(get("swing_style"), tmp_path / "store").load()
    assert sorted(tag_keys) == sorted(keys)  # one row per chart, like the source
    n_haz = len(style.load_rules("hazards"))
    assert A.shape == (24, len(style.LEARNED_TAGS) + len(style.FINE_TAGS) - n_haz) and A.min() >= 0 and A.max() <= 1
    assert get("swing_hazards").derive(tmp_path / "store") == 24
    _, Hz = Store(get("swing_hazards"), tmp_path / "store").load()
    assert Hz.shape == (24, n_haz) and {"art_walls", "dodge_bombs"} <= set(style.load_rules("hazards"))
    assert "bomb_resets" in style.load_rules("style")


def test_fine_tag_rules():
    for conds in style.FINE_TAGS.values():
        assert all(f in NAMES and op in style.OPS for f, op, _ in conds)
    rng = np.random.default_rng(3)
    X = np.abs(rng.normal(size=(500, len(NAMES))))
    m = style.fit(X, X[:100], [[] for _ in range(100)])
    col = NAMES.index("ebpm_p90")
    X[0, col], X[1, col] = 99.0, 0.0
    S, on = style.fine_tags(m, X[:2])
    j = list(style.FINE_TAGS).index("speed")
    assert on[0, j] and not on[1, j] and S[0, j] > S[1, j]
    learned, fine = style.describe(m, X[0])
    assert "speed" in fine and learned == {}


def test_fine_tags_from_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "octantvector.toml").write_text('[swing_tags]\nfast_dots = [["dot_frac", ">=", 0.5], ["ebpm_p90", "q>=", 0.5]]\n')
    assert style.load_rules()["fast_dots"] == [["dot_frac", ">=", 0.5], ["ebpm_p90", "q>=", 0.5]]
    (tmp_path / "octantvector.toml").write_text('[swing_tags]\nbad = [["no_such_feature", ">=", 1]]\n')
    with pytest.raises(GenerationError):
        style.load_rules()


def test_stats_helpers():
    assert dataset.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert dataset.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert dataset.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4)); y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(float)
    folds = dataset.group_folds([i % 37 for i in range(300)], 5)
    assert all(len(set(np.array([i % 37 for i in range(300)])[folds == f])) > 0 for f in range(5))
    assert dataset.auc(dataset.cv_predict(dataset.logistic, X, y, folds, lam=1.0), y) > 0.9
    yr = 2 * X[:, 1] + 0.1 * rng.normal(size=300)
    assert dataset.spearman(dataset.cv_predict(dataset.ridge, X, yr, folds, lam=1.0), yr) > 0.95


def test_evaluate_on_synthetic_rows():
    rng = np.random.default_rng(2)
    rows = []
    for m in range(200):
        tags = ["speed"] if m % 3 == 0 else ["accuracy"]
        for d, diff in enumerate(["Hard", "Expert", "ExpertPlus"]):
            f = rng.normal(size=len(NAMES)) + 1
            f[NAMES.index("pass")] = 2 + d + (2 if "speed" in tags else 0) + 0.1 * rng.normal()
            f[NAMES.index("ebpm_p90")] = (300 if "speed" in tags else 150) + 5 * rng.normal()
            f[NAMES.index("bomb_reset_frac")] = 0.01 * (m % 7)
            f[NAMES.index("bomb_ps")] = 0.1 * (m % 7)
            rows.append({"hash": f"h{m}", "mapper": f"u{m % 40}", "diff": diff, "tags": tags, "f": list(f), "swings": 100,
                         "stars": f[NAMES.index("pass")] * 2 if m % 2 else 0, "bl_stars": 0, "bs_resets": m % 7,
                         "bs_errors": 0, "bs_warns": 0, "kw": ["acc"] if "accuracy" in tags else []})
    res = dataset.evaluate(rows)
    assert res["parity"]["spearman"]["bomb_reset_vs_resets"] > 0.99
    assert "ebpm_p90" in res["keyword_features"]["acc"]["top"]  # acc keyword ~ the slow (accuracy) maps
    assert res["difficulty"]["stars"]["spearman"]["pass"] > 0.95
    assert res["order"]["pass_rises_pct"] > 95
    assert res["style"]["speed"]["auc_logistic_cv"] > 0.9
    assert res["fine_tags"]["speed"]["lift"]["speed"] > 2  # the fine speed tag picks out speed-tagged maps
    model = dataset.train(rows)
    p = style.learned_probs(model, np.array([r["f"] for r in rows]))[:, style.LEARNED_TAGS.index("speed")]
    y = np.array(["speed" in r["tags"] for r in rows])
    assert p[y].mean() > 0.5 > 0.1 > p[~y].mean()


def test_fine_scores_handle_zeros_and_absolute_rules():
    rng = np.random.default_rng(4)
    X = np.abs(rng.normal(size=(1000, len(NAMES))))
    br, ch = NAMES.index("bomb_reset_frac"), NAMES.index("chain_frac")
    X[:700, br] = 0.0  # most maps have no bomb resets
    X[:, ch] = np.linspace(0, 0.04, 1000)
    m = style.fit(X, X[:10], [[] for _ in range(10)])
    S, on = style.fine_tags(m, X)
    names = list(style.FINE_TAGS)
    assert S[:700, names.index("bomb_resets")].max() < 0.5  # zero sits mid-run, not at the top
    c = S[:, names.index("chains")]
    assert c.min() == 0 and c.max() > 0.99 and np.all(np.diff(c) >= 0)  # absolute-only tag still grades maps
    assert len(np.unique(c[on[:, names.index("chains")]])) > 50  # ... including the maps past its threshold


def test_row_labels_modded_and_keywords():
    lab = {"tags": ["tech"], "mapper": "1", "key": "k", "upvotes": 1, "year": 2024, "auto": False}
    r = dataset._diff_labels(lab, {"ne": True, "label": "Vibro hell"}, "Lots of bomb resets and streams")
    assert r["tags"] == ["tech", "modded"] and lab["tags"] == ["tech"]
    assert {"vibro", "bomb_resets", "resets", "stream", "bombs"} <= set(r["kw"])


# ---------------------------------------------------------------- walls and bombs (hazards.py)
from octantvector.generation.swing import hazards  # noqa: E402

FLOW = [(2, 0, 1), (2, 1, 0)] * 40  # 40 s at 120 BPM: long enough for wall and bomb features


def wall(b, x, y, w, h, d=1.0):
    return {"b": b, "x": x, "y": y, "w": w, "h": h, "d": d}


def feats(dat, bpm=120):
    v, both, _ = chart_features(parse(dat, bpm, 16), P)
    return dict(zip(NAMES, v)), both


@pytest.mark.parametrize("w,role", [
    (wall(0, 0, 0, 1, 5), "side"),         # outer lane: in the play space, never reaches the head
    (wall(0, -4, 0, 1, 5), "art"),         # outside the grid
    (wall(0, 1, 0, 1, 5, d=0), "art"),     # zero length
    (wall(0, 1, 0, 1, 5), "dodge"),        # inner lane: lean aside
    (wall(0, 0, 2, 4, 3), "crouch"),       # overhead across the grid
    (wall(0, -2, 0, 8, 5), "unavoidable"),  # covers everything within reach
    (wall(0, 1, 0, 2, 1), "side"),         # low wall below the head
    (wall(0, 1, 4, 2, 1), "art"),          # high above the grid
])
def test_wall_roles(w, role):
    ch = parse(v3(FLOW, walls=[w]), 120, 16)
    assert hazards.classify_walls(ch.walls, P) == [role]


def test_head_moves_for_body_walls_only():
    art = [wall(i, -3, 0, 1, 5, 0.5) for i in range(16)]
    sides = [wall(i, 1 if i % 2 else 2, 0, 1, 5, 0.5) for i in range(16)]
    duck = [wall(i * 2, 0, 2, 4, 3, 1.0) for i in range(8)]
    fa, _ = feats(v3(FLOW, walls=art))
    fs, _ = feats(v3(FLOW, walls=sides))
    fd, _ = feats(v3(FLOW, walls=duck))
    assert fa["art_wall_frac"] == 1 and fa["head_move_mps"] == 0 and fa["dodge_wall_ps"] == 0
    fside, _ = feats(v3(FLOW, walls=[wall(i, 0, 0, 1, 5, 0.5) for i in range(16)]))
    assert fside["side_wall_ps"] > 0 and fside["art_wall_frac"] == 0 and fside["head_move_mps"] == 0
    assert fs["dodge_wall_ps"] > 0 and fs["head_move_mps"] > 0 and fs["dodge_switch_ps"] > 0
    assert fd["crouch_wall_ps"] > 0 and fd["crouch_time_frac"] > 0.1 and fd["dodge_time_frac"] == 0


def test_wall_rhythm_and_buff():
    steady = [wall(i, 1 if i % 2 else 2, 0, 1, 5, 0.25) for i in range(16)]
    ragged = [wall(b, 1, 0, 1, 5, 0.25) for b in (0, 1.13, 1.9, 4.37, 5.1, 7.7, 8.05, 11.6)]
    fs, _ = feats(v3(FLOW, walls=steady))
    fr, _ = feats(v3(FLOW, walls=ragged))
    assert fs["wall_rhythm_frac"] == 1 and fs["wall_sync_frac"] == 1
    assert fr["wall_rhythm_frac"] < 0.5
    plain, _ = feats(v3(FLOW))
    assert fs["pass"] > plain["pass"]  # swinging while dodging is harder (BeatLeader wall buff)


def test_vision_walls():
    f, _ = feats(v3(FLOW, walls=[wall(0, 2, 0, 1, 3, 8)]))
    assert 0.08 < f["vision_wall_frac"] < 0.12  # 8 beats of the 80-note flow sit inside the wall


def test_bomb_roles():
    downs = [(2, 1, 1)] * 16
    ch = parse(v3(downs, bombs=[(i + 0.5, 2, 0) for i in range(15)]), 120, 16)
    _, _, hz = simulate_all(ch, P)
    assert hz["bomb_roles"] == ["reset"] * 15
    far = parse(v3(FLOW, bombs=[(i, -3, 0) for i in range(16)]), 120, 16)
    assert set(simulate_all(far, P)[2]["bomb_roles"]) == {"art"}
    # bombs one lane beside every note, as it is hit
    near = parse(v3(FLOW, bombs=[(i, 3, i % 2) for i in range(32)]), 120, 16)
    roles = simulate_all(near, P)[2]["bomb_roles"]
    assert roles.count("dodge") > len(roles) / 2


def test_bomb_rhythm_and_clusters():
    grid_bombs = [(i, x, 2) for i in range(16) for x in range(4)]
    f, _ = feats(v3(FLOW, bombs=grid_bombs))
    assert f["bomb_rhythm_frac"] == 1 and f["bomb_cluster_frac"] == 1 and f["bomb_ps"] > 0


def test_v2_wall_types_and_mapping_extensions_sizes():
    dat = {"_version": "2.0.0", "_notes": [{"_time": i, "_lineIndex": 2, "_lineLayer": 0, "_type": 1, "_cutDirection": i % 2}
                                           for i in range(32)],
           "_obstacles": [{"_time": 0, "_lineIndex": 0, "_type": 1, "_duration": 1, "_width": 4},
                          {"_time": 2, "_lineIndex": 1, "_type": 0, "_duration": 1, "_width": 1},
                          {"_time": 4, "_lineIndex": 1000, "_type": 1200, "_duration": 1, "_width": 1500}]}
    ch = parse(dat, 120, 16)
    assert [w[2:6] for w in ch.walls] == [(0, 2, 4, 3), (1, 0, 1, 5), (0.0, 0, 0.5, 1.0)]
    assert hazards.classify_walls(ch.walls, P) == ["crouch", "dodge", "side"]


def test_fake_and_crowded_objects_are_art():
    fake_wall = {**wall(0, 1, 0, 1, 5), "customData": {"fake": True}}
    dat = v3([(2, 1, 1)] * 16, bombs=[(i + 0.5, 2, 0) for i in range(15)], walls=[fake_wall])
    dat["customData"] = {"fakeObstacles": [wall(2, 1, 0, 1, 5)]}
    for b in dat["bombNotes"]:
        b["customData"] = {"fake": True}
    ch = parse(dat, 120, 16)
    both, _, hz = simulate_all(ch, P)
    assert hz["wall_roles"] == ["art", "art"] and set(hz["bomb_roles"]) == {"art"}
    assert sum(s.bomb_reset for s in both) == 0  # fake bombs force nothing
    v2 = {"_version": "2.0.0", "_notes": [{"_time": i, "_lineIndex": 2, "_lineLayer": 0, "_type": 1, "_cutDirection": i % 2}
                                          for i in range(32)] + [{"_time": 0.5, "_lineIndex": 2, "_lineLayer": 1, "_type": 1,
                                                                   "_cutDirection": 1, "_customData": {"_fake": True}}],
          "_obstacles": [{"_time": 0, "_lineIndex": 1, "_type": 0, "_duration": 1, "_width": 1,
                          "_customData": {"_interactable": False}}]}
    ch = parse(v2, 120, 16)
    assert len(ch.notes) == 32 and hazards.classify_walls(ch.walls, P) == ["art"]
    # a wall drawing: dozens of thin walls a beat through the play space
    drawing = [wall(i + k / 40, 1 + (k % 8) / 4, k % 3, 0.25, 0.25, 0.1) for i in range(4) for k in range(40)]
    f, _ = feats(v3(FLOW, walls=drawing))
    assert f["art_wall_frac"] == 1 and f["dodge_wall_ps"] == 0 and f["head_move_mps"] == 0


def test_rhythm_walls_need_walls_in_play():
    beat_art = [wall(i, -3, 0, 1, 5, 0.25) for i in range(16)]  # scenery walls on every beat
    f, _ = feats(v3(FLOW, walls=beat_art))
    assert f["wall_rhythm_frac"] == 0 and f["art_wall_rhythm_frac"] == 1


def test_bomb_drawings_force_nothing_and_few_bombs_say_nothing():
    downs = [(2, 1, 1)] * 16
    drawing = [(i + 0.5, x / 4, y / 4) for i in range(15) for x in range(16) for y in range(4)]  # 64 bombs at once
    both, _, hz = simulate_all(parse(v3(downs, bombs=drawing), 120, 16), P)
    assert sum(s.bomb_reset for s in both) == 0 and set(hz["bomb_roles"]) == {"art"}
    f, _ = feats(v3(FLOW, bombs=[(i, 3, i % 2) for i in range(5)]))
    assert f["bomb_ps"] == 0 and f["bomb_dodge_ps"] == 0 and f["bomb_rhythm_frac"] == 0


def test_short_charts_and_impossible_swing_gaps():
    f, _ = feats(v3(FLOW[:20], walls=[wall(i, 1, 0, 1, 5, 0.5) for i in range(8)]))
    assert f["dodge_wall_ps"] == 0 and f["wall_ps"] == 0  # 10 s: too short to say anything about walls
    jank = v3([(2, 0, 1)] * 16)
    jank["colorNotes"] += [{"b": i + 0.01, "x": 2, "y": 1, "c": 1, "d": 0} for i in range(16)]  # 5 ms after each
    both, rat = sim(jank)
    assert len(both) == 16 and rat["pass"] < 10  # merged, not 16 extra swings at thousands of EBPM
    curve = v3([])  # down then down-right 1/16 beat (31 ms) later, once a beat: one curved swing each
    curve["colorNotes"] = [{"b": i + db, "x": 2, "y": y, "c": 1, "d": d} for i in range(16) for db, y, d in ((0, 1, 1), (0.0625, 0, 7))]
    assert len(sim(curve)[0]) == 16
    slow = v3([])  # the same at 150 BPM, 1/8 beat (50 ms) apart: still one slider, not a reset
    slow["colorNotes"] = [{"b": i + db, "x": 2, "y": y, "c": 1, "d": d} for i in range(16) for db, y, d in ((0, 1, 1), (0.125, 0, 7))]
    both, _ = sim(slow, bpm=150)
    assert len(both) == 16
    stream = v3([])  # 1/16-beat arrows curving round for a whole beat: several swings, not one
    stream["colorNotes"] = [{"b": i / 16, "x": 2, "y": 1, "c": 1, "d": 1000 + (i * 30) % 360} for i in range(64)]
    assert len(sim(stream)[0]) >= 8
    vibro = v3([(2, 1, 8)] * 32, step=0.125)  # dots 1/8 beat (75 ms) apart at 100 BPM stay separate swings
    assert len(sim(vibro, bpm=100)[0]) == 32



def test_dot_join_window_and_crouch_tunnels():
    # arrow then dot 1/8 beat later: one swing at 58 ms (130 BPM), two at 67 ms (112 BPM) (BeatLeader replays)
    pair = v3([])
    pair["colorNotes"] = [{"b": i + db, "x": 2, "y": y, "c": 1, "d": d} for i in range(16) for db, y, d in ((0, 1, 1), (0.125, 0, 8))]
    assert len(sim(pair, bpm=130)[0]) == 16 and len(sim(pair, bpm=112)[0]) == 32
    # a crouch tunnel of 0.02 s pieces is one overhead wall, and players crouch a little before and after it
    tunnel = [wall(4 + k * 0.04, 0, 2, 4, 3, 0.04) for k in range(100)]  # 4 beats of 1/50 s pieces at 120 BPM
    ch = parse(v3(FLOW, walls=tunnel), 120, 16)
    hz = simulate_all(ch, P)[2]
    assert hz["wall_roles"] == ["crouch"] and hz["walls"][0][1] == pytest.approx(2.0)
    f, _ = feats(v3(FLOW, walls=tunnel))
    span = ch.notes[-1].t - ch.notes[0].t
    assert f["crouch_time_frac"] * span == pytest.approx(2.0 + 2 * P["head_crouch_margin_s"], abs=0.1)

# Definitions of the stored swing views. When a change to the features, the [swing] defaults or the default tag
# rules makes this fail, bump the encoder's version (encoders/swing.py) and update the pair here; bump the
# derived views too when swing_flow changes, since they are computed from it.
VIEW_DEFINITIONS = {"swing_flow": ("10", "44f88f3658578e53"), "swing_style": ("11", "d2502ed44a830ec2"),
                    "swing_hazards": ("8", "26cd5cc7c6d0a052")}


def test_view_versions_follow_their_definitions(monkeypatch, tmp_path):
    import hashlib
    monkeypatch.chdir(tmp_path)  # no octantvector.toml: defaults only
    fp = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:16]  # noqa: E731
    got = {"swing_flow": (get("swing_flow").version, fp(get("swing_flow").params())),
           "swing_style": (get("swing_style").version, fp(style.load_rules("style"))),
           "swing_hazards": (get("swing_hazards").version, fp(style.load_rules("hazards")))}
    assert got == VIEW_DEFINITIONS
