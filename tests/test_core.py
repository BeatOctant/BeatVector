import hashlib
import gzip
import json
import sqlite3
import zipfile

import numpy as np
import pytest

from octantvector.generation import GenerationError, config
from octantvector.runtime import likes
from octantvector.eval import playdata
from octantvector.generation.archive import MapZip, read_json
from octantvector.eval import results
from octantvector.generation.encoders import REGISTRY, Encoder, get, register
from octantvector.generation.encoders.catalogue_features import NpsNjs
from octantvector.generation.encoders.chart_patterns import NAMES, pattern_features
from octantvector.generation.swing.chart import parse
from octantvector.eval import personal
from octantvector.eval.neighbours import hubness
from octantvector.eval.profile import difficulty_band, fit_penalty, spherical_kmeans
from octantvector.generation.search import Index, parse_views
from octantvector.generation.store import Store


class Toy(Encoder):
    id, dim, kind, norm = "toy", 4, "map", "center"


class Toy2(Encoder):
    id, dim, kind, norm = "toy2", 3, "map", "zscore"


@pytest.fixture(autouse=True)
def toy_encoders():
    REGISTRY.update(toy=Toy, toy2=Toy2)
    yield
    REGISTRY.pop("toy"); REGISTRY.pop("toy2")


def test_registry_has_all_encoders():
    for i in ["npsnjs_std", "chart_patterns", "meta_tags", "swing_flow", "swing_style", "swing_hazards"]:
        e = get(i)
        assert e.id == i and e.dim > 0 and e.kind in ("audio", "zip", "map", "derived") and e.norm in ("center", "zscore")
        json.dumps(e.params())  # store metadata must be serialisable
    for i, cls in REGISTRY.items():
        if i.startswith("toy"):
            continue
        e = cls()
        assert e.inputs, i
        assert e.fingerprint() == cls().fingerprint() and e.fingerprint().startswith("sha256:")
        assert all(s in REGISTRY for s in e.sources()), i
    assert get("swing_style").source == "swing_flow"


def test_register_rejects_bad_declarations():
    for bad in ({"inputs": ()}, {"inputs": ("sound",)}, {"inputs": ("meta:",)}, {"inputs": ("view:toy",)},
                {"kind": "derived"}, {"rows": "notes"}, {"kind": "audio", "rows": "charts"}):
        cls = type("Bad", (Encoder,), {"id": "bad", "dim": 1, "inputs": ("chart",), **bad})
        with pytest.raises(ValueError):
            register(cls)
    assert "bad" not in REGISTRY


def test_fingerprint_follows_source_views(tmp_path, monkeypatch):
    class Derived(Encoder):
        id, dim, kind, inputs = "toy_derived", 2, "derived", ("view:toy",)
    REGISTRY["toy_derived"] = Derived
    try:
        before = Derived().fingerprint()
        Store(Derived(), tmp_path)
        assert json.loads((tmp_path / "toy_derived@1" / "encoder.json").read_text())["fingerprint"] == before
        monkeypatch.setattr(Toy, "params", lambda self: {"retrained": True})  # the source view changed
        assert Derived().fingerprint() != before
        with pytest.raises(GenerationError, match="source"):
            Store(Derived(), tmp_path)  # same version, different source: refuse to mix
    finally:
        REGISTRY.pop("toy_derived")


def test_a_store_without_a_fingerprint_is_refused(tmp_path):
    d = tmp_path / "toy@1"
    d.mkdir()
    (d / "encoder.json").write_text(json.dumps(Toy().definition()))  # written before fingerprints existed
    with pytest.raises(GenerationError, match="another fingerprint"):
        Store(Toy(), tmp_path)


def test_store_roundtrip_resume_and_version_guard(tmp_path):
    s = Store(Toy(), tmp_path, flush_every=3)
    vecs = {f"h{i}": np.arange(4, dtype=np.float32) + i for i in range(5)}
    for h, v in vecs.items():
        s.add(h, v)
    s.flush()
    hs, X = Store(Toy(), tmp_path).load()
    assert set(hs) == set(vecs) and len(Store(Toy(), tmp_path).shards()) == 2
    for h, x in zip(hs, X):
        np.testing.assert_array_equal(x, vecs[h])
    assert Store(Toy(), tmp_path).done() == set(vecs)
    with pytest.raises(ValueError):
        s.add("bad", np.zeros(3))

    class Changed(Toy):
        def params(self):
            return {"changed": True}
    with pytest.raises(GenerationError):
        Store(Changed(), tmp_path)


def _index(tmp_path, n=200, seed=0, **kw):
    rng = np.random.default_rng(seed)
    hs = [f"{i:040x}" for i in range(n)]
    vec = {"toy": (hs, rng.normal(size=(n, 4))), "toy2": (hs[: n - 5], rng.normal(size=(n - 5, 3)))}
    return Index({"toy": 1, "toy2": 3}, tmp_path, vectors=vec, device="cpu", **kw), vec


def test_index_weighting_and_intersection(tmp_path):
    ix, _ = _index(tmp_path)
    assert len(ix) == 195  # only hashes present in every view
    np.testing.assert_allclose(np.linalg.norm(ix.emb, axis=1), 1, atol=1e-5)
    a, b = 3, 17
    cos = ix.view_cosines(a, b)
    assert ix.emb[a] @ ix.emb[b] == pytest.approx(0.25 * cos["toy"] + 0.75 * cos["toy2"], abs=1e-5)
    assert parse_views("a:0.5,b") == {"a": 0.5, "b": 1.0}


def test_saved_run_records_views_and_neighbours(tmp_path):
    from octantvector.eval.neighbours import evaluate
    ix, _ = _index(tmp_path)
    meta = {h: {"key": h[-4:], "song": "s", "artist": "a", "mapper": "m", "bpm": 120, "tags": [], "maxnps": 5.0}
            for h in ix.hashes}
    rows = []
    res = evaluate(ix, meta, k=3, n_random=100, rows=rows)
    path = tmp_path / "eval_n.jsonl.gz"
    results.write(path, results.run_info(ix, "eval-neighbours", res, k=3), rows)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        info, *back = [json.loads(line) for line in f]
    info = info["run"]
    assert info["views"]["toy@1"] == {"weight": 0.25, "fingerprint": Toy().fingerprint()}
    assert info["settings"] == {"k": 3} and info["summary"]["knn"]["nk_max"] == res["knn"]["nk_max"]
    assert len(back) == len(ix) and back[0]["query"] == ix.hashes[0] and len(back[0]["results"]) == 3
    assert all(r[0] != b["query"] for b in back for r in b["results"])


def test_neighbours_by_upvotes_and_isolation(tmp_path):
    from octantvector.eval.neighbours import _quantile_groups, evaluate, isolation
    assert _quantile_groups([1, 2, 3, 4, float("nan")], 2).tolist() == [0, 0, 1, 1, -1]
    ix, _ = _index(tmp_path)
    meta = {h: {"key": h[-4:], "song": f"s{n}", "artist": "a" if n % 2 else "b", "mapper": "m", "bpm": 120, "tags": [],
                "maxnps": 5.0, "up": n} for n, h in enumerate(ix.hashes)}
    iso = isolation(ix, 3)
    assert set(iso) == set(ix.hashes)
    del meta[ix.hashes[0]]  # a version the catalogue no longer has, whose vector the store still keeps
    res = evaluate(ix, meta, k=3, n_random=100, isolation_of={h: v for h, v in list(iso.items())[:-7]})
    up, by_iso = res["by_upvotes"], res["by_isolation"]
    assert list(up) == ["q1 (fewest upvotes)", "q2", "q3", "q4 (most upvotes)"]
    assert sum(g["queries"] for g in up.values()) == len(ix) - 1
    assert sum(g["queries"] for g in by_iso.values()) == len(ix) - 7  # queries without an isolation are left out
    q1, q4 = by_iso["q1 (most isolated)"], by_iso["q4 (least isolated)"]
    assert q1["mean_score"] < q4["mean_score"]  # with the index itself as the reference, isolation is the mean score


def test_knn_matches_brute_force_and_excludes_self(tmp_path):
    ix, _ = _index(tmp_path)
    idx, sc = ix.knn(5)
    S = ix.emb @ ix.emb.T
    np.fill_diagonal(S, -np.inf)
    brute = np.argsort(-S, axis=1)[:, :5]
    assert np.mean([set(a) == set(b) for a, b in zip(idx, brute)]) > 0.99
    assert all(i not in row for i, row in enumerate(idx))
    assert np.all(np.diff(sc, axis=1) <= 1e-6)  # sorted descending


def test_csls_reduces_hubness(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    hs = [f"{i:040x}" for i in range(n)]
    X = rng.normal(size=(n, 4))
    X[:20] = X[:20] * 0.05 + rng.normal(size=4)  # a dense clump of potential hubs
    ix = Index({"toy": 1}, tmp_path, vectors={"toy": (hs, X)}, device="cpu")
    plain = hubness(ix.knn(10)[0], n)
    csls = hubness(ix.knn(10, csls=True)[0], n)
    assert csls["nk_max"] <= plain["nk_max"]
    r = ix.local_density(10)
    assert r.shape == (n,) and np.all(np.isfinite(r))


def test_likes_classify_law_4_1():
    c = likes.classify
    assert c({"play": 1, "finished": 1, "favorite": 1}) == "liked"
    assert c({"play": 1, "finished": 1, "upvote": 1}) == "liked"
    assert c({"play": 2, "finished": 1, "failed": 1, "returned": 1}) == "liked"
    assert c({"play": 3, "finished": 3}) is None  # cleared, but not back on a later day
    assert c({"play": 1, "failed": 1, "favorite": 1}) == "goal"
    assert c({"upvote": 1}) == "goal"  # marked, never played
    assert c({"play": 2, "failed": 2}) == "challenging"
    assert c({"play": 9, "failed": 1, "quit": 8}) is None  # quits are not plays
    assert c({"pd_playcount": 9}) is None  # imported_play alone: neither liked nor a goal
    assert c({"pd_playcount": 9, "favorite": 1}) == "liked"  # clear before OctantPath unknown (inferred)
    assert c({}) is None


def test_config_env_overrides_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert config.load().full_every == config.DEFAULTS["full_every"]
    monkeypatch.setenv("BV_FULL_EVERY", "5")
    (tmp_path / "octantvector.toml").write_text('store = "vectors"\nfull_every = 4\n')
    cfg = config.load()
    assert cfg.full_every == 5 and cfg.store.name == "vectors"
    with pytest.raises(GenerationError):
        cfg.require("archive")


V2 = {"_version": "2.2.0", "_notes": [{"_time": i * 0.5, "_lineIndex": i % 4, "_lineLayer": i % 3, "_type": i % 2,
                                        "_cutDirection": [1, 0][i % 2] if i % 3 else 8} for i in range(40)]
      + [{"_time": 3.0, "_lineIndex": 1, "_lineLayer": 0, "_type": 3, "_cutDirection": 0}],
      "_obstacles": [{"_time": 4, "_duration": 2}]}
V3 = {"version": "3.2.0", "colorNotes": [{"b": n["_time"], "x": n["_lineIndex"], "y": n["_lineLayer"], "c": n["_type"],
                                          "d": n["_cutDirection"]} for n in V2["_notes"] if n["_type"] in (0, 1)],
      "bombNotes": [{"b": 3.0, "x": 1, "y": 0}], "obstacles": [{"b": 4, "d": 2}], "sliders": [], "burstSliders": []}


def test_chart_parser_v2_v3_agree():
    """chart_patterns reads charts through swing/chart.py, the one reader of the formats."""
    c2, c3 = parse(V2, 120, 18), parse(V3, 120, 18)
    key = lambda n: (n.t, n.b, n.x, n.y, n.c, n.ang, n.cut)  # noqa: E731
    assert [key(n) for n in c2.notes] == [key(n) for n in c3.notes]
    assert [b[3] for b in c2.bombs] == [b[3] for b in c3.bombs] == [3.0] and len(c2.walls) == len(c3.walls) == 1
    f2, f3 = pattern_features(c2), pattern_features(c3)
    assert f2.shape == (len(NAMES),) and np.allclose(f2, f3)
    nps = f2[NAMES.index("nps")]
    assert nps == pytest.approx(40 / (39 * 0.5 * 60 / 120))
    assert f2[NAMES.index("njs")] == 18
    assert f2[NAMES.index("wall_time_frac")] == pytest.approx(1.0 / (39 * 0.5 * 60 / 120))  # 2 beats = 1 s
    assert pattern_features(parse({"_notes": V2["_notes"][:5]}, 120, 18)) is None  # too few notes


def test_chart_encoder_on_zip_including_utf16(tmp_path):
    info = {"_beatsPerMinute": 120, "_songFilename": "song.egg", "_difficultyBeatmapSets": [
        {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_beatmapFilename": "Expert.dat", "_noteJumpMovementSpeed": 16},
            {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlus.dat", "_noteJumpMovementSpeed": 18}]}]}
    h = "ab" + "0" * 38
    (tmp_path / "maps" / "ab").mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "maps" / "ab" / f"{h}.zip", "w") as z:
        z.writestr("Info.dat", json.dumps(info))
        z.writestr("Expert.dat", json.dumps(V3))
        z.writestr("ExpertPlus.dat", json.dumps(V2).encode("utf-16"))  # BOM + UTF-16, seen in the wild
    with MapZip(tmp_path, h) as mz:
        v = get("chart_patterns").embed_zip(mz)
        assert read_json(mz.zf, "ExpertPlus.dat")["_version"] == "2.2.0"
    with zipfile.ZipFile(tmp_path / "encodings.zip", "w") as z:
        z.writestr("bom8.dat", b"\xef\xbb\xbf" + json.dumps(V2).encode())
        z.writestr("le16.dat", json.dumps(V2).encode("utf-16-le"))  # UTF-16 without a BOM
    with zipfile.ZipFile(tmp_path / "encodings.zip") as z:
        assert read_json(z, "bom8.dat")["_version"] == read_json(z, "le16.dat")["_version"] == "2.2.0"
    assert sorted(v) == [f"{h}:Standard:Expert", f"{h}:Standard:ExpertPlus"]  # one vector per chart
    assert v[f"{h}:Standard:ExpertPlus"][NAMES.index("njs")] == 18 and v[f"{h}:Standard:Expert"][NAMES.index("njs")] == 16


def test_chart_patterns_keeps_every_play_style_apart(tmp_path):
    info = {"_beatsPerMinute": 120, "_difficultyBeatmapSets": [
        {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_beatmapFilename": "E.dat", "_noteJumpMovementSpeed": 16}]},
        {"_beatmapCharacteristicName": "OneSaber", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_beatmapFilename": "O.dat", "_noteJumpMovementSpeed": 17},
            {"_difficulty": "Hard", "_beatmapFilename": "missing.dat", "_noteJumpMovementSpeed": 15}]}]}
    h = "cd" + "0" * 38
    (tmp_path / "maps" / "cd").mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "maps" / "cd" / f"{h}.zip", "w") as z:
        z.writestr("Info.dat", json.dumps(info))
        z.writestr("E.dat", json.dumps(V3))
        z.writestr("O.dat", json.dumps(V2))
    with MapZip(tmp_path, h) as mz:
        v = get("chart_patterns").embed_zip(mz)
    assert sorted(v) == [f"{h}:OneSaber:Expert", f"{h}:Standard:Expert"]  # a broken chart leaves the others
    assert v[f"{h}:OneSaber:Expert"][NAMES.index("njs")] == 17


def test_chart_keys_and_the_map_level_representative():
    from octantvector.common import charts
    k = charts.key("ab" * 20, "OneSaber", "Expert")
    assert charts.split(k) == ("ab" * 20, "OneSaber", "Expert") and charts.key_hash(k) == "ab" * 20
    assert charts.key_hash("ab" * 20) == "ab" * 20 and not charts.is_chart_key("ab" * 20)
    keys = [charts.key("a" * 40, "Standard", "Hard"), charts.key("a" * 40, "Standard", "Expert"),
            charts.key("a" * 40, "OneSaber", "ExpertPlus"), charts.key("b" * 40, "OneSaber", "Expert")]
    hs, x = charts.representative(keys, np.eye(4))
    assert hs == ["a" * 40] and x.tolist() == [[0, 1, 0, 0]]  # hardest Standard; a map with none is left out
    assert charts.hardest_standard(keys) == keys[1] and charts.hardest_standard(keys[2:]) is None


def test_index_represents_a_chart_view_by_one_chart_per_map(tmp_path):
    from octantvector.common import charts
    enc = get("chart_patterns")
    st = Store(enc, tmp_path)
    rng = np.random.default_rng(0)
    for i in range(6):
        h = f"{i:02x}" + "0" * 38
        st.add_many({charts.key(h, "Standard", d): rng.normal(size=enc.dim) for d in ("Hard", "Expert")})
    st.flush()
    ix = Index({"chart_patterns": 1}, tmp_path, device="cpu")
    keys, X = st.load()
    assert ix.hashes == sorted({charts.key_hash(k) for k in keys})
    assert len(ix) == 6  # a map-level index, one row per map version


def test_npsnjs_fills_missing_difficulties():
    v = {"diffs": [{"characteristic": "Standard", "difficulty": "Hard", "nps": 3.0, "njs": 14},
                   {"characteristic": "Standard", "difficulty": "ExpertPlus", "nps": 6.0, "njs": 19}]}
    x = NpsNjs().embed_map({}, v)
    assert list(x[:5]) == [3.0, 3.0, 3.0, 3.0, 6.0]  # Expert ties Hard/ExpertPlus; the easier one wins
    assert list(x[10:]) == [0, 0, 1, 0, 1]
    assert NpsNjs().embed_map({}, {"diffs": []}) is None


def test_kmeans_and_profiles():
    rng = np.random.default_rng(0)
    centers = np.eye(3)
    X = np.vstack([c + 0.05 * rng.normal(size=(20, 3)) for c in centers])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = spherical_kmeans(X, 3)
    assert sorted(np.argmax(np.abs(C), axis=1).tolist()) == [0, 1, 2]
    E = np.vstack([X, -centers])
    s = personal.view_score(E, np.arange(60), "km3")
    assert s[:60].min() > s[60:].max()
    s = personal.view_score(E, np.arange(20), "mean")
    assert s[:20].min() > s[20:].max()


def test_difficulty_fit():
    lo, hi, sd = difficulty_band(np.array([[4.0, 16], [5.0, 17], [6.0, 18]] * 10))
    pen = fit_penalty([np.array([[5.0, 17]]), np.array([[12.0, 25], [5.0, 17]]), np.array([[12.0, 25]]), np.zeros((0, 2))],
                      lo, hi, sd)
    assert pen[0] == 0 and pen[1] == 0 and pen[2] < pen[0] and pen[3] < 0


def test_personal_loo_finds_planted_taste(tmp_path):
    rng = np.random.default_rng(3)
    n = 400
    hs = [f"{i:040x}" for i in range(n)]
    X = rng.normal(size=(n, 4))
    liked = list(range(30))
    X[liked] = rng.normal(size=4) * 3 + 0.3 * rng.normal(size=(30, 4))  # liked maps share a direction
    ix = Index({"toy": 1}, tmp_path, vectors={"toy": (hs, X)}, device="cpu")
    cand = hs[:150]
    res = personal.run(ix, cand, [hs[i] for i in liked], kind="mean", csls=False)
    assert res["n_liked"] == 30 and res["median_rank"] < res["random_median_rank"] / 4
    res_km = personal.run(ix, cand, [hs[i] for i in liked], kind="km3", csls=True)
    assert res_km["median_rank"] < res_km["random_median_rank"]
    res_comb = personal.run(ix, cand, [hs[i] for i in liked], kind="km3", csls=True, csls_density="combined")
    assert res_comb["median_rank"] < res_comb["random_median_rank"]
    ranked = personal.run(ix, cand, [hs[i] for i in liked], kind="mean", csls=False, keep_ranks=True)
    assert sorted(ranked.pop("ranks")) == sorted(hs[i] for i in liked) and ranked == res


def test_view_density_single_view_equals_combined(tmp_path):
    rng = np.random.default_rng(5)
    hs = [f"{i:040x}" for i in range(200)]
    ix = Index({"toy": 1}, tmp_path, vectors={"toy": (hs, rng.normal(size=(200, 8)))}, device="cpu")
    np.testing.assert_allclose(ix.view_density("toy", 10), ix.local_density(10), atol=1e-5)


def test_labels_keep_quit_fail_unlabeled():
    merged = {"a": {"favorite": 1}, "b": {"play": 3, "quit": 3}, "c": {"play": 4, "failed": 4}, "d": {"installed": 1},
              "e": {"playlist": 2}}
    merged["f"] = {"favorite": 1, "play": 1, "finished": 1}
    lab = personal.labels(merged)  # OctantSpecs law 4.1, the rule recommendations use
    assert lab == {"a": 0, "b": 0, "c": 0, "d": 0, "f": 1}  # e: playlist only is not an interaction


def test_playdata_readers(tmp_path):
    (tmp_path / "votedSongs.json").write_text('﻿{"ABC": {"hash": "ABC", "voteType": "Upvote"}}', encoding="utf-8")
    assert playdata.read_votes(tmp_path / "votedSongs.json") == {"abc": "Upvote"}
    h = "c" * 40
    pd = {"localPlayers": [{"favoritesLevelIds": [f"custom_level_{h.upper()}", "OstLevel"],
                            "levelsStatsData": [{"levelId": f"custom_level_{h.upper()}", "playCount": 4, "validScore": True},
                                                {"levelId": "OstLevel", "playCount": 9}]}]}
    (tmp_path / "PlayerData.dat").write_text(json.dumps(pd))
    fav, pc, cl = playdata.read_player_data(tmp_path / "PlayerData.dat")
    assert fav == {h} and pc == {h: 4} and cl == {h: 1}
    db = tmp_path / "beatsaber.db"
    con = sqlite3.connect(db)
    con.execute("create table MovieCutRecord (songHash text, cleared text, difficulty text, mode text, songSpeedMultiplier text)")
    con.executemany("insert into MovieCutRecord values (?,?,?,?,?)",
                    [(h.upper(), "finished", "Expert+", "SoloStandard", "1.0"), (h.upper(), "pause", "Expert", "SoloStandard", "1.0"),
                     ("", "finished", "Hard", "SoloStandard", "1.0")])
    con.commit(); con.close()
    plays = playdata.read_datarecorder(db)
    assert [p["outcome"] for p in plays] == ["finished", "quit"] and plays[0]["diff"] == "ExpertPlus"
    (tmp_path / "replays").mkdir()
    (tmp_path / "replays" / f"123-Song-Expert-Standard-{h.upper()}-99.bsor").write_bytes(b"")
    sig = playdata.build_signals(votes={h: "Upvote"}, player_data=(fav, pc, cl), plays=plays,
                                 replays=[playdata.replay_hashes(tmp_path / "replays")])
    assert sig[h]["play"] == 2 and sig[h]["finished"] == 1 and sig[h]["replay"] == 1 and sig[h]["favorite"] == 1


def test_difficulty_band_excludes_held_out_key(tmp_path):
    rng = np.random.default_rng(5)
    n = 120
    hs = [f"{i:040x}" for i in range(n)]
    ix = Index({"toy": 1}, tmp_path, vectors={"toy": (hs, rng.normal(size=(n, 4)))}, device="cpu")
    cand, liked = hs[:60], hs[:10]
    key_of = {h: f"k{i}" for i, h in enumerate(cand)}
    diffs = {h: np.array([[5.0, 17.0]]) for h in cand}
    pts = [(f"k{i % 10}", 4.0 + i % 3, 16.0 + i % 3) for i in range(40)]  # plays of the liked keys (any version)
    res = personal.run(ix, cand, liked, kind="mean", csls=False, beta=1.0, difficulties=diffs, finished_points=pts,
                       key_of=key_of)
    assert res["n_liked"] == 10


def test_the_embed_pass_reads_each_zip_from_disk_once(tmp_path, monkeypatch):
    import builtins
    import io
    from octantvector.generation import embed
    info = {"_beatsPerMinute": 120, "_difficultyBeatmapSets": [
        {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_beatmapFilename": "E.dat", "_noteJumpMovementSpeed": 16}]}]}
    h = "ef" + "0" * 38
    (tmp_path / "maps" / "ef").mkdir(parents=True)
    path = tmp_path / "maps" / "ef" / f"{h}.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Info.dat", json.dumps(info))
        z.writestr("E.dat", json.dumps(V3))
    opened = []
    for mod, real in ((io, io.open), (builtins, builtins.open)):
        monkeypatch.setattr(mod, "open", lambda f, *a, _real=real, **k: (opened.append(str(f)), _real(f, *a, **k))[1])
    embed._init(["chart_patterns"], str(tmp_path))
    got, ids, sha, out, err = embed._work((h, ["chart_patterns"]))
    assert err is None and sorted(out["chart_patterns"]) == [f"{h}:Standard:Expert"]
    assert opened.count(str(path)) == 1  # hashed and parsed from the same read
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
