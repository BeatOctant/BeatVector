"""Full distributable DB (dbrelease.py) and its reading side (db.py) on a synthetic archive (no models, no GPU)."""
import io
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq
import pytest

from octantvector.generation import GenerationError, calibration, config, dbrelease, normalisation
from octantvector.common import dbformat
from octantvector.runtime import VersionError
from octantvector.runtime.db import DBError, apply_diff, open_db, read_info, search
from octantvector.generation.encoders import get
from octantvector.generation.ledger import Ledger
from octantvector.generation.search import Index
from octantvector.generation.store import Store

from test_ledger import H, NOTES, mapr, update, version, write_world, write_zip, cfg  # noqa: F401
from datetime import timedelta

VIEWS = ["chart_patterns", "meta_tags"]
CP, MT = "chart_patterns@3", "meta_tags@2"


def ck(h, characteristic="Standard", difficulty="Expert"):
    return f"{h}:{characteristic}:{difficulty}"
AT = datetime(2026, 10, 2, tzinfo=timezone.utc)


class test_world:  # not a test: holds the catalogue the fixture wrote
    maps = None


@pytest.fixture
def world(tmp_path, cfg, monkeypatch):
    rng = np.random.default_rng(0)
    maps = []
    for i in range(1, 32):
        v = version(H(i))
        v["diffs"][0].update(nps=float(rng.uniform(1, 9)), njs=float(rng.uniform(10, 22)))
        v["createdAt"] = "2025-01-01T00:00:00Z"
        tags = [["pop"], ["rock"], ["tech", "metal"], ["dance", "electronic"]][i % 4]
        maps.append(mapr(f"k{i}", [v], up=int(rng.integers(0, 300)), tags=tags))
    maps[0]["versions"] = [version(H(99), "Uploaded")] + maps[0]["versions"]  # an earlier version of k1
    write_world(tmp_path, maps[:30], [H(i) for i in range(1, 31)])
    for i in range(1, 32):  # different charts so chart_patterns is not constant
        notes = [dict(n, _lineIndex=(n["_lineIndex"] + i) % 4, _time=n["_time"] * (1 + i % 3)) for n in NOTES]
        write_zip(tmp_path / "archive", H(i), notes=notes)
    write_zip(tmp_path / "archive", H(5), notes=NOTES[:3])  # too few notes: no chart_patterns vector
    update(cfg, VIEWS)
    monkeypatch.setattr(test_world, "maps", maps)  # for tests that change the world later
    led = Ledger(tmp_path / "store" / "ledger.sqlite")
    pool = [h for h, _ in led.pool("distribution")]
    led.close()
    cal = calibration.load(calibration.save(calibration.compute(tmp_path / "store", VIEWS, pool, "distribution"),
                                            tmp_path / "store"))
    out, diff = dbrelease.release(config.load(), VIEWS, cal, built_at=AT, log=io.StringIO())
    assert diff is None  # the first release is a full
    return tmp_path, cal, pool, out


def test_release_has_the_contract_2_layout(world):
    tmp_path, cal, pool, out = world
    m = json.loads((out / "manifest.json").read_text())
    assert out.name == "2026.10.02" and m["schemaVersion"] == "1.1" and m["baseDbVersion"] is None
    assert m["calibrationId"] == cal["id"] and m["counts"] == {"maps": 30, "charts": 30, "superseded": 1}
    assert m["kind"] == "full" and m["diffsSinceFull"] == 0
    v = m["views"][CP]
    assert v["rows"] == "charts" and v["dtype"] == "float16" and v["role"] == "chart"  # chart views: chart rows
    assert v["fingerprint"] == get("chart_patterns").fingerprint()
    assert m["views"][MT]["role"] == "meta" and m["views"][MT]["rows"] == "maps"
    assert set(m["files"]) == {"maps.parquet", "charts.parquet", "superseded.parquet", f"calibration/{cal['id']}.json"} \
        | {f"views/{k}{s}" for k in m["views"] for s in (".npy", ".rows.npy")}
    maps = pq.read_table(out / "maps.parquet").to_pydict()
    assert maps["hash"] == sorted(pool) and "song" not in maps and "songName" not in maps  # no titles
    charts = pq.read_table(out / "charts.parquet").to_pydict()
    assert "label" not in charts and charts["pass"][0] is None  # Pass/Tech not computed yet
    assert set(charts["play_style"]) == {"Standard"}
    assert pq.read_table(out / "superseded.parquet").to_pydict() == {"hash": [H(99)], "current": [H(1)]}
    assert len(np.load(out / f"views/{CP}.rows.npy")) == 29  # the chart without the view has no row
    again, _ = dbrelease.release(config.load(), VIEWS, built_at=AT, log=io.StringIO())  # same day: never overwritten
    assert again.name == "2026.10.02.2"


def test_db_search_matches_the_working_store(world):
    tmp_path, cal, pool, out = world
    db = open_db(out)
    assert db.version == "2026.10.02" and set(db.views) == {CP, MT}
    for i, key in zip(VIEWS, (CP, MT)):
        ix = Index({i: 1}, tmp_path / "store", restrict=pool, device="cpu", calibration=cal)
        for h in ix.hashes[:10]:  # one Standard Expert chart per map: the map-level index sees the same rows
            got = search(db, ck(h) if key == CP else h, key, 5)["results"]
            _, sc = ix.search_views([ix.pos[h]], 5)[i]
            full = ix._scores(np.array([ix.pos[h]]), weights={i: 1.0})[0]
            np.testing.assert_allclose([r["score"] for r in got], sc[0], atol=2e-3)  # float16
            # the same neighbours, in the same order up to float16 ties: each one scores what the store's rank does
            np.testing.assert_allclose([full[ix.pos[r["hash"]]] for r in got], sc[0], atol=2e-3)


def test_db_search_replaces_old_versions_and_handles_missing_views(world):
    _, _, _, out = world
    db = open_db(out)
    r = search(db, H(99), MT, 3)
    assert r["query"] == H(1) and r["replaced"] == H(99) and len(r["results"]) == 3
    r = search(db, (H(99), "Standard", "Expert"), CP, 3)  # a chart of an old version: the current version's chart
    assert r["query"] == ck(H(1)) and r["replaced"] == ck(H(99)) and len(r["results"]) == 3
    assert r["results"][0].keys() == {"hash", "characteristic", "difficulty", "rank", "score"}
    assert search(db, ck(H(5)), CP, 3)["results"] == []
    assert H(5) not in [x["hash"] for x in search(db, ck(H(1)), CP, 40)["results"]]
    with pytest.raises(KeyError):
        search(db, H(77), MT)
    with pytest.raises(KeyError):
        search(db, ck(H(1), "OneSaber"), CP)


def test_open_db_refuses_a_tampered_or_newer_db(world):
    _, _, _, out = world
    p = out / "views" / f"{MT}.npy"
    good = p.read_bytes()
    p.write_bytes(good[:-2] + b"\0\1")
    with pytest.raises(DBError, match="does not match"):
        open_db(out)
    p.write_bytes(good)
    m = json.loads((out / "manifest.json").read_text())
    (out / "manifest.json").write_text(json.dumps({**m, "schemaVersion": "2.0"}))
    with pytest.raises(VersionError, match="schemaVersion '2.0'"):  # a DBError the host can tell apart
        open_db(out)
    m["schemaVersion"] = "1.3"  # a newer minor: unknown keys and row kinds are ignored
    m["somethingNew"] = 1
    m["views"]["future@1"] = {"rows": "notes", "file": "x.npy", "rowsFile": "x.rows.npy"}
    (out / "manifest.json").write_text(json.dumps(m))
    assert open_db(out).skipped == ["future@1"]


def same_content(a, b):
    assert a["maps"] == b["maps"] and a["charts"] == b["charts"] and a["superseded"] == b["superseded"]
    assert a["calibration"] == b["calibration"] and set(a["views"]) == set(b["views"])
    for k in a["views"]:
        va, vb = a["views"][k], b["views"][k]
        assert va["meta"] == vb["meta"] and set(va["vecs"]) == set(vb["vecs"])
        assert all(va["vecs"][h].tobytes() == vb["vecs"][h].tobytes() for h in va["vecs"])


def next_day(tmp_path, cfg, days=1, **kw):
    update(cfg, VIEWS)
    kw.setdefault("view_ids", VIEWS)
    return dbrelease.release(config.load(), built_at=AT + timedelta(days=days), log=io.StringIO(), **kw)


def change_world(tmp_path):
    maps = test_world.maps
    maps[1]["tags"] = ["jazz"]  # a changed vector (meta_tags)
    maps[2]["versions"][0]["diffs"][0]["nps"] = 99.0  # a changed chart row
    maps[3]["versions"] = [dict(maps[3]["versions"][0], state="Uploaded"), version(H(40))]  # k4 got a new version
    maps[3]["versions"][1]["createdAt"] = "2026-10-03T00:00:00Z"
    del maps[9]  # k10 deleted; k31 is new
    write_world(tmp_path, maps, [H(i) for i in range(1, 32)])


def test_full_plus_diff_equals_the_new_full(world, cfg):
    tmp_path, cal, pool, base = world
    change_world(tmp_path)
    new, diff = next_day(tmp_path, cfg)
    dm = json.loads((diff / "manifest.json").read_text())
    assert dm["kind"] == "diff" and dm["baseDbVersion"] == "2026.10.02" and dm["dbVersion"] == new.name == "2026.10.03"
    assert dm["counts"] == {"added": 2, "replaced": 2, "deleted": 2, "superseded": 2}  # k4's old version and k10 left
    with pytest.raises(DBError, match="not a full"):
        open_db(diff)

    applied = apply_diff(base, diff, tmp_path / "host" / dm["dbVersion"])
    same_content(dbformat.read_content(applied), dbformat.read_content(new))
    ma, mn = (json.loads((p / "manifest.json").read_text()) for p in (applied, new))
    assert {k: v for k, v in ma.items() if k != "files"} == {k: v for k, v in mn.items() if k != "files"}
    assert ma["diffsSinceFull"] == 1
    assert search(open_db(applied), H(4), MT, 3)["query"] == H(40)  # k4's old version -> its new one

    with pytest.raises(DBError, match="starts from"):  # a diff only fits its own base
        apply_diff(new, diff, tmp_path / "host" / "x")


def manifest(folder):
    return json.loads((folder / "manifest.json").read_text())


def newer(folder, bump=()):
    """`folder` rewritten as a newer OctantVector could write it: a column in maps.parquet and in charts.parquet, a
    per-view field, a manifest field, chartMetrics and a newer minor schemaVersion, none of which this one knows.
    bump: the hashes whose values differ from the previous version's (the rows its diff carries)."""
    import pyarrow as pa
    cols = {"maps.parquet": (pa.field("bpm", pa.float64()), lambda h: int(h[:2], 16) + (1000.0 if h in bump else 0)),
            "charts.parquet": (pa.field("segments", pa.list_(pa.float32())), lambda h: [2.5, 1] if h in bump else [.5])}
    for name, (field, value) in cols.items():
        t = pq.read_table(folder / name)
        pq.write_table(t.append_column(field, pa.array([value(h) for h in t.column("hash").to_pylist()], field.type)),
                       folder / name)
    m = manifest(folder)
    m.update(schemaVersion="1.2", chartMetrics={"pass": {"source": "swing@3"}}, somethingNew={"a": [1, 2]})
    for v in m["views"].values():
        v["unit"] = "cosine"
    m["files"] = {n: dbformat.sha256(folder / n) for n in m["files"]}
    (folder / "manifest.json").write_text(json.dumps(m))


def test_a_diff_carries_what_this_octantvector_does_not_know(world, cfg):
    """OctantSpecs octantvector/law.md 2.4 when the DB is newer than this OctantVector: the previous full + the diff is still
    the new full, with the columns, manifest fields, per-view fields and chartMetrics this version does not know."""
    tmp_path, cal, pool, base = world
    change_world(tmp_path)
    new, diff = next_day(tmp_path, cfg)
    changed = set(pq.read_table(diff / "changed.parquet").column("hash").to_pylist())
    for folder, bump in ((base, ()), (new, changed), (diff, changed)):
        newer(folder, bump)
    applied = apply_diff(base, diff, tmp_path / "host" / new.name)
    for name in ("maps.parquet", "charts.parquet", "superseded.parquet"):
        assert pq.read_table(applied / name).equals(pq.read_table(new / name))
    same_content(dbformat.read_content(applied), dbformat.read_content(new))
    ma, mn = manifest(applied), manifest(new)
    assert {k: v for k, v in ma.items() if k != "files"} == {k: v for k, v in mn.items() if k != "files"}
    assert ma["schemaVersion"] == "1.2" and ma["chartMetrics"] == {"pass": {"source": "swing@3"}}
    assert ma["somethingNew"] == {"a": [1, 2]} and {v["unit"] for v in ma["views"].values()} == {"cosine"}
    assert pq.read_table(applied / "maps.parquet").column("bpm").to_pylist()[:2] == [1.0, 1002.0]  # k2 changed
    assert open_db(applied).version == new.name


def test_a_diff_is_refused_when_it_cannot_carry_what_this_octantvector_does_not_know(world, cfg):
    """...so that the caller takes the full instead of keeping a DB that differs from it."""
    tmp_path, cal, pool, base = world
    change_world(tmp_path)
    new, diff = next_day(tmp_path, cfg)
    n = iter(range(100))

    def copy(src, edit=lambda m: None, files=()):
        dst = tmp_path / "copies" / str(next(n)) / src.name
        shutil.copytree(src, dst)
        m = manifest(dst)
        for name in files:
            (dst / name).parent.mkdir(exist_ok=True)
            np.save(dst / name, np.zeros(3, np.int32))
            m["files"][name] = dbformat.sha256(dst / name)
        edit(m)
        (dst / "manifest.json").write_text(json.dumps(m))
        return dst

    def refused(b, d, match):
        out = tmp_path / "host" / str(next(n))
        with pytest.raises(DBError, match=match):
            apply_diff(b, d, out)
        assert not out.exists()

    future = {"rows": "notes", "dim": 3, "file": "views/future@1.npy", "rowsFile": "views/future@1.rows.npy"}
    view = lambda m: m["views"].update({"future@1": dict(future, change="added")})  # noqa: E731
    refused(base, copy(diff, view, [future["file"], future["rowsFile"]]), "view future@1.*take the full of 2026.10.03")
    refused(base, copy(diff, files=["notes.npy"]), "file notes.npy")
    refused(copy(base, lambda m: m.update(somethingNew=1)), diff, "manifest field somethingNew of the base")
    d = copy(diff)
    newer(d, ())
    refused(base, d, "the columns of maps.parquet changed")
    # a view of the base this version cannot read is fine when the diff removes it: the new full has none
    b = copy(base, lambda m: m["views"].update({"future@1": future}), [future["file"], future["rowsFile"]])
    applied = apply_diff(b, copy(diff, lambda m: m["removedViews"].append("future@1")), tmp_path / "host" / "ok")
    same_content(dbformat.read_content(applied), dbformat.read_content(new))


def test_a_release_after_a_newer_octantvector_is_a_full(world, cfg):
    tmp_path, cal, pool, base = world
    newer(base)
    update(cfg, VIEWS)
    log = io.StringIO()
    new, diff = dbrelease.release(config.load(), VIEWS, built_at=AT + timedelta(days=1), log=log)
    assert diff is None and "does not know: column maps.parquet bpm" in log.getvalue()  # a diff would drop it


def test_a_full_is_published_when_a_diff_is_impossible_or_due(world, cfg, monkeypatch):
    tmp_path, cal, pool, base = world
    _, diff = next_day(tmp_path, cfg, full=True)
    assert diff is None
    monkeypatch.setenv("BV_FULL_EVERY", "1")
    _, diff = next_day(tmp_path, cfg, 2)
    assert diff is not None  # nothing changed: an empty diff
    assert json.loads((diff / "manifest.json").read_text())["counts"]["added"] == 0
    _, diff = next_day(tmp_path, cfg, 3)
    assert diff is None  # one diff since the last full is the limit
    other = calibration.compute(tmp_path / "store", VIEWS, pool[:10], "first ten")
    _, diff = dbrelease.release(config.load(), VIEWS, other, built_at=AT + timedelta(days=4), log=io.StringIO())
    assert diff is None  # the statistics of the views changed: coordinates moved


def test_a_new_normalisation_is_published_as_a_full(world, cfg, monkeypatch):
    tmp_path, cal, pool, base = world
    monkeypatch.setattr(normalisation, "VERSION", 2)
    with pytest.raises(GenerationError, match="calibrate again"):  # the previous version's calibration is of version 1
        next_day(tmp_path, cfg)
    cal2 = calibration.load(calibration.save(calibration.compute(tmp_path / "store", VIEWS, pool, "distribution"),
                                             tmp_path / "store"))
    assert cal2["views"][MT]["mean"].tolist() == cal["views"][MT]["mean"].tolist()  # only the version tells them apart
    new, diff = next_day(tmp_path, cfg, view_ids=VIEWS, calibration=cal2)
    assert diff is None  # another coordinate system: a full, under a new calibration id
    assert json.loads((new / "manifest.json").read_text())["calibrationId"] == cal2["id"] != cal["id"]


def test_view_packs_add_replace_and_remove_views(world, cfg, monkeypatch):
    tmp_path, cal, pool, base = world
    # swing_flow is named: added with all its chart rows; chart_patterns@3 goes to version 4; meta_tags is not named
    from octantvector.generation.encoders import REGISTRY
    update(cfg, ["swing_flow"])  # before `first`, so both carry the same Pass and Tech
    first, _ = dbrelease.release(config.load(), VIEWS, cal, built_at=AT + timedelta(days=1), log=io.StringIO(), full=True)
    monkeypatch.setattr(REGISTRY["chart_patterns"], "version", "4")
    update(cfg, ["swing_flow", "chart_patterns"])  # version 4 computes everything again
    new, diff = next_day(tmp_path, cfg, 2, view_ids=["swing_flow", "chart_patterns"])
    dm = json.loads((diff / "manifest.json").read_text())
    assert set(dm["views"]) == {"swing_flow@12", "chart_patterns@4"} and dm["removedViews"] == [MT]
    assert dm["views"]["swing_flow@12"]["change"] == "added" and "replaces" not in dm["views"]["swing_flow@12"]
    assert dm["views"]["swing_flow@12"]["rows"] == "charts"
    assert dm["views"]["chart_patterns@4"]["replaces"] == CP
    assert dm["counts"]["added"] == dm["counts"]["replaced"] == 0  # no map changed
    new_cal = json.loads((diff / dm["calibrationFile"]).read_text())
    assert new_cal["extends"] == cal["id"] and set(new_cal["views"]) >= {"swing_flow@12", "chart_patterns@4"}
    assert new_cal["views"][MT] == json.loads(
        (first / json.loads((first / "manifest.json").read_text())["calibrationFile"]).read_text())["views"][MT]

    applied = apply_diff(first, diff, tmp_path / "host" / dm["dbVersion"])
    same_content(dbformat.read_content(applied), dbformat.read_content(new))
    got = dbformat.read_content(applied)["views"]["swing_flow@12"]["vecs"]
    stored = {k for k in Store(get("swing_flow"), tmp_path / "store").done() if k.split(":")[0] in set(pool)}
    assert got.keys() == stored and got  # the added view has all its chart rows


def test_a_chart_view_describing_maps_is_refused(world, cfg):
    tmp_path, cal, pool, base = world
    update(cfg, ["npsnjs_std"])
    cal2 = calibration.compute(tmp_path / "store", ["npsnjs_std"], pool, "distribution")
    with pytest.raises(GenerationError, match="principle 2"):
        dbrelease.release(config.load(), ["npsnjs_std"], cal2, built_at=AT + timedelta(days=1), log=io.StringIO())


def test_chart_views_compare_charts_of_one_play_style_only(tmp_path, cfg, monkeypatch):
    """Two play styles in the same maps: every chart has its own row, and a search stays inside its play style."""
    import zipfile
    maps = []
    for i in range(1, 9):
        v = version(H(i))
        v["diffs"].append({"characteristic": "OneSaber", "difficulty": "Expert", "nps": 3.0, "njs": 15})
        maps.append(mapr(f"k{i}", [v]))
    write_world(tmp_path, maps, [H(i) for i in range(1, 9)])
    info = {"_beatsPerMinute": 120, "_difficultyBeatmapSets": [
        {"_beatmapCharacteristicName": c, "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_beatmapFilename": f"{c}.dat", "_noteJumpMovementSpeed": 16}]}
        for c in ("Standard", "OneSaber")]}
    for i in range(1, 9):
        p = tmp_path / "archive" / "maps" / H(i)[:2] / f"{H(i)}.zip"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("Info.dat", json.dumps(info))
            for c, shift in (("Standard", i), ("OneSaber", 2 * i)):
                notes = [dict(n, _lineIndex=(n["_lineIndex"] + shift) % 4, _time=n["_time"] * (1 + shift % 3))
                         for n in NOTES]
                z.writestr(f"{c}.dat", json.dumps({"_version": "2.2.0", "_notes": notes, "_obstacles": []}))
    update(cfg, ["chart_patterns"])
    led = Ledger(tmp_path / "store" / "ledger.sqlite")
    pool = [h for h, _ in led.pool("distribution")]
    led.close()
    cal = calibration.compute(tmp_path / "store", ["chart_patterns"], pool, "distribution")
    out, _ = dbrelease.release(config.load(), ["chart_patterns"], cal, built_at=AT, log=io.StringIO())
    charts = pq.read_table(out / "charts.parquet").to_pydict()
    assert sorted(set(charts["play_style"])) == ["OneSaber", "Standard"] and len(charts["hash"]) == 16
    db = open_db(out)
    assert len(db.views[CP][1]) == 16  # one row per chart, both play styles
    for c in ("Standard", "OneSaber"):
        res = search(db, ck(H(1), c), CP, 20)["results"]
        assert len(res) == 7 and {r["characteristic"] for r in res} == {c}


def test_a_charts_table_without_play_style_is_refused(world):
    """A 1.0 DB (no play_style column) is not read: a clear DBError, not a guess from the characteristic."""
    _, _, _, out = world
    t = pq.read_table(out / "charts.parquet")
    pq.write_table(t.drop(["play_style"]), out / "charts.parquet")
    with pytest.raises(DBError, match="no column play_style"):
        dbformat.read_charts(out / "charts.parquet")


def test_chart_rows_match_the_catalogue_by_its_spelling_of_the_difficulty(tmp_path, cfg, monkeypatch):
    """Older zips write "expert" or "Expert+" for the catalogue's "Expert" and "ExpertPlus": the same chart. A zip
    whose difficulty is another one than the catalogue's has no catalogue row, and the log says so."""
    import zipfile
    maps = []
    for i in range(1, 9):
        v = version(H(i))
        if i == 2:
            v["diffs"][0]["difficulty"] = "ExpertPlus"
        maps.append(mapr(f"k{i}", [v]))
    write_world(tmp_path, maps, [H(i) for i in range(1, 9)])
    in_zip = {1: "expert", 2: "Expert+", 3: "Hard"}
    for i in range(1, 9):
        info = {"_beatsPerMinute": 120, "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [{"_difficulty": in_zip.get(i, "Expert"), "_beatmapFilename": "c.dat",
                                         "_noteJumpMovementSpeed": 16}]}]}
        notes = [dict(n, _lineIndex=(n["_lineIndex"] + i) % 4, _time=n["_time"] * (1 + i % 3)) for n in NOTES]
        with zipfile.ZipFile(tmp_path / "archive" / "maps" / H(i)[:2] / f"{H(i)}.zip", "w") as z:
            z.writestr("Info.dat", json.dumps(info))
            z.writestr("c.dat", json.dumps({"_version": "2.2.0", "_notes": notes, "_obstacles": []}))
    update(cfg, ["chart_patterns"])
    led = Ledger(tmp_path / "store" / "ledger.sqlite")
    pool = [h for h, _ in led.pool("distribution")]
    led.close()
    cal = calibration.compute(tmp_path / "store", ["chart_patterns"], pool, "distribution")
    log = io.StringIO()
    out, _ = dbrelease.release(config.load(), ["chart_patterns"], cal, built_at=AT, log=log)
    db = open_db(out)
    assert len(db.views[CP][1]) == 7  # all but the Hard chart the catalogue calls Expert
    assert search(db, ck(H(1)), CP, 3)["results"] and search(db, ck(H(2), difficulty="ExpertPlus"), CP, 3)["results"]
    assert f"1 charts left out, their difficulty is not in the catalogue: {ck(H(3), difficulty='Hard')}" \
        in log.getvalue()


def test_pass_and_tech_come_from_the_stored_swing_simulation(world, cfg):
    """swing_flow fills Pass and Tech of every chart it has a vector for without being one of the views; the others
    stay empty, and chartMetrics names the view."""
    tmp_path, cal, pool, base = world
    assert json.loads((base / "manifest.json").read_text())["chartMetrics"] == {}  # swing_flow not computed: empty
    update(cfg, ["swing_flow"])
    log = io.StringIO()
    out, diff = dbrelease.release(config.load(), VIEWS, cal, built_at=AT + timedelta(days=1), log=log)
    m = json.loads((out / "manifest.json").read_text())
    assert m["chartMetrics"] == {"pass": "swing_flow@12", "tech": "swing_flow@12"} and "swing_flow@12" not in m["views"]
    e = get("swing_flow")
    hs, X = Store(e, tmp_path / "store").load()
    stored = {k: X[n] for n, k in enumerate(hs)}
    t = pq.read_table(out / "charts.parquet").to_pydict()
    for h, ps, tech in zip(t["hash"], t["pass"], t["tech"]):
        v = stored.get(ck(h))
        if v is None:
            assert ps is None and tech is None
        else:
            assert ps == pytest.approx(float(v[e.names.index("pass")])) and tech == pytest.approx(float(v[e.names.index("tech")]))
    assert sum(p is not None for p in t["pass"]) == len([h for h in t["hash"] if ck(h) in stored]) > 0
    assert open_db(out).chart_metrics["pass"].shape == (30,)
    dm = json.loads((diff / "manifest.json").read_text())  # the filled rows travel as changed rows
    assert dm["chartMetrics"] == m["chartMetrics"] and dm["counts"]["replaced"] > 0
    applied = apply_diff(base, diff, tmp_path / "applied")
    assert pq.read_table(applied / "charts.parquet").to_pydict()["pass"] == t["pass"]


def test_a_chart_with_too_many_notes_gets_no_swing_vector(world, cfg, monkeypatch):
    monkeypatch.setattr(type(get("swing_flow")), "max_notes", 5)
    from octantvector.generation.archive import MapZip
    with MapZip(world[0] / "archive", H(1)) as mz:
        assert get("swing_flow").embed_zip(mz) == {}


HANDS = [{"id": "flow", "label": "流れ", "inputs": [{"kind": "seed"}], "handling": "narrow_then_rank",
          "knobs": {"narrow": {"meta_tags": 1}, "rank": {"chart_patterns": 1}}}]


def test_hands_go_into_the_manifest_ride_a_diff_and_are_checked_first(world, cfg, tmp_path):
    _, cal, pool, base = world
    with pytest.raises(GenerationError, match="mert_v1_95m"):  # this DB has no MERT view: refused before anything is written
        next_day(tmp_path, cfg, presets=[dict(HANDS[0], knobs={"weights": {"mert_v1_95m": 1}}, handling="near")])
    path = tmp_path / "hands.json"
    path.write_text(json.dumps(HANDS))
    new, diff = next_day(tmp_path, cfg, presets=path)  # only the hands changed: a diff with no rows
    dm = json.loads((diff / "manifest.json").read_text())
    assert dm["presets"] == HANDS and dm["counts"]["added"] == dm["counts"]["replaced"] == 0
    assert json.loads((new / "manifest.json").read_text())["presets"] == HANDS
    applied = apply_diff(base, diff, tmp_path / "host" / dm["dbVersion"])
    assert json.loads((applied / "manifest.json").read_text())["presets"] == HANDS
    from octantvector.runtime import presets
    assert [p["id"] for p in presets(open_db(applied))["presets"]] == ["flow"]
    newer, _ = next_day(tmp_path, cfg, days=2)  # not given again: kept
    assert json.loads((newer / "manifest.json").read_text())["presets"] == HANDS
    with pytest.raises(GenerationError, match="cannot serve"):  # kept hands are checked like given ones
        next_day(tmp_path, cfg, days=3, view_ids=["chart_patterns"])


def test_the_notice_and_view_credits_go_into_the_manifest_and_ride_a_diff(world, cfg, tmp_path, monkeypatch):
    _, _, _, base = world
    m = json.loads((base / "manifest.json").read_text())
    assert "notice" not in m and "credits" not in m["views"][CP]  # none given, none from the encoders
    monkeypatch.setattr(type(get("chart_patterns")), "credits", ("Pattern source, MIT",))
    fp = get("chart_patterns").fingerprint()
    new, diff = next_day(tmp_path, cfg, notice="# Terms\nCC BY-NC-SA 4.0")  # credits are not the definition: a diff
    assert diff is not None and fp == m["views"][CP]["fingerprint"]
    dm = json.loads((diff / "manifest.json").read_text())
    assert dm["notice"] == "# Terms\nCC BY-NC-SA 4.0" and dm["views"][CP]["credits"] == ["Pattern source, MIT"]
    applied = apply_diff(base, diff, tmp_path / "host" / dm["dbVersion"])
    info = open_db(applied).info()
    assert info["notice"] == "# Terms\nCC BY-NC-SA 4.0"
    assert info["views"][CP]["credits"] == ["Pattern source, MIT"] and info["views"][MT]["credits"] == []
    newer, _ = next_day(tmp_path, cfg, days=2)  # not given again: kept
    assert json.loads((newer / "manifest.json").read_text())["notice"] == "# Terms\nCC BY-NC-SA 4.0"
    with pytest.raises(GenerationError, match="notice"):
        next_day(tmp_path, cfg, days=3, notice=" ")


def test_info_keeps_only_text_from_the_notice_and_credits(world):
    _, _, _, out = world
    m = json.loads((out / "manifest.json").read_text())
    m["notice"] = {"not": "text"}
    m["views"][CP]["credits"] = ["ok", 3, None]
    m["views"][MT]["credits"] = "not a list"
    (out / "manifest.json").write_text(json.dumps(m))
    info = open_db(out).info()
    assert info["notice"] is None and info["views"][CP]["credits"] == ["ok"] and info["views"][MT]["credits"] == []


def test_read_info_says_what_info_says_from_the_manifest_alone(world):
    _, _, _, out = world
    info = open_db(out).info()
    assert read_info(out) == {**info, "views": {k: {f: x for f, x in v.items() if f != "rows"}
                                                for k, v in info["views"].items()}}
    # No file but the manifest is read: a damaged view still shows; opening it is refused
    view = out / json.loads((out / "manifest.json").read_text())["views"][CP]["file"]
    view.write_bytes(view.read_bytes() + b"x")
    assert read_info(out)["dbVersion"] == info["dbVersion"]
    with pytest.raises(DBError):
        open_db(out)


def test_read_info_refuses_what_open_db_refuses_from_the_manifest(world, tmp_path):
    _, _, _, out = world
    with pytest.raises(DBError):
        read_info(tmp_path / "nothing")
    m = json.loads((out / "manifest.json").read_text())
    for name, changed in (("diff", {**m, "kind": "diff"}), ("broken", {**m, "views": [1]}),
                          ("newer", {**m, "schemaVersion": "9.0"})):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps(changed))
        with pytest.raises(VersionError if name == "newer" else DBError):
            read_info(folder)


def test_a_genre_view_lists_its_words_and_a_hand_can_curate_them(world, cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(type(get("meta_tags")), "role", "genre")  # stands in for clap_concepts (needs no model)
    words = list(get("meta_tags").names)
    hands = [{"id": "genre", "label": "ジャンル", "handling": "near", "knobs": {},
              "inputs": [{"kind": "choice", "name": "genre", "label": "ジャンル",
                          "options": [{"value": "tag:rock", "label": "ロック"}, {"value": "tag:metal", "label": "メタル"},
                                      {"value": "enka", "label": "演歌"}]}]}]
    with pytest.raises(GenerationError):  # a hand naming views the DB lacks still stops the release
        next_day(tmp_path, cfg, presets=[{"id": "x", "label": "x", "inputs": [], "handling": "near",
                                          "knobs": {"weights": {"mert_v1_95m": 1}}}])
    new, _ = next_day(tmp_path, cfg, presets=hands, full=True)
    m = json.loads((new / "manifest.json").read_text())
    assert m["views"][MT]["role"] == "genre" and m["views"][MT]["genres"] == words
    from octantvector.runtime import presets
    listed = presets(open_db(new))["presets"]
    assert [o["label"] for o in listed[0]["inputs"][0]["options"]] == ["ロック", "メタル"]  # 演歌 is not a word here
