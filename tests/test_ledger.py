"""Ledger, pools and `octantvector update` on a synthetic archive and catalogue (no models, no GPU)."""
import gzip
import io
import json
import zipfile

import pytest

from octantvector.generation import GenerationError, config, embed, pools
from octantvector.generation.encoders import get
from octantvector.generation.ledger import Ledger
from octantvector.generation.store import Store

NOTES = [{"_time": i * 0.5, "_lineIndex": i % 4, "_lineLayer": i % 3, "_type": i % 2,
          "_cutDirection": [1, 0][i % 2] if i % 3 else 8} for i in range(40)]
INFO = {"_beatsPerMinute": 120, "_songFilename": "song.egg", "_difficultyBeatmapSets": [
    {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
        {"_difficulty": "Expert", "_beatmapFilename": "Expert.dat", "_noteJumpMovementSpeed": 16}]}]}


def H(i):
    return f"{i:02x}" + "0" * 38


def version(h, state="Published", standard=True):
    diffs = [{"characteristic": "Standard" if standard else "OneSaber", "difficulty": "Expert", "nps": 4.0, "njs": 16}]
    return {"hash": h, "state": state, "diffs": diffs}


def mapr(key, versions, up=200, tags=("pop",)):
    return {"id": key, "metadata": {"songName": "s", "songAuthorName": "a", "levelAuthorName": "m", "bpm": 120,
                                    "duration": 100}, "stats": {"upvotes": up}, "tags": list(tags), "versions": versions}


def write_zip(root, h, notes=NOTES, raw=None):
    p = root / "maps" / h[:2] / f"{h}.zip"
    p.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        p.write_bytes(raw)
        return
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("Info.dat", json.dumps(INFO))
        z.writestr("Expert.dat", json.dumps({"_version": "2.2.0", "_notes": notes, "_obstacles": []}))


def write_world(tmp_path, maps, zips):
    arch = tmp_path / "archive"
    for h in zips:
        if not (arch / "maps" / h[:2] / f"{h}.zip").exists():
            write_zip(arch, h)
    (arch / "download.journal.jsonl").write_text(
        "".join(json.dumps({"event": "committed", "hash": h, "commit_seq": n}) + "\n" for n, h in enumerate(zips)))
    with gzip.open(tmp_path / "cat.jsonl.gz", "wt") as f:
        for m in maps:
            f.write(json.dumps(m) + "\n")
    # the contract 4 (OctantHarvest crawl to OctantVector) format markers OctantHarvest's crawlers write
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"schema": "octantharvest-beatsaver-catalogue-capture/v1", "complete": True}))
    (arch / "archive_meta.json").write_text(json.dumps({"format": "beatmap-music-identity-beatsaver-archive-v2"}))


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k, v in {"ARCHIVE": "archive", "CATALOGUE": "cat.jsonl.gz", "STORE": "store", "WORKERS": "1",
                 "DEVICE": "cpu"}.items():
        monkeypatch.setenv(f"BV_{k}", v)
    (tmp_path / "octantvector.toml").write_text('[pools.popular]\nmin_upvotes = 100\nstandard = true\n')
    return config.load


def update(cfg, ids, **kw):
    log = io.StringIO()
    embed.run(cfg(), ids, log=log, **kw)
    return log.getvalue()


def planned(log, enc):
    return int(log.split(f"{enc} (")[1].split(": ")[1].split(" ")[0])


def test_ingest_tracks_states_and_pools(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    defs = {"all": dict(pools.KEYS), "popular": {**pools.KEYS, "min_upvotes": 100, "standard": True}}
    maps = [mapr("a", [version(H(1))]), mapr("b", [version(H(2)), version(H(3), "Uploaded")], up=50),
            mapr("c", [version(H(4), standard=False)])]
    n = led.ingest(maps, {H(1): 0}, {}, defs)
    assert n["new"] == 4 and n["pool:all"]["size"] == 3 and n["pool:popular"]["size"] == 1
    assert dict(led.db.execute("SELECT hash, state FROM versions")) == {
        H(1): "current", H(2): "current", H(3): "old", H(4): "current"}
    assert [h for h, _ in led.pool("all")] == [H(1), H(2), H(4)]  # versions with a zip first

    # a: new version replaces H(1); b: upvoted into "popular"; c: removed from the catalogue
    maps = [mapr("a", [version(H(5))]), mapr("b", [version(H(2)), version(H(3), "Uploaded")], up=150)]
    n = led.ingest(maps, {H(1): 0, H(5): 1}, {}, defs)
    states = dict(led.db.execute("SELECT hash, state FROM versions"))
    assert states[H(1)] == "old" and states[H(4)] == "deleted" and states[H(5)] == "current"
    assert n["old"] == 1 and n["deleted"] == 1 and n["new"] == 1
    assert n["pool:popular"] == {"size": 2, "entered": 2, "left": 1}
    assert {h for h, _ in led.pool("all")} == {H(2), H(5)}
    assert led.db.execute("SELECT left_at IS NOT NULL FROM pools WHERE pool='all' AND hash=?", (H(4),)).fetchone()[0]
    led.close()


def test_pool_definitions_from_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "octantvector.toml").write_text('[pools.fresh]\nmin_upvotes = 3\n[pools.distribution]\nstandard = true\n'
                                              '[pools.trial]\ninclude = "played.txt"\n')
    defs = pools.load()
    assert defs["fresh"] == {**pools.KEYS, "min_upvotes": 3} and defs["distribution"]["standard"] is True
    assert defs["trial"] == {**pools.KEYS, "sample": 20000, "include": "played.txt"}  # the built-in sample is kept
    for bad in ('[pools.bad]\nmax_upvotes = 3\n', '[pools.bad]\nsample = "many"\n', '[pools.bad]\nstandard = 1\n'):
        (tmp_path / "octantvector.toml").write_text(bad)
        with pytest.raises(GenerationError):
            pools.load()


def test_sample_is_stable_and_include_adds_maps(tmp_path):
    current = {f"k{i}": H(i) for i in range(1, 41)}
    eligible = sorted(current.items())
    d = {**pools.KEYS, "sample": 10}
    picked = pools.members(d, eligible, current)
    assert len(picked) == 10 and pools.members(d, list(reversed(eligible)), current) == picked  # order does not matter
    assert pools.members({**d, "seed": "other"}, eligible, current) != picked
    # a map keeps its place when it gets a new version, and a map that leaves does not reshuffle the others
    keys = lambda cur, hs: {k for k, h in cur.items() if h in hs}
    moved = {**current, "k1": H(99)}
    assert keys(moved, pools.members(d, sorted(moved.items()), moved)) == keys(current, picked)
    gone = next(k for k, h in eligible if h not in picked)
    assert pools.members(d, [kv for kv in eligible if kv[0] != gone], current) == picked
    (tmp_path / "played.txt").write_text(f"# my maps\nK2  # by map id, any case\n{H(3).upper()}\n{H(77)}\nnope\n")
    rest = [kv for kv in eligible if kv[1] not in picked]
    inc = pools.members({**pools.KEYS, "sample": 0, "include": str(tmp_path / "played.txt")}, rest, current)
    assert inc == {H(2), H(3)}  # an id or hash that is not a current version adds nothing
    assert pools.members({**d, "include": str(tmp_path / "played.txt")}, eligible, current) == picked | {H(2), H(3)}
    with pytest.raises(GenerationError):
        pools.members({**pools.KEYS, "include": str(tmp_path / "missing.txt")}, eligible, current)


def test_update_computes_only_what_is_missing(tmp_path, cfg):
    maps = [mapr(f"k{i}", [version(H(i))]) for i in range(1, 6)] + [mapr("low", [version(H(9))], up=5)]
    zips = [H(i) for i in range(1, 6)] + [H(9)]
    write_world(tmp_path, maps, zips)
    write_zip(tmp_path / "archive", H(4), notes=NOTES[:3])  # too few notes: chart_patterns skips it
    write_zip(tmp_path / "archive", H(5), raw=b"not a zip")  # unreadable
    ids = ["chart_patterns", "meta_tags", "npsnjs_std"]

    log = update(cfg, ids, pool="popular")
    assert planned(log, "chart_patterns") == 5 and planned(log, "meta_tags") == 5  # pool "popular": up >= 100
    led = Ledger(tmp_path / "store" / "ledger.sqlite")
    res = dict(led.db.execute("SELECT hash, result FROM computed WHERE encoder='chart_patterns@3'"))
    assert res == {H(1): "ok", H(2): "ok", H(3): "ok", H(4): "skip", H(5): "error"}  # results per map version
    assert Store(get("chart_patterns"), tmp_path / "store").done() == {f"{H(i)}:Standard:Expert" for i in (1, 2, 3)}
    assert led.db.execute("SELECT zip_sha256 IS NOT NULL FROM versions WHERE hash=?", (H(1),)).fetchone()[0]
    led.close()

    log = update(cfg, ids, pool="popular")  # nothing changed: nothing to compute, errors are not retried
    assert all(planned(log, i) == 0 for i in ids)

    maps[0]["tags"] = ["rock"]  # a catalogue field meta_tags declares
    maps[1]["stats"]["upvotes"] = 300  # not an input of any view
    write_world(tmp_path, maps, zips)
    log = update(cfg, ids, pool="popular")
    assert planned(log, "meta_tags") == 1 and planned(log, "npsnjs_std") == 0 and planned(log, "chart_patterns") == 0

    log = update(cfg, ["chart_patterns"], pool="popular", retry_errors=True)
    assert planned(log, "chart_patterns") == 1


def test_update_adopts_vectors_computed_before_the_ledger(tmp_path, cfg):
    maps = [mapr(f"k{i}", [version(H(i))]) for i in range(1, 4)]
    write_world(tmp_path, maps, [H(i) for i in range(1, 4)])
    s = Store(get("npsnjs_std"), tmp_path / "store")
    s.add(H(1), get("npsnjs_std").embed_map(maps[0], maps[0]["versions"][0]))
    s.flush()
    log = update(cfg, ["npsnjs_std"])
    assert "1 vectors already in the store" in log and planned(log, "npsnjs_std") == 2


def test_update_rejects_a_changed_zip(tmp_path, cfg):
    write_world(tmp_path, [mapr("k1", [version(H(1))])], [H(1)])
    update(cfg, ["chart_patterns"])
    led = Ledger(tmp_path / "store" / "ledger.sqlite")
    with led.db:
        led.db.execute("UPDATE versions SET zip_sha256='other'")  # as if another zip had been read first
    led.close()
    log = update(cfg, ["swing_flow"])  # another view of the same zip
    assert "zip differs" in log
    led = Ledger(tmp_path / "store" / "ledger.sqlite")
    assert led.db.execute("SELECT result FROM computed WHERE encoder LIKE 'swing_flow@%'").fetchone()[0] == "error"
    led.close()


def test_update_needs_views_and_a_known_pool(tmp_path, cfg):
    write_world(tmp_path, [mapr("k1", [version(H(1))]), mapr("k2", [version(H(2))], up=5)], [H(1), H(2)])
    with pytest.raises(GenerationError, match="name the views"):
        update(cfg, [])
    with pytest.raises(GenerationError, match="distribution, popular, trial"):
        update(cfg, ["meta_tags"], pool="nope")
    assert planned(update(cfg, ["meta_tags"]), "meta_tags") == 2  # default: the distribution pool, every map
    log = update(cfg, ["swing_style"])  # a derived view: its source is computed first
    assert planned(log, "swing_flow") == 2 and "swing_style:" in log


def test_commit_order_skips_a_line_cut_short(tmp_path):
    """A write interrupted mid-line leaves the journal's last line cut short; its zip is not committed."""
    from octantvector.generation import archive
    (tmp_path / "download.journal.jsonl").write_text(
        json.dumps({"event": "committed", "hash": H(1), "commit_seq": 0}) + "\n"
        + '{"event": "committed", "hash": "' + H(2)[:10])
    assert archive.commit_order(tmp_path) == {H(1): 0}


def test_commit_order_reads_only_committed_events(tmp_path):
    """Another event whose line holds "committed" (here as a value of another field) is not a commit."""
    from octantvector.generation import archive
    lines = [{"event": "committed", "hash": H(1), "commit_seq": 0},
             {"event": "hdd_write_error", "hash": H(2), "status": "committed"}]
    (tmp_path / "download.journal.jsonl").write_text("".join(json.dumps(j) + "\n" for j in lines))
    assert archive.commit_order(tmp_path) == {H(1): 0}
