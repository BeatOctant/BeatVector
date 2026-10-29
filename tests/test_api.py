"""Contract 1 (OctantHarvest to OctantVector), generation side (generation/api.py): everything comes in as arguments, nothing from the config file or
the environment, on the synthetic world of test_ledger (no models, no GPU)."""
import json
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from octantvector import generation as gen
from octantvector.generation import GenerationError
from octantvector.common.dbformat import VersionError
from octantvector.runtime.db import open_db

from test_ledger import H, NOTES, mapr, version, write_world, write_zip

VIEWS = ["chart_patterns@3", "meta_tags@2"]
PARAMS = {"workers": 1, "device": "cpu", "pools": {"distribution": {"min_upvotes": 0}}}
AT = datetime(2026, 10, 2, tzinfo=timezone.utc)


@pytest.fixture
def harvest(tmp_path, monkeypatch):
    """The crawl and a store, with a config file and environment that would break everything if they were read."""
    rng = np.random.default_rng(0)
    maps = []
    for i in range(1, 25):
        v = version(H(i))
        v["diffs"][0].update(nps=float(rng.uniform(1, 9)), njs=float(rng.uniform(10, 22)))
        v["createdAt"] = "2025-01-01T00:00:00Z"
        maps.append(mapr(f"k{i}", [v], tags=[["pop"], ["rock"], ["tech", "metal"]][i % 3]))
    write_world(tmp_path, maps, [H(i) for i in range(1, 25)])
    for i in range(1, 25):
        notes = [dict(n, _lineIndex=(n["_lineIndex"] + i) % 4, _time=n["_time"] * (1 + i % 3)) for n in NOTES]
        write_zip(tmp_path / "archive", H(i), notes=notes)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "octantvector.toml").write_text('workers = "not a number"\n[pools.distribution]\nmin_upvotes = 9999\n')
    for k in ("STORE", "ARCHIVE", "CATALOGUE", "WORKERS"):
        monkeypatch.setenv(f"BV_{k}", "/nonexistent")
    crawl = {"archive": tmp_path / "archive", "catalogue": tmp_path / "cat.jsonl.gz"}
    return tmp_path, crawl, tmp_path / "work", maps


def test_compute_is_resumable_and_reads_only_its_arguments(harvest):
    tmp_path, crawl, store, _ = harvest
    got = gen.compute(crawl, store, VIEWS, PARAMS)
    assert got["computed"] == 48 and got["failed"] == 0 and set(got["views"]) == set(VIEWS)
    assert (store / "ledger.sqlite").exists() and not (tmp_path / "store").exists()
    assert gen.compute(crawl, store, VIEWS, PARAMS) == {"computed": 0, "failed": 0, "skipped": 0, "deferred": 0,
                                                         "views": {}}


def test_compute_reports_the_maps_the_gpu_had_no_room_for_and_the_next_call_computes_them(harvest, monkeypatch):
    from octantvector.generation import embed
    _, crawl, store, _ = harvest
    run_heavy = embed._run_heavy
    monkeypatch.setattr(embed, "_run_heavy", lambda *a: {"chart_patterns": 24})  # as if none fitted: nothing recorded
    got = gen.compute(crawl, store, VIEWS, PARAMS)
    assert got["deferred"] == 24 and got["views"]["chart_patterns@3"] == {"deferred": 24} and got["computed"] == 24
    monkeypatch.setattr(embed, "_run_heavy", run_heavy)
    got = gen.compute(crawl, store, VIEWS, PARAMS)
    assert got["computed"] == 24 and got["deferred"] == 0 and set(got["views"]) == {"chart_patterns@3"}


def test_compute_covers_the_named_pool(harvest):
    _, crawl, store, _ = harvest
    params = {**PARAMS, "pools": {"few": {"sample": 5, "seed": 1}}}
    assert gen.compute(crawl, store, VIEWS, params, pool="few")["computed"] == 10
    assert gen.compute(crawl, store, VIEWS, params, pool="few")["computed"] == 0
    assert gen.compute(crawl, store, VIEWS, params)["computed"] == 38  # distribution: only the rest is computed


def test_bad_arguments_are_refused(harvest):
    _, crawl, store, _ = harvest
    with pytest.raises(GenerationError, match="no such setting|unknown setting"):
        gen.compute(crawl, store, VIEWS, {**PARAMS, "wrokers": 2})
    with pytest.raises(GenerationError, match="pass paths"):
        gen.compute(crawl, store, VIEWS, {**PARAMS, "store": "elsewhere"})
    with pytest.raises(GenerationError, match="computes chart_patterns@3"):
        gen.compute(crawl, store, ["chart_patterns@1"], PARAMS)
    for views in (None, [], "meta_tags@2"):  # views are always named
        with pytest.raises(GenerationError, match="views"):
            gen.compute(crawl, store, views, PARAMS)
    with pytest.raises(GenerationError, match="unknown pool 'experiment'; known: distribution, trial"):
        gen.compute(crawl, store, VIEWS, PARAMS, pool="experiment")
    with pytest.raises(GenerationError, match="unknown params"):
        gen.build_index([], store / "pub", {"partSizeMb": 1})


def test_unknown_input_formats_are_refused_before_reading(harvest):
    tmp_path, crawl, store, _ = harvest
    meta = tmp_path / "archive" / "archive_meta.json"
    meta.write_text(json.dumps({"format": "beatmap-music-identity-beatsaver-archive-v3"}))
    with pytest.raises(VersionError, match="archive format"):
        gen.compute(crawl, store, VIEWS, PARAMS)
    meta.write_text(json.dumps({"format": "beatmap-music-identity-beatsaver-archive-v2"}))
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(VersionError, match="catalogue schema None"):
        gen.compute(crawl, store, VIEWS, PARAMS)
    assert not store.exists()


def test_generation_errors_are_one_type_and_never_system_exit(harvest, monkeypatch):
    """Reaudit decision 2: every refusal of the generation side is a GenerationError (VersionError included, still a
    dbformat.VersionError for the runtime's callers), and no module raises SystemExit."""
    import ast
    from pathlib import Path
    from octantvector.common import dbformat
    _, crawl, store, _ = harvest
    assert issubclass(gen.VersionError, GenerationError) and issubclass(gen.VersionError, dbformat.VersionError)

    def bare(*_, **__):  # a format error from below the api, e.g. a base DB of a newer schema
        raise dbformat.VersionError("schemaVersion", "2.0", "1.x", where="base")
    monkeypatch.setattr(gen.api.embed, "run", bare)
    with pytest.raises(GenerationError, match="base: schemaVersion '2.0'") as got:
        gen.compute(crawl, store, VIEWS, PARAMS)
    assert isinstance(got.value, gen.VersionError) and (got.value.found, got.value.reads) == ("2.0", "1.x")

    gone = crawl["archive"].parent / "no-such-version"  # a version folder that is not there: a format error below
    for call in (lambda: gen.build_index([gone], gone.parent / "pub", {"part_size_mb": 1}),
                 lambda: gen.sign(gone, gone.parent / "key"),
                 lambda: gen.drift(store, gone, PARAMS)):
        with pytest.raises(GenerationError, match="no readable manifest.json"):
            call()

    root = Path(gen.__file__).parent
    raised = [f"{p.relative_to(root)}:{n.lineno}" for p in root.rglob("*.py") for n in ast.walk(ast.parse(p.read_text()))
              if isinstance(n, ast.Name) and n.id == "SystemExit"]
    assert raised == []


def test_the_dev_cli_turns_a_generation_error_into_its_exit_message(tmp_path, monkeypatch):
    from octantvector import cli
    monkeypatch.chdir(tmp_path)  # no octantvector.toml: the defaults

    def refuse(cfg, a):
        raise GenerationError("no such thing")
    monkeypatch.setattr(cli, "cmd_check", refuse)  # the dev CLI turns it into its exit message, in one place
    with pytest.raises(SystemExit, match="octantvector: no such thing"):
        cli.main(["check"])


def test_calibrate_release_drift_and_index(harvest, monkeypatch):
    tmp_path, crawl, store, maps = harvest
    from octantvector.common import trusted_keys
    key = gen.keygen(tmp_path / "maintainer.pem")  # what OctantHarvest's approval signs with; its public key as if bundled
    monkeypatch.setattr(trusted_keys, "KEYS", {key["keyId"]: key["publicKey"]})
    gen.compute(crawl, store, VIEWS, PARAMS)
    cal = gen.calibrate(store, VIEWS, PARAMS)
    assert set(cal["views"]) == set(VIEWS) and json.dumps(cal)  # plain JSON
    first = gen.release(crawl, store, VIEWS, cal, None, tmp_path / "releases", PARAMS, built_at=AT)
    assert first["diff"] is None and first["published"] == first["full"] and first["dbVersion"] == "2026.10.02"
    assert open_db(first["full"]).version == "2026.10.02"
    assert max(gen.drift(store, first["full"], PARAMS).values()) < 1e-6  # nothing changed
    lacking = tmp_path / "lacking"  # a DB with a view whose plugin is not installed here: not silently skipped
    lacking.mkdir()
    m = json.loads((first["full"] / "manifest.json").read_text())
    (lacking / "manifest.json").write_text(json.dumps({**m, "views": {**m["views"], "gone_plugin@1": {}}}))
    with pytest.raises(GenerationError, match="gone_plugin@1; install the plugin"):
        gen.drift(store, lacking, PARAMS)
    related = tmp_path / "related"  # a relation view is given to the release, not computed: drift passes it over
    shutil.copytree(first["full"], related)
    (related / "manifest.json").write_text(json.dumps(
        {**m, "views": {**m["views"], "listenbrainz_artists@20261010": {"kind": "relation", "rows": "maps"}}}))
    assert set(gen.drift(store, related, PARAMS)) == set(m["views"])

    for m in maps[:12]:
        m["tags"] = ["metal", "anime", "jazz"]  # half the maps change their tags: meta_tags moves
    write_world(tmp_path, maps, [H(i) for i in range(1, 25)])
    assert gen.compute(crawl, store, VIEWS, PARAMS)["views"] == {"meta_tags@2": {"ok": 12}}
    assert gen.drift(store, first["full"], PARAMS)["meta_tags@2"] > 0.1

    second = gen.release(crawl, store, VIEWS, None, first["full"], tmp_path / "releases", PARAMS,
                         built_at=AT + timedelta(days=1))
    assert second["published"] == second["diff"] and second["diff"].name == "2026.10.03.diff"

    for r in (first, second):
        assert gen.sign(r["published"], tmp_path / "maintainer.pem").name == "manifest.sig"
    pub = tmp_path / "pub"
    idx = json.loads(gen.build_index([first["published"]], pub, {"part_size_mb": 0.01}).read_text())
    assert [e["dbVersion"] for e in idx["entries"]] == ["2026.10.02"]  # only what was passed in
    idx = json.loads(gen.build_index([first["published"], second["published"]], pub,
                                     {"part_size_mb": 0.01, "publish_url": "https://example.org/db"}).read_text())
    full, diff = idx["entries"]
    assert (full["kind"], diff["kind"], diff["baseDbVersion"]) == ("full", "diff", "2026.10.02")
    assert len(full["parts"]) > 1 and full["parts"][0]["url"] == "https://example.org/db/2026.10.02.full.tar.part000"
    with pytest.raises(GenerationError, match="nothing to index"):
        gen.build_index([second["published"]], tmp_path / "pub2", {"part_size_mb": 0.01})
    with pytest.raises(GenerationError, match="part_size_mb is required"):  # the caller's setting, no default here
        gen.build_index([first["published"]], tmp_path / "pub3")
