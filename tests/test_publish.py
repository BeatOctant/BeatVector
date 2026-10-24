"""The contract 5 (OctantHarvest to OctantCompass) index and the host's side of it (generation/index.py, runtime/update.py) on the synthetic world of
test_release. Which versions are approved is the caller's (OctantHarvest's): the index holds only the folders passed in."""
import json
import tarfile
import warnings

import pytest

from octantvector.generation import GenerationError, index
from octantvector.runtime import update
from octantvector.common import dbformat
from octantvector.runtime import VersionError
from octantvector.runtime.db import DBError, apply_diff, open_db

from octantvector.common import signing, trusted_keys
from octantvector.generation import api

from test_release import AT, next_day, same_content, test_world, world  # noqa: F401
from test_ledger import cfg  # noqa: F401


@pytest.fixture
def signed(tmp_path_factory, monkeypatch):
    """A key trusted for the test (as if bundled); -> sign(folder, ...) with it."""
    key = tmp_path_factory.mktemp("key") / "maintainer.pem"
    k = api.keygen(key)
    monkeypatch.setattr(trusted_keys, "KEYS", {k["keyId"]: k["publicKey"]})

    def sign(*folders):
        for f in folders:
            api.sign(f, key)
    return sign


def test_index_lists_approved_versions_and_the_host_reaches_the_newest(world, cfg, tmp_path, monkeypatch, signed):
    _, _, _, base = world
    signed(base)
    pub = tmp_path / "store" / "releases" / "publish"
    with pytest.raises(GenerationError):
        index.build([], pub, 0.01)  # nothing approved yet
    test_world.maps[0]["tags"] = ["jazz"]
    new, diff = next_day(tmp_path, cfg)
    with pytest.raises(DBError, match="no manifest.sig"):
        index.build([base, diff], pub, 0.01)  # an unsigned version is never indexed
    signed(diff)
    with pytest.raises(GenerationError):
        index.build([diff], pub, 0.01)  # a diff without its full
    idx = json.loads(index.build([base], pub, 0.01).read_text())  # ~10 KB parts: every file is split
    assert [e["dbVersion"] for e in idx["entries"]] == ["2026.10.02"]  # the diff is not approved yet
    idx = json.loads(index.build([base, diff], pub, 0.01).read_text())
    full, d = idx["entries"]
    assert (full["kind"], d["kind"], d["baseDbVersion"]) == ("full", "diff", "2026.10.02")
    assert len(full["parts"]) > 1 and full["parts"][0]["url"] is None and full["signature"] is None

    assert update.plan_update(idx, None) == [full, d] and update.plan_update(idx, "2026.10.02") == [d]
    assert update.plan_update(idx, new.name) == []
    host = tmp_path / "host"
    got_full = update.unpack(full, [pub / p["name"] for p in full["parts"]], host / "dl")
    got_diff = update.unpack(d, [pub / p["name"] for p in d["parts"]], host / "dl")
    applied = apply_diff(got_full, got_diff, host / new.name)
    same_content(dbformat.read_content(applied), dbformat.read_content(new))
    assert open_db(applied).version == new.name

    bad = pub / full["parts"][1]["name"]
    bad.write_bytes(b"x" + bad.read_bytes()[1:])
    with pytest.raises(DBError, match="does not match the index"):
        update.unpack(full, [pub / p["name"] for p in full["parts"]], host / "dl2")


def test_urls_and_a_newer_schema(world, tmp_path, signed):
    _, _, _, base = world
    signed(base)
    idx = json.loads(index.build([base], tmp_path / "pub", 1900, "https://example.org/db/").read_text())
    assert idx["entries"][0]["parts"][0]["url"] == "https://example.org/db/2026.10.02.full.tar.part000"
    idx["entries"][0]["schemaVersion"] = "2.0"  # a host that reads only 1.x keeps what it has
    assert update.plan_update(idx, "2026.10.01") == []
    for bad in ({**idx, "indexVersion": "2.0"}, {k: v for k, v in idx.items() if k != "indexVersion"}):
        with pytest.raises(VersionError, match="indexVersion"):  # an index this OctantVector cannot read says so
            update.plan_update(bad, None)


def test_apply_update_reaches_the_newest_and_leaves_the_base_alone(world, cfg, tmp_path, monkeypatch, signed):
    _, _, _, base = world
    test_world.maps[0]["tags"] = ["jazz"]
    new, diff = next_day(tmp_path, cfg)
    signed(base, diff)
    pub = tmp_path / "store" / "releases" / "publish"
    idx = json.loads(index.build([base, diff], pub, 0.01).read_text())
    host = tmp_path / "host"

    def fetched(entries):
        return [(e, [pub / p["name"] for p in e["parts"]]) for e in entries]

    got = update.apply_update(fetched(update.plan_update(idx, None)), None, host / "a")  # full + diff
    same_content(dbformat.read_content(got), dbformat.read_content(new))
    local = update.apply_update(fetched(update.plan_update(idx, None)[:1]), None, host / "local")
    before = sorted(p.name for p in local.rglob("*"))
    got = update.apply_update(fetched(update.plan_update(idx, "2026.10.02")), local, host / "b")  # diff only
    assert open_db(got).version == new.name and sorted(p.name for p in local.rglob("*")) == before

    entries = fetched(update.plan_update(idx, "2026.10.02"))
    bad = entries[0][1][0]
    bad.write_bytes(b"x" + bad.read_bytes()[1:])
    with pytest.raises(DBError, match="does not match the index"):
        update.apply_update(entries, local, host / "c")
    assert not (host / "c").exists() and open_db(local).version == "2026.10.02"
    assert {p.name for p in host.iterdir()} == {"a", "local", "b"}  # no work folders left behind


def test_the_signature_names_the_author_and_does_not_block(world, cfg, tmp_path, monkeypatch, signed):
    """OctantSpecs contracts/, 契約5 署名: manifest.json canonicalised with JCS, Ed25519, manifest.sig next to it. A version
    not signed by a known key is applied with a SignatureWarning (2026-10-11)."""
    _, _, _, base = world
    signed(base)
    sig = json.loads((base / "manifest.sig").read_text())
    assert sig["algorithm"] == "Ed25519" and list(trusted_keys.KEYS) == [sig["keyId"]]
    assert "manifest.sig" not in json.loads((base / "manifest.json").read_text())["files"]
    m = json.loads((base / "manifest.json").read_text())
    (base / "manifest.json").write_text(json.dumps(m, indent=4, sort_keys=False))  # layout is not signed
    assert signing.inspect(base)["status"] == "signed"
    assert open_db(base).version == "2026.10.02"
    with pytest.raises(FileExistsError):
        api.keygen(tmp_path / "k.pem") and api.keygen(tmp_path / "k.pem")  # a key is never overwritten

    pub = tmp_path / "pub"
    idx = json.loads(index.build([base], pub, 0.01).read_text())
    files = [(e, [pub / p["name"] for p in e["parts"]]) for e in update.plan_update(idx, None)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", signing.SignatureWarning)
        assert open_db(update.apply_update(files, None, tmp_path / "host" / "ok")).version == "2026.10.02"

    monkeypatch.setattr(trusted_keys, "KEYS", {})  # signed, but by a key this OctantVector does not bundle
    assert signing.inspect(base)["status"] == "unknownKey"
    with pytest.warns(signing.SignatureWarning, match="does not know"):
        assert open_db(update.apply_update(files, None, tmp_path / "host" / "unknown")).version == "2026.10.02"

    other = tmp_path / "other.pem"
    k = api.keygen(other)
    monkeypatch.setattr(trusted_keys, "KEYS", {k["keyId"]: k["publicKey"]})
    sig2 = {**json.loads((base / "manifest.sig").read_text()), "keyId": k["keyId"]}  # claims a trusted key
    (base / "manifest.sig").write_text(json.dumps(sig2))
    assert signing.inspect(base)["status"] == "mismatch"
    with pytest.raises(DBError, match="does not match"):  # the index takes a signature only once it is verified
        index.build([base], tmp_path / "pub-mismatch", 0.01)
    api.sign(base, other)
    m["dbVersion"] = "2026.10.09"  # the manifest changed after signing
    (base / "manifest.json").write_text(json.dumps(m))
    assert signing.inspect(base)["status"] == "mismatch"


def test_development_trust_extra_keys_and_unsigned(world, tmp_path, monkeypatch):
    """OctantSpecs contracts/, 契約5 署名: apply_update's trust adds known keys and silences the warning for unsigned
    versions (development); build_index takes the same extra keys as trusted_keys."""
    _, _, _, base = world
    monkeypatch.setattr(trusted_keys, "KEYS", {})  # nothing bundled
    pub = tmp_path / "pub"
    with pytest.raises(DBError, match="no manifest.sig"):
        index.build([base], pub, 0.01)
    idx = json.loads(api.build_index([base], pub, {"part_size_mb": 0.01, "allow_unsigned": True}).read_text())
    files = [(e, [pub / p["name"] for p in e["parts"]]) for e in update.plan_update(idx, None)]
    host = tmp_path / "host"
    with pytest.warns(signing.SignatureWarning, match="no manifest.sig"):
        assert open_db(update.apply_update(files, None, host / "a")).version == "2026.10.02"
    with warnings.catch_warnings():
        warnings.simplefilter("error", signing.SignatureWarning)
        assert open_db(update.apply_update(files, None, host / "b", {"allowUnsigned": True})).version == "2026.10.02"

    dev = api.keygen(tmp_path / "dev.pem")  # a developer's own key, trusted only through the setting
    api.sign(base, tmp_path / "dev.pem")
    pub2 = tmp_path / "pub2"
    with pytest.raises(DBError, match="does not know"):  # the author's key is not bundled: name it to the index
        index.build([base], pub2, 0.01)
    dev_keys = {dev["keyId"]: dev["publicKey"]}
    idx = json.loads(api.build_index([base], pub2, {"part_size_mb": 0.01, "trusted_keys": dev_keys}).read_text())
    files = [(e, [pub2 / p["name"] for p in e["parts"]]) for e in update.plan_update(idx, None)]
    with pytest.warns(signing.SignatureWarning, match="does not know"):  # a present signature is still looked up
        update.apply_update(files, None, host / "c", {"allowUnsigned": True})
    trust = {"keys": dev_keys}
    with warnings.catch_warnings():
        warnings.simplefilter("error", signing.SignatureWarning)
        assert open_db(update.apply_update(files, None, host / "d", trust)).version == "2026.10.02"
    for bad in ({"keys": {"0" * 16: dev["publicKey"]}}, {"keys": {dev["keyId"]: "x"}}, {"allowUnsigend": True}):
        with pytest.raises(ValueError, match="trust"):
            update.apply_update(files, None, host / "e", bad)
    assert not (host / "e").exists()
    with pytest.raises(GenerationError, match="trusted_keys"):
        api.build_index([base], tmp_path / "pub3", {"part_size_mb": 0.01, "trusted_keys": {"0" * 16: dev["publicKey"]}})


def test_only_the_indexed_versions_need_a_known_signature(world, cfg, tmp_path, signed):
    """An older version passed in but left out of the index is not looked at: one signed by a key this OctantVector no
    longer bundles does not stop a newer full from being indexed."""
    _, _, _, base = world
    api.keygen(tmp_path / "old.pem")
    api.sign(base, tmp_path / "old.pem")  # a key no longer in the bundled list
    test_world.maps[0]["tags"] = ["jazz"]
    new, diff = next_day(tmp_path, cfg)
    signed(new, diff)
    idx = json.loads(index.build([base, new], tmp_path / "pub", 0.01).read_text())
    assert [e["dbVersion"] for e in idx["entries"]] == [new.name]
    with pytest.raises(DBError, match="does not know"):  # the diff needs its full, so the full is indexed and checked
        index.build([base, diff], tmp_path / "pub2", 0.01)


def test_bundled_keys_are_ed25519_public_keys_under_their_key_id():
    keys, allow_unsigned = signing.check_trust({"keys": trusted_keys.KEYS})
    assert keys == trusted_keys.KEYS and not allow_unsigned


BAD_VERSIONS = ("../x", "a/b", "a\\b", "..", ".", "/abs", "", "C:x", "a\0b", "a" * 101, None, 7)


def test_a_version_that_is_not_a_plain_folder_name_is_refused(tmp_path):
    """The index is not signed and its dbVersion names folders on the host: anything that could leave them is refused
    before any file is touched (reaudit decision 1)."""
    def entry(v, kind="full"):
        return {"dbVersion": v, "schemaVersion": dbformat.SCHEMA_VERSION, "kind": kind, "parts": [], "sha256": ""}

    def idx(*entries):
        return {"indexVersion": dbformat.INDEX_VERSION, "entries": list(entries)}

    good = entry("2026.10.14")
    assert update.plan_update(idx(good), None) == [good]
    for v in BAD_VERSIONS:
        with pytest.raises(DBError, match="not a plain folder name"):
            update.plan_update(idx(good, entry(v)), None)
        with pytest.raises(DBError, match="not a plain folder name"):
            update.unpack(entry(v), [], tmp_path / "dl")
        with pytest.raises(DBError, match="not a plain folder name"):
            update.apply_update([(entry(v), [])], None, tmp_path / "out")
    with pytest.raises(DBError, match="unknown kind"):
        update.plan_update(idx(entry("2026.10.14", "../x")), None)
    assert not (tmp_path / "dl").exists() and list(tmp_path.iterdir()) == []

    dl = tmp_path / "dl"  # a plain name that resolves elsewhere (a link left in the folder) is refused as well
    dl.mkdir()
    (dl / "2026.10.14.full.tar.partial").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(DBError, match="outside"):
        update.unpack(good, [], dl)
    assert not (tmp_path / "elsewhere").exists()


def test_a_full_handed_over_as_one_tar_is_imported_without_an_index(world, cfg, tmp_path, signed):
    """The Dev 版 hand-off (OctantSpecs contracts/distribution.md, 非公開（探求中）の扱い): `pack` writes a full as one tar,
    the same bytes as its index file before it is split, and import_archive checks and extracts it like an update."""
    _, _, _, base = world
    signed(base)
    tar = api.pack(base, tmp_path / "handoff")
    assert tar.name == "2026.10.02.full.tar"
    with pytest.raises(GenerationError, match="already exists"):
        api.pack(base, tmp_path / "handoff")
    idx = json.loads(index.build([base], tmp_path / "pub", 1900).read_text())
    assert idx["entries"][0]["sha256"] == dbformat.sha256(tar)

    host = tmp_path / "host"
    with warnings.catch_warnings():
        warnings.simplefilter("error", signing.SignatureWarning)
        got = update.import_archive(tar, host / "a")
    same_content(dbformat.read_content(got), dbformat.read_content(base))
    assert open_db(got).version == "2026.10.02"
    with pytest.raises(DBError, match="already exists"):
        update.import_archive(tar, host / "a")

    test_world.maps[0]["tags"] = ["jazz"]
    _, diff = next_day(tmp_path, cfg)
    with pytest.raises(GenerationError, match="not a full"):  # the receiver has nothing to apply a diff to
        api.pack(diff, tmp_path / "handoff")
    diff_tar = tmp_path / "diff.tar"
    index.write_tar(diff, diff_tar)
    with pytest.raises(DBError, match="not a full"):
        update.import_archive(diff_tar, host / "b")

    (base / "manifest.sig").unlink()
    unsigned = api.pack(base, tmp_path / "unsigned")
    with pytest.warns(signing.SignatureWarning, match="no manifest.sig"):
        update.import_archive(unsigned, host / "c")
    with warnings.catch_warnings():
        warnings.simplefilter("error", signing.SignatureWarning)
        update.import_archive(unsigned, host / "d", {"allowUnsigned": True})

    broken = tmp_path / "broken.tar"
    with tarfile.open(tar) as t:
        at = t.getmember("2026.10.02/maps.parquet").offset_data
    data = bytearray(tar.read_bytes())
    data[at] ^= 1  # the tar still reads; the file no longer matches the manifest
    broken.write_bytes(bytes(data))
    with pytest.raises(DBError, match="does not match the manifest"):
        update.import_archive(broken, host / "e")
    (tmp_path / "junk.tar").write_bytes(b"not a tar")
    with pytest.raises(DBError, match="not an uncompressed tar"):
        update.import_archive(tmp_path / "junk.tar", host / "f")
    assert {p.name for p in host.iterdir()} == {"a", "c", "d"}  # nothing left of the failures


def test_a_compressed_tar_is_refused_before_anything_is_extracted(world, tmp_path, signed):
    """A compressed archive could expand without bound and fill the disk (a tar bomb), so only the uncompressed tar
    that `pack` writes is accepted (notes/security-audit-2026-10-21.md, 7)."""
    import gzip
    import shutil

    _, _, _, base = world
    signed(base)
    tar = api.pack(base, tmp_path / "handoff")
    bomb = tmp_path / "2026.10.02.full.tar"
    with open(tar, "rb") as src, gzip.open(bomb, "wb") as dst:
        shutil.copyfileobj(src, dst)
    host = tmp_path / "host"
    with pytest.raises(DBError, match="uncompressed"):
        update.import_archive(bomb, host / "a")
    assert not (host / "a").exists()
    # the same bytes uncompressed still import
    update.import_archive(tar, host / "a")
