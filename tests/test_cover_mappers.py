"""Views of the cover image (kind "image"), views whose cosine is kept as it is (norm "none") and the relation view of
the maps' BeatSaver accounts (generation/mappers.py), on the synthetic world of test_ledger (no models, no GPU)."""
import gzip
import json
import zipfile

import numpy as np
import pytest

from octantvector import generation as gen
from octantvector.generation import GenerationError, mappers
from octantvector.generation.encoders import REGISTRY, Encoder, register
from octantvector.generation.normalisation import normalise
from octantvector.plugin import ImageEncoder
from octantvector.runtime.db import open_db
from octantvector.runtime.recommend import recommend

from test_api import AT, PARAMS, VIEWS, harvest  # noqa: F401  (fixture)
from test_ledger import H, INFO, NOTES


@pytest.fixture(autouse=True)
def clean_registry():
    before = dict(REGISTRY)
    yield
    REGISTRY.clear(); REGISTRY.update(before)


def _cover_zip(root, h, cover, name="Cover.JPG", info_name="cover.jpg"):
    p = root / "maps" / h[:2] / f"{h}.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("Info.dat", json.dumps({**INFO, "_coverImageFilename": info_name}))
        z.writestr("Expert.dat", json.dumps({"_version": "2.2.0", "_notes": NOTES, "_obstacles": []}))
        if cover is not None:
            z.writestr(name, cover)


class Covers(ImageEncoder):
    """A cover's vector is (its first byte, its length); a cover starting with "x" cannot be decoded."""
    id, dim, batch = "fake_cover", 2, 3

    def prep(self, data):
        if data[:1] == b"x":
            raise ValueError("not an image")
        return np.array([data[0], len(data)], np.float32)

    def embed_many(self, prepped):
        self.groups = getattr(self, "groups", []) + [len(prepped)]
        return [x for x in prepped]


def test_an_image_encoder_gets_the_cover_named_in_info_dat_and_skips_maps_without_one(harvest):  # noqa: F811
    _, crawl, store, _ = harvest
    for i in range(1, 25):
        _cover_zip(crawl["archive"], H(i), None if i > 20 else b"x" if i == 20 else bytes([i]) * i)
    _cover_zip(crawl["archive"], H(1), b"\x01", name="other.png", info_name="missing.jpg")  # first image instead
    register(Covers)
    got = gen.compute(crawl, store, ["fake_cover"], PARAMS)
    assert got["views"]["fake_cover@1"] == {"ok": 19, "skip": 4, "error": 1}
    from octantvector.generation.store import Store
    hs, X = Store(Covers(), store).load()
    vec = dict(zip(hs, X.tolist()))
    assert vec[H(1)] == [1, 1] and vec[H(7)] == [7, 7] and H(21) not in vec


def test_an_image_encoder_describes_maps_only():
    with pytest.raises(ValueError, match="describe maps"):
        register(type("chart_cover", (Covers,), {"id": "chart_cover", "rows": "charts"}))


def test_norm_none_keeps_the_cosine_of_a_one_hot_view():
    class OneHot(Encoder):
        id, dim, norm, inputs = "one_hot", 3, "none", ("meta:tags",)
    x = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 2], [0, 1, 0]], np.float32)
    y = normalise(OneHot(), x)
    assert np.allclose(y @ y[0], [1, 1, 0, 0])  # 1 for the same category, 0 otherwise: nothing is centred
    with pytest.raises(ValueError, match="norm must be one of"):
        register(type("bad_norm", (Encoder,), {"id": "bad_norm", "dim": 1, "inputs": ("meta:tags",), "norm": "l1"}))


def _uploaders(tmp_path, owners):
    """Rewrite the catalogue so map i was uploaded by owners[i] (None: no uploader)."""
    rows = [json.loads(line) for line in gzip.open(tmp_path / "cat.jsonl.gz", "rt")]
    for n, m in enumerate(rows, 1):
        m.pop("uploader", None)
        if owners.get(n) is not None:
            m["uploader"] = {"id": owners[n], "name": f"name{owners[n]}"}
    with gzip.open(tmp_path / "cat.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(m) + "\n" for m in rows)


def test_the_mapper_relation_is_built_from_the_catalogue_when_views_name_it(harvest):  # noqa: F811
    tmp_path, crawl, store, _ = harvest
    _uploaders(tmp_path, {**{i: 100 + i % 4 for i in range(1, 23)}, 23: None})
    rows = [json.loads(line) for line in gzip.open(tmp_path / "cat.jsonl.gz", "rt")]
    rows[23]["uploader"], rows[23]["collaborators"] = {"id": 7}, [{"id": 101}, {"name": "no id"}]
    with gzip.open(tmp_path / "cat.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(m) + "\n" for m in rows)
    gen.compute(crawl, store, VIEWS, PARAMS)
    cal = gen.calibrate(store, VIEWS, PARAMS)
    first = gen.release(crawl, store, VIEWS + [mappers.NAME], cal, None, tmp_path / "releases", PARAMS, built_at=AT)
    db = open_db(first["full"])
    m = db.manifest["views"][mappers.KEY]
    assert m["kind"] == "relation" and m["role"] == "mapper" and "credits" not in m
    rel = db.relations[mappers.KEY]
    assert sorted(rel.member_pos) == ["beatsaver:100", "beatsaver:101", "beatsaver:102", "beatsaver:103", "beatsaver:7"]
    assert rel.members_of(db.pos[H(23)]) == ()  # no uploader: no members
    close = dict(zip(rel.rows.tolist(), rel.closeness(db.pos[H(1)]).tolist()))
    assert close[db.pos[H(5)]] == 1 and close[db.pos[H(2)]] == 0 and close[db.pos[H(24)]] == 1  # 24: a collaborator
    seed = {"hash": H(1), "characteristic": "Standard", "difficulty": "Expert"}
    out = recommend(db, [], {"seed": seed, "count": 5, "weights": {"meta_tags": 0.1, mappers.NAME: 1}})
    assert {it["hash"] for it in out["items"]} < {H(5), H(9), H(13), H(17), H(21), H(24)}

    second = gen.release(crawl, store, VIEWS, None, first["full"], tmp_path / "releases", PARAMS,
                         built_at=AT.replace(day=3))
    assert mappers.KEY not in open_db(second["full"]).manifest["views"]  # not named: not kept either
    with pytest.raises(GenerationError, match="builds beatsaver_mappers@1"):
        gen.release(crawl, store, VIEWS + ["beatsaver_mappers@2"], None, second["full"], tmp_path / "releases", PARAMS)


def test_compute_and_calibrate_leave_the_mapper_relation_out(harvest):  # noqa: F811
    _, crawl, store, _ = harvest
    assert set(gen.compute(crawl, store, VIEWS + [mappers.NAME], PARAMS)["views"]) == set(VIEWS)
    assert set(gen.calibrate(store, VIEWS + [mappers.NAME], PARAMS)["views"]) == set(VIEWS)
    assert gen.models([mappers.NAME]) == []


def test_the_dev_cli_releases_encoder_ids_and_relation_names_and_refuses_another_version(harvest, monkeypatch):  # noqa: F811
    from octantvector import cli
    from octantvector.generation import config, dbrelease
    monkeypatch.setattr(config, "load", lambda: None)
    got = []
    monkeypatch.setattr(dbrelease, "release", lambda cfg, views, *a, relations=(), **k: got.append((views, relations)))
    cli.main(["release", *VIEWS, "beatsaver_mappers@1"])  # VIEWS name id@version
    assert got == [([v.split("@")[0] for v in VIEWS], [mappers.NAME])]  # versions checked, then the ids release takes
    with pytest.raises(SystemExit, match="builds beatsaver_mappers@1"):
        cli.main(["release", *VIEWS, "beatsaver_mappers@2"])
    assert len(got) == 1


def test_a_cover_that_cannot_be_read_from_the_zip_fails_only_the_image_encoders(tmp_path, monkeypatch):
    from octantvector.generation import embed
    from octantvector.generation.encoders import get
    h = "ab" + "0" * 38
    (tmp_path / "maps" / "ab").mkdir(parents=True)
    p = tmp_path / "maps" / "ab" / f"{h}.zip"
    with zipfile.ZipFile(p, "w") as z:  # stored, so the cover's bytes can be broken in place
        z.writestr("Info.dat", json.dumps({**INFO, "_coverImageFilename": "cover.jpg"}))
        z.writestr("Expert.dat", json.dumps({"_version": "2.2.0", "_notes": NOTES, "_obstacles": []}))
        z.writestr("cover.jpg", b"COVERBYTES")
    p.write_bytes(p.read_bytes().replace(b"COVERBYTES", b"BROKENBYTE"))  # the CRC no longer matches
    monkeypatch.setitem(embed._WORKER, "archive", tmp_path)
    monkeypatch.setitem(embed._WORKER, "encs", {"chart_patterns": get("chart_patterns"), "fake_cover": Covers()})
    _, _, sha, out, err = embed._work((h, ["chart_patterns", "fake_cover"]))
    assert err is None and sha
    assert isinstance(out["fake_cover"], zipfile.BadZipFile)
    assert list(out["chart_patterns"]) == [f"{h}:Standard:Expert"]
