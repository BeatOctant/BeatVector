"""Normalisation, calibration, missing views, per-view top-k and per-request weights (normalisation.py, search.py,
calibration.py)."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from octantvector.common import dbformat
from octantvector.generation import GenerationError, calibration, normalisation
from octantvector.generation.encoders import REGISTRY, Encoder
from octantvector.generation.search import Index
from octantvector.generation.store import Store


class A(Encoder):
    id, dim, kind, norm, inputs = "toy_a", 4, "map", "center", ("meta:a",)


class B(Encoder):
    id, dim, kind, norm, inputs = "toy_b", 3, "map", "zscore", ("meta:b",)


@pytest.fixture(autouse=True)
def toys():
    REGISTRY.update(toy_a=A, toy_b=B)
    yield
    REGISTRY.pop("toy_a"); REGISTRY.pop("toy_b")


def data(n=120, seed=0, drop_b=()):
    rng = np.random.default_rng(seed)
    hs = [f"{i:040x}" for i in range(n)]
    hb = [h for i, h in enumerate(hs) if i not in drop_b]
    return {"toy_a": (hs, rng.normal(size=(n, 4)) + 2.0), "toy_b": (hb, rng.normal(size=(len(hb), 3)) * 5)}


def stored(tmp_path, vec):
    for i, (hs, X) in vec.items():
        s = Store(REGISTRY[i](), tmp_path)
        for h, x in zip(hs, X):
            s.add(h, x)
        s.flush()


def test_calibration_fixes_coordinates_and_keeps_rankings(tmp_path):
    vec = data()
    stored(tmp_path, vec)
    cal = calibration.load(calibration.save(calibration.compute(tmp_path, ["toy_a", "toy_b"]), tmp_path))
    w = {"toy_a": 1, "toy_b": 1}
    plain = Index(w, tmp_path, device="cpu")
    fixed = Index(w, tmp_path, device="cpu", calibration=cal)
    np.testing.assert_allclose(plain.emb, fixed.emb, atol=1e-5)  # same rows: same coordinates, same rankings
    assert (plain.knn(5)[0] == fixed.knn(5)[0]).mean() > 0.99 and fixed.calibration_id == cal["id"]

    more = data(n=160, seed=0)  # first 120 rows identical, 40 new ones
    more = {i: (hs, np.r_[vec[i][1], X[120:] + 3]) for i, (hs, X) in more.items()}
    grown_fixed = Index(w, tmp_path, device="cpu", vectors=more, calibration=cal)
    grown_plain = Index(w, tmp_path, device="cpu", vectors=more)
    old = [grown_fixed.pos[h] for h in fixed.hashes]
    np.testing.assert_allclose(grown_fixed.emb[old], fixed.emb, atol=1e-5)  # existing rows did not move
    assert not np.allclose(grown_plain.emb[old], plain.emb, atol=1e-3)  # without calibration they do


def test_calibration_refuses_another_definition(tmp_path):
    stored(tmp_path, data())
    cal = calibration.compute(tmp_path, ["toy_a"])
    cal["views"]["toy_a@1"]["fingerprint"] = "sha256:other"
    with pytest.raises(GenerationError):
        Index({"toy_a": 1}, tmp_path, device="cpu", calibration=cal)
    with pytest.raises(GenerationError):
        Index({"toy_b": 1}, tmp_path, device="cpu", calibration=cal)  # no statistics for toy_b


# normalisation.py's numbers on fixed inputs, per normalisation.VERSION: changing them without a new version fails
# here (OctantSpecs octantvector/law.md 2.2). When they are meant to change, bump the version and add its digest.
GOLDEN = {1: "de4223331f29aadf"}


def test_the_normalisation_is_pinned_by_its_version():
    e = 2.0 ** -20  # a column whose spread is near the z-score's epsilon
    x = np.array([[1, 0, 1, 1e6], [2, -1, 1 + e, -1e6], [3, 5, 1, 0], [4, 2, 1 + e, 250],
                  [5, np.nan, 1, 1e6], [6, 3, 1 + e, 0], [7, 1, 1, -3], [0.5, 0, 1, 0]])
    given = {"mean": np.array([0.5, 0, 1, 0]), "std": np.array([2, 1, 0.25, 1000])}  # the last row is its mean
    out = []
    for norm in ("center", "zscore"):
        enc = SimpleNamespace(norm=norm)
        st = normalisation.stats(enc, x)
        out += [st["mean"], np.zeros(0) if st["std"] is None else st["std"], normalisation.normalise(enc, x),
                normalisation.normalise(enc, x, given)]
    # rounded to float32, so that last-bit differences between machines do not change the digest
    got = hashlib.sha256(b"".join(np.asarray(a, np.float32).tobytes() for a in out)).hexdigest()[:16]
    assert got == GOLDEN[normalisation.VERSION]


def test_a_new_normalisation_version_needs_a_new_calibration(tmp_path, monkeypatch):
    stored(tmp_path, data())
    old = calibration.compute(tmp_path, ["toy_a"])
    assert "normalisation" not in old  # version 1 is not written: ids made before it existed stay the same
    monkeypatch.setattr(normalisation, "VERSION", 2)
    new = calibration.compute(tmp_path, ["toy_a"])
    assert new["normalisation"] == 2 and new["id"] != old["id"] and new["views"] == old["views"]
    with pytest.raises(GenerationError, match="calibrate again"):
        Index({"toy_a": 1}, tmp_path, device="cpu", calibration=old)
    with pytest.raises(GenerationError, match="calibrate again"):
        calibration.extend(old, tmp_path, ["toy_a", "toy_b"])  # its statistics would mix two normalisations
    Index({"toy_a": 1}, tmp_path, device="cpu", calibration=new)
    assert "normalisation changed" in dbformat.diffable({"views": {}}, old, {"views": {}, "calibration": new})


def test_calibration_refuses_another_norm(tmp_path, monkeypatch):
    stored(tmp_path, data())
    cal = calibration.compute(tmp_path, ["toy_a"])
    monkeypatch.setattr(A, "norm", "zscore")  # not in the fingerprint, but it changes every value
    with pytest.raises(GenerationError, match="center"):
        Index({"toy_a": 1}, tmp_path, device="cpu", calibration=cal)


def test_reweight_keeps_maps_missing_a_view(tmp_path):
    vec = data(drop_b=range(10))  # the first 10 maps have no toy_b vector
    drop = Index({"toy_a": 1, "toy_b": 1}, tmp_path, device="cpu", vectors=vec)
    keep = Index({"toy_a": 1, "toy_b": 1}, tmp_path, device="cpu", vectors=vec, missing="reweight", min_weight=0.5)
    strict = Index({"toy_a": 1, "toy_b": 3}, tmp_path, device="cpu", vectors=vec, missing="reweight", min_weight=0.5)
    assert len(drop) == 110 and len(keep) == 120 and len(strict) == 110  # toy_a alone is 1/4 of the weight there
    a, b, c = keep.pos[vec["toy_a"][0][0]], keep.pos[vec["toy_a"][0][50]], keep.pos[vec["toy_a"][0][60]]
    s = keep._scores(np.array([a, b]))
    cos = keep.view_cosines(b, c)
    assert cos["toy_b"] is not None and keep.view_cosines(a, c)["toy_b"] is None
    assert s[1, c] == pytest.approx(0.5 * cos["toy_a"] + 0.5 * cos["toy_b"], abs=1e-5)
    assert s[0, c] == pytest.approx(0.5 * keep.view_cosines(a, c)["toy_a"], abs=1e-5)  # the missing view counts as 0
    idx, sc = keep.query_rows([a], 5, csls=True)
    assert np.all(np.isfinite(sc)) and a not in idx[0]


def test_reweight_without_missing_views_matches_drop(tmp_path):
    vec = data()
    w = {"toy_a": 1, "toy_b": 3}
    drop = Index(w, tmp_path, device="cpu", vectors=vec)
    keep = Index(w, tmp_path, device="cpu", vectors=vec, missing="reweight")
    np.testing.assert_allclose(drop._scores(np.arange(10)), keep._scores(np.arange(10)), atol=1e-5)
    assert (drop.knn(5, csls=True)[0] == keep.knn(5, csls=True)[0]).mean() > 0.99


def test_search_views_and_request_weights(tmp_path):
    vec = data(drop_b=range(10))
    ix = Index({"toy_a": 1, "toy_b": 1}, tmp_path, device="cpu", vectors=vec, missing="reweight")
    only_a = Index({"toy_a": 1}, tmp_path, device="cpu", vectors=vec)
    rows = np.array([0, 50])
    per = ix.search_views(rows, 5)
    assert set(per) == {"toy_a", "toy_b"}
    a_idx = [[only_a.pos[ix.hashes[j]] for j in r] for r in per["toy_a"][0]]
    np.testing.assert_array_equal(a_idx, only_a.query_rows([only_a.pos[ix.hashes[r]] for r in rows], 5, csls=False)[0])
    missing_b = {ix.pos[h] for h in vec["toy_a"][0][:10]}
    assert np.all(per["toy_b"][0][0] == -1)  # row 0 has no toy_b: empty result
    assert not set(per["toy_b"][0][1]) & missing_b  # maps without toy_b are never returned for it

    by_request = ix.query_rows(rows, 5, csls=False, weights={"toy_a": 1, "toy_b": 0})[0]
    np.testing.assert_array_equal(by_request, per["toy_a"][0])  # weights can change per request
