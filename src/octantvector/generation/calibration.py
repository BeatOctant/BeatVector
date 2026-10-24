"""Calibration: the normalisation statistics of each view, computed once over a pool and then fixed, so vectors
added later land in the same coordinate space (OctantSpecs octantvector/law.md 2.3).

A calibration is a JSON file <store>/calibration/<id>.json:
  {"id": "<sha256 of the content>", "pool": ..., "created_at": ..., "normalisation": <version, absent for 1>,
   "views": {"<id>@<version>": {"norm", "fingerprint", "n", "mean": [...], "std": [...] | null}}}
The statistics are only good for the normalisation version they were made with (normalisation.py); using them
checks it.
Search uses it with Index(..., calibration=load(path)). Only vectors of the same id@version and calibration id are
ever compared.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..common.charts import key_hash
from . import normalisation
from .encoders import get
from .errors import GenerationError
from .normalisation import stats
from .store import Store


def compute(store_root, view_ids, hashes=None, pool=None):
    """Statistics of each view over `hashes` (default every vector in its store); a chart view uses every chart of
    those map versions."""
    views = {}
    for i in view_ids:
        enc = get(i)
        hs, X = Store(enc, store_root).load()
        if hashes is not None:
            want = set(hashes)
            X = X[[n for n, h in enumerate(hs) if key_hash(h) in want]]
        if len(X) == 0:
            raise GenerationError(f"{i}: no vectors to calibrate on")
        st = stats(enc, X)
        views[f"{enc.id}@{enc.version}"] = _plain_stats({
            "norm": st["norm"], "fingerprint": enc.fingerprint(), "n": int(len(X)), "mean": st["mean"], "std": st["std"]})
    return _make(pool, views)


def extend(cal, store_root, view_ids, hashes=None, pool=None):
    """`cal` plus statistics of the views it lacks; the views it has keep theirs exactly, so a DB can add a view
    without moving the others (a view pack). -> a new calibration (new id), or `cal` itself if nothing is missing."""
    missing = [i for i in view_ids if f"{get(i).id}@{get(i).version}" not in cal["views"]]
    if not missing:
        return cal
    normalisation.check(cal)  # the statistics it keeps must mean what the new ones mean
    views = plain(cal)["views"]
    views.update(compute(store_root, missing, hashes, pool)["views"])
    return load_dict({**_make(cal.get("pool"), views), "extends": cal["id"]})


def _make(pool, views):
    body = {"pool": pool, "views": views}
    if normalisation.VERSION != 1:  # in the id, so a new normalisation is a new coordinate system
        body["normalisation"] = normalisation.VERSION
    cid = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return {"id": cid, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **body}


def plain(cal):
    """`cal` as JSON values: the statistics as lists, not arrays (what load_dict reads back)."""
    return {**cal, "views": {k: _plain_stats(st) for k, st in cal["views"].items()}}


def _plain_stats(st):
    return {**st, "mean": np.asarray(st["mean"]).tolist(),
            "std": None if st["std"] is None else np.asarray(st["std"]).tolist()}


def save(cal, store_root):
    p = Path(store_root) / "calibration" / f"{cal['id']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plain(cal)))
    return p


def load(path):
    return load_dict(json.loads(Path(path).read_text()))


def load_dict(cal):
    for st in cal["views"].values():
        st["mean"] = np.asarray(st["mean"])
        st["std"] = None if st["std"] is None else np.asarray(st["std"])
    return cal


def drift(store_root, cal, view_ids, hashes):
    """{view id@version: drift} of `cal`'s means against the current vectors of `hashes`: how far each view's mean
    moved, in units of the spread around the calibration mean (0 = no movement). Views `cal` lacks are left out."""
    want, out = set(hashes), {}
    for i in view_ids:
        e = get(i)
        key = f"{e.id}@{e.version}"
        st = cal["views"].get(key)
        if st is None:
            continue
        hs, X = Store(e, store_root).load()
        X = np.nan_to_num(np.asarray(X[[n for n, h in enumerate(hs) if key_hash(h) in want]], np.float64))
        if not len(X):
            continue
        mu = np.asarray(st["mean"], np.float64)
        spread = np.sqrt(((X - mu) ** 2).sum(1).mean()) + 1e-12
        out[key] = float(np.linalg.norm(X.mean(0) - mu) / spread)
    return out
