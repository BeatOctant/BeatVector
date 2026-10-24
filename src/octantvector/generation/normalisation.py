"""The normalisation of each view: what turns an encoder's vectors into the values of the distributed DB
(dbrelease.py) and of the experiment index (search.py). norm "center" centres a view and L2-normalises it, "zscore"
z-scores and L2-normalises it, with the statistics of a calibration (calibration.py) or of the rows at hand; "none"
only L2-normalises it (its statistics are kept for drift but not used).

This code decides what a value means as much as an encoder's definition does, but no fingerprint covers it, so it
has its own VERSION: bump it whenever stats() or normalise() would give other numbers (OctantSpecs octantvector/law.md
2.2). A calibration records the version it was made with and its id covers it, so after a bump the old calibrations
are refused and calibrating again gives a new calibration id; a DB normalised another way is never a diff of the
previous one (dbformat.diffable). tests/test_search_views.py pins the output of each version.
"""
import numpy as np

from ..common.dbformat import normalisation
from .errors import GenerationError

VERSION = 1  # the first; calibrations do not write it, so their ids are the ones they had before it existed


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def stats(enc, x):
    """Normalisation statistics of one view: {"norm", "mean", "std"} (std only for z-scored views)."""
    x = np.nan_to_num(np.asarray(x, np.float64))
    return {"norm": enc.norm, "mean": x.mean(0), "std": x.std(0) if enc.norm == "zscore" else None}


def normalise(enc, x, st=None):
    x = np.nan_to_num(np.asarray(x, np.float64))
    st = st or stats(enc, x)
    if enc.norm == "zscore":
        return l2((x - st["mean"]) / (np.asarray(st["std"]) + 1e-6))
    if enc.norm == "none":
        return l2(x)
    return l2(x - st["mean"])  # learned embeddings share a large common component


def check(calibration):
    """Stop unless `calibration` was made with this normalisation."""
    if normalisation(calibration) != VERSION:
        raise GenerationError(f"calibration {calibration['id']} was made with normalisation version "
                              f"{normalisation(calibration)}, this OctantVector normalises with version {VERSION}: "
                              "calibrate again")


def calibrated(calibration, enc):
    """enc's statistics in `calibration`: same normalisation version, id@version, fingerprint and norm, or stop."""
    check(calibration)
    key = f"{enc.id}@{enc.version}"
    st = calibration["views"].get(key)
    if st is None:
        raise GenerationError(f"calibration {calibration['id']} has no statistics for {key}")
    if st["fingerprint"] != enc.fingerprint():
        raise GenerationError(f"calibration {calibration['id']} was made for another definition of {key}")
    if st["norm"] != enc.norm:
        raise GenerationError(f"calibration {calibration['id']} normalises {key} by {st['norm']}, "
                              f"the encoder by {enc.norm}")
    return st
