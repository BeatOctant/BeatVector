"""The next distributable DB version from the working store, as a full or as a diff from the previous version
(OctantSpecs contracts/, contract 2 (DB build to recommendation core) and contract 5 (OctantHarvest to OctantCompass);
octantvector/design.md, 配布物). Called by the contract 1 (OctantHarvest to OctantVector) `release` (api.py) and by the
development command `octantvector release`.

Rows are the current versions of the distribution pool in the ledger; old versions get only a row in
superseded.parquet that points to the current version of the same map. Each view holds the rows it has (a map whose
audio failed simply has no row in that view), normalised with the calibration (normalisation.py) and stored as
float16. A view computed from the charts has one row per chart ("charts" rows); a chart view stored per map is
refused, because the DB must not merge what each chart is into its map (OctantSpecs constitution.md, principle 2).

Everything goes to the folder `out` the caller names; approving and uploading are the caller's (OctantHarvest's):
  <dbVersion>/       the complete DB of that version (always written; the base of the next diff)
  <dbVersion>.diff/  the diff from the previous version `base`, when one is possible
A diff is made when every view the previous version also has keeps its fingerprint and calibration statistics,
the normalisation version is the same, the previous version holds nothing this OctantVector does not know (a diff would
lose it: dbformat.py), and fewer than `full_every` diffs were made since the last full; otherwise only the full is
published. A diff may add a view (a view pack: a view newly named), replace one (the encoder's version went
up) or remove one (no longer named); the calibration is then the previous one extended with the new views
(calibration.extend). The files' formats are in dbformat.py.

Not filled in yet: Pass and Tech of charts.parquet (the swing simulation is not in the ledger yet), so those
columns are empty and manifest.chartMetrics is {}. The version is signed afterwards, when it is approved (api.sign).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import artists as artist_files, calibration as calibrations, catalogue, mappers
from ..common import charts, dbformat
from .encoders import get
from .errors import GenerationError
from .ledger import Ledger
from .normalisation import calibrated, normalise
from .store import Store


def build(cfg, view_ids, calibration, log=sys.stderr, relations=()):
    """Content of a DB (dbformat.py) from the ledger, the catalogue and the store. relations: the relation views built
    from the catalogue to put in it (mappers.NAME)."""
    encs = {i: get(i) for i in view_ids}
    for i, e in encs.items():
        if e.role_name() == "chart" and e.row_kind() != "charts":
            raise GenerationError(f"{i} describes charts but stores one vector per map; only views with chart rows "
                                  "can go into the DB (OctantSpecs constitution.md, principle 2)")
    sts = {i: calibrated(calibration, e) for i, e in encs.items()}  # same normalisation and definition, or stop
    with Ledger.of(cfg.store) as led:
        hashes = sorted(led.pool_hashes("distribution"))
        old = led.old_versions()
    if not hashes:
        raise GenerationError("the distribution pool is empty; run `octantvector update` first")
    recs = catalogue.records_for(cfg.require("catalogue"), hashes)
    hashes = [h for h in hashes if h in recs]  # in the ledger but no longer in this catalogue snapshot: left out
    c = dbformat.empty()
    for h in hashes:
        m, v = recs[h]
        md = m.get("metadata", {})  # no song titles: they are never features
        c["maps"][h] = (md.get("songAuthorName"), md.get("levelAuthorName"), _num(md.get("duration")),
                        v.get("createdAt") or m.get("uploaded"))
        c["charts"][h] = [(d.get("characteristic"), d.get("difficulty"), _num(d.get("nps")),
                           _num(d.get("njs")), None, None, charts.play_style(d.get("characteristic")))
                          for d in v.get("diffs", [])]
    current_of_key = {recs[h][0]["id"]: h for h in hashes}
    c["superseded"] = {h: current_of_key[k] for h, k in old if k in current_of_key}
    want = set(hashes)
    chart_keys = {charts.key(h, r[0], r[1]) for h in hashes for r in c["charts"][h]}
    for i, e in encs.items():
        key = f"{e.id}@{e.version}"
        hs, X = Store(e, cfg.store).load()
        keep = [n for n, h in enumerate(hs) if charts.key_hash(h) in want]
        x = normalise(e, X[keep], sts[i]).astype(np.float16) if keep else np.zeros((0, e.dim), np.float16)
        meta = {"role": e.role_name(), "rows": e.row_kind(), "dim": int(x.shape[1]), "dtype": "float16",
                "norm": e.norm, "fingerprint": e.fingerprint(),
                "sources": [f"{get(s).id}@{get(s).version}" for s in e.sources()]}
        if e.all_credits():
            meta["credits"] = e.all_credits()
        if meta["role"] == "genre":  # one word per dimension: the words the request's `genre` can name (contract 2)
            meta["genres"] = list(getattr(e, "names", ()))
        c["views"][key] = {"meta": meta,
                           "vecs": {charts.catalogue_spelling(hs[n]): x[j] for j, n in enumerate(keep)}}
        print(f"{key}: {len(keep)} of {len(hashes)} rows", file=log)
        if e.row_kind() == "charts":  # a chart the catalogue does not list has no row in charts.parquet
            unlisted = sorted(k for k in c["views"][key]["vecs"] if k not in chart_keys)
            if unlisted:
                print(f"{key}: {len(unlisted)} charts left out, their difficulty is not in the catalogue: "
                      f"{', '.join(unlisted[:20])}{' ...' if len(unlisted) > 20 else ''}", file=log)
    if cfg.artists is not None:
        rel = artist_files.relation(cfg.artists, hashes)
        if rel is None:
            print(f"{cfg.artists}: no {artist_files.MAP_FILE} and {artist_files.SIMILAR_FILE}; no artist view", file=log)
        else:
            key, c["relations"][key] = rel
            print(f"{key}: {len(rel[1]['members'])} of {len(hashes)} maps have artists, {len(rel[1]['pairs'])} pairs",
                  file=log)
    if mappers.NAME in relations:
        key, c["relations"][key] = mappers.relation(recs, hashes)
        print(f"{key}: {len(c['relations'][key]['members'])} of {len(hashes)} maps have accounts", file=log)
    c["calibration"] = calibrations.plain(calibration)
    return c


def latest(root):
    """Manifest and folder of the newest complete full version under `root` (None if there is none)."""
    best = None
    for p in Path(root).glob("*/manifest.json"):
        if p.parent.name.endswith(".partial"):
            continue
        m = json.loads(p.read_text())
        if m.get("kind", "full") == "full" and (best is None or m["builtAt"] > best[0]["builtAt"]):
            best = (m, p.parent)
    return best


def release(cfg, view_ids, calibration=None, full=False, built_at=None, log=sys.stderr, out=None, base=...,
            presets=None, notice=None, relations=()):
    """Next version of the views `view_ids` and of the relation views `relations` built from the catalogue
    (mappers.NAME), written under `out` (default <store>/releases). base: folder of the
    previous full version, None when there is none (default: the newest full under `out`). calibration: default
    the base's, extended over the distribution pool with the views it lacks (saved under <store>/calibration).
    presets: the hands (a list, or the path of a JSON file holding one) to write into the manifest; None keeps the
    base's. Every hand, kept ones too, must be one the runtime can serve on this DB, or the release stops (OctantSpecs
    plans/presets.md §8). notice: the DB's terms and credits (Markdown text) for the manifest; None keeps the base's.
    -> (full folder, diff folder or None)."""
    root = Path(out) if out is not None else Path(cfg.store) / "releases"
    if base is ...:
        prev = latest(root)
    elif base is not None:
        prev = (json.loads((Path(base) / "manifest.json").read_text()), Path(base))
        if prev[0].get("kind", "full") != "full":
            raise GenerationError(f"{base} is a diff; the base of a release is a full version")
    else:
        prev = None
    if not view_ids:
        raise GenerationError("name the views to put in the DB")
    if calibration is None:
        if prev is None:
            raise GenerationError("the first release needs a calibration")
        calibration = calibrations.load(prev[1] / prev[0]["calibrationFile"])
        with Ledger.of(cfg.store) as led:
            pool = led.pool_hashes("distribution")
        extended = calibrations.extend(calibration, cfg.store, view_ids, pool, "distribution")
        if extended is not calibration:
            print(f"calibration {extended['id']} = {calibration['id']} + new views; wrote "
                  f"{calibrations.save(extended, cfg.store)}", file=log)
            calibration = extended
    content = build(cfg, view_ids, calibration, log, relations)
    artist = any(r["meta"].get("role") == "artist" for r in content["relations"].values())
    if not artist and prev is not None:  # no artist files given: the previous version's artist view
        for key, r in dbformat.read_relations(prev[1], prev[0]).items():
            if str(r["meta"].get("source", "")).startswith("listenbrainz:"):
                r["meta"]["credits"] = list(artist_files.CREDITS)  # with today's credits
                content["relations"][key] = r
    if presets is None and prev is not None:
        presets = prev[0].get("presets")
    content["presets"] = _hands(presets, content) if presets is not None else None
    if notice is None and prev is not None:
        notice = prev[0].get("notice")
    if notice is not None and not (isinstance(notice, str) and notice.strip()):
        raise GenerationError("the notice must be a non-empty text")
    content["notice"] = notice

    built_at = built_at or datetime.now(timezone.utc).replace(microsecond=0)
    stamp = built_at.isoformat().replace("+00:00", "Z")
    version = built_at.strftime("%Y.%m.%d")
    n = 1
    while (root / version).exists() or (root / f"{version}.diff").exists():
        n += 1
        version = f"{built_at.strftime('%Y.%m.%d')}.{n}"

    base = dbformat.read_content(prev[1]) if prev and not full else None
    why = "--full" if full else "first release" if prev is None else dbformat.diffable(prev[0], base["calibration"], content)
    if why is None and dbformat.unknown(base):  # what this OctantVector cannot write again would be lost by a diff
        why = f"the previous version holds what this OctantVector does not know: {'; '.join(dbformat.unknown(base))}"
    if why is None and prev[0].get("diffsSinceFull", 0) + 1 > cfg.full_every:
        why = f"{cfg.full_every} diffs since the last full (setting full_every)"
    diff = None
    if why is None:
        diff = dbformat.write_diff(root / f"{version}.diff", base, prev[0], content, version, stamp)
        dm = json.loads((diff / "manifest.json").read_text())
        packs = [k + (f" (replaces {v['replaces']})" if v.get("replaces") else "")
                 for k, v in dm["views"].items() if v.get("change") == "added"]
        print(f"wrote {diff}: diff from {prev[0]['dbVersion']}: " + ", ".join(f"{k} {v}" for k, v in dm["counts"].items())
              + (f"; views added {', '.join(packs)}" if packs else "")
              + (f"; views removed {', '.join(dm['removedViews'])}" if dm["removedViews"] else ""), file=log)
    out = dbformat.write_full(root / version, content, version, stamp,
                              0 if diff is None else prev[0].get("diffsSinceFull", 0) + 1)
    print(f"wrote {out}: {len(content['maps'])} maps, views {', '.join(list(content['views']) + list(content['relations']))}"
          + ("" if diff else f" (publish as a full: {why})"), file=log)
    return out, diff


def _hands(presets, content):
    """The hands checked against the content: what the runtime would leave out stops the release here instead."""
    from ..common.hands import check_hands
    hands = json.loads(Path(presets).read_text()) if isinstance(presets, (str, Path)) else presets
    if not isinstance(hands, list):
        raise GenerationError("the presets must be a JSON list of hands")
    views = {k: v["meta"].get("role") for part in ("views", "relations") for k, v in content[part].items()}
    genres = {g for v in content["views"].values() for g in v["meta"].get("genres", [])}
    bad = check_hands(hands, views, genres)
    if bad:
        raise GenerationError("hands this DB cannot serve: " + "; ".join(f"{k}: {why}" for k, why in bad.items()))
    return hands or None


def _num(x):
    return None if x is None else float(x)

