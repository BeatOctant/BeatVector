"""Reading side of the distributed DB (OctantSpecs contracts/: contract 1 (OctantCompass to OctantVector) `open_db`, contract 2 (DB build to recommendation core) format; `search` is
this runtime's own, not in the contract).

open_db checks the manifest before anything is used: the schemaVersion major must be one this code reads, it must be
a full DB (a diff is applied first with apply_diff), and every file must match its SHA-256 in the manifest. Unknown manifest keys, and views whose rows are not ones this version
knows, are ignored (a newer minor version). The signature is not checked here: apply_update checks it on what comes
from outside, and a full made by applying a diff has none (OctantSpecs contracts/, 契約5（OctantHarvest to OctantCompass）署名). The vectors are already normalised with the DB's calibration, so a search is a dot product; the index is
built here on first open and is never distributed.

Views have map rows (one vector per map version: the song, the metadata) or chart rows (one vector per chart). A
chart view is searched with a chart and only returns charts of the same play style (OctantSpecs constitution.md,
principle 2; common/charts.py).

A relation view (contract 2, ビューの種類) holds no vectors: what each map belongs to (its artists) and a score for
pairs of members. Its closeness to a reference map is 1 when they share a member, else the best score of a pair of
their members (the larger of the two directions), else 0 (Relation). `search` is for vector views only.
"""
from pathlib import Path

import numpy as np

from ..common import charts as chartids
from ..common.dbformat import (ROW_KINDS, VIEW_KINDS, FormatError, chart_keys, listed, read_charts, read_hashes,
                               read_manifest, read_relation, read_superseded, read_view, view_kind)
from ..common.dbformat import VersionError, apply_diff  # noqa: F401  (part of the API)


DBError = FormatError  # the DB cannot be used (OctantCompass reports `degraded`)
# VersionError, a DBError: a manifest or index declares a version this OctantVector does not read (OctantSpecs contracts/, 契約の版)


class Relation:
    """A relation view as arrays: the map rows that have members (sorted), each row's member indices (CSR: `ptr`,
    `idx`) and, per member, its neighbours with the larger score of the two directions (CSR: `nptr`, `nidx`, `nscore`)."""

    def __init__(self, members, pairs):
        names = sorted(set(members["member"]) | set(pairs["a"]) | set(pairs["b"]))
        self.member_pos = {m: n for n, m in enumerate(names)}
        n = len(names)
        r = np.asarray(members["row"], np.int64)
        m = np.array([self.member_pos[x] for x in members["member"]], np.int64)
        order = np.lexsort((m, r))
        r, m = r[order], m[order]
        self.rows, starts = np.unique(r, return_index=True)
        self.ptr = np.append(starts, len(r)).astype(np.int64)
        self.idx = m
        self.row_pos = {int(x): k for k, x in enumerate(self.rows)}
        a = np.array([self.member_pos[x] for x in pairs["a"]], np.int64)
        b = np.array([self.member_pos[x] for x in pairs["b"]], np.int64)
        sc = np.asarray(pairs["score"], np.float32)
        i, j, s = np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([sc, sc])
        keep = i != j
        i, j, s = i[keep], j[keep], s[keep]
        order = np.lexsort((-s, j, i))  # per (i, j), the larger score first
        i, j, s = i[order], j[order], s[order]
        first = np.ones(len(i), bool)
        first[1:] = (i[1:] != i[:-1]) | (j[1:] != j[:-1])
        i, j, s = i[first], j[first], s[first]
        self.nptr = np.searchsorted(i, np.arange(n + 1)).astype(np.int64)
        self.nidx, self.nscore = j, s
        self.n_members = n

    def members_of(self, row):
        k = self.row_pos.get(row)
        return () if k is None else tuple(int(x) for x in self.idx[self.ptr[k]:self.ptr[k + 1]])

    def closeness(self, row):
        """Closeness of every row with members (in `rows` order) to the map `row`; None when it has no members."""
        ms = self.members_of(row)
        if not ms:
            return None
        per_member = np.zeros(self.n_members, np.float32)
        for a in ms:
            nb = slice(self.nptr[a], self.nptr[a + 1])
            per_member[self.nidx[nb]] = np.maximum(per_member[self.nidx[nb]], self.nscore[nb])
        per_member[list(ms)] = 1.0
        return np.maximum.reduceat(per_member[self.idx], self.ptr[:-1])

    def reference(self, rows):
        """The mean closeness over the reference maps that have members (several references: their mean, as a vector
        view takes their mean vector); None when none has."""
        found = [c for c in (self.closeness(r) for r in rows) if c is not None]
        return np.mean(found, 0) if found else None


class DB:
    def __init__(self, path, manifest, hashes, chart_keys, play_styles, superseded, views, skipped, chart_metrics=None,
                 relations=None):
        self.path, self.manifest = path, manifest
        self.hashes = hashes
        self.pos = {h: n for n, h in enumerate(hashes)}
        self.chart_keys = chart_keys  # charts.parquet order
        self.chart_pos = {k: n for n, k in enumerate(chart_keys)}
        self.play_styles = play_styles  # per chart row
        self.superseded = superseded
        self.views = views  # {id@version: (vectors float32 (n, dim), row numbers into maps or charts)}
        self.relations = relations or {}  # {id@version: Relation}
        self.rows = {k: manifest["views"][k]["rows"] for k in list(views) + list(self.relations)}
        self.skipped = skipped
        self.chart_metrics = chart_metrics or {}  # charts.parquet numeric columns as float arrays (NaN when empty)
        self._chart_maps = self._chart_map_array = self._hash_array = None
        self._row_pos, self._styles = {}, {}

    @property
    def version(self):
        return self.manifest["dbVersion"]

    def info(self):
        return _info(self.manifest, list(self.views) + list(self.relations), self.skipped,
                     {**{k: (len(r), int(x.shape[1])) for k, (x, r) in self.views.items()},
                      **{k: (len(rel.rows), 0) for k, rel in self.relations.items()}})

    def current(self, h):
        """The current version of the map version `h`: h itself, or the version superseded.parquet points it to;
        None when this DB has neither (what that means is the caller's)."""
        if h in self.pos:
            return h
        cur = self.superseded.get(h)
        return cur if cur in self.pos else None

    def chart_map_rows(self):
        """Map row of each chart row (None when the chart's map version is not in maps.parquet)."""
        if self._chart_maps is None:
            self._chart_maps = [self.pos.get(chartids.key_hash(k)) for k in self.chart_keys]
        return self._chart_maps

    def chart_map_array(self):
        """chart_map_rows as an array, -1 where the chart's map version is not in maps.parquet."""
        if self._chart_map_array is None:
            self._chart_map_array = np.array([-1 if m is None else m for m in self.chart_map_rows()], np.int64)
        return self._chart_map_array

    def hash_array(self):
        """The map hashes as an array, for sorting map rows by hash."""
        if self._hash_array is None:
            self._hash_array = np.array(self.hashes)
        return self._hash_array

    def views_of_role(self, role):
        """Vector views of a role."""
        return [k for k in self.views if self.manifest["views"][k].get("role") == role]

    def roles(self):
        """{view: role} of every view this runtime reads, vector and relation views."""
        return {k: self.manifest["views"][k].get("role") for k in list(self.views) + list(self.relations)}

    def artists(self):
        """The relation view of role `artist` (the first, by key), or None (OctantSpecs law 3.4)."""
        keys = sorted(k for k in self.relations if self.manifest["views"][k].get("role") == "artist")
        return self.relations[keys[0]] if keys else None

    def genres(self):
        """The genre words this DB can filter on: {word: (view, dimension)} from the views with role `genre`
        (contract 2; the word list itself is not decided yet, so a DB may have none)."""
        out = {}
        for k in self.views_of_role("genre"):
            for i, word in enumerate(self.manifest["views"][k].get("genres", [])):
                out.setdefault(word, (k, i))
        return out

    def styles_in_view(self, view):
        """Play style of each row of a chart view."""
        if view not in self._styles:
            self._styles[view] = np.array(self.play_styles, dtype=object)[self.views[view][1]]
        return self._styles[view]

    def row_in_view(self, view, row):
        if view not in self._row_pos:
            self._row_pos[view] = {int(r): n for n, r in enumerate(self.views[view][1])}
        return self._row_pos[view].get(row)


def _reads(v):
    """Whether this runtime reads a view of the manifest; open_db skips the others (skippedViews): rows or a kind of a
    newer minor version."""
    return v.get("rows") in ROW_KINDS and view_kind(v) in VIEW_KINDS and (view_kind(v) == "vector" or v["rows"] == "maps")


def _info(m, read, skipped, sizes=None):
    """What DB.info() and read_info() say of a DB: from its manifest m, the views read (the vector views, then the
    relation views) and the skipped ones. sizes: {view: (rows, dim)} as read; without them (read_info) each view has
    no "rows" and the dim the manifest gives (None when it gives none)."""
    views = {}
    for k in read:
        v = m["views"][k]
        given = v.get("dim") if type(v.get("dim")) is int else None
        rows, dim = sizes[k] if sizes is not None else (None, 0 if view_kind(v) == "relation" else given)
        views[k] = {"role": v.get("role"), "kind": view_kind(v), "rowKind": v["rows"],
                    **({} if rows is None else {"rows": rows}), "dim": dim, "credits": _credits(v)}
    # The genre words: those of the vector views with role `genre` (DB.genres)
    genres = {w for k in read if view_kind(m["views"][k]) == "vector" and m["views"][k].get("role") == "genre"
              for w in m["views"][k].get("genres", [])}
    return {"schemaVersion": m["schemaVersion"], "dbVersion": m["dbVersion"], "builtAt": m.get("builtAt"),
            "calibrationId": m["calibrationId"], "counts": m.get("counts", {}), "views": views,
            "skippedViews": skipped, "genres": sorted(genres),
            "notice": m["notice"] if isinstance(m.get("notice"), str) else None}


def _credits(view):
    """A view's credits as given, keeping only text lines (the DB may come from anyone: contract 5 署名)."""
    c = view.get("credits")
    return [x for x in c if isinstance(x, str)] if isinstance(c, list) else []


TEXT_COLS = ("hash", "characteristic", "difficulty", "play_style")


def _numbers(values):
    """True when a charts.parquet column holds numbers (or nothing): a metric, known (nps, njs, pass, tech) or added by
    a newer DB (its other columns are carried as they are, dbformat.py)."""
    return all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) for v in values)


def open_db(path):
    path = Path(path)
    manifest = read_manifest(path, "full")  # schemaVersion, kind and every file's SHA-256

    hashes = read_hashes(path, manifest)
    ch = read_charts(listed(path, manifest, "charts.parquet"))
    keys = chart_keys(ch)
    superseded = read_superseded(path, manifest)
    n_rows = {"maps": len(hashes), "charts": len(keys)}
    views, relations, skipped = {}, {}, []
    for key, v in manifest.get("views", {}).items():
        if not _reads(v):
            skipped.append(key)
            continue
        if view_kind(v) == "relation":
            relations[key] = Relation(*read_relation(path, manifest, key, len(hashes)))
            continue
        x, rows = read_view(path, manifest, key, n_rows[v["rows"]])
        views[key] = (x.astype(np.float32), rows)
    metrics = {c: np.array([np.nan if v is None else v for v in ch[c]], np.float64)
               for c in ch if c not in TEXT_COLS and _numbers(ch[c])}
    return DB(path, manifest, hashes, keys, list(ch["play_style"]), superseded, views, skipped, metrics, relations)


def read_info(path):
    """What DB.info() says of the full DB at path, from its manifest alone: each view without its "rows" (counted
    when the view is read). No other file is read or checked (open_db checks them all), so it is quick and shows a DB
    that is not open, as it says it is. DBError when there is no readable manifest of a full DB, VersionError when
    its schemaVersion is one this OctantVector does not read."""
    m = read_manifest(path, "full", check_files=False)
    try:
        views = m.get("views", {})
        read = [k for kind in VIEW_KINDS for k, v in views.items() if _reads(v) and view_kind(v) == kind]
        return _info(m, read, [k for k in views if k not in read])
    except (KeyError, TypeError, ValueError, AttributeError) as ex:
        raise FormatError(f"{path}: the manifest is not one of a full DB ({type(ex).__name__}: {ex})") from None


def _current(db, h):
    """-> (current version hash, the old hash it replaced or None)."""
    cur = db.current(h)
    if cur is None:
        raise KeyError(f"{h} is not in DB {db.version}")
    return cur, (h if cur != h else None)


def search(db, query, view, k=10):
    """Top-k of one view (`id@version`).

    A map view takes a version hash and returns {"hash", "rank", "score"}. A chart view takes a chart, as a chart key
    or (hash, characteristic, difficulty), and returns {"hash", "characteristic", "difficulty", "rank", "score"} for
    charts of the same play style only. An old version is replaced by the current version of the same map (for a
    chart: the same characteristic and difficulty of the current version, which may be a remade chart; "replaced"
    says it happened). -> {"view", "query", "replaced", "results"}; empty results when the query has no vector in
    that view."""
    if view not in db.views:
        raise KeyError(f"{view} is not in DB {db.version}; views: {', '.join(db.views)}")
    x, rows = db.views[view]
    if db.rows[view] == "charts":
        h, c, d = chartids.split(query) if isinstance(query, str) else tuple(query)
        cur, replaced = _current(db, h)
        q_key = chartids.key(cur, c, d)
        if q_key not in db.chart_pos:
            raise KeyError(f"{chartids.key(h, c, d)} is not in DB {db.version}")
        out = {"view": view, "query": q_key, "replaced": replaced and chartids.key(h, c, d), "results": []}
        q_row = db.chart_pos[q_key]
        style = db.play_styles[q_row]
        allowed = db.styles_in_view(view) == style
    else:
        cur, replaced = _current(db, query)
        out = {"view": view, "query": cur, "replaced": replaced, "results": []}
        q_row = db.pos[cur]
        allowed = None
    q = db.row_in_view(view, q_row)
    if q is None:
        return out
    s = x @ x[q]
    if allowed is not None:
        s[~allowed] = -np.inf
    s[q] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return out
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top], kind="stable")]
    for n, j in enumerate(top, 1):
        if db.rows[view] == "charts":
            h, c, d = chartids.split(db.chart_keys[int(rows[j])])
            out["results"].append({"hash": h, "characteristic": c, "difficulty": d, "rank": n, "score": float(s[j])})
        else:
            out["results"].append({"hash": db.hashes[int(rows[j])], "rank": n, "score": float(s[j])})
    return out
