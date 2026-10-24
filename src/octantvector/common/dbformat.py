"""Files of the distributed DB (OctantSpecs contracts/, contract 2 (DB build to recommendation core)): writing a full DB, a diff against an earlier one,
and applying a diff. Generation (generation/dbrelease.py) and the runtime (apply_diff) share this code, so "a full"
and "the previous full + a diff" are written by the same function.

The content of a DB, independent of its files:
  maps        {hash: (artist, mapper, duration_s, published_at)}
  charts      {hash: [(characteristic, difficulty, nps, njs, pass, tech, play_style), ...]} in catalogue order
  superseded  {old version hash: current version hash of the same map}
  views       {"id@version": {"meta": {...}, "vecs": {key: float16 vector}}}; the key is the version hash for a
              view with "maps" rows and the chart key (charts.py) for one with "charts" rows
  relations   {"id@version": {"meta": {...}, "members": {hash: (member, ...)}, "pairs": [(a, b, score), ...]}}: the
              relation views (contract 2, ビューの種類): what each map version belongs to (an artist ID) and a score
              for pairs of members, directed, sorted. Their rows are always "maps"
  calibration the calibration JSON (lists, not arrays)
  presets     the hands (手札) to put in the manifest, or None (OctantSpecs plans/presets.md §8). A diff always carries
              the whole list (null when there is none), so applying it gives the new version's hands
  notice      the DB's terms and credits as one Markdown text (manifest `notice`), or None; carried by a diff like
              the hands. What one view owes its sources is that view's `credits` (a list of plain-text lines in its
              meta), so a view keeps its credits wherever it goes
  chart_metrics  the manifest's chartMetrics (which computation filled each metric column of charts.parquet)
  extra_cols  {"maps": [field], "charts": [field]}: columns of those tables this version does not know; their values
              follow the known ones in each row
  extra       manifest fields this version does not know
  schema      the schemaVersion to write, when not this version's (a diff of a newer minor version applied)
Rows are written in hash order, so row numbers depend only on the content. A "charts" view's row numbers point into
charts.parquet, a "maps" view's into maps.parquet. A relation view is a members table (row, member) and a pairs
table (a, b, score) instead of vectors; it is not calibrated.

A diff (manifest "kind": "diff") holds the rows that were added or changed (changed.parquet lists them; maps.parquet,
charts.parquet and each view's vectors carry them), the hashes that left (deleted.parquet) and the whole superseded
table (small, and replacing it is simpler than patching it). It can also add a view (all of its rows: a view pack),
replace a view by another version of the same encoder, or remove one (contract 2, 差分更新). Relation views travel
whole in every diff ("change": "added", replacing the base's view of the same id), since their members change with
the maps and they are small; a diff with a relation view therefore carries every map row. It is only made when
every view both versions have keeps its fingerprint and calibration statistics and the values are normalised the same
way (the calibration's normalisation version); otherwise a full is needed.

"The previous full + a diff" must be the new full even when the DB is newer than this OctantVector (OctantSpecs
octantvector/law.md 2.4), so what this version does not know is carried as it is: columns of maps.parquet and
charts.parquet, manifest fields (a diff's manifest carries every field its full has, apart from the FIELDS that
differ between the two) and per-view fields. What it cannot carry is never dropped silently: a view whose rows it
does not know (it cannot renumber them), a file or a column of superseded.parquet it does not know, a manifest field
only the base has, or tables whose unknown columns differ between the base and the diff make apply_diff refuse, so
the caller takes the full instead. A release whose previous version holds anything this version does not know is a
full (dbrelease.py).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import charts as chartids

SCHEMA_VERSION = "1.1"
INDEX_VERSION = "1.1"  # the contract 5 (OctantHarvest to OctantCompass) index (indexVersion)
MAP_COLS = (("artist", "string"), ("mapper", "string"), ("duration_s", "float64"), ("published_at", "string"))
CHART_COLS = (("characteristic", "string"), ("difficulty", "string"), ("nps", "float64"),
              ("njs", "float64"), ("pass", "float64"), ("tech", "float64"), ("play_style", "string"))
ROW_KINDS = ("maps", "charts")  # the tables a view's rows can point into
FIELDS = ("schemaVersion", "kind", "dbVersion", "builtAt", "baseDbVersion", "diffsSinceFull", "calibrationId",
          "calibrationFile", "counts", "views", "removedViews", "chartMetrics", "presets", "notice", "files")  # known ones
VIEW_FILE_KEYS = ("file", "rowsFile", "membersFile", "pairsFile", "change", "replaces")  # where a view entry's data
# is, not what the view is
VIEW_KINDS = ("vector", "relation")  # contract 2 ビューの種類; a view without "kind" is a vector view


def view_kind(v):
    return v.get("kind", "vector")


class FormatError(Exception):
    pass


class VersionError(FormatError):
    """A file declares a format version this OctantVector does not read (OctantSpecs contracts/, 契約の版).
    `what` names the format, `found` the declared version, `reads` the versions this OctantVector reads."""

    def __init__(self, what, found, reads, where=""):
        self.what, self.found, self.reads = what, found, reads
        super().__init__(f"{where}{': ' if where else ''}{what} {found!r} (this OctantVector reads {reads})")


def major(version):
    return int(str(version).split(".")[0])


def sha256(p):
    with open(p, "rb") as f:
        return "sha256:" + hashlib.file_digest(f, "sha256").hexdigest()


def empty():
    return {"maps": {}, "charts": {}, "superseded": {}, "views": {}, "relations": {}, "calibration": None, "presets": None,
            "notice": None, "chart_metrics": {}, "extra_cols": {"maps": [], "charts": []}, "extra": {}}


def row_keys(content, hashes, rows):
    """Keys of a view's possible rows, in file order: the hashes ("maps") or their chart keys ("charts")."""
    if rows == "charts":
        return [chartids.key(h, r[0], r[1]) for h in hashes for r in content["charts"].get(h, [])]
    return list(hashes)


def view_keys_of(content, h, rows):
    return row_keys(content, [h], rows)


# ---- writing ------------------------------------------------------------------------------------------------------

def _table(cols, rows, hashes, extra=()):
    import pyarrow as pa
    schema = pa.schema([("hash", pa.string())] + [(c, getattr(pa, t)()) for c, t in cols] + list(extra))
    data = [[h] + list(r) for h, r in zip(hashes, rows)]
    return pa.table({n: [d[i] for d in data] for i, n in enumerate(schema.names)}, schema=schema)


def _write_rows(tmp, content, hashes):
    """maps, charts and view files for `hashes` (sorted). -> per-view manifest entries."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    extra = content.get("extra_cols", {})
    pq.write_table(_table(MAP_COLS, [content["maps"][h] for h in hashes], hashes, extra.get("maps", ())),
                   tmp / "maps.parquet")
    ch = [(h, r) for h in hashes for r in content["charts"].get(h, [])]
    pq.write_table(_table(CHART_COLS, [r for _, r in ch], [h for h, _ in ch], extra.get("charts", ())),
                   tmp / "charts.parquet")
    sup = sorted(content["superseded"].items())
    pq.write_table(pa.table({"hash": [a for a, _ in sup], "current": [b for _, b in sup]},
                            schema=pa.schema([("hash", pa.string()), ("current", pa.string())])),
                   tmp / "superseded.parquet")
    (tmp / "views").mkdir()
    views = {}
    keys_of = {kind: row_keys(content, hashes, kind) for kind in ROW_KINDS}
    for key, v in content["views"].items():
        vecs = v["vecs"]
        keys = keys_of[v["meta"].get("rows", "maps")]
        rows = np.array([n for n, k in enumerate(keys) if k in vecs], np.int32)
        dim = v["meta"]["dim"]
        x = np.stack([vecs[keys[n]] for n in rows]).astype(np.float16) if len(rows) else np.zeros((0, dim), np.float16)
        np.save(tmp / "views" / f"{key}.npy", x)
        np.save(tmp / "views" / f"{key}.rows.npy", rows)
        views[key] = {**v["meta"], "file": f"views/{key}.npy", "rowsFile": f"views/{key}.rows.npy"}
    pos = {h: n for n, h in enumerate(hashes)}
    for key, r in content.get("relations", {}).items():
        members = sorted((pos[h], m) for h, ms in r["members"].items() if h in pos for m in ms)
        pq.write_table(pa.table({"row": pa.array([n for n, _ in members], pa.int32()),
                                 "member": pa.array([m for _, m in members], pa.string())}),
                       tmp / "views" / f"{key}.members.parquet")
        pairs = sorted(r["pairs"])
        pq.write_table(pa.table({"a": pa.array([a for a, _, _ in pairs], pa.string()),
                                 "b": pa.array([b for _, b, _ in pairs], pa.string()),
                                 "score": pa.array([x for _, _, x in pairs], pa.float32())}),
                       tmp / "views" / f"{key}.pairs.parquet")
        views[key] = {**r["meta"], "kind": "relation", "rows": "maps",
                      "membersFile": f"views/{key}.members.parquet", "pairsFile": f"views/{key}.pairs.parquet"}
    return views


def _finish(tmp, out, manifest):
    manifest["files"] = {str(p.relative_to(tmp)).replace("\\", "/"): sha256(p)
                         for p in sorted(tmp.rglob("*")) if p.is_file()}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    tmp.rename(out)  # the folder appears only once complete
    return out


def _start(out):
    out = Path(out)
    if out.exists():
        raise FormatError(f"{out} already exists; a DB is never overwritten")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise FormatError(f"{tmp} is left from an interrupted write; remove it first")
    tmp.mkdir(parents=True)
    return out, tmp


def write_full(out, content, db_version, built_at, diffs_since_full=0):
    """A full DB. diffs_since_full: how many diffs led to this content since the last published full (0 for one)."""
    out, tmp = _start(out)
    hashes = sorted(content["maps"])
    views = _write_rows(tmp, content, hashes)
    cal = content["calibration"]
    cal_file = f"calibration/{cal['id']}.json"
    (tmp / "calibration").mkdir()
    (tmp / cal_file).write_text(json.dumps(cal))
    manifest = {"schemaVersion": content.get("schema") or SCHEMA_VERSION, "kind": "full", "dbVersion": db_version,
                "builtAt": built_at, "baseDbVersion": None, "diffsSinceFull": diffs_since_full,
                "calibrationId": cal["id"], "calibrationFile": cal_file,
                "counts": {"maps": len(hashes), "charts": sum(len(content["charts"].get(h, [])) for h in hashes),
                           "superseded": len(content["superseded"])},
                "views": views, "chartMetrics": content.get("chart_metrics") or {}}
    if content.get("presets"):
        manifest["presets"] = content["presets"]
    if content.get("notice"):
        manifest["notice"] = content["notice"]
    manifest.update(content.get("extra") or {})
    return _finish(tmp, out, manifest)


def normalisation(cal):
    """The normalisation version a calibration was made with (generation/normalisation.py). The first, 1, is not
    written, so the calibrations made before the version existed keep their ids."""
    return cal.get("normalisation", 1)


def diffable(base_manifest, base_cal, content):
    """Why a diff from base to content is impossible (None when it is possible): the calibrations must share their
    normalisation version, and every view both have must keep its fingerprint and calibration statistics. Views may
    be added, replaced by another version or removed."""
    new_cal = content["calibration"]
    if normalisation(base_cal) != normalisation(new_cal):
        return f"the normalisation changed (version {normalisation(base_cal)} to {normalisation(new_cal)})"
    for key, v in content["views"].items():
        b = base_manifest.get("views", {}).get(key)
        if b is None:
            continue
        if b.get("fingerprint") != v["meta"]["fingerprint"]:
            return f"{key} has another fingerprint"
        if base_cal["views"].get(key) != new_cal["views"].get(key):
            return f"the calibration of {key} changed"
    return None


def view_changes(base_keys, new_keys):
    """-> ({added key: key it replaces or None}, [removed keys]). A key replaces the base key of the same encoder id."""
    by_id = {k.split("@")[0]: k for k in base_keys if k not in new_keys}
    added = {k: by_id.get(k.split("@")[0]) for k in new_keys if k not in base_keys}
    removed = sorted(k for k in base_keys if k not in new_keys and k not in added.values())
    return added, removed


def changes(base, new):
    """-> (added, replaced, deleted) hashes; rows are compared on the views both contents have."""
    shared = [k for k in new["views"] if k in base["views"]]

    def same(h):
        if base["maps"][h] != new["maps"][h] or base["charts"].get(h, []) != new["charts"].get(h, []):
            return False
        for key in shared:
            for k in view_keys_of(new, h, new["views"][key]["meta"].get("rows", "maps")):
                a, b = base["views"][key]["vecs"].get(k), new["views"][key]["vecs"].get(k)
                if (a is None) != (b is None) or (a is not None and a.tobytes() != np.asarray(b, np.float16).tobytes()):
                    return False
        return True
    added = sorted(set(new["maps"]) - set(base["maps"]))
    deleted = sorted(set(base["maps"]) - set(new["maps"]))
    replaced = sorted(h for h in set(new["maps"]) & set(base["maps"]) if not same(h))
    return added, replaced, deleted


def write_diff(out, base, base_manifest, new, db_version, built_at):
    """Rows added or changed, hashes deleted, and views added (all their rows: a view pack), replaced by another
    version of the same encoder, or removed. A new calibration (the old one plus statistics of added views) travels
    with the diff."""
    why = diffable(base_manifest, base["calibration"], new)
    if why:
        raise FormatError(f"no diff from {base_manifest['dbVersion']}: {why}; a full is needed")
    added, replaced, deleted = changes(base, new)
    new_views, removed_views = view_changes(list(base["views"]), list(new["views"]))
    new_rel, removed_rel = view_changes(list(base.get("relations", {})), list(new.get("relations", {})))
    new_rel.update({k: None for k in new.get("relations", {}) if k not in new_rel})  # kept relations travel whole too
    removed_views = sorted(removed_views + removed_rel)
    out, tmp = _start(out)
    changed = sorted(added + replaced)
    # a new view, or any relation view, needs every row it has
    rows = sorted(set(changed) | (set(new["maps"]) if new_views or new_rel else set()))
    rows_set, changed_set = set(rows), set(changed)
    sub = {**new, "maps": {h: new["maps"][h] for h in rows},
           "views": {k: {**v, "vecs": {h: x for h, x in v["vecs"].items()
                                       if chartids.key_hash(h) in (rows_set if k in new_views else changed_set)}}
                     for k, v in new["views"].items()}}
    views = _write_rows(tmp, sub, rows)
    for k, v in views.items():
        v["change"] = "added" if k in new_views or k in new_rel else "rows"
        if new_views.get(k) or new_rel.get(k):
            v["replaces"] = new_views.get(k) or new_rel[k]
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"hash": pa.array(deleted, pa.string())}), tmp / "deleted.parquet")
    pq.write_table(pa.table({"hash": pa.array(changed, pa.string())}), tmp / "changed.parquet")
    cal = new["calibration"]
    manifest = {"schemaVersion": new.get("schema") or SCHEMA_VERSION, "kind": "diff", "dbVersion": db_version,
                "builtAt": built_at, "baseDbVersion": base_manifest["dbVersion"], "calibrationId": cal["id"],
                "counts": {"added": len(added), "replaced": len(replaced), "deleted": len(deleted),
                           "superseded": len(new["superseded"])},
                "views": views, "removedViews": removed_views, "chartMetrics": new.get("chart_metrics") or {},
                "presets": new.get("presets") or None, "notice": new.get("notice") or None, **(new.get("extra") or {})}
    if cal["id"] != base["calibration"]["id"]:
        manifest["calibrationFile"] = f"calibration/{cal['id']}.json"
        (tmp / "calibration").mkdir()
        (tmp / manifest["calibrationFile"]).write_text(json.dumps(cal))
    return _finish(tmp, out, manifest)


# ---- reading ------------------------------------------------------------------------------------------------------

def read_manifest(path, kind=None, *, check_files=True):
    """The manifest of the DB version at path, refused when its schemaVersion major is not this one's or it is not a
    `kind`. check_files: also check that every file it lists is there with its SHA-256 (False only to show what a
    version says without using it: read_info)."""
    path = Path(path)
    try:
        m = json.loads((path / "manifest.json").read_text())
    except (OSError, ValueError) as ex:
        raise FormatError(f"{path}: no readable manifest.json ({ex})") from None
    try:
        found = major(m["schemaVersion"])
    except (KeyError, ValueError):
        raise FormatError(f"{path}: manifest has no valid schemaVersion") from None
    if found != major(SCHEMA_VERSION):
        raise VersionError("schemaVersion", m["schemaVersion"], f"{major(SCHEMA_VERSION)}.x", where=str(path))
    if kind and m.get("kind", "full") != kind:
        raise FormatError(f"{path} is a {m.get('kind')}, not a {kind}")
    for name, digest in m.get("files", {}).items() if check_files else ():
        p = path / name
        if not p.is_file():
            raise FormatError(f"{path}: {name} is missing")
        if sha256(p) != digest:
            raise FormatError(f"{path}: {name} does not match the manifest")
    return m


def listed(path, m, name):
    if name not in m.get("files", {}):
        raise FormatError(f"{path}: {name} is not listed in the manifest's files")
    return Path(path) / name


def read_view(path, m, key, n_rows):
    """A view's vectors as stored and its row numbers, checked against the n_rows rows of its table."""
    v = m["views"][key]
    x = np.load(listed(path, m, v["file"]))
    rows = np.load(listed(path, m, v["rowsFile"]))
    if len(x) != len(rows) or (len(rows) and (rows.min() < 0 or rows.max() >= n_rows)):
        raise FormatError(f"{path}: {key} rows do not match its vectors or the {v['rows']} table")
    return x, rows


def read_relation(path, m, key, n_rows):
    """A relation view's members ({"row", "member"} columns, rows checked against the n_rows map rows) and pairs
    ({"a", "b", "score"})."""
    import pyarrow.parquet as pq
    v = m["views"][key]
    members = pq.read_table(listed(path, m, v["membersFile"]), columns=["row", "member"]).to_pydict()
    pairs = pq.read_table(listed(path, m, v["pairsFile"]), columns=["a", "b", "score"]).to_pydict()
    rows = members["row"]
    if rows and (min(rows) < 0 or max(rows) >= n_rows):
        raise FormatError(f"{path}: {key} members point outside the maps table")
    return members, pairs


def read_hashes(path, m):
    """The map hashes of maps.parquet, in row order."""
    import pyarrow.parquet as pq
    return pq.read_table(listed(path, m, "maps.parquet"), columns=["hash"]).column("hash").to_pylist()


def read_superseded(path, m):
    """superseded.parquet as {old version hash: current version hash}."""
    import pyarrow.parquet as pq
    sup = pq.read_table(listed(path, m, "superseded.parquet"), columns=["hash", "current"]).to_pydict()
    return dict(zip(sup["hash"], sup["current"]))


def chart_keys(ch):
    """The chart key of each row of charts.parquet read as columns (read_charts)."""
    return [chartids.key(h, c, d) for h, c, d in zip(ch["hash"], ch["characteristic"], ch["difficulty"])]


def read_relations(path, m=None):
    """Only the relation views of a full DB, as in read_content (a release keeps them when no artist files are given)."""
    m = m or read_manifest(path)
    hashes = read_hashes(path, m)
    out = {}
    for key, v in m.get("views", {}).items():
        if view_kind(v) != "relation" or v.get("rows") != "maps":
            continue
        out[key] = _relation_content(path, m, key, hashes)
    return out


def _relation_content(path, m, key, hashes):
    members, pairs = read_relation(path, m, key, len(hashes))
    ms = {}
    for r, member in zip(members["row"], members["member"]):
        ms.setdefault(hashes[r], []).append(member)
    return {"meta": {k: a for k, a in m["views"][key].items() if k not in VIEW_FILE_KEYS + ("kind",)},
            "members": {h: tuple(x) for h, x in ms.items()},
            "pairs": list(zip(pairs["a"], pairs["b"], pairs["score"]))}


def _columns(p, cols):
    """A table as columns, and the fields of the columns this version does not know, in file order."""
    import pyarrow.parquet as pq
    t = pq.read_table(p)
    known = {"hash"} | {c for c, _ in cols}
    return t.to_pydict(), [f for f in t.schema if f.name not in known]


def _files(m):
    """The files of a manifest this version knows what to do with."""
    names = {"maps.parquet", "charts.parquet", "superseded.parquet", m.get("calibrationFile")}
    names |= {v[k] for v in m.get("views", {}).values() for k in ("file", "rowsFile", "membersFile", "pairsFile")
              if k in v and view_kind(v) in VIEW_KINDS}
    return names | ({"deleted.parquet", "changed.parquet"} if m.get("kind") == "diff" else set())


def read_content(path, m=None):
    """Content of a full DB or of a diff (checked first). What this version cannot carry is left out and listed in
    "unread" as (what, name): a view whose rows it does not know, a file it does not know, a column of
    superseded.parquet."""
    m = m or read_manifest(path)
    c = empty()
    c["unread"] = [("file", n) for n in sorted(set(m.get("files", {})) - _files(m))]
    maps, c["extra_cols"]["maps"] = _columns(listed(path, m, "maps.parquet"), MAP_COLS)
    hashes = maps["hash"]
    names = [k for k, _ in MAP_COLS] + [f.name for f in c["extra_cols"]["maps"]]
    c["maps"] = {h: tuple(maps[k][n] for k in names) for n, h in enumerate(hashes)}
    ch, c["extra_cols"]["charts"] = _chart_columns(listed(path, m, "charts.parquet"))
    names = [k for k, _ in CHART_COLS] + [f.name for f in c["extra_cols"]["charts"]]
    for n, h in enumerate(ch["hash"]):
        c["charts"].setdefault(h, []).append(tuple(ch[k][n] for k in names))
    sup, extra = _columns(listed(path, m, "superseded.parquet"), (("current", "string"),))
    c["superseded"] = dict(zip(sup["hash"], sup["current"]))
    c["unread"] += [("column", f"superseded.parquet {f.name}") for f in extra]
    keys_of = {"maps": hashes, "charts": chart_keys(ch)}
    for key, v in m.get("views", {}).items():
        if v.get("rows") not in ROW_KINDS or view_kind(v) not in VIEW_KINDS \
                or (view_kind(v) == "relation" and v["rows"] != "maps"):
            c["unread"].append(("view", key))
            continue
        if view_kind(v) == "relation":
            c["relations"][key] = _relation_content(path, m, key, hashes)
            continue
        keys = keys_of[v["rows"]]
        x, rows = read_view(path, m, key, len(keys))
        c["views"][key] = {"meta": {k: a for k, a in v.items() if k not in VIEW_FILE_KEYS},
                           "vecs": {keys[int(r)]: x[n] for n, r in enumerate(rows)}}
    if m.get("calibrationFile"):
        c["calibration"] = json.loads(listed(path, m, m["calibrationFile"]).read_text())
    c["presets"] = m.get("presets")
    c["notice"] = m.get("notice")
    c["chart_metrics"] = m.get("chartMetrics") or {}
    c["extra"] = {k: v for k, v in m.items() if k not in FIELDS}
    return c


def unknown(content):
    """What a content read by read_content holds that this OctantVector does not know, carried or not (descriptions)."""
    return ([f"{what} {name}" for what, name in content.get("unread", [])]
            + [f"column {table}.parquet {f.name}" for table, fields in content["extra_cols"].items() for f in fields]
            + [f"manifest field {k}" for k in content["extra"]])


def _chart_columns(path):
    ch, extra = _columns(path, CHART_COLS)
    missing = [c for c, _ in CHART_COLS if c not in ch]
    if missing:  # e.g. a 1.0 DB (no play_style): not read
        raise FormatError(f"{path}: no column {', '.join(missing)}")
    return ch, extra


def read_charts(path):
    """charts.parquet as columns."""
    return _chart_columns(path)[0]


def _newer(a, b):
    try:
        return max(a, b, key=lambda v: tuple(int(x) for x in str(v).split(".")))
    except ValueError:
        raise FormatError(f"cannot compare schemaVersion {a!r} with {b!r}") from None


def apply_diff(base_path, diff_path, out):
    """Write base + diff as a new full DB at `out` (contract 5: prepared in another folder, switched to only after
    it is complete). Both are checked first; the diff must start from the base's dbVersion. What this OctantVector does
    not know is carried; when it cannot be carried, FormatError, and the caller takes the full instead."""
    import pyarrow.parquet as pq
    bm, dm = read_manifest(base_path, "full"), read_manifest(diff_path, "diff")
    if dm["baseDbVersion"] != bm["dbVersion"]:
        raise FormatError(f"the diff {dm['dbVersion']} starts from {dm['baseDbVersion']}, not {bm['dbVersion']}")
    base, d = read_content(base_path, bm), read_content(diff_path, dm)
    cant = [f"{what} {name}" for what, name in d["unread"]] \
        + [f"{what} {name} of the base" for what, name in base["unread"] if what == "file"] \
        + [f"manifest field {k} of the base" for k in base["extra"] if k not in d["extra"]] \
        + [f"the columns of {t}.parquet changed" for t in ROW_KINDS if base["extra_cols"][t] != d["extra_cols"][t]] \
        + [f"rows of view {k}, which the base lacks" for k in d["views"]
           if k not in base["views"] and dm["views"][k].get("change") != "added"]
    if cant:
        raise FormatError(f"this OctantVector cannot apply the diff {dm['dbVersion']} without losing what it does not "
                          f"know ({'; '.join(cant)}); take the full of {dm['dbVersion']} instead")
    d["calibration"] = d["calibration"] or base["calibration"]
    if d["calibration"]["id"] != dm["calibrationId"]:
        raise FormatError(f"the diff {dm['dbVersion']} needs calibration {dm['calibrationId']}")
    why = diffable(bm, base["calibration"], d)
    if why:
        raise FormatError(f"the diff does not fit the base: {why}")
    changed = set(pq.read_table(listed(diff_path, dm, "changed.parquet")).column("hash").to_pylist())
    gone = set(pq.read_table(listed(diff_path, dm, "deleted.parquet")).column("hash").to_pylist()) | changed
    new = empty()
    new["calibration"] = d["calibration"]
    new["superseded"] = d["superseded"]
    new["presets"] = d["presets"] if "presets" in dm else base["presets"]
    new["notice"] = d["notice"] if "notice" in dm else base["notice"]
    new.update(schema=_newer(SCHEMA_VERSION, dm["schemaVersion"]), chart_metrics=d["chart_metrics"],
               extra_cols=d["extra_cols"], extra=d["extra"])
    for part in ("maps", "charts"):
        new[part] = {h: r for h, r in base[part].items() if h not in gone}
        new[part].update({h: r for h, r in d[part].items() if h in changed})
    for key, v in d["views"].items():
        if dm["views"][key].get("change") == "added":
            vecs = v["vecs"]
        else:
            vecs = {h: x for h, x in base["views"][key]["vecs"].items() if chartids.key_hash(h) not in gone}
            vecs.update(v["vecs"])
        new["views"][key] = {"meta": v["meta"],
                             "vecs": {h: x for h, x in vecs.items() if chartids.key_hash(h) in new["maps"]}}
    for key, r in d["relations"].items():  # whole in every diff
        new["relations"][key] = {**r, "members": {h: x for h, x in r["members"].items() if h in new["maps"]}}
    return write_full(out, new, dm["dbVersion"], dm["builtAt"], bm.get("diffsSinceFull", 0) + 1)
