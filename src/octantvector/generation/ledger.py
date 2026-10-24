"""The ledger: OctantVector's own state in one SQLite file (<store>/ledger.sqlite). The archive and the catalogue stay
read-only; everything OctantVector decides about them is written here. See OctantSpecs octantvector/design.md, データ.

  versions   one row per BeatSaver version hash: map key, state (current / old / deleted), commit_seq of its zip in
             the archive (NULL = no zip yet), zip SHA-256 (recorded when the zip is first read), first/last seen
  meta_revs  per version and map-kind encoder: digest of the catalogue fields the encoder declares (inputs
             "meta:<field>") and when that digest was first seen; the newest row is the current one
  computed   per version and encoder fingerprint: result (ok / error / skip), the input digest it was computed
             from, the error text and when
  pools      per named pool and version: when it entered and (if so) left the pool

Each step reads only the ledger to decide what is left, so an interrupted run continues where it stopped.
"""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..common.charts import key_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (hash TEXT PRIMARY KEY, key TEXT, state TEXT NOT NULL, seq INTEGER,
                                     zip_sha256 TEXT, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta_revs (hash TEXT NOT NULL, encoder TEXT NOT NULL, digest TEXT NOT NULL,
                                      first_seen TEXT NOT NULL, PRIMARY KEY (hash, encoder, digest));
CREATE TABLE IF NOT EXISTS computed (hash TEXT NOT NULL, fingerprint TEXT NOT NULL, encoder TEXT NOT NULL,
                                     result TEXT NOT NULL, input_digest TEXT, error TEXT, at TEXT NOT NULL,
                                     PRIMARY KEY (hash, fingerprint));
CREATE TABLE IF NOT EXISTS pools (pool TEXT NOT NULL, hash TEXT NOT NULL, entered TEXT NOT NULL, left_at TEXT,
                                  PRIMARY KEY (pool, hash, entered));
CREATE INDEX IF NOT EXISTS pools_open ON pools (pool, hash) WHERE left_at IS NULL;
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def field(m, v, name):
    """Value of a declared catalogue field: "versions.<x>" reads the version record, anything else the map record;
    dots walk into nested objects."""
    obj, path = (v, name.split(".")[1:]) if name.startswith("versions.") else (m, name.split("."))
    for p in path:
        obj = obj.get(p) if isinstance(obj, dict) else None
    return obj


def meta_digest(m, v, fields):
    blob = json.dumps({f: field(m, v, f) for f in sorted(fields)}, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def meta_fields(enc):
    return [i[5:] for i in enc.inputs if i.startswith("meta:")]


class Ledger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    @classmethod
    @contextmanager
    def of(cls, store):
        """The ledger of the working store `store` (<store>/ledger.sqlite), closed when the block ends."""
        led = cls(Path(store) / "ledger.sqlite")
        try:
            yield led
        finally:
            led.close()

    # ---- ingest -----------------------------------------------------------------------------------------------

    def ingest(self, maps, available, encoders, pools):
        """maps: iterable of catalogue map records; available: {hash: commit_seq} of zips in the archive;
        encoders: {id: encoder} whose meta digests to record (map kind only); pools: {name: definition} (pools.py).
        -> counts of what changed."""
        from .catalogue import published_version
        from .pools import matches, members
        t = now()
        seen, eligible, current = set(), {p: [] for p in pools}, {}
        meta_encs = {i: meta_fields(e) for i, e in encoders.items() if e.kind == "map"}
        prev = {h: (k, s) for h, k, s in self.db.execute("SELECT hash, key, state FROM versions")}
        keys = set()
        n = {"new": 0, "old": 0, "deleted": 0, "meta_changed": 0}

        def changed(h, state):
            if h not in prev:
                n["new"] += 1
            elif state != prev[h][1] and state in ("old", "deleted"):
                n[state] += 1
        with self.db:
            for m in maps:
                pub = published_version(m)
                gone = bool(m.get("deletedAt")) or pub is None
                keys.add(m.get("id"))
                for v in m.get("versions", []):
                    h = v["hash"]
                    state = "deleted" if gone else ("current" if v is pub else "old")
                    seen.add(h)
                    changed(h, state)
                    self.db.execute(
                        "INSERT INTO versions (hash, key, state, seq, first_seen, last_seen) VALUES (?,?,?,?,?,?) "
                        "ON CONFLICT(hash) DO UPDATE SET key=excluded.key, state=excluded.state, seq=excluded.seq, "
                        "last_seen=excluded.last_seen", (h, m.get("id"), state, available.get(h), t, t))
                    if state != "current":
                        continue
                    current[m.get("id")] = h
                    for i, fields in meta_encs.items():
                        d = meta_digest(m, v, fields)
                        cur = self.db.execute("SELECT 1 FROM meta_revs WHERE hash=? AND encoder=? AND digest=?",
                                              (h, i, d)).fetchone()
                        if cur is None:
                            n["meta_changed"] += self.db.execute(
                                "SELECT 1 FROM meta_revs WHERE hash=? AND encoder=?", (h, i)).fetchone() is not None
                            self.db.execute("INSERT INTO meta_revs VALUES (?,?,?,?)", (h, i, d, t))
                    for p, definition in pools.items():
                        if matches(definition, m, v):
                            eligible[p].append((m.get("id"), h))
            for h, (k, s) in prev.items():
                if h in seen or s == "deleted":
                    continue
                # no longer listed: replaced by a newer version of a map still in the catalogue, else the map is gone
                state = "old" if k in keys else "deleted"
                if state != s:
                    changed(h, state)
                    self.db.execute("UPDATE versions SET state=? WHERE hash=?", (state, h))
            for p, definition in pools.items():
                n[f"pool:{p}"] = self._set_pool(p, members(definition, eligible[p], current), t)
        return n

    def _set_pool(self, pool, hashes, t):
        inside = {h for (h,) in self.db.execute("SELECT hash FROM pools WHERE pool=? AND left_at IS NULL", (pool,))}
        self.db.executemany("INSERT INTO pools (pool, hash, entered) VALUES (?,?,?)",
                            [(pool, h, t) for h in hashes - inside])
        self.db.executemany("UPDATE pools SET left_at=? WHERE pool=? AND hash=? AND left_at IS NULL",
                            [(t, pool, h) for h in inside - hashes])
        return {"size": len(hashes), "entered": len(hashes - inside), "left": len(inside - hashes)}

    # ---- planning ---------------------------------------------------------------------------------------------

    def pool(self, name):
        """Current members of a pool in archive order (versions without a zip last) -> [(hash, seq)]."""
        return self.db.execute(
            "SELECT v.hash, v.seq FROM pools p JOIN versions v ON v.hash = p.hash "
            "WHERE p.pool=? AND p.left_at IS NULL AND v.state='current' "
            "ORDER BY v.seq IS NULL, v.seq, v.hash", (name,)).fetchall()

    def pool_hashes(self, name):
        """The hashes of pool(name), in the same order."""
        return [h for h, _ in self.pool(name)]

    def old_versions(self):
        """[(hash, map key)] of the versions a newer version of the same map replaced."""
        return self.db.execute("SELECT hash, key FROM versions WHERE state='old'").fetchall()

    def current_digest(self, h, encoder_id):
        r = self.db.execute("SELECT digest FROM meta_revs WHERE hash=? AND encoder=? ORDER BY first_seen DESC, rowid DESC "
                            "LIMIT 1", (h, encoder_id)).fetchone()
        return r[0] if r else None

    def todo(self, enc, fingerprint, members, retry_errors=False):
        """Hashes of `members` ([(hash, seq)]) that `enc` still has to compute: no result for this fingerprint yet,
        an error when retrying, or (map kind) a catalogue digest that changed since it was computed. Audio, image
        and chart encoders need the zip, so versions without one wait."""
        done = {h: (r, d) for h, r, d in self.db.execute(
            "SELECT hash, result, input_digest FROM computed WHERE fingerprint=?", (fingerprint,))}
        out = []
        for h, seq in members:
            if enc.kind in ("audio", "image", "zip") and seq is None:
                continue
            r = done.get(h)
            if r is None or (retry_errors and r[0] == "error"):
                out.append(h)
            elif enc.kind == "map" and r[1] != self.current_digest(h, enc.id):
                out.append(h)
        return out

    # ---- results ----------------------------------------------------------------------------------------------

    def record(self, enc, fingerprint, rows):
        """rows: [(hash, result, input_digest, error)]. Call only after ok vectors are flushed to the store."""
        t = now()
        with self.db:
            self.db.executemany(
                "INSERT INTO computed VALUES (?,?,?,?,?,?,?) ON CONFLICT(hash, fingerprint) DO UPDATE SET "
                "result=excluded.result, input_digest=excluded.input_digest, error=excluded.error, at=excluded.at",
                [(h, fingerprint, f"{enc.id}@{enc.version}", r, d, e, t) for h, r, d, e in rows])

    def adopt(self, enc, fingerprint, keys):
        """Vectors already in the store that the ledger has no ok for (flushed just before a crash, or written by a
        derived encoder) count as ok.
        Map-kind rows get the current digest: the catalogue is assumed unchanged since they were computed. Chart keys
        count for their map version (results are recorded per version)."""
        known = {h for (h,) in self.db.execute("SELECT hash FROM computed WHERE fingerprint=? AND result='ok'",
                                               (fingerprint,))}
        new = [h for h in dict.fromkeys(key_hash(k) for k in keys) if h not in known]
        self.record(enc, fingerprint, [(h, "ok", self.current_digest(h, enc.id) if enc.kind == "map" else None, None)
                                       for h in new])
        return len(new)

    def zip_sha(self, h, sha):
        """Record the zip's SHA-256 the first time it is read. -> False if a different zip was seen for this hash."""
        r = self.db.execute("SELECT zip_sha256 FROM versions WHERE hash=?", (h,)).fetchone()
        if r is None or r[0] is None:
            with self.db:
                self.db.execute("UPDATE versions SET zip_sha256=? WHERE hash=?", (sha, h))
            return True
        return r[0] == sha

    # ---- reporting --------------------------------------------------------------------------------------------

    def results(self):
        """Every computed row as (key "<hash>:<fingerprint>", encoder, result, input digest, error, at)."""
        return set(self.db.execute("SELECT hash || ':' || fingerprint, encoder, result, input_digest, error, at "
                                   "FROM computed"))

    def summary(self):
        states = dict(self.db.execute("SELECT state, count(*) FROM versions GROUP BY state"))
        pools = dict(self.db.execute("SELECT pool, count(*) FROM pools WHERE left_at IS NULL GROUP BY pool"))
        results = {}
        for enc, r, c in self.db.execute("SELECT encoder, result, count(*) FROM computed GROUP BY encoder, result"):
            results.setdefault(enc, {})[r] = c
        return {"versions": states, "pools": pools, "computed": results}
