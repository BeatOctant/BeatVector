"""The contract 5 index (OctantSpecs contracts/, 契約5（OctantHarvest to OctantCompass）: DBの配布と更新), generation side.

  build(folders, out, part_size_mb, publish_url, allow_unsigned, trusted_keys)
      folders: the versions the caller (OctantHarvest) approved, each the folder `release` published (its diff, or its
      full). Writes into `out` one tar per indexed version, split into parts of at most `part_size_mb`, and
      index.json. The index holds the newest of them published as a full and the diffs after it, as long as each
      starts from the one before; nothing that was not passed in is ever indexed.
Uploading is not done here; the URL of each part is `publish_url` + its name (null when unset).
Every version it indexes must be signed by a known key (contract 5, 署名; `sign` writes manifest.sig): one bundled
(common/trusted_keys.py) or in `trusted_keys`, looked up with signing.inspect as apply_update does. An older version
passed in that is left out is not looked at. allow_unsigned (development) indexes a version whatever its signature.
The index's per-entry `signature` is not used and stays null.

  write_tar(folder, path)
      one version folder as one tar: the file the parts are cut from, and what `pack` hands over whole.

The host's side of the index (what to fetch, checking and extracting it) is runtime/update.py.
"""
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from ..common import signing
from ..common.dbformat import FormatError, INDEX_VERSION, read_manifest
from .errors import GenerationError


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def chain(folders, allow_unsigned=False, trusted_keys=None):
    """[(manifest, folder)] to index: the newest full among `folders`, then the diffs that follow it."""
    items = []
    for f in map(Path, folders):
        m = read_manifest(f)  # only complete, intact versions
        items.append((m, f))
    items.sort(key=lambda t: t[0]["builtAt"])
    fulls = [i for i, (m, _) in enumerate(items) if m["kind"] == "full"]
    if not fulls:
        return []
    out = [items[fulls[-1]]]
    for m, f in items[fulls[-1] + 1:]:
        if m["kind"] == "diff" and m["baseDbVersion"] == out[-1][0]["dbVersion"]:
            out.append((m, f))
        else:
            break  # a version missing in between: what follows cannot be applied yet
    if not allow_unsigned:  # the indexed ones only, each signed by a known key (the receiver checks them again)
        for _, f in out:
            sig = signing.inspect(f, {"keys": trusted_keys})
            if sig["status"] != "signed":
                raise FormatError(f"not indexed: {sig['detail']}")
    return out


def build(folders, out, part_size_mb, publish_url=None, allow_unsigned=False, trusted_keys=None):
    """-> path of index.json."""
    pub = Path(out)
    pub.mkdir(parents=True, exist_ok=True)
    limit = max(1, int(float(part_size_mb) * (1 << 20)))
    entries = []
    for m, folder in chain(folders, allow_unsigned, trusted_keys):
        name = f"{m['dbVersion']}.{m['kind']}.tar"
        if not (pub / f"{name}.done").exists():
            _write_parts(folder, pub, name, limit)
        info = json.loads((pub / f"{name}.done").read_text())
        entries.append({"dbVersion": m["dbVersion"], "schemaVersion": m["schemaVersion"], "kind": m["kind"],
                        "baseDbVersion": m.get("baseDbVersion"), "calibrationId": m["calibrationId"],
                        "builtAt": m["builtAt"], "size": info["size"], "sha256": info["sha256"],
                        "parts": [{**p, "url": (publish_url.rstrip("/") + "/" + p["name"]) if publish_url else None}
                                  for p in info["parts"]],
                        "signature": None})
    if not entries:
        raise GenerationError("nothing to index: no version published as a full among the approved ones")
    index = {"indexVersion": INDEX_VERSION, "generatedAt": _now(), "entries": entries}
    p = pub / "index.json"
    p.write_text(json.dumps(index, indent=1))
    return p


def write_tar(folder, path):
    """The version folder as one uncompressed tar at path, its files under the folder's name. Deterministic (sorted
    names, fixed metadata), so the same version always gives the same bytes."""
    with tarfile.open(path, mode="w") as tar:
        for f in sorted(p for p in folder.rglob("*") if p.is_file()):
            ti = tarfile.TarInfo(f"{folder.name}/{f.relative_to(folder).as_posix()}")
            ti.size, ti.mtime, ti.mode = f.stat().st_size, 0, 0o644
            with open(f, "rb") as fh:
                tar.addfile(ti, fh)


def _write_parts(folder, pub, name, limit):
    tmp = pub / f"{name}.tmp"
    write_tar(folder, tmp)
    whole, parts, n = hashlib.sha256(), [], 0
    with open(tmp, "rb") as src:
        while True:
            pn, h, size = f"{name}.part{n:03d}", hashlib.sha256(), 0
            with open(pub / pn, "wb") as dst:
                while size < limit:
                    chunk = src.read(min(1 << 20, limit - size))
                    if not chunk:
                        break
                    dst.write(chunk); h.update(chunk); whole.update(chunk)
                    size += len(chunk)
            if size == 0 and parts:
                (pub / pn).unlink()
                break
            parts.append({"name": pn, "size": size, "sha256": "sha256:" + h.hexdigest()})
            n += 1
    total = tmp.stat().st_size
    tmp.unlink()
    (pub / f"{name}.done").write_text(json.dumps({"size": total, "sha256": "sha256:" + whole.hexdigest(), "parts": parts}))
