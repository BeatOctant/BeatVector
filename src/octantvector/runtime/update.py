"""The host's side of the contract 5 index (OctantSpecs contracts/, 契約5（OctantHarvest to OctantCompass）: DBの配布と更新). OctantVector itself never
downloads anything: OctantCompass fetches the files and calls these.

  plan_update(index, local_version)  which entries to fetch: the diffs after the local version when it is in the
                                     chain, else the full and every diff after it (contract 1 (OctantCompass to OctantVector))
  apply_update(files, base, out, trust)
                                     every fetched entry checked, extracted and applied in order; the new full DB is
                                     written to `out` and the local DB at `base` is never changed (contract 1). A
                                     diff that would lose what this OctantVector does not know is refused (DBError,
                                     dbformat.apply_diff); the full of that version gives the same DB
  unpack                             one entry checked and extracted
  import_archive(path, out, trust)   one full version handed over as a single tar (no index: the Dev 版 hand-off,
                                     contract 5 非公開（探求中）の扱い) checked and extracted to `out` (contract 1)

The index is written by generation/index.py.
"""
import hashlib
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path

from ..common import signing
from ..common.dbformat import INDEX_VERSION, SCHEMA_VERSION, FormatError, VersionError, apply_diff, major, read_manifest

READS_MAJOR = major(SCHEMA_VERSION)
KINDS = ("full", "diff")


MAX_NAME = 100  # bytes of a dbVersion: resolve() does not fail on a name too long for the file system, a later use does


def _folder_name(entry):
    """The entry's dbVersion, which names folders and files on the host. The index is not signed, so anything but a
    single plain path component (a name that could leave the folder it is put in) is refused before any use."""
    v = entry.get("dbVersion")
    if not isinstance(v, str) or v in ("", ".", "..") or "/" in v or "\\" in v or "\0" in v or ":" in v \
            or Path(v).is_absolute() or len(v.encode()) > MAX_NAME:
        raise FormatError(f"the index names a version that is not a plain folder name: {v!r}")
    if entry.get("kind") not in KINDS:
        raise FormatError(f"{v}: unknown kind {entry.get('kind')!r}")
    return v


def _inside(parent, name):
    """parent / name, refused unless it resolves inside parent (the name check above, done again on the real path)."""
    path = Path(parent) / name
    if not path.resolve().is_relative_to(Path(parent).resolve()):
        raise FormatError(f"{name!r} would be written outside {parent}")
    return path


def plan_update(index, local_version=None):
    """Entries to fetch, in order, to reach the newest version this OctantVector can read. [] when up to date.
    VersionError when the index itself is in a format this OctantVector does not read (its indexVersion)."""
    found = index.get("indexVersion")
    try:
        ok_index = major(found) == major(INDEX_VERSION)
    except (TypeError, ValueError):
        ok_index = False
    if not ok_index:
        raise VersionError("indexVersion", found, f"{major(INDEX_VERSION)}.x")
    ok = [e for e in index["entries"] if major(e["schemaVersion"]) == READS_MAJOR]
    for e in ok:
        _folder_name(e)
    if not ok or ok[-1]["dbVersion"] == local_version:
        return []
    versions = [e["dbVersion"] for e in ok]
    if local_version in versions:
        return ok[versions.index(local_version) + 1:]
    start = max(i for i, e in enumerate(ok) if e["kind"] == "full")
    return ok[start:]


def unpack(entry, part_paths, out_dir, trust=None):
    """Check and extract one entry's parts (in index order) into out_dir. Its manifest.sig is looked up against the
    bundled keys and anything but a good signature is a SignatureWarning, not a failure. -> the extracted DB folder."""
    version = _folder_name(entry)
    if len(part_paths) != len(entry["parts"]):
        raise FormatError(f"{version}: {len(entry['parts'])} parts expected, got {len(part_paths)}")
    out_dir = Path(out_dir)
    joined = _inside(out_dir, f"{version}.{entry['kind']}.tar.partial")
    out_dir.mkdir(parents=True, exist_ok=True)
    whole = hashlib.sha256()
    try:
        with open(joined, "wb") as dst:
            for p, want in zip(part_paths, entry["parts"]):
                h, size = hashlib.sha256(), 0
                with open(p, "rb") as src:
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        h.update(chunk); whole.update(chunk); dst.write(chunk)
                        size += len(chunk)
                if size != want["size"] or "sha256:" + h.hexdigest() != want["sha256"]:
                    raise FormatError(f"{p} does not match the index ({want['name']})")
        if "sha256:" + whole.hexdigest() != entry["sha256"]:
            raise FormatError(f"{entry['dbVersion']}: the joined parts do not match the index")
        folder = _extract(joined, out_dir, entry["dbVersion"])
    finally:
        joined.unlink(missing_ok=True)
    m = read_manifest(folder, entry["kind"])
    _warn_unsigned(folder, trust, entry["dbVersion"])
    if m["dbVersion"] != entry["dbVersion"]:
        raise FormatError(f"the archive holds {m['dbVersion']}, the index says {entry['dbVersion']}")
    return folder


def _extract(tar_path, out_dir, label):
    """The one version folder in the tar at tar_path, extracted into out_dir; anything but plain files under a single
    top folder is refused before anything is written. Only an uncompressed tar is accepted (generation writes one):
    a compressed one could expand without bound and fill the disk (a tar bomb), while in a plain tar every member's
    bytes are stored in full, so no more is written than was fetched and checked. A member that declares more than
    the archive holds (a sparse member) is refused for the same reason."""
    try:
        tar = tarfile.open(tar_path, mode="r:")
    except tarfile.TarError as ex:
        raise FormatError(f"{label}: not an uncompressed tar archive ({ex})") from None
    with tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        top = {n.split("/")[0] for n in names}
        if len(top) != 1 or any(n.startswith("/") or ".." in Path(n).parts for n in names) or \
                any(not m.isfile() for m in members):
            raise FormatError(f"{label}: unexpected paths in the archive")
        if sum(m.size for m in members) > Path(tar_path).stat().st_size:
            raise FormatError(f"{label}: the archive declares more content than it holds")
        if (Path(out_dir) / next(iter(top))).exists():
            raise FormatError(f"{Path(out_dir) / next(iter(top))} already exists")
        tar.extractall(out_dir, filter="data")
    return Path(out_dir) / top.pop()


def _warn_unsigned(folder, trust, label):
    sig = signing.inspect(folder, trust)  # the signature names the author and does not block (contract 5, 署名)
    if sig["status"] != "signed" and not (sig["status"] == "unsigned" and signing.check_trust(trust)[1]):
        warnings.warn(signing.SignatureWarning(f"{label}: {sig['detail']}"), stacklevel=3)


def import_archive(path, out, trust=None):
    """path: one full version as a single tar (generation's `pack`, the same bytes as its unsplit index file), handed
    over without an index. -> `out`, the full DB folder, which appears only once the files match the manifest; on any
    failure nothing is left at `out`. trust as apply_update's; a version not signed by a known key is still used, with
    a SignatureWarning."""
    signing.check_trust(trust)
    out = Path(out)
    if out.exists():
        raise FormatError(f"{out} already exists; a DB is never overwritten")
    work = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=out.parent if out.parent.exists() else None))
    try:
        try:
            folder = _extract(path, work, Path(path).name)
        except tarfile.TarError as ex:
            raise FormatError(f"{Path(path).name}: not a DB archive ({ex})") from None
        m = read_manifest(folder, "full")
        if folder.name != _folder_name(m):
            raise FormatError(f"{Path(path).name}: the folder {folder.name!r} holds {m['dbVersion']}")
        _warn_unsigned(folder, trust, m["dbVersion"])
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(out))
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def apply_update(files, base, out, trust=None):
    """files: [(entry, part paths)] in plan_update's order; base: the local full DB folder (None when plan_update
    asked for a full first). -> `out`, a full DB folder that appears only once every step passed. On any failure
    nothing is left at `out` and `base` is untouched. trust: {"keys": {keyId: public key base64} known besides the
    bundled ones, "allowUnsigned": no warning for versions without manifest.sig (development)}; None = the bundled
    keys only. A version that is not signed by a known key is still applied, with a SignatureWarning."""
    signing.check_trust(trust)  # a malformed trust fails before anything is unpacked
    out = Path(out)
    if out.exists():
        raise FormatError(f"{out} already exists; a DB is never overwritten")
    if not files:
        raise FormatError("nothing to apply: plan_update returned no entries")
    for entry, _ in files:
        _folder_name(entry)
    work = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=out.parent if out.parent.exists() else None))
    try:
        cur = Path(base) if base is not None else None
        for n, (entry, parts) in enumerate(files):
            folder = unpack(entry, parts, work / f"{n}.unpacked", trust)
            if entry["kind"] == "full":
                cur = folder
            elif cur is None:
                raise FormatError(f"{entry['dbVersion']}: a diff needs a base DB")
            else:
                cur = apply_diff(cur, folder, _inside(work / f"{n}.applied", entry["dbVersion"]))
        read_manifest(cur, "full")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cur), str(out))
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)
