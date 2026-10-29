"""Contract 1, generation side (OctantSpecs contracts/, 契約1（OctantHarvest to OctantVector）生成側の関数): the functions OctantHarvest calls.

Every input comes as an argument: the crawl (contract 4 (OctantHarvest crawl to OctantVector)) as {"archive": folder, "catalogue": catalogue.jsonl.gz}
and optionally "artists": the folder of map_artists.jsonl.gz and similar_artists.jsonl.gz (the relation view
listenbrainz_artists@<YYYYMMDD>, role `artist`),
`views`, the views to compute or put in a DB: encoders by id@version (or a bare id), and the relation views built
from the catalogue by name (RELATIONS; nothing to compute or calibrate, they only go into a release),
the working store's folder (it holds the ledger), and `params`, the settings of OctantSpecs octantvector/config.md by
name (numbers, `device`, `swing_model`, and the `pools`, `swing` and `swing_tags` tables). Defaults fill what
`params` leaves out. Nothing here reads the config file or the environment, and nothing talks to the network: the
models the encoders read are listed by `models(views)`, fetched by the caller and passed as `models_dir`.

When to run them, which version to approve and where the files go are the caller's.
"""
import functools
import io
import json
from pathlib import Path

from ..common import dbformat, signing
from ..common.dbformat import read_manifest
from . import calibration as calibrations, config, dbrelease, embed, index as indexes, mappers
from .encoders import get, registry
from .encoders.base import not_provided, plugin_errors
from .errors import GenerationError, VersionError
from .ledger import Ledger

PATHS = ("archive", "catalogue", "artists", "store", "models_dir")
RELATIONS = (mappers.NAME,)  # relation views a release builds from the catalogue when `views` names them
# the contract 4 formats this OctantVector reads (OctantSpecs contracts/, 契約の版): the catalogue capture's manifest.json
# `schema` next to catalogue.jsonl.gz, and the archive's archive_meta.json `format`
CATALOGUE_SCHEMAS = ("octantharvest-beatsaver-catalogue-capture/v1",)
ARCHIVE_FORMATS = ("beatmap-music-identity-beatsaver-archive-v2",)


def _contract(f):
    """A contract 1 function: a format version the generation side does not read reaches the caller as the generation
    VersionError (a GenerationError), never as the bare dbformat one, and any other format error (a version folder
    that is missing, broken or not signed by a known key) as a GenerationError."""
    @functools.wraps(f)
    def call(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except VersionError:
            raise
        except dbformat.VersionError as ex:
            raise VersionError.of(ex) from ex
        except dbformat.FormatError as ex:
            raise GenerationError(str(ex)) from ex
    return call


def _declared(path, key):
    try:
        return json.loads(Path(path).read_text()).get(key)
    except (OSError, ValueError):
        return None


def _check_input(input):
    """VersionError unless the crawl declares formats this OctantVector reads (a capture without manifest.json is
    incomplete and declares nothing)."""
    cat = Path(input["catalogue"]).parent / "manifest.json"
    found = _declared(cat, "schema")
    if found not in CATALOGUE_SCHEMAS:
        raise VersionError("catalogue schema", found, ", ".join(CATALOGUE_SCHEMAS), where=str(cat))
    meta = Path(input["archive"]) / "archive_meta.json"
    found = _declared(meta, "format")
    if found not in ARCHIVE_FORMATS:
        raise VersionError("archive format", found, ", ".join(ARCHIVE_FORMATS), where=str(meta))


def _values(input, store, params, models_dir=None):
    v = dict(params or {})
    passed = set(v) & set(PATHS)
    if passed:
        raise GenerationError(f"{', '.join(sorted(passed))}: pass paths as the arguments, not in params")
    if input is not None:
        _check_input(input)
        v["archive"], v["catalogue"] = str(input["archive"]), str(input["catalogue"])
        if input.get("artists"):
            v["artists"] = str(input["artists"])
    v["store"] = str(store)
    if models_dir is not None:
        v["models_dir"] = str(models_dir)
    return v


def _ids(views):
    """`id@version` (or a bare id) -> encoder ids; a version other than the current one is refused. Views are always
    named by the caller. The relation views (RELATIONS) are left out: they are not computed."""
    return _split(views)[0]


def _split(views):
    """-> (encoder ids, relation view names) of `views`."""
    if not views or isinstance(views, str):
        raise GenerationError(f"views: a non-empty list of id@version is required, got {views!r}")
    out, rel = [], []
    for view in views:
        i, _, ver = str(view).partition("@")
        if i in RELATIONS:
            if ver and f"{i}@{ver}" != mappers.KEY:
                raise GenerationError(f"{view}: this OctantVector builds {mappers.KEY}")
            rel.append(i)
            continue
        if i not in registry():
            raise GenerationError(f"{view}: no such encoder (OctantSpecs octantvector/encoders.md){not_provided()}")
        if ver and ver != str(registry()[i].version):
            raise GenerationError(f"{view}: this OctantVector computes {i}@{registry()[i].version}")
        out.append(i)
    return out, rel


def _pool(store):
    with Ledger.of(store) as led:
        return led.pool_hashes("distribution")


def _results(store):
    with Ledger.of(store) as led:
        return led.results()


@_contract
def compute(input, store, views, params=None, pool="distribution", models_dir=None, log=None):
    """Ingest the crawl into the ledger and compute what it says is missing for `views` (and the sources of derived
    ones) over the pool `pool` (params["pools"], OctantSpecs octantvector/config.md). models_dir: the folder holding the
    models `models(views)` lists. Resumable; a second call right after computes nothing. When the GPU has no room the
    models take fewer maps at a time; a map that does not fit even alone is left uncomputed and unrecorded ("deferred"),
    so the next call computes it.
    -> {"computed": n, "failed": n, "skipped": n, "deferred": n,
        "views": {id@version: {"ok"|"error"|"skip"|"deferred": n}}}, this call's."""
    ids = _ids(views)
    with config.given(_values(input, store, params, models_dir)):
        before = _results(store)
        no_room = embed.run(config.load(), ids, pool=pool, log=log or io.StringIO())
        rows = _results(store) - before  # results written or changed by this call
    per = {}
    for _, enc, r, *_ in rows:
        per.setdefault(enc, {})[r] = per.get(enc, {}).get(r, 0) + 1
    for i, k in no_room.items():
        per.setdefault(f"{i}@{get(i).version}", {})["deferred"] = k
    total = {r: sum(v.get(r, 0) for v in per.values()) for r in ("ok", "error", "skip", "deferred")}
    return {"computed": total["ok"], "failed": total["error"], "skipped": total["skip"], "deferred": total["deferred"],
            "views": per}


@_contract
def models(views):
    """The models `views` read (with their sources'), for the caller to fetch into models_dir before `compute`.
    -> [{"repo", "revision", "files": [...], "path": "hf/<repo>/<revision>"}] (each once, sorted)."""
    with config.given({}):  # some encoders read settings as they are made; nothing here comes from the file
        found = {m.key: m for i in _ids(views) for m in get(i).required_models()}
    return [{"repo": m.repo, "revision": m.revision, "files": list(m.files), "path": m.path}
            for _, m in sorted(found.items())]


@_contract
def drift(store, base, params=None):
    """{view id@version: drift} of the full DB `base`'s calibration against the store's current distribution pool:
    how far each view's mean moved, in units of the spread around the calibration mean (0 = no movement). A vector view
    of `base` that no installed encoder computes stops it (a missing plugin must not pass unnoticed)."""
    try:
        m = json.loads((Path(base) / "manifest.json").read_text())
    except (OSError, ValueError) as ex:
        raise GenerationError(f"{base}: no readable manifest.json ({ex})") from ex
    vec = [k for k, v in m["views"].items() if dbformat.view_kind(v) != "relation"]  # relations are given, not computed
    views = [k.split("@")[0] for k in vec]
    unknown = [k for k, i in zip(vec, views) if i not in registry()]
    if unknown:
        raise GenerationError(f"{base} has views no installed encoder computes: {', '.join(unknown)}; install the "
                              f"plugin that provides them{plugin_errors()}")
    with config.given(_values(None, store, params)):
        cal = calibrations.load(Path(base) / m["calibrationFile"])
        return calibrations.drift(store, cal, views, _pool(store))


@_contract
def calibrate(store, views, params=None):
    """A new calibration of `views` over the distribution pool, also saved under <store>/calibration.
    -> the calibration (plain JSON: id, pool, created_at, views)."""
    with config.given(_values(None, store, params)):
        cal = calibrations.compute(store, _ids(views), _pool(store), "distribution")
        return json.loads(calibrations.save(cal, store).read_text())


@_contract
def release(input, store, views, calibration, base, out, params=None, full=False, built_at=None, log=None, presets=None,
            notice=None):
    """The next DB version (contract 2 (DB build to recommendation core)) under the folder `out`: always the full, and the diff from `base` (the
    previous full's folder, or None) when one is possible. views: the views to put in it (required). calibration:
    one returned by `calibrate`, or None to keep the base's (extended with new views).
    presets: the hands to write into the manifest (a list, or the path of a JSON file); None keeps the base's. Either
    way they are checked against the new DB (OctantSpecs plans/presets.md §8).
    notice: the DB's terms and credits as one Markdown text for the manifest; None keeps the base's. Each view's
    credits come from its encoder.
    -> {"dbVersion", "full": folder, "diff": folder or None, "published": what to approve and index}."""
    ids, rel = _split(views)
    with config.given(_values(input, store, params)):
        cal = calibrations.load_dict(json.loads(json.dumps(calibration))) if calibration is not None else None
        full_dir, diff_dir = dbrelease.release(config.load(), ids, cal, full=full,
                                               built_at=built_at, log=log or io.StringIO(), out=out, base=base,
                                               presets=presets, notice=notice, relations=rel)
    return {"dbVersion": full_dir.name, "full": full_dir, "diff": diff_dir, "published": diff_dir or full_dir}


@_contract
def keygen(path):
    """A new Ed25519 private key written to `path` (an existing file is never overwritten). -> {"publicKey": base64,
    "keyId"}; the public key goes into OctantVector's bundled list (common/trusted_keys.py) by a release."""
    pub, kid = signing.keygen(path)
    return {"publicKey": pub, "keyId": kid}


@_contract
def sign(folder, key_path):
    """Sign the version folder `release` returned as "published" with the private key file. -> path of manifest.sig."""
    read_manifest(folder)  # a complete, intact version only
    return signing.sign(folder, key_path)


@_contract
def pack(folder, out):
    """The full version folder `release` returned as "full" as one tar in the folder `out`, named
    <dbVersion>.full.tar: what is handed over without an index (OctantSpecs contracts/distribution.md, 非公開（探求中）の扱い),
    the same bytes as that version's file in the index before it is split. Sign the folder first (`sign`) for the
    receiver to see the author. An existing file is never overwritten. -> path of the tar."""
    folder = Path(folder)
    m = read_manifest(folder, "full")  # a complete, intact full only: the receiver has nothing to apply a diff to
    if folder.name != m["dbVersion"]:
        raise GenerationError(f"{folder}: the folder of {m['dbVersion']} must be named after it")
    path = Path(out) / f"{m['dbVersion']}.full.tar"
    if path.exists():
        raise GenerationError(f"{path} already exists; it is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        indexes.write_tar(folder, tmp)
        tmp.rename(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


@_contract
def build_index(releases, out, params=None):
    """The contract 5 (OctantHarvest to OctantCompass) index and the split files, in the folder `out`, for the approved versions `releases` (each the
    folder `release` returned as "published"). params: part_size_mb (the largest file it writes; required, the
    caller's setting), publish_url, trusted_keys ({keyId: public key base64} known besides the bundled ones, as
    apply_update's trust["keys"]: the author's key when this OctantVector does not bundle it), allow_unsigned (index
    versions whatever their signature; development). Each version indexed must be signed by a known key.
    -> path of index.json."""
    p = dict(params or {})
    unknown = set(p) - {"part_size_mb", "publish_url", "trusted_keys", "allow_unsigned"}
    if unknown:
        raise GenerationError(f"build_index: unknown params {', '.join(sorted(unknown))}")
    if p.get("part_size_mb") is None:
        raise GenerationError("build_index: params.part_size_mb is required")
    try:
        signing.check_trust({"keys": p.get("trusted_keys")})
    except (ValueError, TypeError) as ex:
        raise GenerationError(f"build_index: params.trusted_keys: {ex}") from None
    return indexes.build(releases, out, p["part_size_mb"], p.get("publish_url"), bool(p.get("allow_unsigned", False)),
                         p.get("trusted_keys"))
