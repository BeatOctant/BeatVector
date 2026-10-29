"""The recommendation runtime: what OctantCompass calls (OctantSpecs contracts/engine-runtime.md).

It reads only the DB it is given and takes every number as an argument: it reads no settings file, holds no user
state and never touches the archive, the catalogue, the working store or the encoders. It imports only `common`
(tests/test_layers.py checks this), so it runs on numpy and pyarrow alone.

Contract 1 (OctantCompass to OctantVector) (what OctantCompass calls):
  open_db(path)                                db.py: check and read a full DB (views, genre words, the notice
                                               and each view's credits: DB.info());
                                               DBError when it cannot be used, VersionError (a DBError) when its
                                               schemaVersion is one this OctantVector does not read
  read_info(path)                              db.py: what DB.info() says of a DB without opening it, from its
                                               manifest alone (each view without its rows; no file checked): to
                                               show a DB that is not open (DBError, VersionError as open_db)
  recommend(db, history, request, params)      recommend.py: ranked charts with a reason each (RequestError when
                                               the request does not fit the DB; SeedNotInDBError, a RequestError,
                                               when the seed is not in it; SeedNotEmbeddedError, a
                                               SeedNotInDBError, when it is listed but has no vector to compare)
  presets(db)                                  presets.py: the hands (手札) this DB and this OctantVector can serve, from the
                                               DB's manifest or the bundled defaults (OctantSpecs plans/presets.md §8)
  expand_preset(db, request)                   presets.py: a request naming a hand turned into contract 1 conditions
                                               (UnknownPresetError, a RequestError, for an id outside the list)
  describe(db, history, params)                recommend.py: a display summary of the player
  plan_update(index, local_version)            update.py: entries to fetch from the contract 5 (OctantHarvest to OctantCompass) index (VersionError
                                               when the index's indexVersion is one this OctantVector does not read)
  apply_update(files, base, out, trust)        update.py: fetched entries checked and applied into a new DB folder
                                               (SignatureWarning, not a failure, when a version is not signed by a
                                               known key)
  import_archive(path, out, trust)             update.py: one full version handed over as a single tar, checked and
                                               extracted into a new DB folder (SignatureWarning as apply_update)
Inside the runtime (not in the contract; tools and tests use them):
  search(db, query, view, k)                   db.py: top-k of one view
  apply_diff, unpack                           the steps apply_update is made of
  history.py, skill.py, likes.py               the history as read, the skill as distance, the 好みの判定ルール
                                               (OctantSpecs octantvector/law.md); the numbers' defaults are
                                               common/params.py
"""
# the runtime API's version (OctantSpecs contracts/engine-runtime.md, 版の付け方): separate from the package version, so
# an implementation change that keeps the API keeps it; OctantCompass checks it at startup
API_VERSION = "1.1"

from .db import DB, DBError, VersionError, apply_diff, open_db, read_info, search  # noqa: F401
from .recommend import RequestError, SeedNotEmbeddedError, SeedNotInDBError, describe, recommend  # noqa: F401
from .presets import UnknownPresetError, expand_preset, presets  # noqa: F401
from .update import apply_update, import_archive, plan_update, unpack  # noqa: F401
from ..common.signing import SignatureWarning  # noqa: F401
