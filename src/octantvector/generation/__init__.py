"""DB generation: from the crawl (contract 4 (OctantHarvest crawl to OctantVector)) through the ledger, the encoders and the working store to the
distributed DB (contract 2 (DB build to recommendation core)) and its index (contract 5 (OctantHarvest to OctantCompass)). OctantHarvest calls it through the functions below (OctantSpecs
contracts/engine-generation.md); OctantCompass never does.

It imports `common` and never the runtime: the DB is the only thing that passes from here to the runtime
(OctantSpecs constitution.md, principle 8; tests/test_layers.py checks this).
"""

# the generation API's version (OctantSpecs contracts/engine-generation.md, 版の付け方): separate from the package
# version and from the runtime's; OctantHarvest checks it before a run
API_VERSION = "1.0"

from .api import build_index, calibrate, compute, drift, keygen, models, pack, release, sign  # noqa: E402,F401
from .errors import GenerationError, VersionError  # noqa: E402,F401  (VersionError: an unknown crawl or DB format)
