"""All settings and their defaults (the single source of truth; OctantSpecs octantvector/config.md describes them).

Values come from, in order: environment variable > config file > default below. The config file is TOML,
found at $BV_CONFIG or ./octantvector.toml, with keys named like the variables without the BV_ prefix,
lower-case (e.g. `archive = "/path/to/archive"`).

The contract 1 (OctantHarvest to OctantVector) functions (api.py) read neither: inside `given(values)` every setting and table comes from `values`
only (OctantSpecs contracts/, 契約1 生成側の関数). The development commands (cli.py) use the file and the environment.
"""
import contextvars
import os
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .errors import GenerationError

DEFAULTS = {
    "archive": None,              # BV_ARCHIVE: folder with maps/<hash[:2]>/<hash>.zip and download.journal.jsonl
    "catalogue": None,            # BV_CATALOGUE: BeatSaver catalogue snapshot (catalogue.jsonl.gz)
    "artists": None,              # BV_ARTISTS: optional folder with map_artists.jsonl.gz and similar_artists.jsonl.gz
                                  # (the relation view listenbrainz_artists@<YYYYMMDD>, role `artist`;
                                  # generation/artists.py)
    "catalogue_sha256": None,     # BV_CATALOGUE_SHA256: optional expected hash, checked by `octantvector check`
    "store": "store",             # BV_STORE: vector store root (one sub-folder per encoder id@version)
    "models_dir": None,           # BV_MODELS_DIR: the model files encoders read, as <models_dir>/hf/<repo>/<revision>/
                                  # (put there by `octantharvest models`; OctantVector never downloads them)
    "device": "auto",             # BV_DEVICE: "auto" (CUDA if available) or "cpu"
    "workers": 12,                # BV_WORKERS: CPU worker processes for embedding
    "worker_memory_gb": 6,        # BV_WORKER_MEMORY_GB: a worker over this much memory of its own is stopped and its map
                                  # recorded as failed (0: no limit; Linux only)
    "full_every": 30,             # BV_FULL_EVERY: diffs between two full releases (octantvector release)
    "swing_model": None,          # BV_SWING_MODEL: style model file written by `octantvector swing-train`
}


_GIVEN = contextvars.ContextVar("given", default=None)


@contextmanager
def given(values: dict):
    """Settings and tables from `values` only (no file, no environment) while the block runs."""
    token = _GIVEN.set(dict(values))
    try:
        yield
    finally:
        _GIVEN.reset(token)


def current():
    """The values set by `given` (None outside it); passed on to worker processes."""
    return _GIVEN.get()


def _file_values():
    g = _GIVEN.get()
    if g is not None:
        return g
    p = Path(os.environ.get("BV_CONFIG", "octantvector.toml"))
    if p.exists():
        with open(p, "rb") as f:
            return tomllib.load(f)
    return {}


def section(name) -> dict:
    """A table of the config file (e.g. [swing]); its keys and defaults are owned by the module that reads it."""
    v = _file_values().get(name, {})
    if not isinstance(v, dict):
        raise GenerationError(f"'{name}' in the config file must be a table ([{name}])")
    return v


@dataclass(frozen=True)
class Config:
    archive: Path | None
    catalogue: Path | None
    artists: Path | None
    catalogue_sha256: str | None
    store: Path
    models_dir: Path | None
    device: str
    workers: int
    worker_memory_gb: float
    full_every: int
    swing_model: Path | None

    def require(self, name):
        v = getattr(self, name)
        if v is None:
            raise GenerationError(f"setting '{name}' is not configured: set BV_{name.upper()} or '{name}' in "
                                  f"octantvector.toml (see https://github.com/BeatOctant/OctantSpecs/blob/main/octantvector/config.md)")
        return v


def load():
    vals = dict(DEFAULTS)
    vals.update({k: v for k, v in _file_values().items() if k in DEFAULTS})
    unknown = set(_GIVEN.get() or {}) - set(DEFAULTS) - {"pools", "swing", "swing_tags"}
    if unknown:
        raise GenerationError(f"unknown setting(s): {', '.join(sorted(unknown))} (OctantSpecs octantvector/config.md)")
    for k in DEFAULTS if _GIVEN.get() is None else ():
        env = os.environ.get(f"BV_{k.upper()}")
        if env is not None:
            vals[k] = env
    for k in ("workers", "full_every"):
        vals[k] = int(vals[k])
    vals["worker_memory_gb"] = float(vals["worker_memory_gb"])
    for k in ("archive", "catalogue", "artists", "store", "models_dir", "swing_model"):
        vals[k] = Path(vals[k]).expanduser() if vals[k] else None
    return Config(**vals)
