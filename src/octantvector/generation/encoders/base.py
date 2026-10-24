"""Encoder interface and registry. The list of encoders and their known issues is in OctantSpecs octantvector/encoders.md.

Every encoder has a stable `id`, a `version` (bump whenever its output would change), a fixed `dim`, a `kind`
and a `norm`:
  kind "audio"   prep(y, sr) runs in CPU worker processes (picklable result); embed(prepped) runs in the
                 main process (GPU when available) and returns a (dim,) vector, or None when the song has
                 nothing this view describes (recorded as a skip). An encoder with `batch` > 1 gets that many
                 maps at once in embed_many (one output each), to keep the GPU busy.
  kind "image"   prep(data) runs in the CPU worker on the map's cover image (its file's bytes); embed(prepped) runs in
                 the main process like an audio encoder's (with `batch` and embed_many the same way). A map without a
                 readable cover is recorded as a skip.
  kind "zip"     embed_zip(mapzip) runs in the CPU worker on the open map zip (generation/archive.py MapZip).
  kind "map"     embed_map(m, v) gets the catalogue map record and version record.
  kind "derived" derive(store_root) computes vectors from other stores.
  norm "center"  search centres and L2-normalises the view (learned embeddings);
  norm "zscore"  search z-scores and L2-normalises it (hand-made / interpretable features);
  norm "none"    search only L2-normalises it (a view whose cosine is meant as it is, e.g. a one-hot category: the
                 cosine of two maps is 1 when they share the category and 0 otherwise).

Each encoder also declares `inputs`, what its values depend on, so a later change of exactly those inputs is what
triggers recomputation:
  "audio"            the song file          "chart"          Info.dat and the difficulty files
  "cover"            the cover image file
  "meta:<field>"     a catalogue field      "view:<id>"      another encoder's vectors (derived encoders)
  "file:<setting>"   a file named by a setting (e.g. a trained model)
Which views go into a DB is the caller's (contract 1, `views`). `role` names what a view describes
(music, chart, meta): a label the runtime uses for its defaults and filters (hands name views, never roles); it is
written to the DB manifest and is not part of the definition (changing it needs no version bump). By default it follows the kind (ROLES); a derived view takes
the role of its source.

`rows` says what one vector describes (OctantSpecs constitution.md, principle 2): "maps" (one map version, keyed by its
hash: the song and the metadata) or "charts" (one chart, keyed by charts.key: everything computed from a chart). A
"charts" encoder returns {chart key: vector} for a map instead of one vector (embed_zip / embed_map); a map without a
usable chart returns an empty dict or None. `play_styles` lists the play styles it computes (None: every play style;
charts.play_style); a chart of another play style simply gets no vector. A derived view takes its rows from its
source (row_kind).
Only "charts" views may describe charts in the distributed DB (dbrelease.py).

The fingerprint hashes the definition (id, version, dim, kind, params) together with the fingerprints of the source
views, so a change anywhere upstream changes it too. The store refuses to mix fingerprints under one version.

`models` declares the model files an encoder reads (Model: a Hugging Face repo pinned to a commit). OctantVector never
downloads them: the caller puts them under the setting `models_dir` (OctantHarvest's `octantharvest models`, from the
contract 1 function `models(views)`), and the encoder reads them from there only (model_dir). Each model's
repo@revision is part of the definition, so a model cannot change without the fingerprint changing.

`credits` lists what the view owes its sources (a model, ported code, a data set) as plain-text lines: what it is,
by whom, where, on what terms. The release writes them, with those of the stem and the source views (all_credits),
into the view's entry of the DB manifest, so the credits travel with the view (contract 2). Like `role`, they are
not part of the definition: changing them needs no version bump. A view built from BeatSaver's data alone needs none
(the DB's own notice credits BeatSaver).

Plugins (other packages) register encoders the same way through the entry-point group `octantvector.encoders`, using
only `octantvector.plugin`; a plugin built for another PLUGIN_API is not loaded (load_plugins).
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...common.dbformat import ROW_KINDS
from .. import config
from ..errors import GenerationError
from ..store import Store

REGISTRY = {}
INPUT_KINDS = ("audio", "cover", "chart", "meta:", "view:", "file:")
NORMS = ("center", "zscore", "none")
PLUGIN_GROUP = "octantvector.encoders"
# the unstable plugin API (octantvector.plugin): any change to what it exports bumps it, and a plugin declaring another
# number is not loaded
PLUGIN_API = "0.3"  # 0.2: Encoder.credits; 0.3: ImageEncoder, norm "none"
PLUGIN_ERRORS = {}  # entry-point name -> why that plugin's encoders are not registered
PLUGINS = set()  # entry-point names of the plugins whose encoders are registered
ROLES = {"audio": "music", "image": "cover", "zip": "chart", "map": "meta"}
_staged = None  # while a plugin module is imported: the classes it registers, committed only if its API matches
_plugins_loaded = False


def _plain(name):
    return isinstance(name, str) and name not in ("", ".", "..") and not any(c in name for c in "/\\:\0")


@dataclass(frozen=True)
class Model:
    """Model files an encoder reads: `files` of the Hugging Face repo `repo` at the commit `revision`, kept in
    <models_dir>/hf/<repo>/<revision>/ (path)."""
    repo: str
    revision: str
    files: tuple

    @property
    def key(self):
        return f"{self.repo}@{self.revision}"

    @property
    def path(self):
        return f"hf/{self.repo}/{self.revision}"

    def check(self):
        """-> why this declaration is unusable, or None. The repo and files become folder and file names."""
        parts = self.repo.split("/") if isinstance(self.repo, str) else []
        if len(parts) != 2 or not all(map(_plain, parts)):
            return f"repo must be 'owner/name', got {self.repo!r}"
        if not (isinstance(self.revision, str) and len(self.revision) == 40
                and set(self.revision) <= set("0123456789abcdef")):
            return f"{self.repo}: revision must be a full 40-hex commit, got {self.revision!r}"
        if not isinstance(self.files, tuple) or not self.files or not all(map(_plain, self.files)):
            return f"{self.key}: files must be a non-empty tuple of file names, got {self.files!r}"
        return None


def register(cls):
    if cls.id in REGISTRY or any(c.id == cls.id for c in _staged or ()):
        raise ValueError(f"duplicate encoder id {cls.id}")
    if not cls.inputs or any(not i.startswith(INPUT_KINDS) or i.endswith(":") for i in cls.inputs):
        raise ValueError(f"{cls.id}: inputs must be a non-empty tuple of {INPUT_KINDS}, got {cls.inputs!r}")
    if (cls.kind == "derived") != any(i.startswith("view:") for i in cls.inputs):
        raise ValueError(f"{cls.id}: derived encoders, and only they, read other views ('view:<id>' inputs)")
    if cls.rows not in ROW_KINDS or (cls.rows == "charts" and cls.kind in ("audio", "image")):
        raise ValueError(f"{cls.id}: rows must be one of {ROW_KINDS} (audio and image views describe maps), "
                         f"got {cls.rows!r}")
    if cls.norm not in NORMS:
        raise ValueError(f"{cls.id}: norm must be one of {NORMS}, got {cls.norm!r}")
    if not isinstance(cls.models, tuple) or not all(isinstance(m, Model) for m in cls.models):
        raise ValueError(f"{cls.id}: models must be a tuple of Model, got {cls.models!r}")
    bad = next((why for why in (m.check() for m in cls.models) if why), None)
    if bad:
        raise ValueError(f"{cls.id}: {bad}")
    if not isinstance(cls.credits, tuple) or not all(isinstance(c, str) and c.strip() for c in cls.credits):
        raise ValueError(f"{cls.id}: credits must be a tuple of non-empty strings, got {cls.credits!r}")
    if _staged is not None:
        _staged.append(cls)
    else:
        REGISTRY[cls.id] = cls
    return cls


def load_plugins(eps=None):
    """Import the modules other packages name under the entry-point group `octantvector.encoders`. What a module
    registers while it is imported is kept only if the module declares OCTANTVECTOR_PLUGIN_API == PLUGIN_API. A plugin
    that fails is recorded in PLUGIN_ERRORS, not raised, so the other views keep working; get() names it."""
    global _staged
    if eps is None:
        from importlib.metadata import entry_points
        eps = entry_points(group=PLUGIN_GROUP)
    for ep in eps:
        _staged = []
        try:
            module = ep.load()
        except Exception as ex:
            PLUGIN_ERRORS[ep.name] = f"{ep.value} failed to import: {ex!r}"
            PLUGINS.discard(ep.name)
            continue
        else:
            staged = _staged
        finally:
            _staged = None
        api = getattr(module, "OCTANTVECTOR_PLUGIN_API", None)
        if api != PLUGIN_API:
            # imported before (directly, not through here): what it registered then went straight to REGISTRY
            for i in [i for i, c in REGISTRY.items() if _from(c, module)]:
                del REGISTRY[i]
            PLUGIN_ERRORS[ep.name] = (f"{ep.value} is built for plugin API {api!r}, this OctantVector has {PLUGIN_API!r}"
                                      if api is not None else f"{ep.value} declares no OCTANTVECTOR_PLUGIN_API")
            PLUGINS.discard(ep.name)
            continue
        PLUGIN_ERRORS.pop(ep.name, None)
        PLUGINS.add(ep.name)
        REGISTRY.update((c.id, c) for c in staged)


def _from(cls, module):
    return cls.__module__ == module.__name__ or cls.__module__.startswith(module.__name__ + ".")


def registry():
    """REGISTRY with the plugins loaded (on first use, not at import: a plugin imports octantvector.plugin, which
    imports this package, so loading plugins while it is being imported would find it half-made)."""
    global _plugins_loaded
    if not _plugins_loaded:
        _plugins_loaded = True
        load_plugins()
    return REGISTRY


def plugin_errors():
    """'' or why some plugins are not loaded, to add to a message about a missing view."""
    if not PLUGIN_ERRORS:
        return ""
    return "; plugins not loaded: " + "; ".join(f"{n} ({why})" for n, why in sorted(PLUGIN_ERRORS.items()))


def not_provided():
    """What to add to a message about a view id no registered encoder has: the plugins that are loaded (none when the
    plugin that provides it is not installed, e.g. one built for another OctantVector version) and why others are not."""
    return f"; no loaded plugin provides it (loaded: {', '.join(sorted(PLUGINS)) or 'none'}){plugin_errors()}"


def get(encoder_id):
    try:
        return registry()[encoder_id]()
    except KeyError:
        raise GenerationError(f"unknown encoder {encoder_id!r}; available: {', '.join(sorted(REGISTRY))}"
                              + plugin_errors()) from None


def clips(y, sr, clip_s, n):
    """n windows of clip_s seconds spread over 15-75% of the song (skips intro/outro); short songs are padded."""
    L = int(clip_s * sr)
    if len(y) <= L:
        return [np.pad(y, (0, L - len(y)))]
    return [y[int(s):int(s) + L] for s in np.linspace(0.15, 0.75, n) * (len(y) - L)]


def resample(y, sr, target):
    if sr == target:
        return y
    import soxr
    return soxr.resample(y, sr, target)


class Encoder:
    kind = "map"
    norm = "zscore"
    version = "1"
    inputs = ()
    role = None
    rows = "maps"
    play_styles = None
    models = ()
    credits = ()

    def role_name(self):
        if self.role:
            return self.role
        return get(self.source).role_name() if self.kind == "derived" else ROLES[self.kind]

    def row_kind(self):
        return get(self.source).row_kind() if self.kind == "derived" else self.rows

    def params(self):
        return {}

    def own_models(self):
        """The models this encoder itself reads (ClipAudioEncoder adds its stem's)."""
        return self.models

    def own_credits(self):
        """The credits of this encoder itself (ClipAudioEncoder adds its stem's)."""
        return self.credits

    def all_credits(self):
        """What the view owes: its own credits and its sources' (no repeats, in order). Written to the DB manifest."""
        out = list(self.own_credits())
        for s in self.sources():
            out += [c for c in get(s).all_credits() if c not in out]
        return out

    def required_models(self):
        """Every model needed to compute this view: its own and its sources' (no repeats, in order)."""
        out = list(self.own_models())
        for s in self.sources():
            out += [m for m in get(s).required_models() if m not in out]
        return out

    def model_dir(self, model=None):
        """The local folder of `model` (default: the first declared) under the setting models_dir, after checking
        every declared file is there. OctantVector never downloads a model."""
        m = model or self.own_models()[0]
        root = config.load().models_dir
        d = Path(root) / m.path if root else None
        if d is None or not all((d / f).is_file() for f in m.files):
            raise GenerationError(f"{m.key} is not in {root or 'models_dir (not set)'}: run `octantharvest models`")
        return d

    def sources(self):
        """Ids of the views this encoder is computed from."""
        return [i[5:] for i in self.inputs if i.startswith("view:")]

    @property
    def source(self):
        """The single source view of a derived encoder."""
        (s,) = self.sources()
        return s

    def definition(self):
        """What the store records in encoder.json and compares on open (JSON-normalised)."""
        params = self.params()
        if self.own_models():
            params = {**params, "models": [m.key for m in self.own_models()]}
        d = {"id": self.id, "version": self.version, "dim": self.dim, "kind": self.kind, "params": params,
             "rows": self.rows, "play_styles": list(self.play_styles) if self.play_styles else None}
        return json.loads(json.dumps(d))

    def fingerprint(self):
        up = {s: get(s).fingerprint() for s in self.sources()}
        blob = json.dumps({"definition": self.definition(), "sources": up}, sort_keys=True)
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()



class DerivedEncoder(Encoder):
    """A view computed row by row from one source view's stored vectors: values(X, device) maps (rows, source dim)
    to (rows, dim). derive reads the source store and writes only the rows this view does not have yet; the hook is
    not called when there are none (so a model it loads is not loaded for nothing)."""
    kind = "derived"

    def values(self, X, device="cpu"):
        raise NotImplementedError

    def derive(self, store_root, device="cpu"):
        dst = Store(self, store_root)
        done = dst.done()
        hs, X = Store(get(self.source), store_root).load()
        new = [i for i, h in enumerate(hs) if h not in done]
        if new:
            for i, v in zip(new, self.values(X[new], device)):
                dst.add(hs[i], v)
            dst.flush()
        return len(dst.done())


class AudioEncoder(Encoder):
    kind = "audio"
    norm = "center"
    inputs = ("audio",)
    sr = None

    def setup_worker(self):  # CPU-side objects (feature extractors), once per worker process
        pass

    def setup_model(self, device):  # model, main process only
        pass

    batch = 1  # > 1: the heavy pass hands embed_many this many maps at once

    def embed_many(self, prepped):
        """-> one output (vector or None) per prepped map."""
        return [self.embed(x) for x in prepped]


class ImageEncoder(Encoder):
    """A view of the map's cover image: prep(data) in the CPU worker gets the image file's bytes and returns a picklable
    value (decoded and resized); embed(prepped) in the main process (GPU when available) returns a (dim,) vector or
    None. With `batch` > 1, embed_many gets that many maps at once, as for audio."""
    kind = "image"
    norm = "center"
    inputs = ("cover",)

    def setup_worker(self):
        pass

    def setup_model(self, device):
        pass

    batch = 1

    def prep(self, data):
        raise NotImplementedError

    def embed(self, prepped):
        raise NotImplementedError

    def embed_many(self, prepped):
        return [self.embed(x) for x in prepped]
