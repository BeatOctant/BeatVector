"""Encoder registry: the built-in encoders register when this package is imported, plugin encoders when the registry
is first used (registry(), get()). List, known issues and how to add one: OctantSpecs octantvector/encoders.md."""
from .base import (REGISTRY, AudioEncoder, Encoder, ImageEncoder, Model, get, load_plugins, register,  # noqa: F401
                   registry)
from .audio import ClipAudioEncoder, HiddenMean  # noqa: F401
from . import chart_patterns, catalogue_features, swing  # noqa: F401,E402
