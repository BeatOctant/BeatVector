"""The one entry point for packages that add encoders (views) to OctantVector. UNSTABLE: any change to what it exports
bumps PLUGIN_API, and a plugin built for another number is not loaded (the reason is shown when its view is asked
for). Documented in OctantSpecs octantvector/ once settled; it is not part of contract 1.

A plugin is a module named under the entry-point group `octantvector.encoders` that sets
`OCTANTVECTOR_PLUGIN_API = "<PLUGIN_API it was written for>"` and decorates its encoder classes with `register`. It
imports only this module. Its models are declared (Model) and read from models_dir; OctantVector never downloads them.
"""
from .generation.encoders.audio import ClipAudioEncoder, HiddenMean
from .generation.encoders.base import (PLUGIN_API, AudioEncoder, DerivedEncoder, Encoder, ImageEncoder, Model, clips,
                                       register, resample)
from .generation.errors import GenerationError

__all__ = ["PLUGIN_API", "Encoder", "AudioEncoder", "ImageEncoder", "DerivedEncoder", "ClipAudioEncoder", "HiddenMean",
           "Model", "register", "clips", "resample", "GenerationError"]
