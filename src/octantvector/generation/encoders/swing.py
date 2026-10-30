"""Swing-simulation views (method: OctantSpecs octantvector/swing.md; the simulation is octantvector.generation.swing)."""
import numpy as np

from ...common import charts
from .. import config
from ..errors import GenerationError
from ..swing import NAMES, chart_features, params, standard_charts, style
from .base import DerivedEncoder, Encoder, register


@register
class SwingFlow(Encoder):
    """Interpretable swing-simulation features of every Standard chart. The simulation models two sabers and
    Standard play, so other play styles get no vector until a simulation is validated for them. A chart with more than
    `max_notes` notes gets none either: the simulation of a 500,000-note chart does not finish."""
    kind = "zip"
    id = "swing_flow"
    version = "12"
    max_notes = 50000
    rows = "charts"
    play_styles = ("Standard",)
    inputs = ("chart",)
    credits = ("BeatLeader analyzer, https://github.com/BeatLeader/beatleader-analyzer, formulas ported into "
               "OctantVector (THIRD_PARTY_NOTICES.md), MIT",
               "JoshaParity (Joshabi), https://github.com/Joshabi/JoshaParity, formulas ported into OctantVector "
               "(THIRD_PARTY_NOTICES.md), MIT")

    def __init__(self):
        self.names, self.P = NAMES, params.load()
        self.dim = len(NAMES)

    def params(self):
        return {"charts": "every Standard chart", "max_notes": self.max_notes, "features": list(self.names),
                "swing": self.P}

    def embed_zip(self, mz):
        """-> {chart key: features} for every Standard chart with enough notes and at most max_notes."""
        out = {}
        for d, ch in standard_charts(mz):
            if len(ch.notes) > self.max_notes:
                continue
            v = chart_features(ch, self.P)[0]
            if v is not None:
                out[charts.key(mz.hash, "Standard", d)] = v
        return out


class _SwingTags(DerivedEncoder):
    """Shared by the tag views derived from `swing_flow`; `group` picks the fine tags (style.load_rules)."""
    inputs = ("view:swing_flow", "file:swing_model")
    group = None
    learned = False

    def __init__(self):
        self.path = config.load().swing_model
        self.rules = style.load_rules(self.group)
        self.names = ([f"p:{t}" for t in style.LEARNED_TAGS] if self.learned else []) + list(self.rules)
        self.dim = len(self.names)

    def params(self):
        return {"source": self.source, "names": self.names, "rules": self.rules,
                "model": style.model_hash(self.path) if self.path and self.path.exists() else None}

    def derive(self, store_root, device="cpu"):
        # checked before the store is opened: without the model, params() would record "model": None there
        if not self.path or not self.path.exists():
            raise GenerationError(f"{self.id} needs a style model: run `octantvector swing-train` and set swing_model "
                                  "(https://github.com/BeatOctant/OctantSpecs/blob/main/octantvector/config.md)")
        self.model = style.load_model(self.path)
        return super().derive(store_root, device)

    def values(self, X, device="cpu"):  # row by row: quantiles and weights come from the model, not from X
        S, _ = style.fine_tags(self.model, X, self.rules)
        if self.learned:
            S = np.c_[np.nan_to_num(style.learned_probs(self.model, X), nan=0.0), S]
        return S


@register
class SwingStyle(_SwingTags):
    """Play-style tags from `swing_flow`: probability of each BeatSaver style tag (learned) followed by the score of
    each fine tag (0..1) that is not about walls or bombs. Needs the model file from `octantvector swing-train`
    (setting `swing_model`)."""
    id = "swing_style"
    version = "13"
    group = "style"
    learned = True


@register
class SwingHazards(_SwingTags):
    """Wall and bomb tags from `swing_flow` (art / workout / rhythm walls, art / dodge / rhythm bombs, ...): the
    score of each (0..1). Kept apart from `swing_style` so recommendations can weigh them separately."""
    id = "swing_hazards"
    version = "10"
    group = "hazards"
