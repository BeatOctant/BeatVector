"""Swing simulation of Beat Saber charts: expected parity, wrist roll, rhythm and movement, BeatLeader-style
pass/tech ratings and style tags (learned BeatSaver tags + fine tags). Method and sources: OctantSpecs octantvector/swing.md."""
from . import params  # noqa: F401
from .chart import bpm_of, parse, standard_beatmaps
from .features import NAMES, chart_features  # noqa: F401


def standard_charts(mz):
    """[(difficulty, Chart)] for the Standard characteristic of an open MapZip, easiest first."""
    return [(d, parse(mz.json(fn), bpm_of(mz.info), njs)) for d, fn, njs in standard_beatmaps(mz.info)]
