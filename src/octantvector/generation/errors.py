"""The one error type of the generation side (reaudit decision 2): a call that cannot be done as asked raises
GenerationError, never SystemExit, so OctantHarvest can record it as the run's error and carry on."""
from ..common import dbformat


class GenerationError(Exception):
    """A generation call could not be done as asked; the message says why."""


class VersionError(GenerationError, dbformat.VersionError):
    """dbformat.VersionError raised by the generation side: `except GenerationError` catches it, and code that
    catches dbformat.VersionError is unaffected."""

    @classmethod
    def of(cls, ex):
        """The same error as a generation VersionError (message and fields kept)."""
        out = cls.__new__(cls, *ex.args)
        out.args, out.what, out.found, out.reads = ex.args, ex.what, ex.found, ex.reads
        return out
