"""Named pools: which current versions a computation covers. The caller names the pool (api.compute, `octantvector
update --pool`); calibration and releases always use `distribution`. The defaults are below; the config file's
[pools.<name>] tables (contract 1: params["pools"]) change them or add pools. Keys:

  min_upvotes  int, default 0: upvote floor
  standard     bool, default false: only versions with a Standard chart
  sample       int or null: keep only this many maps, the same ones every time for the same seed and catalogue
               (ranked by sha256("<seed>:<map id>"); the map id stays with the map across its versions)
  seed         str or int, default "octantvector"
  include      path or null: a text file, one BeatSaver map id or version hash per line ('#' starts a comment);
               those maps are members whenever that version is current, on top of the others

  distribution  every current version (what the distributed DB holds)
  trial         20,000 maps sampled from every current version (trying a view without computing everything)
"""
import hashlib
from pathlib import Path

from . import config
from .catalogue import standard_diffs
from .errors import GenerationError

KEYS = {"min_upvotes": 0, "standard": False, "sample": None, "seed": "octantvector", "include": None}
DEFAULTS = {"distribution": {}, "trial": {"sample": 20000}}


def _checked(name, d):
    bad = [k for k, ok in (("min_upvotes", isinstance(d["min_upvotes"], int) and not isinstance(d["min_upvotes"], bool)),
                           ("standard", isinstance(d["standard"], bool)),
                           ("sample", d["sample"] is None or (isinstance(d["sample"], int)
                                                              and not isinstance(d["sample"], bool) and d["sample"] >= 0)),
                           ("seed", isinstance(d["seed"], (str, int)) and not isinstance(d["seed"], bool)),
                           ("include", d["include"] is None or isinstance(d["include"], (str, Path))))
           if not ok]
    if bad:
        raise GenerationError(f"[pools.{name}]: wrong type of {', '.join(bad)} (pools.py)")
    return d


def load():
    """{name: definition} with every key filled in."""
    given = {name: {} for name in DEFAULTS}
    for name, d in config.section("pools").items():
        if not isinstance(d, dict) or set(d) - set(KEYS):
            raise GenerationError(f"[pools.{name}] may only set {', '.join(KEYS)}")
        given.setdefault(name, {}).update(d)
    return {name: _checked(name, {**KEYS, **DEFAULTS.get(name, {}), **d}) for name, d in given.items()}


def matches(d, m, v):
    """The per-version filters of a definition (min_upvotes, standard); sample and include apply to the whole pool."""
    return m.get("stats", {}).get("upvotes", 0) >= d["min_upvotes"] and (not d["standard"] or bool(standard_diffs(v)))


def _included(path):
    try:
        lines = Path(path).expanduser().read_text().splitlines()
    except OSError as ex:
        raise GenerationError(f"pool include file {path}: {ex}") from None
    return {s.lower() for s in (line.split("#", 1)[0].strip() for line in lines) if s}


def members(d, eligible, current):
    """Hashes of the pool. eligible: [(map id, hash)] of the current versions that pass `matches`; current: {map id:
    hash} of every current version."""
    if d["sample"] is None:
        out = {h for _, h in eligible}
    else:
        rank = sorted(eligible, key=lambda t: (hashlib.sha256(f"{d['seed']}:{t[0]}".encode()).hexdigest(), t[1]))
        out = {h for _, h in rank[:d["sample"]]}
    if d["include"] is not None:
        hashes = {h.lower(): h for h in current.values()}
        keys = {str(k).lower(): h for k, h in current.items()}
        out |= {keys.get(x) or hashes[x] for x in _included(d["include"]) if x in keys or x in hashes}
    return out
