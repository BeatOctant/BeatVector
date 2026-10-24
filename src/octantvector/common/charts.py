"""Chart identity and play style (OctantSpecs constitution.md, principles 1 and 2; contracts/, contract 2 (DB build to recommendation core)).

A chart is what a player plays: one difficulty of one characteristic of one map version. Views computed from the
charts are stored per chart under the key "<version hash>:<characteristic>:<difficulty>"; views computed from the
song or the map's metadata are stored per version hash. key_hash() turns either kind of key into the version hash,
so the ledger, pools and calibration keep working per map version.

Each chart has a play style. Charts of different play styles are never compared as if they were the same thing: a
chart view only measures similarity between charts of the same play style. The play style is the characteristic's
name for now; a play style that the characteristic alone cannot tell apart would be decided here.
"""
import numpy as np

SEP = ":"
DIFF_ORDER = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
STANDARD = "Standard"


def key(version_hash, characteristic, difficulty):
    return f"{version_hash}{SEP}{characteristic}{SEP}{difficulty}"


_DIFF_SPELLING = {**{d.lower(): d for d in DIFF_ORDER}, "expert+": "ExpertPlus"}


def difficulty(name):
    """A difficulty name spelled as BeatSaver's catalogue spells it ("normal" -> "Normal", "Expert+" ->
    "ExpertPlus"); any other name is returned as it is."""
    return _DIFF_SPELLING.get(name.lower(), name) if isinstance(name, str) else name


def difficulty_rank(name):
    """Position of a difficulty from Easy (0) to ExpertPlus (4); -1 for a name outside DIFF_ORDER."""
    return DIFF_ORDER.index(name) if name in DIFF_ORDER else -1


def catalogue_spelling(chart_key):
    """The chart key with its difficulty spelled as BeatSaver's catalogue spells it. Some older zips write
    "normal" or "Expert+" in Info.dat for the catalogue's "Normal" and "ExpertPlus"; any other key is returned as
    it is (a difficulty that differs in more than spelling is a different chart)."""
    if not is_chart_key(chart_key):
        return chart_key
    h, c, d = split(chart_key)
    return key(h, c, difficulty(d))


def split(chart_key):
    """-> (version hash, characteristic, difficulty)"""
    h, c, d = chart_key.split(SEP, 2)
    return h, c, d


def key_hash(k):
    """Version hash of a chart key; a version hash (a map row) is returned as it is."""
    return k.split(SEP, 1)[0]


def is_chart_key(k):
    return SEP in k


def play_style(characteristic):
    return characteristic or STANDARD


def _rank(k):
    _, c, d = split(k)
    r = difficulty_rank(d)
    return r if play_style(c) == STANDARD and r >= 0 else None


def hardest_standard(keys):
    """The hardest Standard chart (up to ExpertPlus) among chart keys, or None."""
    ranked = [(_rank(k), k) for k in keys if _rank(k) is not None]
    return max(ranked)[1] if ranked else None


def representative(keys, vecs):
    """One chart per map version for map-level experiments (search, eval-personal): the hardest Standard
    difficulty up to ExpertPlus, which is what the chart views described before they were stored per chart.
    -> (version hashes, vectors)."""
    best = {}
    for n, k in enumerate(keys):
        rank, h = _rank(k), key_hash(k)
        if rank is not None and (h not in best or rank > best[h][0]):
            best[h] = (rank, n)
    hs = sorted(best)
    x = np.stack([vecs[best[h][1]] for h in hs]) if hs else np.zeros((0, np.asarray(vecs).shape[-1]), np.float32)
    return hs, x
