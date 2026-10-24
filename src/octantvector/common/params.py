"""Defaults of the numbers `recommend` and `describe` take in `params` (OctantSpecs contracts/, contract 1 (OctantCompass/OctantHarvest to OctantVector)).

OctantCompass passes the values from the settings JSON; a key it leaves out uses the default here. Unknown keys are
ignored. Where OctantSpecs already gives a default, the value is that one and the comment names the table:
  difficulty-fit.md, 設定                 the difficulty keys
The others are provisional values of this runtime, not decided in OctantSpecs.
"""

DEFAULTS = {
    "minViewWeight": 0.5,          # share of the view weight a candidate must have vectors for (OctantSpecs law 3.1;
                                   # also what the development search keeps, cli.py)
    "includeKnown": False,         # provisional: also recommend maps the player already played, liked or voted on
    "recentPlays": 200,            # provisional: plays that count as "recent" for the genre entry (OctantSpecs Q2: undecided)
    "genreTopShare": 0.1,          # provisional: share of maps, closest to the word, kept by a genre filter
    "narrowTopMaps": 300,          # provisional: maps kept by `narrow` before ranking by `weights`
    "narrowWiden": 3,              # provisional: narrowTopMaps is multiplied by this while fewer than `count` are left (2026-10-13)
    "seedTasteWeight": 0.0,        # provisional: how much the player's taste is mixed into a seeded recommendation
    "mapsPerSong": 1,              # provisional: maps of one song (other mappers' maps of it) in a list; 0: no limit
    "mapsPerArtist": 3,            # maps of one artist (the same artist set) in a list; 0: no limit (OctantSpecs law 3.4, 2026-10-12)
    "sameSongCosine": 0.9,         # provisional: mean music-view cosine from which two maps are the same song
    # difficulty (difficulty-fit.md, 設定)
    "features": {"nps": 1.0, "njs": 1.0},  # chart features and their weights for the distance
    "k": 5,                        # nearest strong charts averaged into a distance
    "normalPercentile": 80,        # cleared charts' distances up to this percentile are "normal"
    "strongRecentClears": 200,     # most recent clears that form the strong set
    "minClears": 50,               # fewer clears in a play style: no difficulty fitting for it (section 6)
    "minFailsForCurve": 30,        # fewer fails: no clear-chance curve; fewer since the strong set's oldest clear: no
                                   # weak directions
    "directionShrinkPlays": 100,   # provisional: plays (clears and fails) at which the learned weak-direction weights
                                   # and the configured ones count half each (n0 of OctantSpecs
                                   # personal-data-overfitting.md, 個人化の強さ)
    "toleranceSd": 1.0,            # allowed distance from the target, in standard deviations of the cleared distances
    "toleranceMaxSd": 2.0,         # widened up to this when too few candidates are left
    "penaltyWeight": 1.0,          # weight of the distance-from-target penalty (`beta`)
    "challengeClearChance": 0.5,   # clear chance the challenge aim targets
    "warmupScale": 0.5,            # warmup target = this x the normal target
}

AIMS = ("normal", "challenge", "warmup")

# What each number may be: (low, high, whole). The bounds are this runtime's own guard, not decided in OctantSpecs:
# wide enough for any sensible setting, narrow enough that no value can run the host away (the tolerance widening
# loop in recommend.py steps 0.5 at a time up to toleranceMaxSd, an unbounded percentile or share breaks numpy, and
# a hand in a DB carries params too: notes/security-audit-2026-10-21.md, 8). `whole` means no fraction; 300.0 passes.
BOUNDS = {
    "minViewWeight": (0, 1, False),
    "recentPlays": (0, 10**6, True),
    "genreTopShare": (0, 1, False),
    "narrowTopMaps": (1, 10**7, True),
    "narrowWiden": (2, 1000, True),
    "seedTasteWeight": (0, 1, False),
    "mapsPerSong": (0, 10**4, True),
    "mapsPerArtist": (0, 10**4, True),
    "sameSongCosine": (-1, 1, False),
    "k": (1, 10**4, True),
    "normalPercentile": (0, 100, False),
    "strongRecentClears": (1, 10**6, True),
    "minClears": (0, 10**6, True),
    "minFailsForCurve": (1, 10**6, True),
    "directionShrinkPlays": (0, 10**12, False),
    "toleranceSd": (0, 100, False),
    "toleranceMaxSd": (0, 100, False),
    "penaltyWeight": (0, 10**6, False),
    "challengeClearChance": (0.001, 0.999, False),  # log(c / (1 - c)) must stay finite (skill.py)
    "warmupScale": (0, 100, False),
}


def _number(value, low, high, whole):
    import math
    import numbers
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        return False
    if whole and value != int(value):
        return False
    return low <= value <= high


def why_invalid(params):
    """Why these params cannot be used, or None. Unknown keys are ignored as resolve ignores them (contract 1
    (OctantCompass/OctantHarvest to OctantVector)); a known key with a value outside BOUNDS, or of another type, is refused
    before it reaches the handling. Checked by recommend and describe (the request fails) and by the hand checks in
    common/hands.py (the hand is left out of the list)."""
    for key, value in (params or {}).items():
        if key not in DEFAULTS:
            continue
        if key == "includeKnown":
            if not isinstance(value, bool):
                return "includeKnown must be true or false"
        elif key == "features":
            if not isinstance(value, dict) or not all(isinstance(f, str) and _number(w, 0, 10**6, False)
                                                      for f, w in value.items()):
                return "features must map feature names to finite weights between 0 and 1000000"
        else:
            low, high, whole = BOUNDS[key]
            kind = "a whole number" if whole else "a number"
            if not _number(value, low, high, whole):
                return f"{key} must be {kind} between {low} and {high}"
    return None


def resolve(params):
    """DEFAULTS with the known keys of `params` (OctantCompass's values) put over them."""
    out = dict(DEFAULTS)
    for k, v in (params or {}).items():
        if k in DEFAULTS:
            out[k] = v
    return out
