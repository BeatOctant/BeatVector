"""What a hand (手札) may say, checked without the runtime (OctantSpecs plans/presets.md §8). The runtime lists and
expands hands with it (runtime/presets.py), and generation refuses a hand the runtime would leave out before it is
written into a DB (generation/dbrelease.py). `views` is {view key: role} and `genres` the DB's genre words."""
import numbers

from .params import AIMS, DEFAULTS, why_invalid


def is_text(value):
    """A shown text: a non-empty string, or {language: non-empty string} with at least one language (OctantSpecs
    plans/presets.md, 多言語化 2026-10-11). It is passed through as is; OctantPath picks the language it shows."""
    if isinstance(value, str):
        return bool(value)
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in value.items()))


def view_keys(name, views):
    """The view keys a weight's name stands for: the view itself (id@version) or every version of a view id the DB
    has (a bare id, so a hand that names one embedding survives a new version of it). A role is a label, not a name
    a hand can use (2026-10-14): which views a DB has is its maker's choice, so a hand says which views it means.
    `views` holds the DB's view keys; [] when the DB has none of them."""
    if name in views:
        return [name]
    return [k for k in views if k.split("@")[0] == name]


def _views(value, views, genres):
    if not isinstance(value, dict) or not value:
        return "needs views with weights"
    for name, w in value.items():
        if not isinstance(w, numbers.Real) or isinstance(w, bool) or w <= 0:
            return f"the weight of {name!r} is not a positive number"
        if not view_keys(name, views):
            return f"the DB has no view {name!r}"
    return None


def _view_groups(value, views, genres):
    if not isinstance(value, list) or len(value) < 2:
        return "all is not a list of two or more views with weights"
    for group in value:
        why = _views(group, views, genres)
        if why:
            return why
    return None


def _difficulty(value, views, genres):
    if value in AIMS or value == "auto":
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and 0 <= value <= 1:
        return None
    return f"difficulty {value!r} is not an aim or a clear chance"


def _genre(value, views, genres):
    return None if value in genres else f"the DB has no genre word {value!r}"


def _genre_from(value, views, genres):
    if not isinstance(value, list) or not value:
        return "genreFrom is not a list of genre words"
    missing = [w for w in value if w not in genres]
    return f"the DB has no genre words {missing!r}" if missing else None


def _params(value, views, genres):
    if not isinstance(value, dict):
        return "params is not an object"
    unknown = sorted(set(value) - set(DEFAULTS))
    if unknown:
        return f"unknown params {unknown}"
    # A DB carries hands, so their values are checked like a request's (a runaway value must not come from a DB)
    return why_invalid(value)


def _hand_ids(value, views, genres):
    """The hands a mixed hand draws from: two or more distinct ids. Whether each one is in the list, and can be
    served, is checked against the whole list (why_not_mixing), not here."""
    if not isinstance(value, list) or len(value) < 2:
        return "hands is not a list of two or more hand ids"
    if not all(isinstance(h, str) and h for h in value):
        return "hands has an entry that is not a hand id"
    if len(set(value)) != len(value):
        return "hands names a hand twice"
    return None


# Handlings this OctantVector knows: {handling: {knob: (check, contract 1 (OctantCompass/OctantHarvest to OctantVector) condition it becomes)}}.
#   near              rank by the weighted views around the seed (or the history without one); without a seed,
#                     `genreFrom` picks the word of the list the liked maps are closest to and keeps its maps.
#                     `all` in place of `weights`: two or more views with weights, each of which the map must be
#                     close in (recommend ranks by the lowest percentile: "close in this and in that")
#   narrow_then_rank  keep the maps closest by `narrow`, then rank them by `rank` (recommend's `narrow` condition)
#   mixed             one list drawn from the hands named in `hands` (ids of other hands in the same list, not mixed
#                     ones), each hand's share of the places set by the player's reactions to its earlier drafts
#                     (recommend's `mix` condition; spikes/2026-10-29-preset-reactions). The seed, the difficulty and
#                     the params of the request reach every hand; a hand's own knobs fill what the request leaves out
HANDLINGS = {
    "near": {"weights": (_views, "weights"), "all": (_view_groups, "all"), "difficulty": (_difficulty, "difficulty"), "genre": (_genre, "genre"),
             "genreFrom": (_genre_from, "genreFrom"), "params": (_params, "params")},
    "narrow_then_rank": {"narrow": (_views, "narrow"), "rank": (_views, "weights"),
                         "difficulty": (_difficulty, "difficulty"), "params": (_params, "params")},
    "mixed": {"hands": (_hand_ids, "mix"), "difficulty": (_difficulty, "difficulty"), "params": (_params, "params")},
}
REQUIRED_KNOBS = {"narrow_then_rank": ("narrow", "rank"), "mixed": ("hands",)}
EITHER_KNOBS = (("weights", "all"),)  # knobs a hand has one of at most (also one condition slot: presets._expand)
CHOICE_KNOBS = ("genre", "difficulty")  # knobs a choice input can fill (one value each)

def inputs(hand, views, genres):
    """(inputs as listed, with choice options filled and filtered, or None) and the reason when None."""
    knobs = HANDLINGS[hand["handling"]]
    out = []
    for item in hand.get("inputs", []):
        if not isinstance(item, dict):
            return None, "an input is not an object"
        item = dict(item)
        kind = item.get("kind")
        if "label" in item and not is_text(item["label"]):
            return None, "an input's label is not a text"
        if kind == "choice":
            name = item.get("name")
            if name not in knobs or name not in CHOICE_KNOBS:
                return None, f"choice {name!r} fills no knob of {hand['handling']}"
            if "options" in item:
                if not isinstance(item["options"], list):
                    return None, f"the options of choice {name!r} are not a list"
                item["options"] = [o for o in item["options"]
                                   if isinstance(o, dict) and knobs[name][0](o.get("value"), views, genres) is None
                                   and ("label" not in o or is_text(o["label"]))]
            elif name == "genre":
                # The DB declares no display names for its genre words yet; show the word itself.
                item["options"] = [{"value": g, "label": g} for g in sorted(genres)]
            if not item.get("options"):
                if item.get("optional"):
                    continue
                return None, f"choice {name!r} has no option this DB can serve"
        elif kind != "seed":
            if item.get("optional"):
                continue  # an input this OctantVector cannot fill: left out of an optional place
            return None, f"input kind {kind!r} is unknown"
        out.append(item)
    return out, None


def why_not(hand, views, genres):
    """None when the hand can be served by this DB (`views`: {view key: role}, `genres`: the words), else why not."""
    if not isinstance(hand, dict):
        return "not an object"
    if not isinstance(hand.get("id"), str) or not hand["id"]:
        return "no id"
    if not is_text(hand.get("label")):
        return "no label"
    handling = hand.get("handling")
    if handling not in HANDLINGS:
        return f"handling {handling!r} is unknown to this OctantVector"
    knobs = hand.get("knobs", {})
    if not isinstance(knobs, dict):
        return "knobs is not an object"
    for name in REQUIRED_KNOBS.get(handling, ()):
        if name not in knobs:
            return f"{handling} needs the knob {name!r}"
    for pair in EITHER_KNOBS:
        if all(name in knobs for name in pair):
            return f"knobs {' and '.join(pair)} are one or the other"
    for name, value in knobs.items():
        if name not in HANDLINGS[handling]:
            return f"knob {name!r} is unknown to {handling}"
        why = HANDLINGS[handling][name][0](value, views, genres)
        if why:
            return why
    if not isinstance(hand.get("inputs", []), list):
        return "inputs is not a list"
    return inputs(hand, views, genres)[1]


def needs_seed(hand, views, genres):
    """True when the hand cannot be asked without a seed (a seed input that is not optional)."""
    listed = inputs(hand, views, genres)[0] or []
    return any(item["kind"] == "seed" and not item.get("optional") for item in listed)


def why_not_mixing(hand, served, views, genres):
    """None when a mixed hand can draw from every hand it names, else why not. `served`: {id: hand} of the hands of
    the same list that passed why_not. A named hand must be there and not be mixed itself (one level: the reactions
    are recorded per hand, so a mix of mixes would have no reactions of its own to learn from); when the mixed hand
    can be asked without a seed, so must every hand it names, or that hand could not be asked."""
    if hand.get("handling") != "mixed":
        return None
    for name in hand["knobs"]["hands"]:
        other = served.get(name)
        if other is None or name == hand["id"]:
            return f"mixed names the hand {name!r}, which is not in the list"
        if other.get("handling") == "mixed":
            return f"mixed names the hand {name!r}, which is mixed itself"
        if not needs_seed(hand, views, genres) and needs_seed(other, views, genres):
            return f"mixed may be asked without a seed but the hand {name!r} needs one"
        if any(item["kind"] == "choice" and not item.get("optional") for item in inputs(other, views, genres)[0] or []):
            return f"the hand {name!r} needs a choice, which a mixed hand does not gather"
    return None


def check_hands(hands, views, genres):
    """{index or id: reason} for every hand that would be left out (and duplicate ids)."""
    out, seen, served = {}, set(), {}
    for n, hand in enumerate(hands):
        why = why_not(hand, views, genres)
        key = hand.get("id") if isinstance(hand, dict) and isinstance(hand.get("id"), str) else n
        if why is None and key in seen:
            why = "the id is used by an earlier hand"
        if why:
            out[key] = why
        else:
            served[key] = hand
        seen.add(key)
    for hand in hands:  # a mixed hand draws from the hands that passed, wherever they are in the list
        if isinstance(hand, dict) and hand.get("handling") == "mixed" and hand.get("id") not in out:
            why = why_not_mixing(hand, served, views, genres)
            if why:
                out[hand["id"]] = why
    return out
