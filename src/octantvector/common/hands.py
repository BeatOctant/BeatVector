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


def _difficulty(value, views, genres):
    if value in AIMS or value == "auto":
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and 0 <= value <= 1:
        return None
    return f"difficulty {value!r} is not an aim or a clear chance"


def _genre(value, views, genres):
    return None if value in genres else f"the DB has no genre word {value!r}"


def _params(value, views, genres):
    if not isinstance(value, dict):
        return "params is not an object"
    unknown = sorted(set(value) - set(DEFAULTS))
    if unknown:
        return f"unknown params {unknown}"
    # A DB carries hands, so their values are checked like a request's (a runaway value must not come from a DB)
    return why_invalid(value)


# Handlings this OctantVector knows: {handling: {knob: (check, contract 1 (OctantCompass/OctantHarvest to OctantVector) condition it becomes)}}.
#   near              rank by the weighted views around the seed (or the history without one)
#   narrow_then_rank  keep the maps closest by `narrow`, then rank them by `rank` (recommend's `narrow` condition)
HANDLINGS = {
    "near": {"weights": (_views, "weights"), "difficulty": (_difficulty, "difficulty"), "genre": (_genre, "genre"),
             "params": (_params, "params")},
    "narrow_then_rank": {"narrow": (_views, "narrow"), "rank": (_views, "weights"),
                         "difficulty": (_difficulty, "difficulty"), "params": (_params, "params")},
}
REQUIRED_KNOBS = {"narrow_then_rank": ("narrow", "rank")}
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
    for name, value in knobs.items():
        if name not in HANDLINGS[handling]:
            return f"knob {name!r} is unknown to {handling}"
        why = HANDLINGS[handling][name][0](value, views, genres)
        if why:
            return why
    if not isinstance(hand.get("inputs", []), list):
        return "inputs is not a list"
    return inputs(hand, views, genres)[1]


def check_hands(hands, views, genres):
    """{index or id: reason} for every hand that would be left out (and duplicate ids)."""
    out, seen = {}, set()
    for n, hand in enumerate(hands):
        why = why_not(hand, views, genres)
        key = hand.get("id") if isinstance(hand, dict) and isinstance(hand.get("id"), str) else n
        if why is None and key in seen:
            why = "the id is used by an earlier hand"
        if why:
            out[key] = why
        seen.add(key)
    return out
