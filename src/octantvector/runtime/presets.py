"""Presets (手札): named ways to use the embeddings that OctantPath shows as buttons (OctantSpecs plans/presets.md §8).

A hand is one handling (how the embeddings are used; the handling itself is code here, never in the DB) plus its
knobs, with an id, a label and the inputs OctantPath gathers before asking. The DB may carry its own hands in the
manifest (`presets`), which then replace the defaults below; without them the defaults are used.

What a hand needs is read from its knobs, not declared by its author (2026-10-10): a hand is listed only when
  - this OctantVector knows its handling, and every knob is one that handling takes, with a value it accepts
  - every view its knobs name, and every genre word, is in the DB (a weight names a view id of any version, or
    id@version: common.hands.view_keys; roles are labels only)
  - every input it needs is one this OctantVector can fill: a seed, or a choice whose `name` is a knob of the handling
    and that has options left (a `genre` choice without options takes the DB's words)
A hand that breaks any of these is left out on its own; the rest of the list and the DB are still used.

  presets(db)                 -> {"revision", "presets": [{id, group?, label, hint?, inputs, genreLabels?}]}
A shown text (label, group, hint, an input's or option's label) is a string or {language: string}, e.g.
{"en": "Similar songs", "ja": "似た曲"}; it is passed through as is and OctantPath picks the language it shows.
A hand may also carry `genreLabels` ({genre word: shown text}): the names OctantPath shows for the genre word a list
was narrowed to (each item's reason names it, recommend.py); the words the DB lacks and the names that are not texts
are left out of the listing, and a hand without any shows the word itself.
  expand_preset(db, request)  -> the request with `preset` and `choices` turned into contract 1 (OctantCompass to OctantVector) conditions; conditions
                                 the caller sent themselves win; a request without `preset` is returned as it is
A mixed hand (handling `mixed`, knob `hands`: ids of other hands of the list) expands into `mix`: one entry per
named hand, each with the request that hand would get on its own (the caller's conditions, then the mixed hand's
knobs, then the named hand's knobs). recommend splits the places between the entries (runtime/recommend.py).
The checks are in common/hands.py, which generation also uses to refuse a hand before it is written into a DB.
Expanding only combines conditions; it never scores, sorts or trims candidates.
"""
import hashlib
import json

from ..common.hands import EITHER_KNOBS, HANDLINGS, inputs, is_text, why_not, why_not_mixing
from .recommend import RequestError


# The list for a DB without hands of its own (OctantSpecs plans/presets.md §3): only "all", since every other hand names
# views and which views a DB has is its maker's choice (2026-10-14). The English is a guess (2026-10-11:
# English first, Japanese kept).
DEFAULT_PRESETS = [
    {"id": "all", "group": {"en": "Song & map", "ja": "曲と譜面"}, "label": {"en": "All", "ja": "すべて"},
     "hint": {"en": "Close to the selected map in every way", "ja": "選んでいる譜面に近いものを、どの軸でも"},
     "inputs": [{"kind": "seed"}], "handling": "near", "knobs": {}},
]


class UnknownPresetError(RequestError):
    """The request names a preset that is not in the current list."""


def _db_facts(db):
    return db.roles(), set(db.genres())


def _hands(db):
    """{id: (hand, inputs as listed)} for the hands this DB and this OctantVector can serve, in list order."""
    hands = db.manifest.get("presets")
    hands = hands if isinstance(hands, list) and hands else DEFAULT_PRESETS
    views, genres = _db_facts(db)
    out = {}
    for hand in hands:
        if why_not(hand, views, genres) is None and hand["id"] not in out:
            out[hand["id"]] = (hand, inputs(hand, views, genres)[0])
    served = {k: h for k, (h, _) in out.items()}
    return {k: v for k, v in out.items() if why_not_mixing(v[0], served, views, genres) is None}


def presets(db):
    listed = []
    genres = _db_facts(db)[1]
    for hand, shown in _hands(db).values():
        out = {"id": hand["id"], "label": hand["label"]}
        for key in ("group", "hint"):
            if is_text(hand.get(key)):
                out[key] = hand[key]
        names = hand.get("genreLabels")
        names = {w: t for w, t in names.items() if w in genres and is_text(t)} if isinstance(names, dict) else {}
        if names:
            out["genreLabels"] = names
        out["inputs"] = shown
        listed.append(out)
    digest = hashlib.sha256(json.dumps([db.version, listed], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {"revision": digest[:12], "presets": listed}


def expand_preset(db, request):
    request = dict(request or {})
    if request.get("preset") is None:
        return request  # `choices` belongs to a preset; without one it is not ours to drop (contract 1)
    preset_id = request.pop("preset")
    choices = request.pop("choices", None) or {}
    hands = _hands(db)
    if preset_id not in hands:
        raise UnknownPresetError(f"preset {preset_id!r} is not in the current list")
    hand, inputs = hands[preset_id]
    request = _expand(hand, inputs, request, choices, preset_id)
    if hand["handling"] == "mixed":
        # every named hand gets the request as it stands (the caller's conditions, then the mixed hand's knobs) and
        # fills the rest from its own knobs; its inputs were settled when the mixed hand was listed (why_not_mixing)
        base = {k: v for k, v in request.items() if k != "mix"}
        request["mix"] = [{"hand": name, "request": _expand(hands[name][0], [], dict(base), {}, name)}
                          for name in request["mix"]]
    request.setdefault("difficulty", "auto")
    return request


def _expand(hand, inputs, request, choices, preset_id):
    """`request` with the hand's inputs (from `choices`) and knobs put in where the request has nothing; a hand's
    `difficulty` knob is kept so recommend can tell it from the request's own (a mixed hand's entries, where the
    request's "auto" would otherwise cover a hand's own aim)."""
    knobs = HANDLINGS[hand["handling"]]
    for item in inputs:
        if item["kind"] == "seed":
            if request.get("seed") is None and not item.get("optional"):
                raise RequestError(f"preset {preset_id} needs a seed")
        elif item["kind"] == "choice":
            value = choices.get(item["name"])
            if value is None and item.get("optional"):
                continue
            if value not in {o.get("value") for o in item["options"]}:
                raise RequestError(f"preset {preset_id} needs a choice for {item['name']}")
            request.setdefault(knobs[item["name"]][1], value)
    for name, value in hand.get("knobs", {}).items():
        condition = knobs[name][1]
        if any(condition in pair and any(request.get(c) is not None for c in pair) for pair in EITHER_KNOBS):
            continue  # the request's own weights (or all) stand for the hand's
        if condition == "params":
            request["params"] = {**value, **(request.get("params") or {})}
        else:
            request.setdefault(condition, value)
    return request
