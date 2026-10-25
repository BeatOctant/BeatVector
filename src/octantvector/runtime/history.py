"""The player's history as OctantCompass passes it: contract 3 (OctantPath to OctantCompass) `events`, unfiltered (OctantSpecs contracts/, contract 1 (OctantCompass to OctantVector)).

Each event has `type` (upvote / downvote / like / play / user_delete / recommendation / imported_play), `hash` and
`at`. A play also has `characteristic`, `difficulty`, `result` (clear / fail / quit / restart) and may have `modifiers` and a
score.
`imported_play` (`count` per chart, made once by OctantPath from the game's PlayerData when OctantPath is installed)
alone makes a map neither liked nor a goal (likes.py): the count includes quits and has no results (OctantSpecs
octantvector/law.md 4.1). Its chart still says which charts of a liked map the player played, so it can stand in for
plays as a taste reference (recommend.py). One with `cleared` (PlayerData has a score for the chart: cleared at least
once, modifiers unknown) fills the strong set behind the player's own clears (skill.py), so the skill is there from
the first play; it never counts as a clear or a fail anywhere else.
Granularity is kept: plays are per chart, the other signals per map (OctantSpecs constitution.md, principle 2).

An event on an old version is moved to the current version of the same map; a play moves to the chart with the same
characteristic and difficulty, which may be a remade chart. Every move is reported. Events on maps the DB does not
know are kept out and counted.
"""
from dataclasses import dataclass, field

from ..common import charts as chartids
from . import likes


@dataclass
class Play:
    hash: str
    chart: str | None  # chart key in the DB, or None when the current version has no such chart
    result: str        # the event's result as sent (clear / fail / quit / restart); the counts fold restart into quit
    at: str
    plain: bool        # speed 1.0 and no modifiers: usable for the skill (difficulty-fit.md, 2)


@dataclass
class History:
    counts: dict = field(default_factory=dict)    # hash -> merged counts in likes.py's form
    plays: list = field(default_factory=list)     # Play, oldest first
    replaced: list = field(default_factory=list)  # {"from", "to"} for every old version moved to the current one
    imported: dict = field(default_factory=dict)  # hash -> chart keys in the DB its imported_play events name
    imported_clears: dict = field(default_factory=dict)  # chart key in the DB -> PlayerData play count, for the
                                                         # imported_play events with `cleared`
    unknown: int = 0                              # events on maps the DB does not have

    def liked(self):
        return {h for h, s in self.counts.items() if likes.classify(s) == "liked"}

    def known(self):
        """Maps the player already interacted with (played, liked, voted on, deleted)."""
        return {h for h, s in self.counts.items() if any(s.values())}

    def goals(self):
        """Goals (a favourite or an upvote on a map not cleared yet, law 4.1), except maps the player deleted, which
        stay out like any other known map. Inferred, not decided in OctantSpecs: law 4.1 only makes a deleted map less
        likely, and how strongly is open (open.md)."""
        return {h for h, s in self.counts.items() if likes.classify(s) == "goal" and not s["deleted"]}


def _plain(e):
    """Speed 1.0 and no other modifier (`modifiers`: {"speed": 1.0, "noFail": false}, contract 3)."""
    mods = e.get("modifiers") or {}
    return mods.get("speed") in (None, 1, 1.0) and not any(v for k, v in mods.items() if k != "speed")


def _current(db, h, replaced, seen):
    cur = db.current(h)
    if cur is not None and cur != h and h not in seen:
        seen.add(h)
        replaced.append({"from": h, "to": cur})
    return cur


def parse(db, events):
    hist, seen = History(), set()
    votes, first_clear = {}, {}
    for e in sorted(events or (), key=lambda e: str(e.get("at", ""))):
        t, h = e.get("type"), e.get("hash")
        if not h or t not in ("upvote", "downvote", "like", "like_removed", "play", "user_delete", "recommendation", "imported_play"):
            continue
        if t == "recommendation":  # never a taste signal (contract 1, 好みの判定ルール)
            continue
        cur = _current(db, h, hist.replaced, seen)
        if cur is None:
            hist.unknown += 1
            continue
        s = hist.counts.setdefault(cur, {"favorite": 0, "upvote": 0, "play": 0, "finished": 0, "failed": 0,
                                         "quit": 0, "deleted": 0, "pd_playcount": 0, "returned": 0})
        if t in ("upvote", "downvote"):
            votes[cur] = t  # the latest vote wins (votedSongs.json overwrites too)
        elif t == "like":
            s["favorite"] = 1
        elif t == "like_removed":  # back to no favorite, not a dislike; the latest like/like_removed wins
            s["favorite"] = 0
        elif t == "imported_play":
            s["pd_playcount"] += int(e.get("count") or 0)
            key = chartids.key(cur, e.get("characteristic"), e.get("difficulty"))
            if key in db.chart_pos:
                hist.imported.setdefault(cur, set()).add(key)
                if e.get("cleared") is True:
                    hist.imported_clears[key] = hist.imported_clears.get(key, 0) + int(e.get("count") or 0)
        elif t == "user_delete":
            s["deleted"] += 1
        else:
            res = e.get("result")
            s["play"] += 1
            s[{"clear": "finished", "fail": "failed"}.get(res, "quit")] += 1
            day = str(e.get("at", ""))[:10]  # `at` is UTC (contract 3)
            if res in ("clear", "fail") and cur in first_clear and day > first_clear[cur]:
                s["returned"] = 1  # played again on a later day than the first clear (law 4.1)
            if res == "clear":
                first_clear.setdefault(cur, day)
            key = chartids.key(cur, e.get("characteristic"), e.get("difficulty"))
            hist.plays.append(Play(cur, key if key in db.chart_pos else None, res, str(e.get("at", "")), _plain(e)))
    for h, v in votes.items():
        hist.counts[h]["upvote"] = int(v == "upvote")
    return hist
