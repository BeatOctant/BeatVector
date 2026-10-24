"""What counts as a liked (positive) map in a player's own data. The rule is OctantSpecs octantvector/law.md 4.1; its
rationale is in octantvector/design.md.

The rule (`classify`, 2026-10-10; history.py, and the CLI evaluation through eval/personal.py):
  liked        cleared at least once, and a favourite or an upvote (either one), or played again on a later day
               than the first clear
  goal         a favourite or an upvote on a map not cleared yet: not part of the taste, kept for challenges
  challenging  played repeatedly (MIN_CHALLENGE_PLAYS clears + fails or more) without a clear: only a difficulty hint
  None         anything else (a single clear, quits, fails, unplayed maps): unlabeled, never a negative
Plays that count are the ones that ended in a clear or a fail; quits (stopping mid-song) and restarts do not.
`imported_play` (the game's PlayerData count from before OctantPath, with no results or dates) alone makes a map
neither liked nor a goal. Inferred, not decided in OctantSpecs (open.md): a favourite or an upvote on a map known only
from `imported_play` is liked, because whether it was cleared before OctantPath is unknown and most such marks were
on cleared maps (Spike F13: 142 of 165); "repeatedly" is MIN_CHALLENGE_PLAYS; a "day" is the date of `at` (UTC).
"""
MIN_CHALLENGE_PLAYS = 2  # inferred: "repeatedly" in OctantSpecs law 4.1


def classify(s: dict):
    """s: one map's counts as history.py keeps them (`returned`: a clear or a fail on a later day than the first
    clear). -> "liked" | "goal" | "challenging" | None (OctantSpecs law 4.1)."""
    marked = bool(s.get("favorite") or s.get("upvote"))
    if s.get("finished", 0):
        return "liked" if marked or s.get("returned") else None
    if s.get("play", 0) == 0 and s.get("pd_playcount", 0):  # cleared before OctantPath or not: unknown
        return "liked" if marked else None
    if marked:
        return "goal"
    if s.get("failed", 0) >= MIN_CHALLENGE_PLAYS:
        return "challenging"
    return None

