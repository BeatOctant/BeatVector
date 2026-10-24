"""The relation view listenbrainz_artists@<YYYYMMDD> (the day of fetchedAt), role `artist` (OctantSpecs contracts/,
contract 2 (DB build to recommendation core) ビューの種類; decisions.md, 近いアーティスト) from the two optional files of
the crawl (contract 4 (OctantHarvest crawl to OctantVector)), in the folder `artists`:
  map_artists.jsonl.gz      {hash, mbids}: the MusicBrainz artist IDs of a map version (no names: OctantSpecs law 3.2)
  similar_artists.jsonl.gz  a header {algorithm, fetchedAt}, then {mbid, similar: [[mbid, score], ...]} from
                            ListenBrainz
Members are the IDs of the maps in the DB. A pair's score is the ListenBrainz score divided by the largest score of
its list, so each ID's best neighbour scores 1; only pairs whose two IDs are members are kept, at most `per_member`
per ID (the best ones). The view's `credits` name both sources.
"""
from pathlib import Path

from .catalogue import read_jsonl

MAP_FILE, SIMILAR_FILE = "map_artists.jsonl.gz", "similar_artists.jsonl.gz"
PER_MEMBER = 30
CREDITS = ("MusicBrainz artist IDs, https://musicbrainz.org/, core data under CC0",
           "ListenBrainz similar artists, https://listenbrainz.org/, data under CC0 (checked 2026-10-17)")


def relation(folder, hashes, per_member=PER_MEMBER):
    """-> (view key, relation content as in common/dbformat.py) for the maps `hashes`, or None when the folder lacks
    the files."""
    folder = Path(folder)
    if not (folder / MAP_FILE).is_file() or not (folder / SIMILAR_FILE).is_file():
        return None
    want = {h.lower() for h in hashes}
    members = {}
    for r in read_jsonl(folder / MAP_FILE):
        h = str(r.get("hash", "")).lower()
        ids = tuple(sorted(set(r.get("mbids") or ())))
        if h in want and ids:
            members[h] = ids
    used = {m for ids in members.values() for m in ids}
    lines = read_jsonl(folder / SIMILAR_FILE)
    header = next(lines, {}) or {}
    pairs = []
    for r in lines:
        a = r.get("mbid")
        sims = [(b, float(s)) for b, s in r.get("similar") or () if b != a]
        if a not in used or not sims:
            continue
        top = max(s for _, s in sims)
        if top <= 0:
            continue
        kept = sorted(((b, s / top) for b, s in sims if b in used), key=lambda t: (-t[1], t[0]))[:per_member]
        pairs += [(a, b, s) for b, s in kept]
    algorithm = header.get("algorithm") or "unknown"
    fetched = header.get("fetchedAt")
    key = f"listenbrainz_artists@{(fetched or 'unknown')[:10].replace('-', '')}"
    meta = {"role": "artist", "rows": "maps", "source": f"listenbrainz:{algorithm}", "fetchedAt": fetched,
            "perMember": per_member, "fingerprint": f"listenbrainz:{algorithm}", "sources": [],
            "credits": list(CREDITS)}
    return key, {"meta": meta, "members": members, "pairs": sorted(pairs)}
