"""The relation view beatsaver_mappers@1, role `mapper` (OctantSpecs contracts/, contract 2 (DB build to recommendation
core) ビューの種類): which BeatSaver accounts made each map version, from the catalogue (contract 4 (OctantHarvest crawl
to OctantVector)). Members are "beatsaver:<account id>" of the uploader (`uploader.id`) and of the collaborators the
catalogue lists (`collaborators[].id`); every version of a map has its map's members. There are no pairs: two maps are
close (1) only when they share an account. Account IDs only, never names (OctantSpecs law 3.2).

It is built from the catalogue at every release, so it follows the maps the release has. A DB has it when its maker
names it among the views (NAME); it needs no credits beyond the DB's own notice of BeatSaver.
"""
NAME = "beatsaver_mappers"
KEY = f"{NAME}@1"


def _accounts(m):
    people = [m.get("uploader")] + list(m.get("collaborators") or ())
    return tuple(sorted({f"beatsaver:{p['id']}" for p in people if isinstance(p, dict) and p.get("id") is not None}))


def relation(recs, hashes):
    """-> (view key, relation content as in common/dbformat.py) for the maps `hashes`; recs: {hash: (map, version)}
    from the catalogue."""
    members = {h: ids for h in hashes if h in recs and (ids := _accounts(recs[h][0]))}
    meta = {"role": "mapper", "rows": "maps", "source": "beatsaver:uploader+collaborators",
            "fingerprint": "beatsaver:uploader+collaborators/1", "sources": []}
    return KEY, {"meta": meta, "members": members, "pairs": []}
