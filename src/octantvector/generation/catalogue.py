"""BeatSaver catalogue snapshot (catalogue.jsonl.gz, one map per line) -> records and display metadata."""
import gzip
import json
from pathlib import Path


def read_jsonl(path: Path):
    """The records of a gzipped JSON-lines file of the crawl (the catalogue, the artist files), blank lines skipped."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def published_version(m):
    return next((v for v in m["versions"] if v["state"] == "Published"), None)


def standard_diffs(v):
    return [d for d in v.get("diffs", []) if d["characteristic"] == "Standard"]


def records_for(catalogue: Path, hashes) -> dict:
    """version hash -> (map record, version record) for the given hashes (one streaming pass)."""
    want, out = set(hashes), {}
    for m in read_jsonl(catalogue):
        for v in m["versions"]:
            if v["hash"] in want:
                out[v["hash"]] = (m, v)
    return out


def load_meta(catalogue: Path, cache: Path) -> dict:
    """version hash -> display metadata for every catalogue version; cached as JSON next to the store.
    Titles/artists are for display and record linking only, never similarity features (OctantSpecs octantvector/design.md)."""
    if cache.exists() and cache.stat().st_mtime >= Path(catalogue).stat().st_mtime:
        return json.loads(cache.read_text())
    out = {}
    for m in read_jsonl(catalogue):
        md = m["metadata"]
        for v in m["versions"]:
            std = standard_diffs(v)
            out[v["hash"]] = {"key": m["id"], "song": md["songName"], "artist": md["songAuthorName"],
                              "mapper": md["levelAuthorName"], "bpm": md["bpm"], "up": m["stats"]["upvotes"],
                              "tags": m.get("tags", []), "maxnps": max((d["nps"] for d in std), default=None)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False))
    return out
