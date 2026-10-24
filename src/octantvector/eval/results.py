"""Saved evaluation runs: per-query results stored together with what produced them (view id@version, fingerprint,
weight, settings), so encoders can be compared later without recomputing. Format: JSON lines (gzip when the path
ends in .gz); the first line is {"run": ...}, every other line one query."""
import gzip
import json
from datetime import datetime, timezone

from .. import __version__


def run_info(index, command, summary=None, **settings):
    return {"command": command, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "octantvector": __version__, "n_items": len(index), "calibration": index.calibration_id,
            "missing": index.missing,
            "views": {f"{e.id}@{e.version}": {"weight": index.views[i][1], "fingerprint": e.fingerprint()}
                      for i, e in index.encoders.items()},
            "settings": settings, "summary": summary}


def write(path, info, rows):
    with (gzip.open(path, "wt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "w", encoding="utf-8")) as f:
        f.write(json.dumps({"run": info}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")

