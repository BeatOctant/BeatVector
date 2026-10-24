"""Encoders computed from the catalogue record alone (no zip access): NPS/NJS per difficulty, metadata/tags."""
import numpy as np

from ...common.charts import DIFF_ORDER
from .base import Encoder, register


@register
class NpsNjs(Encoder):
    """NPS and NJS for Standard Easy..ExpertPlus (a missing slot copies the nearest present one) + presence flags.
    A map-level summary of chart values for experiments only: dbrelease.py refuses it (chart views must have chart
    rows; per-chart NPS and NJS are in the DB's charts table)."""
    id = "npsnjs_std"
    dim = 15
    role = "chart"  # describes the charts' density, though it is read from the catalogue
    inputs = ("meta:versions.diffs",)

    def params(self):
        return {"source": "catalogue versions[].diffs", "characteristic": "Standard", "layout": "nps[5], njs[5], present[5]"}

    def embed_map(self, m, v):
        by = {d["difficulty"]: d for d in v["diffs"] if d["characteristic"] == "Standard"}
        if not by:
            return None
        present = [n in by for n in DIFF_ORDER]
        nps, njs = [], []
        for i in range(len(DIFF_ORDER)):
            j = i if present[i] else min((j for j, p in enumerate(present) if p), key=lambda j: abs(j - i))
            nps.append(by[DIFF_ORDER[j]]["nps"]); njs.append(by[DIFF_ORDER[j]]["njs"])
        return np.array(nps + njs + [float(p) for p in present], np.float32)


TAGS = ["accuracy", "balanced", "challenge", "dance-style", "fitness", "speed", "tech", "alternative", "ambient",
        "anime", "classical-orchestral", "comedy-meme", "dance", "drum-and-bass", "dubstep", "electronic", "folk-acoustic",
        "funk-disco", "hardcore", "hip-hop-rap", "holiday", "house", "indie", "instrumental", "j-pop", "j-rock", "jazz",
        "k-pop", "kids-family", "metal", "nightcore", "pop", "punk", "rb", "rock", "soul", "speedcore",
        "swing", "tv-movie-soundtrack", "trance", "video-game-soundtrack", "vocaloid", "poodle", "techno", "ai"]


@register
class MetaTags(Encoder):
    """BeatSaver tags + year, length, BPM and flags. Deliberately no popularity and no mapper identity, and nothing
    computed from the charts (version 2 dropped has_chroma_ne_me and n_std_diffs: chart information stays per chart)."""
    id = "meta_tags"
    version = "2"
    inputs = tuple(f"meta:{f}" for f in ["tags", "uploaded", "createdAt", "metadata.duration", "metadata.bpm", "ranked",
                                         "blRanked", "qualified", "blQualified", "curator", "curatedAt", "automapper",
                                         "declaredAi"])
    names = [f"tag:{t}" for t in TAGS] + ["year", "log_duration", "bpm", "ranked_any", "curated", "automapper",
                                          "declared_ai"]
    dim = len(names)

    def params(self):
        return {"source": "catalogue", "features": self.names, "note": "no popularity, no mapper identity"}

    def embed_map(self, m, v):
        md, tags = m["metadata"], set(m.get("tags", []))
        up = m.get("uploaded") or m.get("createdAt") or "2018-06"
        f = [float(t in tags) for t in TAGS]
        f += [int(up[:4]) + int(up[5:7]) / 12, np.log1p(md.get("duration") or 0), md.get("bpm") or 0,
              float(bool(m.get("ranked") or m.get("blRanked") or m.get("qualified") or m.get("blQualified"))),
              float(bool(m.get("curator") or m.get("curatedAt"))), float(bool(m.get("automapper"))),
              float(m.get("declaredAi", "None") != "None")]
        return np.array(f, np.float32)
