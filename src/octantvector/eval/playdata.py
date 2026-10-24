"""Readers for a player's own data. All functions only read the paths they are given; nothing is copied,
uploaded or written except where the caller saves the returned values. What each signal means for labels:
OctantSpecs octantvector/design.md. Typical sources (paths are the user's own; nothing here assumes a location):

  votedSongs.json        BeatSaverVoting (UserData/)          read_votes
  *.bplist               playlists (Playlists/)               read_playlists
  PlayerData.dat         game profile (AppData/LocalLow/...)  read_player_data
  beatsaber.db           DataRecorder (UserData/DataRecorder) read_datarecorder
  *.bsor / *.dat         BeatLeader / ScoreSaber replays      replay_hashes (file names only)
  CustomLevels/          installed maps                       installed_keys
"""
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from ..common.charts import difficulty

HASH_RE = re.compile(r"[0-9A-Fa-f]{40}")


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_votes(path) -> dict:
    """-> {version hash: "Upvote" | "Downvote"}"""
    return {h.lower(): v["voteType"] for h, v in _load_json(path).items()}


def read_playlists(paths) -> Counter:
    """-> Counter of version hashes over the given .bplist files."""
    c = Counter()
    for p in paths:
        for s in _load_json(p).get("songs", []):
            if s.get("hash"):
                c[s["hash"].lower()] += 1
    return c


def read_player_data(path):
    """-> (favourite hashes, {hash: total playCount}, {hash: number of difficulties with a valid score})."""
    pd = _load_json(path)["localPlayers"][0]
    fav = {lid[13:53].lower() for lid in pd.get("favoritesLevelIds", []) if lid.startswith("custom_level_")}
    plays, cleared = Counter(), Counter()
    for st in pd.get("levelsStatsData", []):
        if st["levelId"].startswith("custom_level_") and st.get("playCount", 0) > 0:
            h = st["levelId"][13:53].lower()
            plays[h] += st["playCount"]
            cleared[h] += bool(st.get("validScore"))
    return fav, plays, cleared


def read_datarecorder(db_path) -> list:
    """DataRecorder SQLite (opened read-only, immutable) -> one dict per play of a custom level:
    hash, outcome ("finished" | "failed" | "quit"), difficulty, mode, speed."""
    con = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro&immutable=1", uri=True)
    out = []
    for h, cleared, diff, mode, speed in con.execute(
            "select songHash, cleared, difficulty, mode, songSpeedMultiplier from MovieCutRecord "
            "where songHash is not null and songHash != ''"):
        if not HASH_RE.fullmatch(h or ""):
            continue  # built-in levels
        outcome = {"finished": "finished", "failed": "failed", "softFail": "failed"}.get(cleared, "quit")
        out.append({"hash": h.lower(), "outcome": outcome, "diff": difficulty(diff), "mode": mode,
                    "speed": float(speed) if speed not in (None, "None") else 1.0})
    con.close()
    return out


def replay_hashes(folder) -> Counter:
    """Version hashes found in replay file names (BeatLeader .bsor, ScoreSaber .dat); files are not opened."""
    c = Counter()
    for f in Path(folder).iterdir():
        m = HASH_RE.search(f.name)
        if m:
            c[m.group(0).lower()] += 1
    return c


def installed_keys(custom_levels) -> set:
    """BeatSaver keys of installed maps (folder names "<key> (<title> - <mapper>)")."""
    return {f.name.split(" ")[0] for f in Path(custom_levels).iterdir() if f.is_dir()}


def build_signals(votes=None, playlists=None, player_data=None, plays=None, replays=(), installed=None, key_to_hash=None):
    """Combine the readers' outputs into {version hash: {signal: count}}. Signals: upvote, downvote, playlist,
    favorite, pd_playcount, pd_cleared, play, finished, failed, quit, replay, installed."""
    sig = defaultdict(Counter)
    for h, t in (votes or {}).items():
        sig[h]["upvote" if t == "Upvote" else "downvote"] += 1
    for h, n in (playlists or {}).items():
        sig[h]["playlist"] += n
    if player_data:
        fav, pc, cl = player_data
        for h in fav:
            sig[h]["favorite"] += 1
        for h, n in pc.items():
            sig[h]["pd_playcount"] += n
        for h, n in cl.items():
            sig[h]["pd_cleared"] += n
    for p in plays or []:
        sig[p["hash"]]["play"] += 1
        sig[p["hash"]][p["outcome"]] += 1
    for c in replays:
        for h, n in c.items():
            sig[h]["replay"] += n
    for k in installed or ():
        if key_to_hash and k in key_to_hash:
            sig[key_to_hash[k]]["installed"] += 1
    return sig


INTERACTION = ("installed", "play", "replay", "pd_playcount", "favorite", "upvote")


def merge_by_key(sig, meta, pool_hash_of_key):
    """Merge signals of all versions of a map onto its pool version hash -> {pool hash: Counter}.
    meta: catalogue.load_meta output (hash -> {key, ...}); pool_hash_of_key: {key: pool hash}."""
    out = defaultdict(Counter)
    for h, s in sig.items():
        ph = pool_hash_of_key.get(meta.get(h, {}).get("key"))
        if ph:
            out[ph].update(s)
    return out

