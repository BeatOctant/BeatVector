"""The one reader of a map's chart files: the charts Info.dat (v2 or v4) lists, and one difficulty (.dat, formats
v2/v3/v4) -> timed objects, for the swing simulation and the pattern statistics (encoders/chart_patterns.py).

Times are converted to seconds through the chart's BPM changes (v2 `_events` type 100, v3 `bpmEvents`; v2
`_BPMChanges` is editor data and ignored); v4 keeps BPM changes in a separate audio file, so v4 charts use the
Info.dat BPM throughout. Noodle Extensions note and bomb placement (`coordinates` / `_position`) is used.
Mapping Extensions precision positions (|x| >= 1000), wall sizes (>= 1000) and angles (cut 1000..1360) are mapped
back to grid units and degrees. Noodle Extensions fake walls and bombs are kept but flagged (scenery the player
cannot touch); fake colour notes are left out.
"""
from dataclasses import dataclass, field

from ...common.charts import difficulty_rank

DIR_DEG = {0: 90.0, 1: 270.0, 2: 180.0, 3: 0.0, 4: 135.0, 5: 45.0, 6: 225.0, 7: 315.0}  # 8 = dot (None)


def beatmaps(info):
    """[(characteristic, difficulty, filename, njs)] for every chart listed in Info.dat (v2 or v4)."""
    out = []
    for s in info.get("_difficultyBeatmapSets", []):
        c = s.get("_beatmapCharacteristicName")
        out += [(c, d["_difficulty"], d["_beatmapFilename"], d.get("_noteJumpMovementSpeed", 0))
                for d in s["_difficultyBeatmaps"]]
    for d in info.get("difficultyBeatmaps", []):
        out.append((d.get("characteristic"), d["difficulty"], d["beatmapDataFilename"], d.get("noteJumpMovementSpeed", 0)))
    return out


def standard_beatmaps(info):
    """[(difficulty, filename, njs)] for the Standard characteristic, easiest first."""
    return sorted(((d, fn, njs) for c, d, fn, njs in beatmaps(info) if c == "Standard"),
                  key=lambda t: difficulty_rank(t[0]))


def bpm_of(info):
    return info.get("_beatsPerMinute") or info.get("audio", {}).get("bpm") or 120


@dataclass
class Note:
    t: float            # seconds
    b: float            # beats (json time)
    x: float            # lane 0..3
    y: float            # row 0..2
    c: int              # 0 red (left hand), 1 blue (right hand)
    ang: float | None   # cut direction in degrees, None for a dot
    tail: tuple | None = None  # chain head: (tail x, tail y)
    cut: int = 8        # the cut direction as written (0..7, 8 = dot; Mapping Extensions 1000..1360)


@dataclass
class Chart:
    notes: list
    bombs: list = field(default_factory=list)   # [(t, x, y, beat, fake)]
    walls: list = field(default_factory=list)   # [(t, duration_s, x, y, w, h, beat, fake)]
    njs: float = 0.0
    bpm: float = 120.0
    arcs: int = 0
    chains: int = 0

    @property
    def length_s(self):
        return (self.notes[-1].t - self.notes[0].t) if self.notes else 0.0


def _pos(v):
    v = float(v or 0)
    if v >= 1000:
        return v / 1000 - 1
    if v <= -1000:
        return v / 1000 + 1
    return v


def _size(v):
    """Wall width/height; Mapping Extensions precision sizes (>= 1000) are 1000 + 1000 * units."""
    v = float(v or 0)
    return v / 1000 - 1 if v >= 1000 else v


def _fake(o):
    """Noodle Extensions objects the player cannot touch (v2 `_fake` / `_interactable: false`, v3 `fake` /
    `uninteractable`): scenery only."""
    cd = o.get("_customData") if "_time" in o else o.get("customData")
    if not isinstance(cd, dict):
        return False
    return cd.get("_fake") is True or cd.get("_interactable") is False or cd.get("fake") is True or cd.get("uninteractable") is True


def _v2_wall_rows(kind):
    """v2 `_type` -> (bottom row, height in rows): 0 full height, 1 overhead (crouch); Mapping Extensions
    1000..4000 is a full-height wall of (kind - 1000) / 1000 * 5 rows. Anything else is taken as full height."""
    kind = int(kind or 0)
    if kind == 1:
        return 2, 3
    if 1000 <= kind <= 4000:
        return 0, (kind - 1000) / 1000 * 5
    return 0, 5


def _cut(d):
    return int(d or 0)


def _ang(d, offset=0.0):
    d = int(d or 0)
    if 1000 <= d <= 1360:  # Mapping Extensions: degrees counter-clockwise from "down"
        return (270.0 + d - 1000) % 360
    if d not in DIR_DEG:
        return None
    return (DIR_DEG[d] + float(offset or 0)) % 360


class _Clock:
    """json beat -> seconds through BPM changes [(beat, bpm)]."""

    def __init__(self, bpm, changes):
        pts = sorted((float(b), float(m)) for b, m in changes if m and float(m) > 0)
        self.seg = [(0.0, 0.0, float(bpm))]  # (beat, seconds at beat, bpm from here)
        for b, m in pts:
            b0, s0, m0 = self.seg[-1]
            if b < b0:
                continue
            self.seg.append((b, s0 + (b - b0) * 60 / m0, m))

    def __call__(self, beat):
        for b0, s0, m0 in reversed(self.seg):
            if beat >= b0:
                return s0 + (beat - b0) * 60 / m0
        return beat * 60 / self.seg[0][2]


def _changes(dat):
    """[(beat, bpm)] from the BPM changes the game itself uses: v3 `bpmEvents` and v2 `_events` type 100.
    v2 `_BPMChanges` (top level or in `_customData`) is editor grid data the game ignores: v2 note times are already
    in Info.dat-BPM beats, so honouring it ran some clocks twice as fast (exp16). Malformed entries are skipped."""
    src = dat.get("bpmEvents")
    if isinstance(src, list) and src:
        return [(e.get("b", 0), e.get("m")) for e in src if isinstance(e, dict)]
    ev = dat.get("_events")
    if isinstance(ev, list):
        return [(e.get("_time", 0), e.get("_floatValue")) for e in ev if isinstance(e, dict) and e.get("_type") == 100]
    return []


def _noodle_xy(o, x, y):
    """Noodle Extensions placement (v3 customData.coordinates, v2 _customData._position: [x, y] with x = 0 at the
    grid centre) overrides the grid position."""
    cd = o.get("customData") if "b" in o else o.get("_customData")
    c = cd.get("coordinates" if "b" in o else "_position") if isinstance(cd, dict) else None
    if isinstance(c, list) and len(c) >= 2 and all(isinstance(v, (int, float)) for v in c[:2]):
        return c[0] + 2.0, float(c[1])
    return x, y


def parse(dat, bpm, njs) -> Chart:
    clock = _Clock(bpm, _changes(dat))
    notes, bombs, walls, arcs, chains, tails = [], [], [], 0, 0, {}
    if "_notes" in dat:  # v2
        for n in dat["_notes"]:
            t = n.get("_type")
            if t in (0, 1) and not _fake(n):
                x, y = _noodle_xy(n, _pos(n.get("_lineIndex")), _pos(n.get("_lineLayer")))
                notes.append(Note(clock(n["_time"]), n["_time"], x, y, t, _ang(n.get("_cutDirection")),
                                  cut=_cut(n.get("_cutDirection"))))
            elif t == 3:
                bombs.append((clock(n["_time"]), *_noodle_xy(n, _pos(n.get("_lineIndex")), _pos(n.get("_lineLayer"))),
                              n["_time"], _fake(n)))
        for o in dat.get("_obstacles", []):
            b, d = o.get("_time", 0), o.get("_duration", 0)
            y, h = _v2_wall_rows(o.get("_type", 0))
            walls.append((clock(b), clock(b + d) - clock(b), _pos(o.get("_lineIndex")), y, _size(o.get("_width", 1)), h, b,
                          _fake(o)))
        arcs = len(dat.get("_sliders", []))
    elif "colorNotesData" in dat:  # v4: objects index into *Data arrays
        nd, bd, od = dat.get("colorNotesData", []), dat.get("bombNotesData", []), dat.get("obstaclesData", [])
        for n in dat.get("colorNotes", []):
            d = nd[n.get("i", 0)] if nd else {}
            notes.append(Note(clock(n.get("b", 0)), n.get("b", 0), _pos(d.get("x")), _pos(d.get("y")), d.get("c", 0),
                              _ang(d.get("d"), d.get("a")), cut=_cut(d.get("d"))))
        for n in dat.get("bombNotes", []):
            d = bd[n.get("i", 0)] if bd else {}
            bombs.append((clock(n.get("b", 0)), _pos(d.get("x")), _pos(d.get("y")), n.get("b", 0), False))
        for o in dat.get("obstacles", []):
            d = od[o.get("i", 0)] if od else {}
            b = o.get("b", 0)
            walls.append((clock(b), clock(b + d.get("d", 0)) - clock(b), _pos(d.get("x")), _pos(d.get("y")),
                          _size(d.get("w", 1)), _size(d.get("h", 5)), b, False))
        cd = dat.get("chainsData", [])
        for ch in dat.get("chains", []):
            head = nd[ch.get("i", 0)] if nd else {}
            tail = cd[ch.get("ci", 0)] if cd else {}
            tails[(round(ch.get("hb", 0), 3), head.get("c", 0))] = (_pos(tail.get("tx")), _pos(tail.get("ty")))
        arcs, chains = len(dat.get("arcs", [])), len(dat.get("chains", []))
    else:  # v3
        cd = dat.get("customData") if isinstance(dat.get("customData"), dict) else {}
        for n in dat.get("colorNotes", []):  # Noodle's customData.fakeColorNotes are never hit: left out
            if not _fake(n):
                x, y = _noodle_xy(n, _pos(n.get("x")), _pos(n.get("y")))
                notes.append(Note(clock(n.get("b", 0)), n.get("b", 0), x, y, n.get("c", 0),
                                  _ang(n.get("d"), n.get("a")), cut=_cut(n.get("d"))))
        for fake, src in ((False, dat.get("bombNotes", [])), (True, cd.get("fakeBombNotes", []))):
            for n in src:
                bombs.append((clock(n.get("b", 0)), *_noodle_xy(n, _pos(n.get("x")), _pos(n.get("y"))), n.get("b", 0),
                              fake or _fake(n)))
        for fake, src in ((False, dat.get("obstacles", [])), (True, cd.get("fakeObstacles", []))):
            for o in src:
                b = o.get("b", 0)
                walls.append((clock(b), clock(b + o.get("d", 0)) - clock(b), _pos(o.get("x")), _pos(o.get("y")),
                              _size(o.get("w", 1)), _size(o.get("h", 5)), b, fake or _fake(o)))
        for ch in dat.get("burstSliders", []):
            tails[(round(ch.get("b", 0), 3), ch.get("c", 0))] = (_pos(ch.get("tx")), _pos(ch.get("ty")))
        arcs, chains = len(dat.get("sliders", [])), len(dat.get("burstSliders", []))
    for n in notes:
        n.tail = tails.get((round(n.b, 3), n.c))
    notes.sort(key=lambda n: (n.t, n.c))
    bombs.sort()
    return Chart(notes, bombs, sorted(walls), float(njs or 0), float(bpm), arcs, chains)
