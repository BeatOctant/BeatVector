"""Walls and bombs by what they mean for the player (OctantSpecs octantvector/swing.md, "壁と爆弾").

Walls only matter to the head (sabers pass through them), so each wall is judged against a simulated head:
  art          scenery: a Noodle fake wall, part of a crowd no player can react to (wall art), too short,
               zero-sized, or outside the play space
  side         in the play space but clear of the standing head in the centre (outer-lane and low walls): frames
               the play, asks for no movement
  dodge        reaches the standing head; the player steps or leans sideways
  crouch       reaches the standing head but not the crouched one (overhead walls)
  unavoidable  covers every head position within reach (fake or modded walls)
The head itself is simulated over time: at each step it takes the free position closest to the centre and to
where it was, crouching only when that is cheaper than leaning. Its travel, time off-centre or crouched and
left/right switches measure how much exercise the walls give.

Bombs only matter to the sabers, so each bomb is judged against the simulated saber paths of both hands:
  reset        forces a bomb reset (simulate.assign_parity)
  dodge        near a saber path around the time it passes: the player has to steer around it
  art          far from both saber paths, a Noodle fake bomb, or part of a large cluster (bomb drawings): scenery
Rhythm is judged from the onsets of the walls and bombs in play (not art): on the 1/4-beat grid and part of an
evenly spaced run (rhythmic), or at the same time as a note (synced); art walls get their own rhythm share.
"""
import bisect

import numpy as np


def on_grid(b, step, tol):
    """True when beat b is within tol beats of the grid of `step` beats."""
    r = (b / step) % 1
    return min(r, 1 - r) * step <= tol


def rhythm(onsets, note_beats, P):
    """Onset beats (sorted) -> (rhythmic flags, synced flags), one per distinct onset; the distinct onsets too."""
    tol = P["snap_tolerance_beats"]
    ons = []
    for b in onsets:
        if not ons or b - ons[-1] > tol:
            ons.append(b)
    n = len(ons)
    ioi = np.diff(ons) if n > 1 else np.zeros(0)
    even = np.zeros(n, bool)
    for i in range(1, n - 1):  # onset i is in an even run if the gap before it equals the gap after it
        if ioi[i - 1] <= P["rhythm_max_ioi_beats"] and abs(ioi[i] - ioi[i - 1]) <= tol:
            even[i - 1:i + 2] = True
    snapped = np.array([on_grid(b, P["rhythm_grid_beats"], tol) for b in ons], bool)
    sync = np.array([_near(note_beats, b, tol) for b in ons], bool)
    return even & snapped, sync, ons


def _near(sorted_vals, v, tol):
    i = bisect.bisect_left(sorted_vals, v - tol)
    return i < len(sorted_vals) and sorted_vals[i] <= v + tol


# ---------------------------------------------------------------- walls
class Head:
    """Candidate head positions: lanes around the centre (x) x standing / crouched (row height)."""

    def __init__(self, P):
        c, r, st = P["head_x_lanes"], P["head_reach_lanes"], P["head_step_lanes"]
        k = int(round(r / st))
        xs = c + st * np.arange(-k, k + 1)
        self.x = np.concatenate([xs, xs])
        self.y = np.concatenate([np.full(len(xs), P["head_stand_row"]), np.full(len(xs), P["head_crouch_row"])])
        self.crouch = self.y == P["head_crouch_row"]
        self.hw, self.c = P["head_half_width_lanes"], c
        self.neutral = int(np.argmin(np.abs(xs - c)))  # index of (centre, standing)
        self.neutral_crouch = self.neutral + len(xs)

    def blocked(self, walls):
        """walls: (k, 4) array of x, y, w, h -> (k, candidates) bool."""
        if not len(walls):
            return np.zeros((0, len(self.x)), bool)
        x, y, w, h = (walls[:, i:i + 1] for i in range(4))
        lo, hi = np.minimum(x, x + w), np.maximum(x, x + w)  # Mapping Extensions allows negative sizes
        bot, top = np.minimum(y, y + h), np.maximum(y, y + h)
        return (lo < self.x + self.hw) & (hi > self.x - self.hw) & (bot < self.y) & (top > self.y)


def _crowded(times, max_ps, max_together):
    """Per object: True when it starts among more distinct onsets per second (in the second around it), or more
    objects at the same instant, than a player can react to: a wall or bomb drawing, not an obstacle."""
    t = np.asarray(times, float)
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    starts = np.r_[True, np.diff(ts) > 1e-3]
    on = ts[starts]
    together = np.diff(np.r_[np.where(starts)[0], len(ts)])  # objects per distinct onset
    group = np.cumsum(starts) - 1
    per_s = np.searchsorted(on, on + 0.5, "right") - np.searchsorted(on, on - 0.5, "left")
    bad = (per_s > max_ps) | (together > max_together)
    out = np.zeros(len(t), bool)
    out[order] = bad[group]
    return out


def join_walls(walls, P):
    """Walls with the same box (and fake flag) following each other within wall_join_gap_s -> one wall: a crouch
    tunnel built from many 0.02 s pieces is one long wall to the player."""
    out, last = [], {}
    for w in sorted(walls, key=lambda w: w[0]):
        key = (round(w[2], 3), round(w[3], 3), round(w[4], 3), round(w[5], 3), w[7])
        k = last.get(key)
        if k is not None and w[0] <= out[k][0] + out[k][1] + P["wall_join_gap_s"]:
            o = out[k]
            out[k] = (o[0], max(o[1], w[0] + w[1] - o[0])) + tuple(o[2:])
            continue
        last[key] = len(out)
        out.append(tuple(w))
    return out


def classify_walls(walls, P):
    """-> roles ("art", "side", "dodge", "crouch" or "unavoidable"), one per wall. Noodle fake walls and walls in crowds no player can react to
    (wall art, lightshows) are art whatever their position."""
    if not walls:
        return []
    head = Head(P)
    W = np.array([[w[2], w[3], w[4], w[5]] for w in walls], float)
    B = head.blocked(W)
    dur = np.array([w[1] for w in walls])
    size = np.abs(W[:, 2] * W[:, 3])
    crowd = _crowded([w[0] for w in walls], P["art_wall_max_onsets_ps"], P["art_wall_max_together"])
    roles = []
    for i in range(len(walls)):
        if walls[i][7] or crowd[i] or dur[i] < P["art_wall_min_s"] or size[i] == 0:
            roles.append("art")
        elif not B[i, head.neutral]:
            lo, hi = sorted((W[i, 0], W[i, 0] + W[i, 2]))
            bot = min(W[i, 1], W[i, 1] + W[i, 3])
            inside = lo < 4 and hi > 0 and bot < P["side_wall_max_row"]
            roles.append("side" if inside else "art")
        elif B[i].all():
            roles.append("unavoidable")
        elif not B[i, head.neutral_crouch]:
            roles.append("crouch")
        else:
            roles.append("dodge")
    return roles


def head_path(walls, roles, P):
    """Simulated head over the body walls (dodge / crouch) -> dict(travel_m, off_centre_s, crouch_s, switches)."""
    out = {"travel_m": 0.0, "off_centre_s": 0.0, "crouch_s": 0.0, "switches": 0}
    body = [w for w, r in zip(walls, roles) if r in ("dodge", "crouch")]
    if not body:
        return out
    head = Head(P)
    step = P["head_time_step_s"]
    starts = np.array([w[0] for w in body]); ends = starts + np.array([max(w[1], 0.0) for w in body])
    crouch = np.array([r == "crouch" for r in roles if r in ("dodge", "crouch")])
    starts = starts - crouch * P["head_crouch_margin_s"]; ends = ends + crouch * P["head_crouch_margin_s"]
    boxes = np.array([[w[2], w[3], w[4], w[5]] for w in body], float)
    B = head.blocked(boxes)
    order = np.argsort(starts)
    t0, t1 = starts.min(), ends.max() + 2 * step
    cur = head.neutral
    side = 0
    active = []
    nxt = 0
    lane_m, depth_m = P["lane_spacing_m"], P["crouch_depth_m"]
    for t in np.arange(t0, t1, step):
        while nxt < len(order) and starts[order[nxt]] <= t + step:
            active.append(order[nxt]); nxt += 1
        active = [i for i in active if ends[i] >= t]
        free = ~B[active].any(0) if active else np.ones(len(head.x), bool)
        if not free.any():
            continue  # nowhere to go: stay put
        cost = (P["head_move_cost"] * np.abs(head.x - head.x[cur]) + np.abs(head.x - head.c)
                + P["head_crouch_cost"] * head.crouch)
        cost[~free] = np.inf
        new = int(np.argmin(cost))
        out["travel_m"] += abs(head.x[new] - head.x[cur]) * lane_m + (head.crouch[new] != head.crouch[cur]) * depth_m
        cur = new
        off = head.x[cur] - head.c
        if abs(off) > 1e-9:
            out["off_centre_s"] += step
            s = 1 if off > 0 else -1
            if side and s != side:
                out["switches"] += 1
            side = s
        if head.crouch[cur]:
            out["crouch_s"] += step
    return out


def wall_spans(walls, roles, P):
    """Merged [(start, end, buff)] windows in which swings get BeatLeader's wall buff (dodge / crouch walls)."""
    out = []
    m = P["wall_buff_margin_s"]
    for w, r in zip(walls, roles):
        if r in ("dodge", "crouch"):
            out.append((w[0] - m, w[0] + max(w[1], 0.0) + m, P[f"wall_buff_{r}"]))
    return sorted(out)


def wall_buffs(times, spans):
    """Sorted swing times -> the largest buff of the spans covering each (1.0 when none)."""
    out, active, k = [], [], 0
    for t in times:
        while k < len(spans) and spans[k][0] <= t:
            active.append(spans[k]); k += 1
        active = [sp for sp in active if sp[1] >= t]
        out.append(max((sp[2] for sp in active), default=1.0))
    return out


# ---------------------------------------------------------------- bombs
def _hand_track(swings, P):
    """Key times and saber positions of one hand: entry shortly before the swing, exit shortly after."""
    T, X, Y = [], [], []
    for i, s in enumerate(swings):
        lead = P["swing_lead_s"]
        if i > 0:
            lead = min(lead, max(s.t - swings[i - 1].t_end, 0.0) / 2)
        lag = P["swing_lead_s"]
        if i + 1 < len(swings):
            lag = min(lag, max(swings[i + 1].t - s.t_end, 0.0) / 2)
        T += [s.t - lead, s.t_end + lag]; X += [s.entry[0], s.exit[0]]; Y += [s.entry[1], s.exit[1]]
    T = np.maximum.accumulate(np.array(T))
    return T, np.array(X), np.array(Y)


def inert_bombs(bombs, P):
    """Per bomb: True for Noodle fake bombs and bombs in crowds no player can react to (bomb drawings). They never
    force resets and are art."""
    if not bombs:
        return []
    crowd = _crowded([b[0] for b in bombs], P["art_bomb_max_onsets_ps"], P["art_bomb_max_together"])
    return [bool(b[4] or c) for b, c in zip(bombs, crowd)]


def classify_bombs(bombs, both, grid, P, inert=None):
    """-> roles ("art", "near", "dodge" or "reset"; near: neither far nor close enough to steer around), one per bomb
    (chart.bombs order). Noodle fake bombs and bombs in large clusters
    (drawings) are art (`inert_bombs`)."""
    if not bombs:
        return []
    inert = inert_bombs(bombs, P) if inert is None else inert
    reset = set(k for s in both if s.bomb_reset for k in s.reset_bombs)
    bt = np.array([b[0] for b in bombs])
    bx = np.array([grid.x(b[1]) for b in bombs]); by = np.array([grid.y(b[2]) for b in bombs])
    w = P["bomb_window_s"]
    ts = bt[:, None] + np.linspace(-w, w, 5)[None, :]
    dist = np.full(len(bombs), np.inf)
    for h in (0, 1):
        ss = [s for s in both if s.hand == h]
        if not ss:
            continue
        T, X, Y = _hand_track(ss, P)
        hx, hy = np.interp(ts, T, X), np.interp(ts, T, Y)
        dist = np.minimum(dist, np.hypot(hx - bx[:, None], hy - by[:, None]).min(1))
    roles = []
    for k in range(len(bombs)):
        if k in reset:
            roles.append("reset")
        elif inert[k]:
            roles.append("art")
        elif dist[k] <= P["bomb_near_m"]:
            roles.append("dodge")
        elif dist[k] >= P["bomb_far_m"]:
            roles.append("art")
        else:
            roles.append("near")
    return roles


# ---------------------------------------------------------------- summary
def _frac(a):
    return float(np.mean(a)) if len(a) else 0.0


def _offgrid(v):
    return abs(v - round(v)) > 1e-6


def summary(chart, both, walls, wall_roles, bomb_roles, span, P):
    """Wall and bomb features (the hazard group of features.NAMES) from the joined walls (analyse); all 0 for charts
    shorter than hazard_min_length_s."""
    if span < P["hazard_min_length_s"]:
        from .features import HAZARD_NAMES
        return dict.fromkeys(HAZARD_NAMES, 0.0)
    f = {}
    note_beats = sorted(n.b for n in chart.notes)
    bombs = chart.bombs
    roles = np.array(wall_roles)
    n_w = len(walls)
    art = roles == "art" if n_w else np.zeros(0, bool)
    f["wall_ps"] = float((~art).sum()) / span  # walls in play
    f["art_wall_ps"] = float(art.sum()) / span
    f["art_wall_frac"] = _frac(art)
    f["side_wall_ps"] = float((roles == "side").sum()) / span if n_w else 0.0
    f["art_wall_offgrid_frac"] = _frac([any(_offgrid(v) for v in w[2:6]) for w, a in zip(walls, art) if a])
    for r in ("dodge", "crouch", "unavoidable"):
        f[f"{r}_wall_ps"] = float((roles == r).sum()) / span if n_w else 0.0
    hp = head_path(walls, wall_roles, P)
    f["head_move_mps"] = hp["travel_m"] / span
    f["dodge_time_frac"] = min(hp["off_centre_s"] / span, 1.0)
    f["crouch_time_frac"] = min(hp["crouch_s"] / span, 1.0)
    f["dodge_switch_ps"] = hp["switches"] / span
    body = sorted(w[6] for w, r in zip(walls, wall_roles) if r in ("dodge", "crouch"))
    rb, sy, _ = rhythm(body, note_beats, P) if len(body) >= P["hazard_min_count"] else ([], [], [])
    f["wall_rhythm_frac"], f["wall_sync_frac"] = _frac(rb), _frac(sy)  # walls the body reacts to
    ra, _, _ = rhythm(sorted(w[6] for w, a in zip(walls, art) if a), note_beats, P)
    f["art_wall_rhythm_frac"] = _frac(ra)  # scenery moving with the music
    f["vision_wall_frac"] = _vision(chart)

    br = np.array(bomb_roles)
    f["bomb_art_frac"] = _frac(br == "art") if len(br) else 0.0
    beats = sorted(b[3] for b, r in zip(bombs, bomb_roles) if r != "art")  # bombs in play, not drawings
    if len(beats) < P["hazard_min_count"]:  # a handful of bombs says nothing about the map's style
        beats = []
    f["bomb_ps"] = len(beats) / span
    f["bomb_dodge_ps"] = float((br == "dodge").sum()) / span if beats else 0.0
    rh, sy, ons = rhythm(beats, note_beats, P)
    f["bomb_rhythm_frac"], f["bomb_sync_frac"] = _frac(rh), _frac(sy)
    tol = P["snap_tolerance_beats"]
    real = sorted(b[3] for b in bombs if not b[4])  # a cluster in play counts every real bomb at that instant
    sizes = [bisect.bisect_right(real, o + tol) - bisect.bisect_left(real, o - tol) for o in ons]
    f["bomb_cluster_frac"] = _frac([s >= P["bomb_cluster_n"] for s in sizes])
    return f


def _vision(chart):
    """Share of notes that arrive inside a wall's box (hidden behind or within it while they approach)."""
    if not chart.walls or not chart.notes:
        return 0.0
    nt = [n.t for n in chart.notes]
    hidden = np.zeros(len(nt), bool)
    for t, d, x, y, w, h, *_ in chart.walls:
        if d <= 0:
            continue
        lo, hi = min(x, x + w), max(x, x + w)
        bot, top = min(y, y + h), max(y, y + h)
        for k in range(bisect.bisect_left(nt, t), bisect.bisect_right(nt, t + d)):
            n = chart.notes[k]
            if lo < n.x + 1 and hi > n.x and bot < n.y + 1 and top > n.y:
                hidden[k] = True
    return float(hidden.mean())


def analyse(chart, both, grid, P, inert=None):
    """-> dict(walls: the chart's walls joined (join_walls), wall_roles: one per joined wall, bomb_roles)."""
    walls = join_walls(chart.walls, P)
    return {"walls": walls, "wall_roles": classify_walls(walls, P),
            "bomb_roles": classify_bombs(chart.bombs, both, grid, P, inert)}

