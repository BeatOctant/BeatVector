"""Every numeric constant of the swing simulation, with its default (the single source of truth; OctantSpecs octantvector/swing.md
explains the model). Override any of them in the `[swing]` table of octantvector.toml, e.g.

    [swing]
    reset_cost = 0.4
    slider_max_gap_s = 0.06

Encoders record the effective values in their encoder.json, so changing one without bumping the encoder version
is caught by the store (store.py) instead of mixing outputs.
"""
from .. import config
from ..errors import GenerationError

DEFAULTS = {
    # --- grouping notes into swings
    "simultaneous_s": 0.001,      # notes of one colour closer than this are hit by the same swing
    "slider_max_gap_s": 0.07,     # same-colour notes closer than this can still be one swing (sliders, dot spam)
    "slider_max_beats": 0.125,    # ... and arrows this many beats apart (1/8), if also within ...
    "slider_max_gap_long_s": 0.1,  # ... this many seconds
    "slider_max_span_s": 0.3,     # a slider (or chain of joined notes) lasts at most this long ...
    "slider_max_turn": 180.0,     # ... and its arrows turn at most this many degrees in total
    "slider_max_angle": 90.0,     # ... if their arrows differ by at most this many degrees (curved sliders)
    "slider_geom_angle": 67.5,    # ... or the step between them points within this of the arrow
    "dot_max_gap_s": 0.06,        # a dot joins a neighbouring note into one swing only this close (replays: 65 ms is two)
    "swing_merge_s": 0.04,        # same-colour notes closer than this are one swing whatever their arrows
    "min_swing_dt_s": 0.05,       # time between swings of one hand is taken as at least this (caps EBPM at 600)
    # --- player model (positions in metres; grid as in BeatLeader's analyzer)
    "lane_spacing_m": 0.6,
    "row_y_m": [0.0, 0.55, 1.05],
    "note_radius_m": 0.3,         # entry/exit points sit this far before/after the note along the swing
    "neutral_right_forehand": 247.5,  # most comfortable swing direction (degrees, 0 = right, 90 = up)
    "neutral_right_backhand": 67.5,
    "neutral_left_forehand": 292.5,
    "neutral_left_backhand": 112.5,
    "dot_geom_blend": 0.5,        # a lone dot swings opposite to the previous swing, turned this far toward the step
    # --- parity search (dynamic programming over forehand/backhand per hand)
    "parity_strain_weight": 0.8,  # weight of the wrist strain of a parity
    "parity_rotation_weight": 0.75,  # weight of (degrees away from the expected direction / 180)^2
    "reset_cost": 0.6,            # cost of keeping the same parity (lifting the saber back without a hit)
    "parity_long_gap_s": 0.65,    # from this gap between swings of one hand on, the *_long weights apply:
    "parity_strain_weight_long": 1.0,  # players reset more freely between close swings than the model did,
    "parity_rotation_weight_long": 1.0,  # but not after long gaps (BeatLeader replays, exp19)
    "reset_cost_long": 0.5,
    "bomb_reset_cost": 0.05,      # ... when bombs sit where the saber rests after the previous swing
    "bomb_block_cost": 1.0,       # cost of alternating through such bombs
    "bomb_rest_dist_m": 0.75,     # bombs within this distance of the previous exit point, ahead of the swing
    "strain_falloff_start_s": 0.25,   # angle strain fades for gaps longer than this ...
    "strain_falloff_base": 4.0,       # ... by this factor per second
    "inverted_roll_deg": 90.0,    # wrist roll beyond this counts as an inverted (upside-down) hit
    "linear_tolerance_deg": 22.5,
    # --- per-swing difficulty (BeatLeader pass/tech formulas, see OctantSpecs octantvector/swing.md)
    "strain_scale": 4.0,
    "distance_falloff": 2.668,
    "speed_falloff_base": 1.4,
    "stress_falloff": 2.0,
    "reset_speed_mult": 2.0,
    "stream_bonus": 1.05,
    "stress_w_strain": 0.05,
    "stress_w_reposition": 0.3,
    "stress_w_rotation": 0.2,
    "reposition_perp_scale": 0.6875,
    "reposition_avg_scale": 0.0625,
    "njs_buff_from": 24.0,
    "njs_buff_per": 0.01,
    "pass_windows": [8, 16, 32, 64, 128],
    "one_saber_nerf": 0.5,
    "pass_calibration": 0.825,
    "tech_skip_frac": 0.25,       # tech averages the hardest (1 - this) share of swings
    "tech_scale": 14.0,
    "tech_pass_base": 1.4,
    "wall_buff_dodge": 1.1,       # swing difficulty multiplier while dodging a wall (BeatLeader) ...
    "wall_buff_crouch": 1.2,      # ... or crouching under one
    "wall_buff_margin_s": 0.5,    # ... from this long before a wall until this long after it
    # --- walls (hazards.py); positions in lanes (0..4 across the grid) and rows (0 floor .. 3+ above the top row)
    "head_x_lanes": 2.0,          # standing head: centre of the grid
    "head_half_width_lanes": 0.15,
    "head_stand_row": 2.5,        # head height standing (overhead walls start at row 2) ...
    "head_crouch_row": 1.5,       # ... and crouched
    "head_reach_lanes": 1.5,      # how far the head can step or lean sideways
    "head_step_lanes": 0.25,      # resolution of the head positions tried
    "head_time_step_s": 0.05,
    "head_move_cost": 0.5,        # cost per lane moved from the previous head position ...
    "head_crouch_cost": 1.0,      # ... and of crouching (in lanes off-centre); being off-centre costs 1 per lane
    "crouch_depth_m": 0.5,        # head travel counted for crouching down or standing up
    "art_wall_min_s": 0.02,       # shorter walls never get in the way: scenery ...
    "wall_join_gap_s": 0.1,       # ... once same-box walls following each other this closely are joined (tunnels)
    "head_crouch_margin_s": 0.3,  # players crouch this long before a crouch wall and stay down this long after it
    "side_wall_max_row": 3.0,     # walls clear of the head starting below this row are side walls, higher ones art
    "art_wall_max_onsets_ps": 8.0,  # walls starting among more distinct onsets per second than this ...
    "art_wall_max_together": 6,     # ... or more walls at one instant than this are wall art
    # --- bombs (hazards.py)
    "swing_lead_s": 0.08,         # the saber is at a swing's entry this long before it and at its exit this long after
    "bomb_window_s": 0.1,         # saber positions checked this long before and after a bomb passes
    "bomb_near_m": 0.65,          # a saber passing this close (about one lane) must steer around the bomb
    "bomb_far_m": 1.2,            # bombs at least this far from both sabers are scenery
    "bomb_cluster_n": 4,          # this many bombs at once form a cluster (bomb walls, shapes)
    "art_bomb_max_onsets_ps": 12.0,  # bombs among more distinct onsets per second than this ...
    "art_bomb_max_together": 12,     # ... or more bombs at one instant than this are bomb drawings
    "hazard_min_length_s": 30.0,  # shorter charts (memes, jingles) get no wall or bomb features
    "hazard_min_count": 10,       # fewer bombs (or body walls) in play than this give no bomb (wall rhythm) features
    # --- rhythm of walls and bombs
    "rhythm_grid_beats": 0.25,    # onsets on this grid ...
    "rhythm_max_ioi_beats": 2.0,  # ... in evenly spaced runs with at most this many beats between onsets are rhythmic
    # --- summary features
    "gap_bins_s": [0.15, 0.25, 0.4, 0.7],  # same-hand time between swings, histogram edges
    "full_swing_s": 0.4,          # a hand with at least this long between swings can take a full (acc) swing
    "wide_jump_m": 1.0,           # entry-to-entry hand travel counted as a jump ...
    "wide_jump_max_s": 0.3,       # ... when the hand has less than this long
    "sustained_swings": 8,        # window for the peak sustained swing rate (EBPM)
    "snap_tolerance_beats": 0.02,
    "min_notes": 16,              # charts with fewer colour notes get no features
}

LIST_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, list)}


def load(overrides: dict | None = None) -> dict:
    """Defaults < `[swing]` table of the config file < overrides."""
    p = dict(DEFAULTS)
    for src in (config.section("swing"), overrides or {}):
        unknown = set(src) - set(DEFAULTS)
        if unknown:
            raise GenerationError(f"unknown [swing] setting(s): {', '.join(sorted(unknown))} (see swing/params.py)")
        for k, v in src.items():
            p[k] = [float(x) for x in v] if k in LIST_KEYS else type(DEFAULTS[k])(v)
    p["pass_windows"] = [int(w) for w in p["pass_windows"]]
    if len(p["gap_bins_s"]) != len(DEFAULTS["gap_bins_s"]) or len(p["row_y_m"]) != 3:
        raise GenerationError("[swing] gap_bins_s needs 4 edges and row_y_m 3 heights (they size the feature vector)")
    return p
