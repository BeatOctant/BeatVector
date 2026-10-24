"""Pattern statistics of every chart (each difficulty of each characteristic), read by swing/chart.py (beatmap
formats v2, v3 and v4: times through the chart's BPM changes, Noodle and Mapping Extensions positions, objects the
player cannot touch left out). Rotation events are ignored. The statistics mean different things for different play
styles (one saber, no arrows, rotation); search only compares charts of the same play style (charts.py)."""
import numpy as np

from ...common import charts
from ..swing.chart import beatmaps, bpm_of, parse
from .base import Encoder, register

GAPS = (0.126, 0.26, 0.51, 1.01)  # same-hand gap thresholds in beats (~1/8, 1/4, 1/2, 1)
NAMES = (["nps", "njs", "bomb_ps", "wall_ps", "wall_time_frac", "arc_ps", "chain_ps", "red_frac", "double_frac",
          "dot_frac", "up_down_frac", "horiz_frac", "diag_frac", "same_dir_repeat", "angle_change_mean", "flip180_frac",
          "x_spread", "y_mean", "outer_lane_frac", "top_row_frac", "bottom_row_frac", "crossover_frac",
          "density_cv", "density_p90_over_mean", "length_s"]
         + [f"gap_le_{g}" for g in GAPS] + ["gap_gt_1"])


def pattern_features(chart):
    """Feature vector (len(NAMES),) for one parsed difficulty (swing/chart.py Chart), or None if it has fewer than
    16 colour notes."""
    if len(chart.notes) < 16:
        return None
    notes = sorted(chart.notes, key=lambda n: n.b)
    b = np.array([n.b for n in notes], float)
    sec = np.array([n.t for n in notes], float)
    span = max(sec[-1] - sec[0], 1.0)
    x = np.array([n.x for n in notes], float); y = np.array([n.y for n in notes], float)
    c = np.array([n.c for n in notes]); d = np.array([n.cut for n in notes])
    ang = [n.ang for n in notes]
    bombs = [o for o in chart.bombs if not o[4]]
    walls = [o for o in chart.walls if not o[7]]
    double = np.mean(np.concatenate([[False], np.diff(b) < 1e-3]))
    gaps, angs, repeats, flips, cross = [], [], [], [], []
    for col in (0, 1):
        m = c == col
        bb, dd, xx = b[m], d[m], x[m]
        if len(bb) > 1:
            gaps += list(np.diff(bb))
            a = [v for v, k in zip(ang, m) if k]
            for a0, a1 in zip(a[:-1], a[1:]):
                if a0 is not None and a1 is not None:
                    ch = abs((a1 - a0 + 180) % 360 - 180)
                    angs.append(ch); flips.append(ch >= 135)
            repeats += list((dd[1:] == dd[:-1]) & (dd[1:] != 8))
        cross += list((xx >= 2) if col == 0 else (xx <= 1))  # red on the right half / blue on the left
    gaps = np.array(gaps) if gaps else np.array([1.0])
    hist = np.histogram(sec, bins=max(int(span // 4), 1))[0] / 4.0  # notes/s in 4 s windows
    v = [len(notes) / span, chart.njs, len(bombs) / span, len(walls) / span, sum(w[1] for w in walls) / span,
         chart.arcs / span, chart.chains / span, float(np.mean(c == 0)), double,
         np.mean(d == 8), np.mean(np.isin(d, (0, 1))), np.mean(np.isin(d, (2, 3))), np.mean(np.isin(d, (4, 5, 6, 7))),
         np.mean(repeats) if repeats else 0, np.mean(angs) if angs else 0, np.mean(flips) if flips else 0,
         x.std(), y.mean(), np.mean((x == 0) | (x == 3)), np.mean(y == 2), np.mean(y == 0), np.mean(cross) if cross else 0,
         hist.std() / (hist.mean() + 1e-9), np.percentile(hist, 90) / (hist.mean() + 1e-9), span]
    prev = 0
    for g in GAPS:
        v.append(np.mean((gaps > prev) & (gaps <= g))); prev = g
    v.append(np.mean(gaps > GAPS[-1]))
    return np.nan_to_num(np.array(v, np.float32))


@register
class ChartPatterns(Encoder):
    kind = "zip"
    id = "chart_patterns"
    version = "3"  # 3: read by swing/chart.py (BPM changes, Noodle / Mapping Extensions, untouchable objects left out)
    rows = "charts"
    inputs = ("chart",)
    names = NAMES
    dim = len(NAMES)

    def params(self):
        return {"charts": "every chart", "features": list(self.names)}

    def embed_zip(self, mz):
        """-> {chart key: features} for every chart with enough notes."""
        out, first_error = {}, None
        for c, d, fn, njs in beatmaps(mz.info):
            if self.play_styles and charts.play_style(c) not in self.play_styles:
                continue
            try:
                v = pattern_features(parse(mz.json(fn), bpm_of(mz.info), njs))
            except Exception as ex:  # one broken chart leaves the others usable
                first_error = first_error or ex
                continue
            if v is not None:
                out[charts.key(mz.hash, c, d)] = v
        if not out and first_error is not None:
            raise first_error
        return out
