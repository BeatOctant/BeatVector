"""Weighted multi-view cosine k-NN with optional CSLS hub reduction.

Each view is normalised on its own (normalisation.py; encoder.norm: "center" = centre + L2, "zscore" = z-score +
L2). The statistics come from a calibration file when one is given (calibration.py), so the coordinates do not move
when rows are added; otherwise they are computed from the rows loaded. The combined score is the weighted mean of
per-view cosines; with missing="drop" only hashes present in every view are indexed and the views are scaled by
sqrt(w / sum(w)) and concatenated, which gives the same score in one product.

With missing="reweight" a hash is kept if its views carry at least `min_weight` of the total weight. A view missing
on either side of a pair counts as cosine 0 with its full weight (the score is divided by the total weight, not by
the shared weight), so a pair that shares fewer views cannot outrank one that shares them all by agreeing on a
light view alone. search_views() returns the
top-k of each view on its own (the base output); the weighted combination is the layer on top (query_rows), and
its weights can change per request. Similarity/top-k run in row chunks on CUDA via torch when available (device
"auto"), otherwise numpy; results are numpy either way.

The index is map-level (an experiment tool: search, eval-neighbours, eval-personal). A view with chart rows is
represented by one chart per map version, the hardest Standard difficulty (charts.representative); the distributed
DB keeps every chart (runtime/db.py).
"""
import numpy as np

from ..common import charts
from ..common.params import DEFAULTS
from .encoders import get
from .normalisation import calibrated, normalise
from .store import Store

CHUNK = 2048


def parse_views(spec: str) -> dict:
    """'clap_htsat_unfused:0.5,meta_tags:0.5' -> {id: weight} (weight defaults to 1)."""
    out = {}
    for part in spec.split(","):
        i, _, w = part.partition(":")
        out[i.strip()] = float(w or 1)
    return out


def resolve_device(device="auto"):
    """The CUDA device for the setting `device` ("auto" or "cpu"), or None when it is "cpu" or there is none."""
    if device == "cpu":
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch.device("cuda") if torch.cuda.is_available() else None


def _topk_np(s, k):
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.take_along_axis(s, idx, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(idx, order, 1)


class Index:
    def __init__(self, weights: dict, store_root, restrict=None, device="auto", vectors=None, calibration=None,
                 missing="drop", min_weight=DEFAULTS["minViewWeight"]):
        """weights: {encoder id: weight}. vectors: optional {encoder id: (hashes, array)} instead of reading stores.
        calibration: a calibration (calibration.load) holding fixed statistics for every view.
        missing: "drop" (only hashes in every view) or "reweight" (see the module docstring)."""
        if missing not in ("drop", "reweight"):
            raise ValueError(f"missing must be 'drop' or 'reweight', got {missing!r}")
        raw = {}
        for i in weights:
            enc = get(i)
            hs, vs = vectors[i] if vectors is not None else Store(enc, store_root).load()
            if enc.row_kind() == "charts" and any(charts.is_chart_key(h) for h in hs):
                hs, vs = charts.representative(hs, vs)
            raw[i] = (enc, dict(zip(hs, vs)))
        sets = [set(d) for _, d in raw.values()]
        keep = set.intersection(*sets) if missing == "drop" else set.union(*sets)
        if restrict is not None:
            keep &= set(restrict)
        total = sum(weights.values())
        if missing == "reweight":  # enough of the weight must be present for a hash to be searchable
            keep = {h for h in keep if sum(weights[i] for i, (_, d) in raw.items() if h in d) >= min_weight * total}
        self.hashes = sorted(keep)
        self.pos = {h: n for n, h in enumerate(self.hashes)}
        self.missing = missing
        self.encoders = {i: enc for i, (enc, _) in raw.items()}
        self.calibration_id = calibration["id"] if calibration else None
        self.views, self.present, parts = {}, {}, []
        for i, (enc, d) in raw.items():
            mask = np.array([h in d for h in self.hashes], bool)
            st = calibrated(calibration, enc) if calibration else None
            dim = len(next(iter(d.values()))) if d else enc.dim
            x = np.zeros((len(self.hashes), dim), np.float32)
            if mask.any():
                x[mask] = normalise(enc, np.stack([d[h] for h, m in zip(self.hashes, mask) if m]), st)
            self.views[i] = (x, weights[i] / total)
            self.present[i] = mask
            parts.append(np.sqrt(weights[i] / total) * x)
        self.emb = np.hstack(parts).astype(np.float32) if missing == "drop" else None
        self.device = resolve_device(device)
        self._t, self._r, self._vt = None, {}, {}

    def __len__(self):
        return len(self.hashes)

    def _view_t(self, view):
        if view not in self._vt:
            import torch
            x, _ = self.views[view]
            self._vt[view] = (torch.from_numpy(x).to(self.device),
                              torch.from_numpy(self.present[view]).to(self.device))
        return self._vt[view]

    def _weighted(self, rows, weights):
        """Weighted mean of per-view cosines of `rows` to all items; a view missing on either side counts as 0.
        Pairs with no shared view get -inf."""
        tot = sum(weights.values())
        if self.device is not None:
            import torch
            rt = torch.as_tensor(rows, device=self.device)
            num = den = 0
            for v, w in weights.items():
                x, m = self._view_t(v)
                pair = (m[rt][:, None] & m[None, :]).float() * (w / tot)
                num = num + pair * (x[rt] @ x.T)
                den = den + pair
            return torch.where(den > 0, num, torch.tensor(-torch.inf, device=self.device))
        num = den = 0
        for v, w in weights.items():
            x, m = self.views[v][0], self.present[v]
            pair = (m[rows][:, None] & m[None, :]) * (w / tot)
            num = num + pair * (x[rows] @ x.T)
            den = den + pair
        return np.where(den > 0, num, -np.inf).astype(np.float32)

    def _scores(self, rows, r=None, weights=None):
        """Similarity of `rows` to all items with self excluded; CSLS-adjusted if r (local densities) is given.
        weights: per-request {view: weight} (default the index weights)."""
        if weights is not None or self.emb is None:
            s = self._weighted(rows, weights or {v: w for v, (_, w) in self.views.items()})
            if self.device is not None:
                import torch
                rt = torch.as_tensor(rows, device=self.device)
                s[torch.arange(len(rows), device=self.device), rt] = -torch.inf
                if r is not None:
                    r_t = torch.as_tensor(r, device=self.device)
                    s = 2 * s - r_t[rt][:, None] - r_t[None, :]
                return s
            s[np.arange(len(rows)), rows] = -np.inf
            if r is not None:
                s = 2 * s - r[rows][:, None] - r[None, :]
            return s
        if self.device is not None:
            import torch
            if self._t is None:
                self._t = torch.from_numpy(self.emb).to(self.device)
            rt = torch.as_tensor(rows, device=self.device)
            s = self._t[rt] @ self._t.T
            s[torch.arange(len(rows), device=self.device), rt] = -torch.inf
            if r is not None:
                r_t = torch.as_tensor(r, device=self.device)
                s = 2 * s - r_t[rt][:, None] - r_t[None, :]
            return s
        s = self.emb[rows] @ self.emb.T
        s[np.arange(len(rows)), rows] = -np.inf
        if r is not None:
            s = 2 * s - r[rows][:, None] - r[None, :]
        return s

    def _topk(self, s, k):
        if self.device is not None:
            import torch
            v, i = torch.topk(s, k, dim=1)
            return i.cpu().numpy(), v.cpu().numpy()
        i = _topk_np(s, k)
        return i, np.take_along_axis(s, i, 1)

    def local_density(self, k=10):
        """CSLS r_i: mean cosine of each item to its k nearest neighbours."""
        if k not in self._r:
            r = np.empty(len(self), np.float32)
            for a in range(0, len(self), CHUNK):
                rows = np.arange(a, min(a + CHUNK, len(self)))
                r[rows] = self._topk(self._scores(rows), k)[1].mean(1)
            self._r[k] = r
        return self._r[k]

    def view_density(self, view, k=10):
        """CSLS r_i inside one view only: mean cosine of each item to its k nearest neighbours in that view."""
        key = (view, k)
        if key not in self._r and self.emb is None:  # missing views: masked scores; rows without the view get 0
            r = np.zeros(len(self), np.float32)
            have = np.flatnonzero(self.present[view])
            for a in range(0, len(have), CHUNK):
                rows = have[a:a + CHUNK]
                v = self._topk(self._scores(rows, weights={view: 1.0}), k)[1]
                r[rows] = _finite_mean(v)
            self._r[key] = r
        if key not in self._r:
            x = self.views[view][0]
            r = np.empty(len(self), np.float32)
            for a in range(0, len(self), CHUNK):
                rows = np.arange(a, min(a + CHUNK, len(self)))
                if self.device is not None:
                    import torch
                    xt = torch.from_numpy(x).to(self.device)
                    s = xt[torch.as_tensor(rows, device=self.device)] @ xt.T
                    s[torch.arange(len(rows), device=self.device), torch.as_tensor(rows, device=self.device)] = -torch.inf
                    r[rows] = torch.topk(s, k, dim=1).values.mean(1).cpu().numpy()
                else:
                    s = x[rows] @ x.T
                    s[np.arange(len(rows)), rows] = -np.inf
                    r[rows] = np.take_along_axis(s, _topk_np(s, k), 1).mean(1)
            self._r[key] = r
        return self._r[key]

    def query_rows(self, rows, k=10, csls=True, csls_k=10, weights=None):
        """Top-k neighbours of the given row indices by the weighted combination -> (indices, scores).
        weights: per-request {view: weight}; CSLS densities are those of the index weights."""
        rows = np.asarray(rows)
        return self._topk(self._scores(rows, self.local_density(csls_k) if csls else None, weights), k)

    def search_views(self, rows, k=10):
        """Top-k of each view on its own -> {view: (indices, scores)}; items missing the view are never returned
        and a row missing it gets an empty result."""
        rows = np.asarray(rows)
        out = {}
        for v in self.views:
            have = self.present[v][rows]
            idx = np.full((len(rows), k), -1, np.int64)
            sc = np.full((len(rows), k), -np.inf, np.float32)
            if have.any():
                i, s = self._topk(self._scores(rows[have], weights={v: 1.0}), k)
                idx[have], sc[have] = i, s
            out[v] = (idx, sc)
        return out

    def knn(self, k=10, csls=False):
        parts = [self.query_rows(np.arange(a, min(a + CHUNK, len(self))), k, csls, k) for a in range(0, len(self), CHUNK)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def view_cosines(self, a, b):
        """Per-view cosine of items a and b; None where either lacks the view."""
        return {i: float(x[a] @ x[b]) if self.present[i][a] and self.present[i][b] else None
                for i, (x, _) in self.views.items()}


def _finite_mean(v):
    v = np.asarray(v, np.float64)
    ok = np.isfinite(v)
    return np.where(ok.any(1), np.where(ok, v, 0).sum(1) / np.maximum(ok.sum(1), 1), 0.0)
