"""octantvector command line. Settings: OctantSpecs octantvector/config.md. Usage examples: README.md."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .generation import GenerationError, config

def _meta(cfg):
    from .generation.catalogue import load_meta
    return load_meta(cfg.require("catalogue"), cfg.store / "meta.json")


def cmd_check(cfg, a):
    from .common.dbformat import sha256
    from .generation.encoders import registry
    from .generation.encoders.base import PLUGIN_ERRORS
    from .generation.store import Store
    print(f"archive:   {cfg.archive} ({'ok' if cfg.archive and (cfg.archive / 'maps').is_dir() else 'missing'})")
    if cfg.catalogue and cfg.catalogue.exists():
        h = sha256(cfg.catalogue).removeprefix("sha256:")
        ok = "" if not cfg.catalogue_sha256 else (" (matches)" if h == cfg.catalogue_sha256 else " (MISMATCH)")
        print(f"catalogue: {cfg.catalogue} sha256 {h}{ok}")
    else:
        print(f"catalogue: {cfg.catalogue} (missing)")
    print(f"store:     {cfg.store}")
    encs = registry()
    for i in sorted(encs):
        if not (cfg.store / f"{i}@{encs[i].version}").exists():
            print(f"  {i:22s} -")
            continue
        try:
            print(f"  {i:22s} {len(Store(encs[i](), cfg.store).done())} vectors")
        except GenerationError as ex:
            print(f"  {i:22s} DEFINITION MISMATCH: {ex}")
    for name, why in sorted(PLUGIN_ERRORS.items()):
        print(f"plugin {name} not loaded: {why}")
    if (cfg.store / "ledger.sqlite").exists():
        from .generation.ledger import Ledger
        with Ledger.of(cfg.store) as led:
            print(f"ledger:    {json.dumps(led.summary())}")


def cmd_update(cfg, a):
    from .generation.embed import run
    run(cfg, a.encoders, pool=a.pool, n=a.n, seed=a.seed, retry_errors=a.retry_errors)


def _index(cfg, a):
    """Index from the --views / --calibration / --missing options shared by search and the evaluations; with
    --missing reweight it keeps the maps the runtime would (minViewWeight, common/params.py)."""
    from .generation.search import Index, parse_views
    cal = None
    if a.calibration:
        from .generation.calibration import load
        cal = load(a.calibration)
    return Index(parse_views(a.views), cfg.store, device=cfg.device, calibration=cal, missing=a.missing)


def cmd_calibrate(cfg, a):
    from .generation import calibration
    hashes = None
    if a.pool:
        from .generation.ledger import Ledger
        with Ledger.of(cfg.store) as led:
            hashes = led.pool_hashes(a.pool)
        if not hashes:
            raise SystemExit(f"pool {a.pool!r} is empty; run `octantvector update` first")
    cal = calibration.compute(cfg.store, a.views, hashes, pool=a.pool)
    path = calibration.save(cal, cfg.store)
    print(f"wrote {path}: " + ", ".join(f"{v} ({st['n']} vectors)" for v, st in cal["views"].items()))


def cmd_release(cfg, a):
    from .generation import calibration, dbrelease
    from .generation.api import _split
    views, relations = _split(a.views)  # encoder ids and relation names, checked as the api checks them
    dbrelease.release(cfg, views, calibration.load(a.calibration) if a.calibration else None, full=a.full,
                      presets=a.presets, notice=Path(a.notice).read_text(encoding="utf-8") if a.notice else None,
                      relations=relations)


def cmd_apply(cfg, a):
    from .runtime.db import DBError, apply_diff
    try:
        print(f"wrote {apply_diff(a.base, a.diff, a.out)}")
    except DBError as ex:
        raise SystemExit(f"cannot apply the diff: {ex}") from None


def cmd_db(cfg, a):
    """Open a distributed DB as OctantCompass would (checks included) and optionally search it. A query is a BeatSaver
    key or a version hash; chart views take "<key or hash>:<characteristic>:<difficulty>", or use the hardest
    Standard chart of the map when only the map is given."""
    from .common import charts
    from .runtime.db import DBError, open_db, search
    try:
        db = open_db(a.path)
    except DBError as ex:
        raise SystemExit(f"cannot use the DB: {ex}") from None
    print(json.dumps(db.info(), indent=1))
    md = _meta(cfg) if cfg.catalogue else {}
    by_key = {m["key"]: h for h, m in md.items() if h in db.pos}
    for q in a.query:
        m, _, chart = q.partition(":")
        h = by_key.get(m, m)
        for v in a.view or list(db.views):
            query = h
            if db.rows[v] == "charts":
                if chart:
                    query = f"{h}:{chart}"
                else:
                    query = charts.hardest_standard(k for k in db.chart_keys if charts.key_hash(k) == db.current(h))
                    if query is None:
                        print(f"\n== {q} in {v}: no Standard chart; name one as {m}:<characteristic>:<difficulty>")
                        continue
            r = search(db, query, v, a.k)
            print(f"\n== {r['query']} in {v}" + (f" (replaced {r['replaced']} by the current version)" if r["replaced"] else ""))
            for x in r["results"]:
                label = md.get(x["hash"], {}).get("key", x["hash"][:10])
                if "difficulty" in x:
                    label += f" {x['characteristic']}/{x['difficulty']}"
                print(f"{x['rank']:3d} {label:>24} {x['score']:7.3f}")


def cmd_search(cfg, a):
    md = _meta(cfg)
    ix = _index(cfg, a)
    by_key = {md[h]["key"]: h for h in ix.hashes if h in md}
    print(f"index: {len(ix)} maps; views " + ", ".join(f"{v}={w:.2f}" for v, (_, w) in ix.views.items())
          + ("" if a.no_csls else "; CSLS"))
    for q in a.query:
        h = by_key.get(q) or (q if q in ix.pos else None)
        if h is None:
            print(f"\n{q}: not in the index (missing selected views)")
            continue
        m = md.get(h, {})
        print(f"\n== {m.get('key', q)}  {m.get('artist', '')} - {m.get('song', '')}")
        if a.per_view:
            for v, (idx, score) in ix.search_views([ix.pos[h]], a.k).items():
                print(f"-- {v}: " + ", ".join(f"{md.get(ix.hashes[j], {}).get('key', '?')} {s:.3f}"
                                              for j, s in zip(idx[0], score[0]) if j >= 0))
        idx, score = ix.query_rows([ix.pos[h]], a.k, csls=not a.no_csls)
        print(f"{'#':>3} {'key':>6} {'score':>7} " + " ".join(f"{v[:10]:>10}" for v in ix.views) + "  title / mapper")
        for rank, (j, s) in enumerate(zip(idx[0], score[0]), 1):
            mm, cos = md.get(ix.hashes[j], {}), ix.view_cosines(ix.pos[h], j)
            print(f"{rank:3d} {mm.get('key', '?'):>6} {s:7.3f} "
                  + " ".join(f"{cos[v]:10.3f}" if cos[v] is not None else f"{'-':>10}" for v in ix.views)
                  + f"  {mm.get('artist', '')} - {mm.get('song', '')} / {mm.get('mapper', '')}")


def cmd_eval_neighbours(cfg, a):
    from .eval import results
    from .eval.neighbours import evaluate
    ix = _index(cfg, a)
    rows = [] if a.out else None
    iso = None
    if a.isolation_view:
        from argparse import Namespace
        from .eval.neighbours import isolation
        ref = ix if a.views == a.isolation_view else _index(cfg, Namespace(**{**vars(a), "views": a.isolation_view}))
        iso = isolation(ref, a.k)
    res = evaluate(ix, _meta(cfg), a.k, csls=a.csls, rows=rows, isolation_of=iso)
    print(json.dumps(res, indent=1))
    if a.out:
        settings = {"k": a.k, "csls": a.csls, **({"isolation_view": a.isolation_view} if a.isolation_view else {})}
        results.write(a.out, results.run_info(ix, "eval-neighbours", res, **settings), rows)
        print(f"wrote {a.out}: {len(rows)} queries", file=sys.stderr)


def cmd_playdata(cfg, a):
    """Build a signals file from the user's own data files (written only to --out)."""
    from pathlib import Path
    from .eval import playdata as pd
    md = _meta(cfg) if a.custom_levels else {}
    key_to_hash = {}
    for h, m in md.items():
        key_to_hash.setdefault(m["key"], h)
    plays = pd.read_datarecorder(a.datarecorder) if a.datarecorder else []
    sig = pd.build_signals(
        votes=pd.read_votes(a.votes) if a.votes else None,
        playlists=pd.read_playlists(a.playlists) if a.playlists else None,
        player_data=pd.read_player_data(a.player_data) if a.player_data else None,
        plays=plays, replays=[pd.replay_hashes(d) for d in a.replays],
        installed=pd.installed_keys(a.custom_levels) if a.custom_levels else None, key_to_hash=key_to_hash)
    Path(a.out).write_text(json.dumps({"signals": sig, "plays": plays}))
    print(f"wrote {a.out}: {len(sig)} maps with signals, {len(plays)} play records")


def cmd_eval_personal(cfg, a):
    from .generation import catalogue
    from .eval import personal
    from .eval.playdata import merge_by_key
    data = json.loads(open(a.signals).read())
    md = _meta(cfg)
    ix = _index(cfg, a)
    pool_of_key = {md[h]["key"]: h for h in ix.hashes if h in md}
    merged = merge_by_key(data["signals"], md, pool_of_key)
    lab = personal.labels(merged)
    cand = sorted(lab)
    liked = [h for h in cand if lab[h]]
    diffs, pts = None, None
    if a.beta:
        recs = catalogue.records_for(cfg.require("catalogue"), set(cand) | {p["hash"] for p in data["plays"]})
        diffs = {h: np.array([[d["nps"], d["njs"]] for d in catalogue.standard_diffs(recs[h][1])]) for h in cand if h in recs}
        pts = []
        for p in data["plays"]:
            if p["outcome"] == "finished" and p["mode"] == "SoloStandard" and p["speed"] == 1.0 and p["hash"] in recs:
                d = {x["difficulty"]: x for x in catalogue.standard_diffs(recs[p["hash"]][1])}.get(p["diff"])
                if d:
                    pts.append((md.get(p["hash"], {}).get("key"), d["nps"], d["njs"]))
    res = personal.run(ix, cand, liked, kind=a.profile, csls=not a.no_csls, beta=a.beta, difficulties=diffs,
                       finished_points=pts, csls_density=a.csls_density, key_of={h: md[h]["key"] for h in cand},
                       keep_ranks=bool(a.out))
    ranks = res.pop("ranks", {})
    print(json.dumps(res, indent=1))
    if a.out:
        from .eval import results
        info = results.run_info(ix, "eval-personal", res, profile=a.profile, beta=a.beta, csls=not a.no_csls,
                                csls_density=a.csls_density)
        results.write(a.out, info, [{"query": h, "rank": r} for h, r in ranks.items()])
        print(f"wrote {a.out}: ranks of {len(ranks)} liked maps", file=sys.stderr)


def cmd_swing_dataset(cfg, a):
    from .generation.swing import dataset
    dataset.build(cfg, a.out, a.min_upvotes, n=a.n, seed=a.seed)


def cmd_swing_eval(cfg, a):
    from .generation.swing import dataset
    print(json.dumps(dataset.evaluate(dataset.load(a.rows), k=a.folds, lam=a.lam), indent=1))


def cmd_swing_trace(cfg, a):
    """Simulated swings of one map: per-swing table, ratings, and style tags when a style model is configured."""
    from .generation.archive import MapZip
    from .generation.swing import NAMES, chart_features, params, standard_charts, style
    from .generation.swing.features import HAZARD_NAMES
    from .generation.swing.simulate import simulate_all
    md = _meta(cfg) if cfg.catalogue else {}
    h = next((x for x, m in md.items() if m["key"] == a.map), a.map)
    P = params.load()
    with MapZip(cfg.require("archive"), h) as mz:
        charts = standard_charts(mz)
    charts = [c for c in charts if c[0] == a.diff] if a.diff else charts[-1:]
    if not charts:
        raise SystemExit(f"no Standard {a.diff} in {a.map}")
    for diff, ch in charts:
        v, both, rat = chart_features(ch, P)
        if v is None:
            print(f"{diff}: too few notes"); continue
        print(f"== {md.get(h, {}).get('key', h)} {diff}: " + ", ".join(f"{k} {x:.2f}" for k, x in rat.items()))
        _, _, hz = simulate_all(ch, P)
        for what, roles in (("walls", hz["wall_roles"]), ("bombs", hz["bomb_roles"])):
            if roles:
                print(f"{what}: " + ", ".join(f"{r} {roles.count(r)}" for r in dict.fromkeys(roles)))
        f = dict(zip(NAMES, v))
        print("walls/bombs: " + ", ".join(f"{n} {f[n]:.2f}" for n in HAZARD_NAMES if f[n]))
        if a.swings:
            print(f"{'time':>7} {'hand':>4} {'kind':>6} {'dir':>4} {'par':>3} {'roll':>5} {'reset':>5} {'ebpm':>5} {'strain':>6} {'rot':>5} {'repos':>5} {'diff':>5}")
            for s in both:
                print(f"{s.t:7.2f} {'LR'[s.hand]:>4} {s.kind:>6} {s.dir:4.0f} {'FB'[not s.forehand]:>3} {s.roll:5.0f} "
                      f"{('bomb' if s.bomb_reset else 'yes') if s.reset else '':>5} {60 / (2 * s.dt) if s.dt else 0:5.0f} "
                      f"{s.strain:6.2f} {s.rotation:5.2f} {s.reposition:5.2f} {s.diff:5.2f}")
        if cfg.swing_model and cfg.swing_model.exists():
            learned, fine = style.describe(style.load_model(cfg.swing_model), v)
            print("style (BeatSaver tags, learned): " + ", ".join(f"{t} {p:.2f}" for t, p in sorted(learned.items(), key=lambda t: -t[1])))
            print("fine tags: " + (", ".join(fine) or "-"))


def cmd_swing_train(cfg, a):
    from .generation.swing import dataset, style
    model = dataset.train(dataset.load(a.rows), lam=a.lam)
    style.save(model, a.out)
    print(f"wrote {a.out}: quantiles from {model['n_rows']} charts, learned tags "
          + ", ".join(f"{t} ({m['n_pos']} maps)" for t, m in model["learned"].items()))


def _index_args(p, missing=("drop", "reweight")):
    p.add_argument("--calibration", help="fixed normalisation statistics (file written by `octantvector calibrate`)")
    p.add_argument("--missing", choices=missing, default="drop",
                   help="drop: only maps with every view; reweight: keep maps with enough of the weight "
                        "(minViewWeight, as recommendations do); a missing view scores 0")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="octantvector", description="Multi-encoder vector search over a local BeatSaver archive")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="show configured paths, catalogue hash, and vectors per encoder")
    p = sub.add_parser("update", help="ingest catalogue and archive into the ledger, then compute what is missing")
    p.add_argument("encoders", nargs="+", help="encoder ids (OctantSpecs octantvector/encoders.md)")
    p.add_argument("--pool", default="distribution", help="pool to compute over (pools.py; default %(default)s)")
    p.add_argument("--n", type=int, help="random subset of the pool")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--retry-errors", action="store_true", help="compute again what failed before")
    p = sub.add_parser("calibrate", help="fix the normalisation statistics of views over a pool")
    p.add_argument("views", nargs="+", help="encoder ids")
    p.add_argument("--pool", help="ledger pool to compute them on (default: every vector in each store)")
    p = sub.add_parser("release", help="write the next distributable DB version under <store>/releases, as a diff when "
                                       "possible (for experiments; OctantHarvest runs the real releases)")
    p.add_argument("views", nargs="+", help="encoder ids")
    p.add_argument("--calibration", help="file written by `octantvector calibrate` covering every view "
                                         "(default: the previous version's; needed for the first release)")
    p.add_argument("--full", action="store_true", help="publish a full even when a diff is possible")
    p.add_argument("--presets", help="JSON file with the hands (手札) to put in the manifest (default: the previous "
                                     "version's)")
    p.add_argument("--notice", help="Markdown file with the DB's terms and credits to put in the manifest (default: the "
                                    "previous version's)")
    p = sub.add_parser("apply", help="apply a diff to a full DB into a new folder, as OctantCompass does")
    p.add_argument("base")
    p.add_argument("diff")
    p.add_argument("out")
    p = sub.add_parser("db", help="open a distributed DB with its checks, print its info, optionally search it")
    p.add_argument("path")
    p.add_argument("query", nargs="*", help="BeatSaver keys or version hashes")
    p.add_argument("--view", action="append", help="id@version (default: every view)")
    p.add_argument("--k", type=int, default=10)
    p = sub.add_parser("search", help="nearest neighbours of maps by BeatSaver key or version hash")
    p.add_argument("query", nargs="+")
    p.add_argument("--views", required=True, help="id:weight,...")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--no-csls", action="store_true", help="plain cosine instead of CSLS hub reduction")
    p.add_argument("--per-view", action="store_true", help="also print each view's own top-k")
    _index_args(p)
    p = sub.add_parser("eval-neighbours", help="catalogue-only neighbour quality and hubness")
    p.add_argument("--views", required=True, help="id:weight,...")
    _index_args(p)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--csls", action="store_true")
    p.add_argument("--isolation-view", help="also break the agreement down by how isolated each query is in this "
                                            "view (e.g. clap_htsat_unfused:1); keep it fixed to compare --views")
    p.add_argument("--out", help="also save every query's neighbours with the run settings (.jsonl[.gz])")
    p = sub.add_parser("playdata", help="build a signals file from your own play data (local only)")
    for f in ("votes", "player-data", "datarecorder", "custom-levels"):
        p.add_argument(f"--{f}")
    p.add_argument("--playlists", nargs="*", default=[])
    p.add_argument("--replays", nargs="*", default=[], help="replay folders (only file names are read)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("eval-personal", help="leave-one-out against your liked maps (OctantSpecs octantvector/design.md)")
    p.add_argument("--signals", required=True, help="file written by `octantvector playdata`")
    p.add_argument("--views", required=True, help="id:weight,...")
    _index_args(p, missing=("drop",))  # the per-view profile scores need every view
    p.add_argument("--profile", default="km3", help="mean | km<k>")
    p.add_argument("--beta", type=float, default=0.0, help="difficulty-fit weight")
    p.add_argument("--no-csls", action="store_true")
    p.add_argument("--csls-density", choices=("view", "combined"), default="view",
                   help="hub density per view (default) or from all views combined")
    p.add_argument("--out", help="also save each liked map's rank with the run settings (.jsonl[.gz]; personal data)")
    p = sub.add_parser("swing-dataset", help="swing features + catalogue labels for every Standard difficulty of the pool")
    p.add_argument("--out", required=True, help="output .jsonl.gz")
    p.add_argument("--n", type=int, help="random subset of the pool")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--min-upvotes", type=int, default=100, help="upvote floor of the maps (default %(default)s)")
    p = sub.add_parser("swing-eval", help="check the swing simulation against stars, style tags and parity (OctantSpecs octantvector/swing.md)")
    p.add_argument("--rows", required=True, help="file written by swing-dataset")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--lam", type=float, default=10.0, help="L2 strength of the cross-validated models")
    p = sub.add_parser("swing-train", help="style model (learned BeatSaver tags + fine-tag quantiles) from swing-dataset rows")
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True, help="model file; point the swing_model setting at it")
    p.add_argument("--lam", type=float, default=10.0, help="L2 strength")
    p = sub.add_parser("swing-trace", help="simulated swings and ratings of one map")
    p.add_argument("map", help="BeatSaver key or version hash")
    p.add_argument("--diff", help="difficulty (default: hardest Standard)")
    p.add_argument("--swings", action="store_true", help="print every swing")
    a = ap.parse_args(argv)
    try:  # the one place a generation error becomes the command's exit message
        cfg = config.load()
        {"check": cmd_check, "update": cmd_update, "calibrate": cmd_calibrate, "release": cmd_release, "apply": cmd_apply, "db": cmd_db, "search": cmd_search, "eval-neighbours": cmd_eval_neighbours,
         "playdata": cmd_playdata, "eval-personal": cmd_eval_personal, "swing-dataset": cmd_swing_dataset,
         "swing-eval": cmd_swing_eval, "swing-train": cmd_swing_train, "swing-trace": cmd_swing_trace}[a.cmd](cfg, a)
    except GenerationError as ex:
        raise SystemExit(f"octantvector: {ex}") from None


if __name__ == "__main__":
    sys.exit(main())
