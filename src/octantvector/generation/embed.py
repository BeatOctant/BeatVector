"""`octantvector update`: ingest the catalogue and archive into the ledger, re-evaluate the pools, then compute only
what the ledger says is missing (ledger.py). Running it again right after computes nothing.

Order: map encoders (catalogue only), then one pass over the zips for audio, image and zip encoders, then derived
encoders. Zips are read in journal commit_seq order (sequential on disk); each song is decoded once and shared by every
audio encoder, and the cover image is read once for every image encoder. The caller names the views and the pool
(pools.py) they are computed over.
"""
import hashlib
import multiprocessing as mp
import os
import random
import signal
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import closing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from . import archive, catalogue, config, pools
from ..common.charts import key_hash
from .encoders import get, registry
from .errors import GenerationError
from .ledger import Ledger, meta_digest, meta_fields
from .search import resolve_device
from .store import Store

_WORKER = {}
MAIN_KINDS = ("audio", "image")  # prepared in the workers, embedded in the main process (GPU)


def _init(ids, archive_root, given=None):
    if given is not None:  # a contract 1 (OctantHarvest to OctantVector) call: the worker sees the caller's settings, not the file
        config._GIVEN.set(given)
    try:  # zip-only encoders (chart_patterns, swing_flow) run without torch
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    _WORKER["archive"] = Path(archive_root)
    _WORKER["encs"] = {}
    for i in ids:
        e = get(i)
        if hasattr(e, "setup_worker"):
            e.setup_worker()
        _WORKER["encs"][i] = e


def _work(task):
    """-> (hash, encoder ids, zip sha256, {encoder id: output | None | Exception}, map-level error)"""
    h, ids = task
    out, sha = {}, None
    try:
        raw = archive.zip_path(_WORKER["archive"], h).read_bytes()  # read once: hashed, then opened from memory
        sha = hashlib.sha256(raw).hexdigest()
        with archive.MapZip(_WORKER["archive"], h, raw) as mz:
            for i in ids:
                e = _WORKER["encs"][i]
                if e.kind == "zip":
                    try:
                        out[i] = e.embed_zip(mz)
                    except Exception as ex:  # a broken chart only fails that encoder
                        out[i] = ex
            audio = [i for i in ids if _WORKER["encs"][i].kind == "audio"]
            data = mz.song_bytes() if audio else None
            image = [i for i in ids if _WORKER["encs"][i].kind == "image"]
            try:
                cover = mz.cover_bytes() if image else None
            except Exception as ex:  # a cover that cannot be read from the zip only fails the image encoders
                cover = ex
        for i in image:  # a map without a cover is a skip; one that cannot be read or decoded fails that encoder
            if isinstance(cover, Exception):
                out[i] = cover
                continue
            try:
                out[i] = None if cover is None else _WORKER["encs"][i].prep(cover)
            except Exception as ex:
                out[i] = ex
        if audio:
            try:
                y, sr = archive.decode_audio(data)
            except Exception as ex:  # a song that cannot be decoded only fails the audio encoders
                out.update((i, ex) for i in audio)
            else:
                for i in audio:
                    out[i] = _WORKER["encs"][i].prep(y, sr)
        return h, ids, sha, out, None
    except Exception as ex:
        return h, ids, sha, out, repr(ex)


def run(cfg, encoder_ids, pool="distribution", n=None, seed=2, retry_errors=False, log=sys.stderr):
    """Compute `encoder_ids` (and the sources of derived ones) over the pool `pool`. n: random subset of the pool
    (tests and quick experiments)."""
    arch, cat = cfg.require("archive"), cfg.require("catalogue")
    if not encoder_ids:
        raise GenerationError("name the views to compute")
    ids = list(dict.fromkeys(_with_sources(encoder_ids)))
    encs = {i: get(i) for i in ids}
    defs = pools.load()
    if pool not in defs:
        raise GenerationError(f"unknown pool {pool!r}; known: {', '.join(sorted(defs))}")
    led = Ledger(Path(cfg.store) / "ledger.sqlite")
    try:
        all_map = {i: c() for i, c in registry().items() if c.kind == "map"}  # digests of every map encoder stay current
        counts = led.ingest(catalogue.read_jsonl(cat), archive.commit_order(arch), all_map, defs)
        print("ingest: " + ", ".join(f"{k} {v}" for k, v in counts.items()), file=log)

        pending = {}
        stores = {i: Store(e, cfg.store, on_flush=lambda hs, i=i: _flushed(led, encs[i], stores[i], pending, hs))
                  for i, e in encs.items()}
        members = led.pool(pool)
        if n is not None:
            members = random.Random(seed).sample(members, min(n, len(members)))
            members.sort(key=lambda r: (r[1] is None, r[1] or 0, r[0]))
        plan = {}
        for i, e in encs.items():
            adopted = led.adopt(e, stores[i].fingerprint, stores[i].done())
            if adopted:
                print(f"{i}: {adopted} vectors already in the store recorded in the ledger", file=log)
            if e.kind == "derived":
                continue
            plan[i] = led.todo(e, stores[i].fingerprint, members, retry_errors)
            print(f"{i} (pool {pool}): {len(plan[i])} to compute", file=log)

        mapped = [i for i in plan if encs[i].kind == "map"]
        if any(plan[i] for i in mapped):
            recs = catalogue.records_for(cat, {h for i in mapped for h in plan[i]})
            for i in mapped:
                _run_map(led, encs[i], stores[i], pending, plan[i], recs)
        heavy = [i for i in plan if encs[i].kind in MAIN_KINDS + ("zip",) and plan[i]]
        if heavy:
            _run_heavy(cfg, arch, led, encs, stores, pending, {i: plan[i] for i in heavy}, log)
        for i in [i for i in encs if encs[i].kind == "derived"]:
            # derived views follow their source store; whatever they add is recorded like any other result
            try:
                print(f"{i}: {encs[i].derive(cfg.store, _device(cfg))} vectors", file=log)
            except GenerationError as ex:  # e.g. no style model configured: the other views are still done
                print(f"{i}: skipped ({ex})", file=log)
                continue
            led.adopt(encs[i], stores[i].fingerprint, Store(encs[i], cfg.store).done())
        print("ledger: " + str(led.summary()), file=log)
    finally:
        led.close()


def _with_sources(ids):
    """Sources first, so a derived view finds them computed."""
    for i in ids:
        yield from _with_sources(get(i).sources())
        yield i


def _flushed(led, enc, store, pending, keys):
    hashes = dict.fromkeys(key_hash(k) for k in keys)  # chart rows: one result per map version
    led.record(enc, store.fingerprint, [(h, "ok", pending.pop((enc.id, h), None), None) for h in hashes])


def _add(store, h, x):
    """One map's output: a vector (map rows) or {chart key: vector} (chart rows). -> False if there was nothing."""
    if isinstance(x, dict):
        if not x:
            return False
        store.add_many(x)
    else:
        store.add(h, x)
    return True


def _run_map(led, enc, store, pending, todo, recs):
    fields, rows = meta_fields(enc), []
    for h in todo:
        if h not in recs:
            continue
        m, v = recs[h]
        d = meta_digest(m, v, fields)
        vec = enc.embed_map(m, v)
        if vec is None or (isinstance(vec, dict) and not vec):
            rows.append((h, "skip", d, None))
        else:
            pending[(enc.id, h)] = d
            _add(store, h, vec)
    store.flush()
    led.record(enc, store.fingerprint, rows)


def _run_heavy(cfg, arch, led, encs, stores, pending, plan, log):
    heavy = list(plan)
    want = {}
    for i, hs in plan.items():
        for h in hs:
            want.setdefault(h, []).append(i)
    seq = dict(led.db.execute("SELECT hash, seq FROM versions WHERE seq IS NOT NULL"))
    tasks = sorted(want.items(), key=lambda t: seq.get(t[0], 0))
    print(f"{len(tasks)} maps to read for {heavy}", file=log)
    device = _device(cfg)
    for i in heavy:
        if encs[i].kind in MAIN_KINDS:
            encs[i].setup_model(device)
    t0 = time.time()
    try:
        if cfg.workers <= 1:  # in-process (tests, debugging)
            _init(heavy, str(arch))
            results = map(_work, tasks)
            _consume(led, encs, stores, results, len(tasks), t0, log)
        else:
            limit = cfg.worker_memory_gb * 2**30 if cfg.worker_memory_gb > 0 else None
            with closing(parallel(_work, tasks, cfg.workers, _init, (heavy, str(arch), config.current()),
                                  memory_limit=limit)) as results:
                _consume(led, encs, stores, (_unpack(t, r, log) for t, r in results), len(tasks), t0, log)
    finally:
        for i in heavy:
            stores[i].flush()


# numeric libraries start one thread per CPU in every worker; with many workers those threads only fight each other
ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# a worker is replaced after this many maps, so memory that builds up in a long-lived worker is given back
TASKS_PER_WORKER = 200


class MemoryLimitExceeded(Exception):
    """A worker went over the memory limit on this task and was stopped (only that task is lost)."""


def _tracked(fn, notes, i, task):
    """Runs in the worker: note which task this process holds, so the parent can name and stop it."""
    note = Path(notes) / str(os.getpid())
    note.write_text(str(i))
    try:
        return fn(task)
    finally:  # a worker that is replaced (TASKS_PER_WORKER) leaves no note behind whose pid could be reused
        note.unlink()


def _anon(pid):
    """Bytes of the process's own (anonymous) memory. Not the whole resident size: the shared libraries a worker maps
    (torch's alone are several GB) count there too, although they cost no memory of their own."""
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    return 0


def _watch(notes, limit, killed, stop):
    """Parent thread: stop any worker whose own memory goes over `limit` bytes, noting the task it held."""
    while not stop.wait(0.2):
        for note in Path(notes).iterdir():
            try:
                pid = int(note.name)
                if _anon(pid) > limit:
                    killed.add(int(note.read_text()))
                    os.kill(pid, signal.SIGKILL)
            except (OSError, ValueError):  # the worker ended or is just writing its note
                continue


def parallel(fn, tasks, workers, initializer, initargs, in_flight=4, memory_limit=None):
    """-> (task, fn(task)) for every task, run in worker processes, as they finish (any order). At most in_flight per
    worker wait to be consumed, so a slow consumer does not pile results up in memory.

    memory_limit (bytes, Linux): a worker whose own (anonymous) memory goes over it is stopped; its task comes back as
    (task, MemoryLimitExceeded) and the other tasks go on in fresh workers. A worker that dies any other way (crashed
    in native code, killed from outside) stops the run with the tasks it may have held: a rerun continues from the
    ledger."""
    saved = {k: os.environ.get(k) for k in ONE_THREAD}
    for k in ONE_THREAD:  # read by each worker as it starts (workers start as tasks are submitted); a caller's own
        os.environ.setdefault(k, "1")  # values are kept, and everything is restored when the run ends
    watch = bool(memory_limit) and os.path.exists("/proc/self/status")
    todo = deque(enumerate(tasks))
    try:
        while todo:
            with tempfile.TemporaryDirectory() as notes:
                killed, stop = set(), threading.Event()
                watcher = threading.Thread(target=_watch, args=(notes, memory_limit, killed, stop), daemon=True)
                # spawn, not fork: the main process may already hold CUDA / downloader threads (forked workers deadlock)
                ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=initializer,
                                         initargs=initargs, max_tasks_per_child=TASKS_PER_WORKER)
                if watch:
                    watcher.start()
                running = {}
                try:
                    while todo or running:
                        broken = False
                        while todo and len(running) < workers * in_flight:
                            i, task = todo.popleft()
                            try:
                                running[ex.submit(_tracked, fn, notes, i, task)] = (i, task)
                            except BrokenProcessPool:  # a worker died before its result was collected
                                todo.appendleft((i, task))
                                broken = True
                                break
                        if broken:
                            break
                        done, _ = wait(running, return_when=FIRST_COMPLETED)
                        for f in done:
                            i, task = running.pop(f)
                            try:
                                r = f.result()
                            except BrokenProcessPool:
                                running[f] = (i, task)
                                broken = True
                                continue
                            yield task, r
                        if broken:
                            break
                finally:
                    ex.shutdown(wait=True, cancel_futures=True)
                    stop.set()
                    if watcher.is_alive():
                        watcher.join()
                if not running and not todo:
                    return
                if not killed:
                    held = [t[0] if isinstance(t, tuple) else t for _, t in running.values()]
                    raise GenerationError(f"a worker process died (crashed in native code, or killed from outside); "
                                          f"maps it may have held: {', '.join(map(str, held))}. Run again to continue.")
                for i, task in sorted(running.values(), key=lambda t: t[0], reverse=True):
                    if i in killed:
                        yield task, MemoryLimitExceeded(f"used more than {memory_limit / 2**30:.1f} GiB of memory")
                    else:  # held by a worker that was only stopped with the pool: computed again
                        todo.appendleft((i, task))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _unpack(task, r, log):
    """A task stopped for its memory fails every encoder that wanted that map, like a zip that cannot be read."""
    if isinstance(r, MemoryLimitExceeded):
        h, ids = task
        print(f"{h}: worker stopped, it {r}", file=log, flush=True)
        return h, ids, None, {}, f"worker {r}"
    return r


# groups handed to the model thread before the main process waits for the oldest: enough that the GPU always has the
# next group while the main process takes in the workers' results, few enough that prepped maps do not pile up
GPU_AHEAD = 2


def _consume(led, encs, stores, results, total, t0, log):
    """Audio and image encoders with `batch` > 1 get their maps in groups (embed_many), so the GPU works on several
    maps at once; a group's results are recorded when the group is embedded (and the rest at the end).

    The models run in one thread of their own, so the main process goes on taking in the workers' results (and the
    workers go on decoding) while the GPU works: neither waits for the other. Only the main process writes the ledger
    and the stores; results are recorded in the order the groups were handed over."""
    groups, ahead = {}, deque()
    model = ThreadPoolExecutor(1, thread_name_prefix="octantvector-model")

    def hand_over(i, group):
        ahead.append((i, group, model.submit(_run_model, encs[i], [x for _, x in group])))
        while ahead and (len(ahead) > GPU_AHEAD or ahead[0][2].done()):
            _record_group(led, encs, stores, *ahead.popleft())

    try:
        for k, (h, ids, sha, out, err) in enumerate(results, 1):
            if sha and not led.zip_sha(h, sha):
                print(f"{h}: zip differs from the one first read for this hash; not used", file=log)
                err = f"zip sha256 {sha} differs from the recorded one"
            fails = {}
            for i, x in out.items():
                if err:
                    fails[i] = ("error", err)
                elif x is None or (isinstance(x, dict) and not x):
                    fails[i] = ("skip", None)
                elif isinstance(x, Exception):
                    fails[i] = ("error", repr(x))
                elif encs[i].kind in MAIN_KINDS:
                    g = groups.setdefault(i, [])
                    g.append((h, x))
                    if len(g) >= getattr(encs[i], "batch", 1):
                        hand_over(i, groups.pop(i))
                else:
                    try:
                        _embedded(led, encs[i], stores[i], h, x)
                    except Exception as ex:
                        fails[i] = ("error", repr(ex))
            if err:  # the zip could not be read at all: every encoder that wanted it failed
                for i in ids:
                    fails.setdefault(i, ("error", err))
            for i, (r, e) in fails.items():
                led.record(encs[i], stores[i].fingerprint, [(h, r, None, e)])
            if k % 100 == 0:
                rate = k / (time.time() - t0)
                print(f"{k}/{total}  {rate:.1f} maps/s  eta {(total - k) / rate / 60:.1f} min", file=log, flush=True)
    finally:  # also when the run stops: what was read is not computed again
        try:
            for i, g in groups.items():
                hand_over(i, g)
            while ahead:
                _record_group(led, encs, stores, *ahead.popleft())
        finally:
            model.shutdown(wait=True)


def _embedded(led, enc, store, h, v):
    """One map's output. None from an audio or image encoder means it found nothing to describe (e.g. no voice in the
    song): recorded as a skip."""
    if v is None:
        led.record(enc, store.fingerprint, [(h, "skip", None, None)])
    else:
        _add(store, h, v)


def _run_model(enc, xs):
    """In the model thread: one output per prepped map (embed_many for an encoder with a batch, else embed)."""
    return enc.embed_many(xs) if getattr(enc, "batch", 1) > 1 else [enc.embed(x) for x in xs]


def _record_group(led, encs, stores, i, group, future):
    """Record a group handed to the model thread, waiting for it if the model is still at work."""
    enc, store = encs[i], stores[i]
    try:
        vs = future.result()
    except Exception as ex:
        led.record(enc, store.fingerprint, [(h, "error", None, repr(ex)) for h, _ in group])
        return
    for (h, _), v in zip(group, vs):
        try:
            _embedded(led, enc, store, h, v)
        except Exception as ex:
            led.record(enc, store.fingerprint, [(h, "error", None, repr(ex))])


def _device(cfg):
    """The device the models run on ("cuda" or "cpu"), by the setting `device` (search.resolve_device)."""
    return "cpu" if resolve_device(cfg.device) is None else "cuda"
