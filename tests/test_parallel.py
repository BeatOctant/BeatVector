"""The worker pool of the heavy pass (embed.parallel)."""
import io
import os
import sys
import time

import pytest

from octantvector.generation import GenerationError, embed


def _noop(*args):
    pass


def _square(task):
    return task * task


def _threads(task):
    return task, os.environ.get("OPENBLAS_NUM_THREADS")


def _die_on_three(task):
    if task == 3:
        os._exit(1)  # as the kernel's out-of-memory killer or a crash in native code would end it
    return task


def _hog(task):
    if task == 5:
        block = bytearray(400 * 2**20)  # resident once written
        for i in range(0, len(block), 4096):
            block[i] = 1
        time.sleep(30)
    return task


def test_every_task_comes_back_once():
    got = sorted(embed.parallel(_square, range(50), 3, _noop, ()))
    assert got == [(i, i * i) for i in range(50)]


def test_workers_start_with_one_numeric_thread_and_the_caller_keeps_its_environment(monkeypatch):
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    assert {t for _, (_, t) in embed.parallel(_threads, range(4), 2, _noop, ())} == {"1"}
    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_a_worker_that_dies_stops_the_run_instead_of_waiting_forever():
    got = []
    with pytest.raises(GenerationError, match="worker process died.*3"):
        for _, r in embed.parallel(_die_on_three, range(8), 2, _noop, ()):
            got.append(r)
    assert 3 not in got


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the memory watch reads /proc")
def test_a_worker_over_the_memory_limit_fails_only_its_task_and_the_rest_go_on():
    got = dict(embed.parallel(_hog, range(12), 2, _noop, (), memory_limit=300 * 2**20))
    assert sorted(got) == list(range(12))
    assert isinstance(got[5], embed.MemoryLimitExceeded)
    assert all(got[i] == i for i in got if i != 5)


def test_a_map_stopped_for_its_memory_is_a_map_level_error():
    h, ids, sha, out, err = embed._unpack(("ab" * 20, ["clap_htsat_unfused"]),
                                          embed.MemoryLimitExceeded("used more than 6.0 GiB of memory"), io.StringIO())
    assert (h, ids, sha, out) == ("ab" * 20, ["clap_htsat_unfused"], None, {}) and "6.0 GiB" in err


def _map_file(path):
    import mmap
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        return sum(m[i] for i in range(0, len(m), 4096))  # every page resident, none of it the worker's own


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the memory watch reads /proc")
def test_mapped_files_do_not_count_against_the_memory_limit(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"\x01" * (200 * 2**20))
    tasks = [str(big)] * 3
    got = list(embed.parallel(_map_file, tasks, 1, _noop, (), memory_limit=100 * 2**20))
    assert [r for _, r in got] == [51200] * 3


class _BreaksOnThirdSubmit:
    def __init__(self, *a, **k):
        self.n = 0

    def submit(self, fn, *args):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        self.n += 1
        if self.n == 3:
            raise BrokenProcessPool("a child process terminated abruptly")
        f = Future()
        f.set_exception(BrokenProcessPool("a child process terminated abruptly"))
        return f

    def shutdown(self, **k):
        pass


def test_a_pool_found_broken_while_handing_out_work_stops_the_run_cleanly(monkeypatch):
    monkeypatch.setattr(embed, "ProcessPoolExecutor", _BreaksOnThirdSubmit)
    with pytest.raises(GenerationError, match="worker process died"):
        list(embed.parallel(_square, range(10), 1, _noop, (), in_flight=5))
