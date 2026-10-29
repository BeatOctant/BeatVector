"""Song decoding in the heavy pass (archive.decode_audio, embed._work)."""
import io
import json
import zipfile

import numpy as np
import pytest

from octantvector.generation import archive, embed
from octantvector.generation.encoders import get

sf = pytest.importorskip("soundfile")


def _wav(seconds, sr=8000):
    b = io.BytesIO()
    sf.write(b, np.zeros((int(seconds * sr), 2), np.float32), sr, format="WAV")
    return b.getvalue()


def test_a_song_longer_than_the_limit_is_refused_before_it_is_decoded():
    y, sr = archive.decode_audio(_wav(3))
    assert sr == 8000 and y.shape == (24000,)
    with pytest.raises(ValueError, match="longer than"):
        archive.decode_audio(_wav(3), max_s=2)


class _Audio:
    kind = "audio"

    def prep(self, y, sr):
        return len(y)


def test_a_song_that_cannot_be_decoded_fails_only_the_audio_encoders(tmp_path, monkeypatch):
    info = {"_beatsPerMinute": 120, "_songFilename": "song.egg", "_difficultyBeatmapSets": [
        {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_beatmapFilename": "Expert.dat", "_noteJumpMovementSpeed": 16}]}]}
    notes = {"_version": "2.2.0", "_notes": [{"_time": t / 2, "_lineIndex": t % 4, "_lineLayer": 0, "_type": t % 2,
                                              "_cutDirection": 1} for t in range(64)]}
    h = "cd" + "0" * 38
    (tmp_path / "maps" / "cd").mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "maps" / "cd" / f"{h}.zip", "w") as z:
        z.writestr("Info.dat", json.dumps(info))
        z.writestr("Expert.dat", json.dumps(notes))
        z.writestr("song.egg", _wav(3))
    monkeypatch.setattr(archive.decode_audio, "__defaults__", (2,))
    monkeypatch.setitem(embed._WORKER, "archive", tmp_path)
    monkeypatch.setitem(embed._WORKER, "encs", {"chart_patterns": get("chart_patterns"), "audio": _Audio()})
    _, _, sha, out, err = embed._work((h, ["chart_patterns", "audio"]))
    assert err is None and sha
    assert isinstance(out["audio"], ValueError)
    assert list(out["chart_patterns"]) == [f"{h}:Standard:Expert"]


def test_an_audio_encoder_that_finds_nothing_records_a_skip_not_an_error():
    class Led:
        rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"
        added = []

        def add(self, h, x):
            self.added.append(h)

    class Quiet(_Audio):
        def embed(self, x):
            return None if x == 0 else np.ones(2, np.float32)

    led, store = Led(), Store()
    results = [("a" * 40, ["q"], "s", {"q": 0}, None), ("b" * 40, ["q"], "s", {"q": 1}, None)]
    embed._consume(led, {"q": Quiet()}, {"q": store}, iter(results), 2, 0, io.StringIO())
    assert led.rows == [("a" * 40, "skip", None, None)] and store.added == ["b" * 40]


def test_an_encoder_with_a_batch_gets_its_maps_in_groups_and_the_rest_at_the_end():
    class Led:
        rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"
        added = []

        def add(self, h, x):
            self.added.append(h)

    class Grouped(_Audio):
        batch = 2
        calls = []

        def embed_many(self, xs):
            self.calls.append(len(xs))
            return [None if x == 0 else np.ones(2, np.float32) for x in xs]

    enc, led, store = Grouped(), Led(), Store()
    results = [(str(n) * 40, ["g"], "s", {"g": n % 2}, None) for n in range(5)]
    embed._consume(led, {"g": enc}, {"g": store}, iter(results), 5, 0, io.StringIO())
    assert enc.calls == [2, 2, 1]
    assert store.added == ["1" * 40, "3" * 40] and [r[1] for r in led.rows] == ["skip"] * 3


def test_the_model_works_while_the_main_process_takes_in_further_results():
    import threading

    class Led:
        rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"
        added = []

        def add(self, h, x):
            self.added.append(h)

    taken = threading.Event()

    class Slow(_Audio):
        overlapped = []

        def embed(self, x):  # the first map waits until the main process has taken in the third result
            if x == 0:
                self.overlapped.append(taken.wait(5))
            return np.ones(2, np.float32)

    def results():
        for n in range(4):
            if n == 2:  # asked for while the model holds the first map
                taken.set()
            yield str(n) * 40, ["s"], "s", {"s": n}, None

    enc, store = Slow(), Store()
    embed._consume(Led(), {"s": enc}, {"s": store}, results(), 4, 0, io.StringIO())
    assert enc.overlapped == [True] and store.added == [str(n) * 40 for n in range(4)]


def test_a_group_the_model_fails_on_is_recorded_as_errors_and_the_others_go_on():
    class Led:
        rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"
        added = []

        def add(self, h, x):
            self.added.append(h)

    class Flaky(_Audio):
        batch = 2

        def embed_many(self, xs):
            if 0 in xs:
                raise RuntimeError("the model broke")
            return [np.ones(2, np.float32) for _ in xs]

    led, store = Led(), Store()
    results = [(str(n) * 40, ["f"], "s", {"f": n}, None) for n in range(4)]
    embed._consume(led, {"f": Flaky()}, {"f": store}, iter(results), 4, 0, io.StringIO())
    assert [(r[0], r[1]) for r in led.rows] == [("0" * 40, "error"), ("1" * 40, "error")]
    assert "the model broke" in led.rows[0][3] and store.added == ["2" * 40, "3" * 40]


class OutOfMemoryError(RuntimeError):
    """Named like torch.cuda.OutOfMemoryError, which embed tells by its name."""


def _ledger_and_store():
    class Led:
        def __init__(self):
            self.rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"

        def __init__(self):
            self.added = []

        def add(self, h, x):
            self.added.append(h)

    return Led(), Store()


def test_a_group_the_gpu_has_no_room_for_is_split_and_the_encoder_keeps_the_smaller_size():
    class Tight(_Audio):
        id = "t"
        batch = 4
        calls = []

        def embed_many(self, xs):
            self.calls.append(len(xs))
            if len(xs) > 1:
                raise OutOfMemoryError("CUDA out of memory")
            return [np.ones(2, np.float32) for _ in xs]

    (led, store), enc, log = _ledger_and_store(), Tight(), io.StringIO()
    results = [(str(n) * 40, ["t"], "s", {"t": n}, None) for n in range(8)]
    assert embed._consume(led, {"t": enc}, {"t": store}, iter(results), 8, 0, log) == {}
    assert enc.calls == [4, 2, 1, 1, 1, 1, 1, 1, 1, 1]  # the second group goes one map at a time from the start
    assert store.added == [str(n) * 40 for n in range(8)] and led.rows == []
    assert "t: the GPU had no room, 2 maps at a time from now" in log.getvalue()


def test_a_map_the_gpu_has_no_room_for_even_alone_is_not_recorded_so_a_later_run_computes_it():
    class Huge(_Audio):
        id = "h"
        tries = []

        def embed(self, x):
            if x == 1:
                self.tries.append(x)
                raise OutOfMemoryError("CUDA out of memory")
            return np.ones(2, np.float32)

    (led, store), enc = _ledger_and_store(), Huge()
    results = [(str(n) * 40, ["h"], "s", {"h": n}, None) for n in range(3)]
    assert embed._consume(led, {"h": enc}, {"h": store}, iter(results), 3, 0, io.StringIO()) == {"h": 1}
    assert enc.tries == [1, 1]  # tried again once with the cache emptied
    assert store.added == ["0" * 40, "2" * 40] and led.rows == []
