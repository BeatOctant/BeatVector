"""Read-only access to the local map archive: maps/<hash[:2]>/<hash>.zip plus download.journal.jsonl."""
import io
import json
import zipfile
from pathlib import Path


def zip_path(archive: Path, version_hash: str) -> Path:
    return Path(archive) / "maps" / version_hash[:2] / f"{version_hash}.zip"


def commit_order(archive: Path) -> dict[str, int]:
    """version hash -> commit_seq for every verified zip (sequential on-disk order; read zips in this order)."""
    seq = {}
    with open(Path(archive) / "download.journal.jsonl") as f:
        for line in f:
            if '"committed"' in line:  # skips most other events without parsing them; the event is checked below
                try:
                    j = json.loads(line)
                except ValueError:  # a line cut short by an interrupted write; its zip is not committed
                    continue
                if j.get("event") == "committed":
                    seq[j["hash"]] = j["commit_seq"]
    return seq


def read_json(zf: zipfile.ZipFile, name: str):
    """Map .dat files are JSON, usually UTF-8 (with or without BOM), occasionally UTF-16. json.loads tells the
    encoding from the BOM or, without one, from the first bytes (RFC 8259 / 4627)."""
    return json.loads(zf.read(name))


class MapZip:
    """Context manager over one map zip, opened read-only. `data` is the zip's bytes when the caller has already
    read them, so the file is not read from disk a second time."""

    def __init__(self, archive: Path, version_hash: str, data: bytes | None = None):
        self.path = zip_path(archive, version_hash)
        self.hash = version_hash
        self.data = data

    def __enter__(self):
        self.zf = zipfile.ZipFile(self.path if self.data is None else io.BytesIO(self.data))
        self.names = {n.lower(): n for n in self.zf.namelist()}  # Info.dat vs info.dat etc.
        self.info = read_json(self.zf, self.names["info.dat"])
        return self

    def __exit__(self, *exc):
        self.zf.close()

    def song_bytes(self) -> bytes:
        fn = self.info.get("_songFilename") or self.info.get("audio", {}).get("songFilename") or "song.egg"
        name = self.names.get(fn.lower()) or next(
            n for n in self.zf.namelist() if n.lower().endswith((".egg", ".ogg", ".wav")))
        return self.zf.read(name)

    def cover_bytes(self) -> bytes | None:
        """The cover image named in Info.dat (`_coverImageFilename`, v4 `coverImageFilename`), matched to the zip's
        names without case; else the first .jpg, .jpeg or .png in the zip; None when there is none."""
        fn = self.info.get("_coverImageFilename") or self.info.get("coverImageFilename")
        name = self.names.get(fn.lower()) if isinstance(fn, str) and fn else None
        if name is None:
            name = next((n for n in self.zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))), None)
        return None if name is None else self.zf.read(name)

    def json(self, filename: str):
        return read_json(self.zf, self.names[filename.lower()])


# longest song decoded: a whole song is decoded into memory as float32, and libsndfile crashes the process on very
# long ones (a 9-hour song, map 5149c); BeatSaber songs over half an hour are joke or test maps
MAX_SONG_S = 1800


def decode_audio(data: bytes, max_s: float = MAX_SONG_S):
    """-> (mono float32 samples, sample rate). libsndfile first, ffmpeg (via librosa) as fallback. A song whose
    header says it is longer than max_s seconds raises ValueError before anything is decoded."""
    import soundfile as sf
    try:
        f = sf.SoundFile(io.BytesIO(data))
    except Exception:
        f = None
    if f is not None:
        with f:
            if f.frames > max_s * f.samplerate:
                raise ValueError(f"song is {f.frames / f.samplerate / 60:.0f} min, longer than the "
                                 f"{max_s / 60:.0f} min that is decoded")
            try:
                return f.read(dtype="float32", always_2d=True).mean(1), f.samplerate
            except Exception:
                pass
    import librosa, tempfile
    with tempfile.NamedTemporaryFile(suffix=".egg") as t:
        t.write(data); t.flush()
        y, sr = librosa.load(t.name, sr=None, mono=True, duration=max_s + 1)
    if len(y) > max_s * sr:
        raise ValueError(f"song is longer than the {max_s / 60:.0f} min that is decoded")
    return y, sr
