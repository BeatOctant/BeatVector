"""The vocal stem's Demucs path (generation/encoders/stem.py) against demucs itself, on a tiny random HTDemucs saved
the way the Hugging Face repo holds htdemucs (htdemucs.yaml + <sig>.safetensors with the export metadata): the local
load builds the bag demucs.hf.get_hf_model builds, and the separation is the one before the stem moved to core."""
import hashlib
import json
import sys
import types
from collections import OrderedDict

import numpy as np
import pytest

pytest.importorskip("demucs")
pytest.importorskip("yaml")
pytest.importorskip("safetensors")
torch = pytest.importorskip("torch")

from octantvector.generation.encoders import stem  # noqa: E402

SIG = "955717e8"
TINY = {"sources": ["drums", "bass", "other", "vocals"], "channels": 16, "depth": 2, "nfft": 512, "t_layers": 1,
        "t_heads": 2, "segment": 1, "samplerate": stem.SEP_SR}


@pytest.fixture
def models_dir(tmp_path):
    from demucs.htdemucs import HTDemucs
    from safetensors.torch import save_file
    torch.manual_seed(0)
    model = HTDemucs(**TINY)
    folder = tmp_path / "hf" / "adefossez" / "HTDemucs" / stem.HTDEMUCS.revision
    folder.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, folder / f"{SIG}.safetensors",
              metadata={"klass": "demucs.htdemucs.HTDemucs", "args": "[]", "kwargs": json.dumps(TINY)})
    (folder / "htdemucs.yaml").write_text(f"models: ['{SIG}']\n")
    return folder


def _hub_bag(folder, monkeypatch):
    """demucs 4.1.0's own load of `htdemucs`, its hub download answered from `folder`."""
    from demucs.hf import get_hf_model
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo_id, filename, **kw: str(folder / filename)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return get_hf_model("htdemucs").eval()


class OldSeparator:
    """The Separator before the stem moved to core (vocal.py), with the bag passed in instead of get_model'd."""

    def __init__(self, model, device, half, sep_clips):
        from demucs.apply import apply_model
        self.torch, self.device, self.half, self.sep_clips = torch, device, half, sep_clips
        self.model, self._apply = model, apply_model
        self._stem = list(self.model.sources).index("vocals")
        self.segment = int(self.model.models[0].segment * stem.SEP_SR) if hasattr(self.model, "models") else \
            int(self.model.segment * stem.SEP_SR)
        self._cache, self._keep = OrderedDict(), 4

    def vocals(self, mix):
        key = hashlib.blake2b(mix.tobytes(), digest_size=16).digest() + str(mix.shape).encode()
        if key in self._cache:
            return self._cache[key]
        torch = self.torch
        out = []
        cuda = str(self.device).startswith("cuda")
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=cuda and self.half):
            for s in range(0, len(mix), self.sep_clips):
                x = torch.from_numpy(mix[s:s + self.sep_clips]).to(self.device)[:, None].expand(-1, 2, -1)
                mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True) + 1e-8
                split = x.shape[-1] > self.segment
                y = self._apply(self.model, (x - mu) / sd, shifts=0, split=split, overlap=0.25, device=self.device)
                out.append((y[:, self._stem].float() * sd + mu).mean(1).cpu().numpy())
        voc = np.concatenate(out)
        self._cache[key] = voc
        return voc


def test_the_local_load_builds_the_bag_demucs_builds(models_dir, monkeypatch):
    ours, theirs = stem.load_htdemucs(models_dir), _hub_bag(models_dir, monkeypatch)
    assert list(ours.sources) == list(theirs.sources) and ours.weights == theirs.weights
    assert [(type(m), m.segment) for m in ours.models] == [(type(m), m.segment) for m in theirs.models]
    a, b = ours.state_dict(), theirs.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_the_separation_is_the_old_one(models_dir, monkeypatch):
    monkeypatch.setattr(stem.Separator, "_models", {})
    monkeypatch.setattr(stem.Separator, "_caches", {})
    sep = stem.Separator(models_dir, "cpu")
    old = OldSeparator(_hub_bag(models_dir, monkeypatch), "cpu", stem.SEP_HALF, stem.SEP_CLIPS)
    rng = np.random.default_rng(0)
    for n in (stem.SEP_SR // 2, int(stem.SEP_SR * 1.5)):  # within one segment (no split), and longer (split)
        mix = (rng.standard_normal((3, n)) * 0.1).astype(np.float32)
        mix[1] *= 0.0  # a silent clip: the normalisation's epsilon
        new = sep.vocals(mix)
        assert new.shape == mix.shape and np.abs(new - old.vocals(mix)).max() < 1e-5
