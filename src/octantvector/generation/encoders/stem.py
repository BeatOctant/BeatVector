"""The vocal stem, a step before any clip audio encoder (ClipAudioEncoder with `stem = "vocal"`): the clips are separated
with Demucs (htdemucs) and only the vocal stem is embedded. Role `vocal`, so a hand can ask for maps whose singing is
alike apart from the whole mix. A song with too little voice in its clips (an instrumental) gets no vector, so a vocal
view is simply missing for it.

Spike (OctantVector spikes/vocal_probe.py, 771 maps, 2026-10-10): added to the mix CLAP, each vocal view raised "the 10
nearest are by the same artist" from 10.6 % to 12.2-13.2 %, and 40-56 % of the same-artist maps a vocal view finds are
not in the mix CLAP's top 10. Taking the vocal-richest 3 of 7 windows was hardly better than the fixed 3 clips (13.2 vs
12.9 %), which need only 3 clips separated. The clips are at the same places as the music views' but 7.8 s long (the
spike used 10 s): htdemucs takes 7.8 s at a time, so a 10 s clip cost two overlapping forwards. Workers decode songs to
mono, so the separator gets the mono mix on both channels (the spike used the stereo file; guessed to cost a little
separation quality).

Cost: separation is most of the GPU time, and reading the archive is the other big part. The vocal views share both:
one heavy pass reads each song once for every view named, and the stem of a group of maps is separated once and reused
by the other vocal views in the same pass (the separation cache below). The heavy pass hands `batch` maps at once
(embed_many); their clips go through the separator `sep_clips` at a time and through the embedding model together.
Demucs' random time shifts are off (shifts=0), so the same song always gives the same vector.

The separator's weights are a declared model like any other (HTDEMUCS), read from models_dir only."""
import hashlib
from collections import OrderedDict

import numpy as np

from ..errors import GenerationError
from .base import Model, clips, resample

SEP_SR = 44100
HTDEMUCS = Model("adefossez/HTDemucs", "cbc8a9b1a87023b7fd74e7b3412e6321c0eab003",
                 ("htdemucs.yaml", "955717e8.safetensors"))
CREDITS = ("Demucs (Alexandre Défossez), https://github.com/adefossez/demucs, code under MIT; "
           "HTDemucs weights, https://huggingface.co/adefossez/HTDemucs, terms in the model card (not checked)",)
# what a class that turns the vocal stem on gets unless it sets them itself (ClipAudioEncoder.__init_subclass__)
VOCAL = {"role": "vocal",
         "batch": 8,  # maps per embed_many
         "clip_s": 7.8}  # htdemucs' training segment: one forward per clip, no overlapping chunks
SEP_CLIPS = 12  # clips per separator call (GPU memory)
SEP_HALF = True  # the separator runs under fp16 autocast on CUDA
MIN_VOCAL_SHARE = 0.3  # provisional (spike: 3 of 18 maps tagged instrumental stay above it, 608 of 771 maps do)


def params():
    """What the stem adds to a stem encoder's definition."""
    return {"separator": "demucs htdemucs", "stem": "vocals", "shifts": 0,
            "sep_precision": "fp16 autocast on cuda" if SEP_HALF else "fp32",
            "input": "mono decode on both channels", "min_vocal_share": MIN_VOCAL_SHARE,
            "vocal_share": "mean over clips of 100 ms frames with vocal energy >= 10% of the mix's"}


def vocal_share(voc, mix, sr, frame=0.1, ratio=0.1):
    """Per clip: share of `frame`-second frames whose vocal energy is at least `ratio` of the mix's."""
    n = int(frame * sr)
    k = voc.shape[-1] // n
    ev = (voc[..., :k * n].reshape(len(voc), k, n) ** 2).mean(-1)
    em = (mix[..., :k * n].reshape(len(mix), k, n) ** 2).mean(-1) + 1e-10
    return ((ev / em) >= ratio).mean(-1)


def prep(y, sr, clip_s, n_clips):
    """The song's clips at SEP_SR, each exactly clip_s long (resampling may be a sample off)."""
    n = int(clip_s * SEP_SR)
    cl = [resample(c, sr, SEP_SR)[:n] for c in clips(y, sr, clip_s, n_clips)]
    return np.stack([np.pad(c, (0, n - len(c))) for c in cl]).astype(np.float32)


def load_htdemucs(folder):
    """The htdemucs bag of models from its local files (as demucs.hf.get_hf_model builds it, without the hub)."""
    import yaml
    from demucs.apply import BagOfModels
    from demucs.hf import load_safetensors_model
    bag = yaml.safe_load((folder / "htdemucs.yaml").read_text())
    missing = [s for s in bag["models"] if f"{s}.safetensors" not in HTDEMUCS.files]
    if missing:
        raise GenerationError(f"{HTDEMUCS.key}: htdemucs.yaml names models not declared: {', '.join(missing)}")
    models = [load_safetensors_model(folder / f"{s}.safetensors") for s in bag["models"]]
    return BagOfModels(models, bag.get("weights"), bag.get("segment"))


class Separator:
    """One Demucs model per device in the process, with the stems of the last few groups kept (per precision) so the
    vocal views of one pass separate each group once."""
    _models, _caches = {}, {}

    def __init__(self, folder, device, keep=4):
        import torch
        from demucs.apply import apply_model
        self.torch, self.device = torch, device
        key = str(device)
        if key not in Separator._models:
            Separator._models[key] = load_htdemucs(folder).to(device).eval()
        self.model, self._apply = Separator._models[key], apply_model
        self._stem = list(self.model.sources).index("vocals")
        self.segment = int(self.model.models[0].segment * SEP_SR) if hasattr(self.model, "models") else \
            int(self.model.segment * SEP_SR)
        self._cache, self._keep = Separator._caches.setdefault((key, SEP_HALF), OrderedDict()), keep

    def vocals(self, mix):
        """(clips, n) mono at SEP_SR -> (clips, n) mono vocal stem."""
        key = hashlib.blake2b(mix.tobytes(), digest_size=16).digest() + str(mix.shape).encode()
        if key in self._cache:
            return self._cache[key]
        torch = self.torch
        out = []
        cuda = str(self.device).startswith("cuda")
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=cuda and SEP_HALF):
            for s in range(0, len(mix), SEP_CLIPS):
                x = torch.from_numpy(mix[s:s + SEP_CLIPS]).to(self.device)[:, None].expand(-1, 2, -1)
                mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True) + 1e-8  # as demucs.separate normalises
                split = x.shape[-1] > self.segment  # clips of one segment go through in one forward
                y = self._apply(self.model, (x - mu) / sd, shifts=0, split=split, overlap=0.25, device=self.device)
                out.append((y[:, self._stem].float() * sd + mu).mean(1).cpu().numpy())
        voc = np.concatenate(out)
        self._cache[key] = voc
        while len(self._cache) > self._keep:
            self._cache.popitem(last=False)
        return voc


def embed_many(enc, prepped):
    """Separate a group of songs' clips once, then embed the voiced songs' stems with `enc` (enc.stem_vectors:
    (clips at SEP_SR) -> (clips, dim)); a song's vector is the mean of its clips', an unvoiced song gets None."""
    mix = np.concatenate(prepped)
    voc = enc.vocals(mix)
    bounds = np.cumsum([0] + [len(p) for p in prepped])
    voiced = [float(vocal_share(voc[a:b], mix[a:b], SEP_SR).mean()) >= MIN_VOCAL_SHARE
              for a, b in zip(bounds[:-1], bounds[1:])]
    keep = [j for c, a, b in zip(voiced, bounds[:-1], bounds[1:]) if c for j in range(a, b)]  # voiced clips
    if not keep:
        return [None] * len(prepped)
    vecs = dict(zip(keep, enc.stem_vectors(voc[keep])))
    return [np.mean([vecs[i] for i in range(a, b)], 0) if c else None
            for c, a, b in zip(voiced, bounds[:-1], bounds[1:])]
