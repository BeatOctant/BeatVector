"""The shapes audio views take, shared by the built-in views and plugins (octantvector.plugin).

ClipAudioEncoder  n_clips clips of clip_s seconds (base.clips), resampled to `sr`, turned into model input by
                  `features` in the CPU workers; the model gives one vector per clip (`clip_vectors`), L2-normalised
                  per clip when `l2_clips`, and the song's vector is their mean. `stem = "vocal"` embeds the clips'
                  vocal stem instead of the mix (stem.py).
HiddenMean        a ClipAudioEncoder whose clip vector is the time-mean of the outputs of the encoder layers `layers`
                  (1-based), averaged over those layers.

The first of `models` is the one model_dir() and the load hooks read. A subclass supplies load_processor, load_model
and (for ClipAudioEncoder) clip_vectors; heavy imports belong inside them.
"""
import numpy as np

from . import stem as stems
from .base import AudioEncoder, clips, resample


class ClipAudioEncoder(AudioEncoder):
    clip_s = 10
    n_clips = 3
    l2_clips = False
    stem = None  # or "vocal"

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.stem not in (None, "vocal"):
            raise ValueError(f"{cls.__name__}: stem must be None or 'vocal', got {cls.stem!r}")
        if cls.stem == "vocal" and "stem" in cls.__dict__:  # the class that turns the stem on takes its settings
            for k, v in stems.VOCAL.items():
                if k not in cls.__dict__:
                    setattr(cls, k, v)

    def own_models(self):
        return tuple(self.models) + ((stems.HTDEMUCS,) if self.stem else ())

    def own_credits(self):
        return tuple(self.credits) + (stems.CREDITS if self.stem else ())

    def params(self):
        p = {"clip_s": self.clip_s, "n_clips": self.n_clips, "l2_clips": self.l2_clips,
             "pool": "mean over clips" + (" of L2-normed clip vectors" if self.l2_clips else "")}
        if self.stem:
            p["stem"] = stems.params()
        return p

    # hooks
    def load_processor(self, folder):
        """-> the feature extractor read from the model's local folder."""
        raise NotImplementedError

    def load_model(self, folder, device):
        """-> the model read from the model's local folder, on `device`, in eval mode."""
        raise NotImplementedError

    def features(self, clips):
        """Clips at self.sr -> the model input (picklable: it crosses from the worker processes)."""
        return self.proc(clips, sampling_rate=self.sr, return_tensors="np")["input_values"].astype(np.float32)

    def clip_vectors(self, x):
        """Model input of n clips -> (n, dim) float32 numpy."""
        raise NotImplementedError

    # the heavy pass
    def setup_worker(self):
        if not self.stem:  # with a stem the clips only exist after separation, in the main process
            self.proc = self.load_processor(self.model_dir())

    def setup_model(self, device):
        self.device = device
        self.model = self.load_model(self.model_dir(), device)
        if self.stem:
            self.proc = self.load_processor(self.model_dir())
            self._sep = stems.Separator(self.model_dir(stems.HTDEMUCS), device)

    def prep(self, y, sr):
        if self.stem:
            return stems.prep(y, sr, self.clip_s, self.n_clips)
        return self.features([resample(c, sr, self.sr) for c in clips(y, sr, self.clip_s, self.n_clips)])

    def _rows(self, v):
        v = np.asarray(v, np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12) if self.l2_clips else v

    def embed(self, x):
        if self.stem:
            return self.embed_many([x])[0]
        return self._rows(self.clip_vectors(x)).mean(0)

    def embed_many(self, prepped):
        if self.stem:
            return stems.embed_many(self, prepped)
        return super().embed_many(prepped)

    def vocals(self, mix):
        return self._sep.vocals(mix)

    def stem_vectors(self, voc):
        """Vocal stems of clips at SEP_SR -> (clips, dim)."""
        return self._rows(self.clip_vectors(self.features([resample(v, stems.SEP_SR, self.sr) for v in voc])))


class HiddenMean(ClipAudioEncoder):
    layers = ()  # 1-based encoder layer numbers
    half = False  # run the model in fp16 on CUDA

    def params(self):
        return {**super().params(), "layers": list(self.layers),
                "pool": "time-mean per layer, mean over layers and clips"}

    def encoder_layers(self):
        """The model's encoder layers, in order (hidden state i is the output of layer i)."""
        return self.model.encoder.layers

    def setup_model(self, device):
        super().setup_model(device)
        if self.half and str(device).startswith("cuda"):
            self.model = self.model.half()
        # nothing reads the layers after the last one taken, so they are dropped: same output, less work
        del self.encoder_layers()[max(self.layers):]
        self._cap = {}
        for i in self.layers:
            self.encoder_layers()[i - 1].register_forward_hook(
                lambda mod, inp, out, i=i: self._cap.__setitem__(i, out[0] if isinstance(out, tuple) else out))

    def clip_vectors(self, x):
        """(clips, samples) feature-extracted input -> (clips, dim): time-mean per layer, mean over layers."""
        import torch
        dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            self._cap.clear()
            self.model(torch.from_numpy(x).to(self.device, dtype))
            return torch.stack([self._cap[i].float() for i in self.layers]).mean(2).mean(0).cpu().numpy()
