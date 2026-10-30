"""Declared models, the shared audio shapes and plugin loading (encoders/base.py, encoders/audio.py, encoders/stem.py,
octantvector.plugin), with tiny fake encoders: no model files, no torch unless a test says so."""
import types

import numpy as np
import pytest

from octantvector import generation as gen, plugin
from octantvector.generation import GenerationError, config
from octantvector.generation.encoders import REGISTRY, Encoder, get, load_plugins, register
from octantvector.generation.encoders import base, stem
from octantvector.plugin import ClipAudioEncoder, HiddenMean, Model

from test_api import PARAMS, harvest  # noqa: F401  (fixture)

A = Model("owner/a", "a" * 40, ("config.json", "weights.bin"))
B = Model("owner/b", "b" * 40, ("b.bin",))


@pytest.fixture(autouse=True)
def clean_registry():
    before, errors, plugins = dict(REGISTRY), dict(base.PLUGIN_ERRORS), set(base.PLUGINS)
    yield
    REGISTRY.clear(); REGISTRY.update(before)
    base.PLUGIN_ERRORS.clear(); base.PLUGIN_ERRORS.update(errors)
    base.PLUGINS.clear(); base.PLUGINS.update(plugins)


def _cls(name, parent=Encoder, **kw):
    return type(name, (parent,), {"id": name, "dim": 2, "inputs": ("meta:tags",), **kw})


class Fake(ClipAudioEncoder):
    """Clips go through a stand-in processor; a clip's vector is (its mean, its index in the call)."""
    id, dim, sr = "fake_clip", 2, 8000
    models = (A,)

    def load_processor(self, folder):
        self.read = folder
        return lambda cl, sampling_rate, return_tensors: {"input_values": np.stack(cl)}

    def clip_vectors(self, x):
        return np.stack([x.mean(1), np.arange(len(x))], 1).astype(np.float32)


class FakeVocal(Fake):
    id, stem = "fake_vocal", "vocal"


def test_a_model_is_checked_when_its_encoder_registers():
    for bad in (Model("owner/a", "a" * 39, ("x",)), Model("owner/a", "A" * 40, ("x",)), Model("owner/a", "a" * 40, ()),
                Model("a", "a" * 40, ("x",)), Model("owner/..", "a" * 40, ("x",)), Model("owner/a", "a" * 40, ("../x",)),
                Model("owner/a", "a" * 40, ["x"]), "owner/a"):
        with pytest.raises(ValueError):
            register(_cls("bad_model", models=(bad,)))
    with pytest.raises(ValueError):
        register(_cls("bad_models", models=[A]))
    assert "bad_model" not in REGISTRY and "bad_models" not in REGISTRY
    assert A.path == f"hf/owner/a/{'a' * 40}" and A.key == f"owner/a@{'a' * 40}"


def test_models_lists_what_views_read_with_their_sources_and_the_stem():
    register(Fake)
    register(FakeVocal)
    register(_cls("fake_derived", kind="derived", inputs=("view:fake_clip",), models=(B,)))
    got = gen.models(["fake_derived@1", "fake_vocal", "fake_clip"])
    assert [(m["repo"], m["revision"]) for m in got] == [
        ("adefossez/HTDemucs", stem.HTDEMUCS.revision), ("owner/a", "a" * 40), ("owner/b", "b" * 40)]
    assert got[1] == {"repo": "owner/a", "revision": "a" * 40, "files": ["config.json", "weights.bin"],
                      "path": f"hf/owner/a/{'a' * 40}"}
    assert gen.models(["chart_patterns"]) == []
    with pytest.raises(GenerationError, match="no such encoder"):
        gen.models(["no_such_view"])


def test_credits_are_checked_and_gathered_with_the_stem_and_the_sources():
    for bad in (["x"], ("",), (1,), "x"):
        with pytest.raises(ValueError):
            register(_cls("bad_credits", credits=bad))
    assert "bad_credits" not in REGISTRY
    register(type("CreditedClip", (Fake,), {"id": "credited_clip", "credits": ("Model A",)}))
    register(type("CreditedVocal", (Fake,), {"id": "credited_vocal", "stem": "vocal", "credits": ("Model A",)}))
    register(_cls("credited_derived", kind="derived", inputs=("view:credited_clip",), credits=("Words B", "Model A")))
    assert get("credited_clip").all_credits() == ["Model A"]
    assert get("credited_vocal").all_credits() == ["Model A", *stem.CREDITS]
    assert get("credited_derived").all_credits() == ["Words B", "Model A"]  # no repeats
    assert get("chart_patterns").all_credits() == []
    assert "credits" not in get("credited_clip").definition()  # not part of the definition


def test_the_models_are_part_of_the_definition():
    register(Fake)
    register(FakeVocal)
    assert get("fake_clip").definition()["params"]["models"] == [A.key]
    assert get("fake_vocal").definition()["params"]["models"] == [A.key, stem.HTDEMUCS.key]
    moved = type("Moved", (Fake,), {"models": (Model("owner/a", "c" * 40, A.files),)})
    assert moved().fingerprint() != Fake().fingerprint()
    assert "models" not in get("chart_patterns").definition()["params"]


def test_model_dir_reads_only_a_complete_folder_under_models_dir(tmp_path):
    with config.given({}):
        with pytest.raises(GenerationError, match="not set.*octantharvest models"):
            Fake().model_dir()
    with config.given({"models_dir": str(tmp_path)}):
        with pytest.raises(GenerationError, match=r"owner/a@a{40} is not in .*: run `octantharvest models`"):
            Fake().model_dir()
        d = tmp_path / A.path
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        with pytest.raises(GenerationError, match="is not in"):  # one declared file is still missing
            Fake().model_dir()
        (d / "weights.bin").write_bytes(b"")
        assert Fake().model_dir() == d
        with pytest.raises(GenerationError, match="HTDemucs"):
            FakeVocal().model_dir(stem.HTDEMUCS)


def test_compute_reads_models_from_the_models_dir_argument_only(harvest, tmp_path):  # noqa: F811
    _, crawl, store, _ = harvest

    class Reads(Encoder):
        id, dim, inputs, models = "fake_reads", 1, ("meta:tags",), (B,)

        def embed_map(self, m, v):
            return np.array([float((self.model_dir() / "b.bin").read_text())], np.float32)
    register(Reads)
    with pytest.raises(GenerationError, match="models_dir"):
        gen.compute(crawl, store, ["fake_reads"], {**PARAMS, "models_dir": str(tmp_path)})
    with pytest.raises(GenerationError, match="octantharvest models"):
        gen.compute(crawl, store, ["fake_reads"], PARAMS, models_dir=tmp_path / "models")
    (tmp_path / "models" / B.path).mkdir(parents=True)
    (tmp_path / "models" / B.path / "b.bin").write_text("7")
    assert gen.compute(crawl, store, ["fake_reads"], PARAMS, models_dir=tmp_path / "models")["views"]["fake_reads@1"] \
        == {"ok": 24}


class EP:
    def __init__(self, name, attrs=None, classes=(), fail=None):
        self.name, self.value = name, f"pkg_{name}"
        self.attrs, self.classes, self.fail = attrs or {}, classes, fail

    def load(self):
        for c in self.classes:  # what the module's @register lines do as it is imported
            register(c)
        if self.fail:
            raise self.fail
        return types.SimpleNamespace(**{"__name__": self.value, **self.attrs})


def test_a_plugin_is_loaded_only_for_the_same_plugin_api():
    base.PLUGINS.clear(); base.PLUGIN_ERRORS.clear()  # the plugins installed beside OctantVector are not this test's
    ok = {"OCTANTVECTOR_PLUGIN_API": plugin.PLUGIN_API}
    load_plugins([EP("good", ok, [_cls("plug_good")]),
                  EP("old", {"OCTANTVECTOR_PLUGIN_API": "0.0"}, [_cls("plug_old")]),
                  EP("bare", {}, [_cls("plug_bare")]),
                  EP("broken", ok, [_cls("plug_broken")], fail=ImportError("no module named torch")),
                  EP("twice", ok, [_cls("plug_good")])])
    assert get("plug_good").dim == 2
    assert not {"plug_old", "plug_bare", "plug_broken"} & set(REGISTRY)
    err = base.PLUGIN_ERRORS
    assert "'0.0'" in err["old"] and plugin.PLUGIN_API in err["old"]
    assert "declares no OCTANTVECTOR_PLUGIN_API" in err["bare"]
    assert "no module named torch" in err["broken"] and "duplicate encoder id plug_good" in err["twice"]
    with pytest.raises(GenerationError, match=r"available: .*plug_good.*plugins not loaded: .*old \(pkg_old is built "
                                              r"for plugin API '0.0'"):
        get("plug_old")
    with pytest.raises(GenerationError, match="plugins not loaded: .*bare"):
        gen.models(["plug_old"])
    with pytest.raises(GenerationError, match=r"plug_none: no such encoder .*no loaded plugin provides it \(loaded: "
                                              r"good\); plugins not loaded"):
        gen.models(["plug_none"])


def test_a_view_whose_plugin_is_not_installed_says_no_plugin_is_loaded():
    base.PLUGINS.clear(); base.PLUGIN_ERRORS.clear()
    with pytest.raises(GenerationError, match=r"uninstalled_view@1: no such encoder .*no loaded plugin provides it "
                                              r"\(loaded: none\)$"):
        gen.models(["uninstalled_view@1"])  # no installed plugin may provide it


def test_a_plugin_imported_before_loading_is_dropped_for_another_plugin_api():
    register(_cls("plug_early", __module__="pkg_early.views"))  # `import pkg_early` before registry(): not staged
    register(_cls("plug_early_ok", __module__="pkg_fine"))
    load_plugins([EP("early", {"__name__": "pkg_early", "OCTANTVECTOR_PLUGIN_API": "0.0"}),
                  EP("fine", {"__name__": "pkg_fine", "OCTANTVECTOR_PLUGIN_API": plugin.PLUGIN_API})])
    assert "plug_early" not in REGISTRY and "'0.0'" in base.PLUGIN_ERRORS["early"]
    assert get("plug_early_ok").dim == 2


def test_the_plugin_module_exports_the_shapes():
    assert plugin.PLUGIN_API == base.PLUGIN_API
    assert set(plugin.__all__) == {"PLUGIN_API", "Encoder", "AudioEncoder", "ImageEncoder", "DerivedEncoder",
                                   "ClipAudioEncoder", "HiddenMean", "Model", "register", "clips", "resample",
                                   "GenerationError"}
    assert all(hasattr(plugin, n) for n in plugin.__all__)


def test_a_derived_encoder_computes_only_the_rows_it_lacks(tmp_path):
    from octantvector.generation.store import Store
    src = register(_cls("drv_src"))
    calls = []

    class Twice(plugin.DerivedEncoder):
        id, dim, inputs = "drv_twice", 2, ("view:drv_src",)

        def values(self, X, device="cpu"):
            calls.append(len(X))
            return 2 * X
    register(Twice)
    s = Store(src(), tmp_path)
    s.add("a" * 40, np.array([1, 2], np.float32))
    s.flush()
    assert get("drv_twice").derive(tmp_path) == 1
    s.add("b" * 40, np.array([3, 4], np.float32))
    s.flush()
    assert get("drv_twice").derive(tmp_path) == 2 and calls == [1, 1]
    assert get("drv_twice").derive(tmp_path) == 2 and calls == [1, 1]  # nothing new: the hook is not called
    hs, X = Store(get("drv_twice"), tmp_path).load()
    assert dict(zip(hs, X.tolist())) == {"a" * 40: [2, 4], "b" * 40: [6, 8]}
    assert get("drv_twice").role_name() == "meta"


def test_a_clip_encoder_cuts_clips_and_averages_their_vectors(tmp_path):
    (tmp_path / A.path).mkdir(parents=True)
    for f in A.files:
        (tmp_path / A.path / f).write_text("")
    e = Fake()
    with config.given({"models_dir": str(tmp_path)}):
        e.setup_worker()
    assert e.read == tmp_path / A.path
    y = np.arange(8000 * 100, dtype=np.float32)
    x = e.prep(y, 8000)
    assert x.shape == (3, 80000) and x[0, 0] == int(0.15 * (len(y) - 80000))  # clips over 15-75 % of the song
    assert e.embed(x).tolist() == pytest.approx([x.mean(1).mean(), 1.0])
    e.l2_clips = True
    v = e.clip_vectors(x)
    assert e.embed(x).tolist() == pytest.approx((v / np.linalg.norm(v, axis=1, keepdims=True)).mean(0).tolist())
    p = Fake().params()
    assert p["clip_s"] == 10 and p["n_clips"] == 3 and p["l2_clips"] is False and "stem" not in p
    assert e.batch == 1 and e.role_name() == "music"


def test_a_stem_encoder_takes_the_stem_settings():
    e = FakeVocal()
    assert (e.role_name(), e.batch, e.clip_s, e.kind) == ("vocal", 8, 7.8, "audio")
    assert e.params()["stem"]["separator"] == "demucs htdemucs" and e.params()["clip_s"] == 7.8
    assert type("Own", (Fake,), {"stem": "vocal", "batch": 2})().batch == 2  # a class may still set its own
    with pytest.raises(ValueError):
        type("Odd", (Fake,), {"stem": "drums"})


def test_a_stem_encoder_embeds_only_the_voiced_songs_of_a_group():
    e = FakeVocal()
    e.sr = stem.SEP_SR
    e.proc = e.load_processor(None)
    loud = np.ones((3, stem.SEP_SR), np.float32)
    e.vocals = lambda m: m * (m[:, :1] > 1.5)  # the clips of the second song are all voice, the first none
    out = e.embed_many([loud, loud * 2])
    assert out[0] is None and out[1].tolist() == [2.0, 1.0]  # clip vectors (2, 0), (2, 1), (2, 2)
    e.vocals = lambda m: np.zeros_like(m)
    assert e.embed(loud) is None


def test_the_stem_clips_are_exactly_one_separator_segment():
    pytest.importorskip("soxr")
    e = FakeVocal()
    for sr in (22050, 44100, 48000):
        x = e.prep(np.zeros(sr * 40, np.float32), sr)
        assert x.shape == (3, int(7.8 * stem.SEP_SR))
    assert stem.vocal_share(x[:, :stem.SEP_SR] + 1, x[:, :stem.SEP_SR] + 2, stem.SEP_SR).tolist() == [1.0] * 3


def test_hidden_mean_takes_the_time_mean_of_the_named_layers():
    """Against the old MERT code path: the outputs of layers 2 and 3 (1-based) of a 5-layer stand-in, time-mean,
    mean over layers; the layers after the last one taken are dropped."""
    torch = pytest.importorskip("torch")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.proj = torch.nn.Linear(1, 6)
            self.encoder = torch.nn.Module()
            self.encoder.layers = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Tanh())
                                                      for _ in range(5))

        def forward(self, x):
            h = self.proj(x[..., None])
            for layer in self.encoder.layers:
                h = layer(h)
            return h

    class Tiny(HiddenMean):
        id, dim, sr, layers, models = "tiny_hidden", 6, 8000, (2, 3), (A,)

        def load_model(self, folder, device):
            return Net().to(device).eval()

    x = np.random.default_rng(0).normal(size=(3, 50)).astype(np.float32)
    ref = Net().eval()
    with torch.no_grad():
        h, outs = ref.proj(torch.from_numpy(x)[..., None]), []
        for layer in ref.encoder.layers:
            h = layer(h)
            outs.append(h)
    want = torch.stack([outs[1], outs[2]]).mean(2).mean(0).numpy()
    e = Tiny()
    e.model_dir = lambda model=None: None
    e.setup_model("cpu")
    assert len(e.model.encoder.layers) == 3
    np.testing.assert_allclose(e.clip_vectors(x), want, atol=1e-6)
    np.testing.assert_allclose(e.embed(x), want.mean(0), atol=1e-6)
    assert e.params()["layers"] == [2, 3]
