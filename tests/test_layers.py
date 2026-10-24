"""The dependency direction inside the package (OctantSpecs constitution.md, principle 8; contracts/, contract 1 (OctantCompass/OctantHarvest to OctantVector)):

  common      imports nothing else from octantvector
  runtime     imports common only: no settings, working store, encoders, archive or user-data readers
  generation  imports common, never runtime: the distributed DB is the only thing passed between them
  plugin      the entry point for plugin packages (octantvector.plugin): imports generation only; nothing in the
              package imports it
  eval, cli   development tools; may import anything

The runtime must also load without the generation side's heavy dependencies (torch, transformers, librosa ...)."""
import ast
import subprocess
import sys
from pathlib import Path

import octantvector

SRC = Path(octantvector.__file__).parent
ALLOWED = {"common": {"common"}, "runtime": {"common", "runtime"}, "generation": {"common", "generation"}}


def _imports(path):
    """octantvector subpackages imported by a module: ('runtime', 'common', '' for the package root ...)."""
    rel = path.relative_to(SRC).with_suffix("").parts
    pkg = list(rel[:-1])
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[:len(pkg) - (node.level - 1)] + (node.module.split(".") if node.module else [])
            elif node.module and node.module.split(".")[0] == "octantvector":
                base = node.module.split(".")[1:]
            else:
                continue
            if base:
                yield base[0]
            else:  # from octantvector import x, y / from .. import x
                yield from (a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "octantvector":
                    yield (a.name.split(".") + [""])[1]


def test_layers():
    bad = []
    for layer, allowed in ALLOWED.items():
        for p in sorted((SRC / layer).rglob("*.py")):
            for target in _imports(p):
                if target not in allowed and target != "__version__":
                    bad.append(f"{p.relative_to(SRC)} imports octantvector.{target}")
    assert not bad, "\n".join(bad)


def test_plugin_is_only_an_entry_point():
    assert set(_imports(SRC / "plugin.py")) == {"generation"}
    users = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if p.name != "plugin.py" and "plugin" in _imports(p)]
    assert not users, users


def test_runtime_loads_alone():
    code = ("import sys, octantvector.runtime as r; r.open_db; "
            "print(sorted(m for m in sys.modules if m.startswith('octantvector.') or m.split('.')[0] in "
            "('torch', 'transformers', 'librosa', 'soundfile', 'soxr')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    loaded = eval(out)
    assert all(m.split(".")[1] in ("common", "runtime") for m in loaded if m.startswith("octantvector.")), loaded
    assert not [m for m in loaded if not m.startswith("octantvector.")], loaded


def test_api_versions():
    """Each side names its contract 1 API version as "major.minor", apart from the package version."""
    from octantvector import generation, runtime
    for v in (runtime.API_VERSION, generation.API_VERSION):
        major, minor = v.split(".")
        assert major.isdigit() and minor.isdigit()
