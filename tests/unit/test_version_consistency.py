"""The package version is declared three times in pyproject.toml; they agree.

`[project]` is what pip and PyPI see, `[tool.pixi.workspace]` and
`[tool.pixi.package]` are what `pixi publish` stamps on the conda package. A
release that bumps one and not the others ships two packages with different
versions from one commit. `pixi run bump-version` sets all three.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_the_three_version_declarations_agree():
    meta = tomllib.loads(PYPROJECT.read_text())
    versions = {
        "[project]": meta["project"]["version"],
        "[tool.pixi.workspace]": meta["tool"]["pixi"]["workspace"]["version"],
        "[tool.pixi.package]": meta["tool"]["pixi"]["package"]["version"],
    }
    assert len(set(versions.values())) == 1, versions
