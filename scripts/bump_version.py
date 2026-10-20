#!/usr/bin/env python
"""Set the package version everywhere `pyproject.toml` declares it.

    pixi run bump-version patch        # 1.0.0 -> 1.0.1
    pixi run bump-version minor        # 1.0.0 -> 1.1.0
    pixi run bump-version major        # 1.0.0 -> 2.0.0
    pixi run bump-version 1.2.0        # an explicit version

The version is declared three times: `[project]` (read by pip, PyPI and the
wheel), `[tool.pixi.workspace]` and `[tool.pixi.package]` (read by
`pixi publish` for the conda package). They must agree, and
`tests/unit/test_version_consistency.py` fails when they do not.

`conda-recipe/recipe.yaml` is deliberately not touched. It is the original
conda-forge submission recipe, kept for reference; the feedstock's bot opens
its own version bump once a release reaches PyPI, and that recipe's `sha256`
can only be known after the sdist is uploaded.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
TABLES = ("project", "tool.pixi.workspace", "tool.pixi.package")
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def current(text: str) -> dict[str, str]:
    """The `version` value in each of the three tables, by table name."""
    found, table = {}, None
    for line in text.splitlines():
        head = re.match(r"^\[([^\]]+)\]\s*$", line)
        if head:
            table = head.group(1)
            continue
        m = re.match(r'^version\s*=\s*"([^"]+)"', line)
        if m and table in TABLES and table not in found:
            found[table] = m.group(1)
    missing = [t for t in TABLES if t not in found]
    if missing:
        sys.exit(f"no version found in [{'], ['.join(missing)}] of {PYPROJECT}")
    return found


def target(old: str, arg: str) -> str:
    if SEMVER.match(arg):
        return arg
    m = SEMVER.match(old)
    if not m:
        sys.exit(f"current version {old!r} is not X.Y.Z; pass an explicit version")
    major, minor, patch = map(int, m.groups())
    bumps = {"major": f"{major + 1}.0.0", "minor": f"{major}.{minor + 1}.0",
             "patch": f"{major}.{minor}.{patch + 1}"}
    if arg not in bumps:
        sys.exit(f"expected major, minor, patch or X.Y.Z; got {arg!r}")
    return bumps[arg]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    text = PYPROJECT.read_text()
    found = current(text)
    if len(set(found.values())) != 1:
        print("the three declarations disagree; setting all of them:",
              ", ".join(f"[{t}] {v}" for t, v in found.items()))
    old = found["project"]
    new = target(old, sys.argv[1])

    out, table = [], None
    for line in text.splitlines(keepends=True):
        head = re.match(r"^\[([^\]]+)\]\s*$", line)
        if head:
            table = head.group(1)
        elif table in TABLES and re.match(r'^version\s*=\s*"', line):
            line = re.sub(r'"[^"]+"', f'"{new}"', line, count=1)
            table = None  # only the first `version` in each table
        out.append(line)
    PYPROJECT.write_text("".join(out))

    after = current(PYPROJECT.read_text())
    assert set(after.values()) == {new}, after
    print(f"{old} -> {new} in [{'], ['.join(TABLES)}]")
    print(f"next: commit, merge to main, then publish a GitHub Release tagged v{new}")


if __name__ == "__main__":
    main()
