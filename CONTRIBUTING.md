# Contributing to pyvae

## Branch model

| Branch | Purpose |
|---|---|
| `develop` | **Default & integration branch.** All work merges here first. |
| `main` | **Protected release branch.** Updated only by promoting `develop` via PR; tags on `main` trigger publishing. |

Because `develop` is the default branch, new pull requests target it automatically.
**Do not open PRs against `main`** (it requires a PR and rejects direct pushes).

## Making a change

```bash
git switch develop && git pull          # ff-only
git switch -c fix/<issue>-<short-desc>   # or feat/<short-desc>
# ...commit...
git push -u origin fix/<issue>-<short-desc>
gh pr create --base develop --head fix/<issue>-<short-desc>
```

Link the issue in the PR (e.g. "Closes #2"). Keep PRs focused and atomic.

## Releasing (maintainers)

Publishing runs **only** when a **GitHub Release** with a `v*` tag is published
and that tag's commit is on `main`. A bare `git push origin vX.Y.Z` does **not**
publish.

```bash
# 1. bump the version in pyproject.toml ([project] + [tool.pixi.*]) on develop, then:
gh pr create --base main --head develop --title "Release vX.Y.Z"

# 2. after the PR merges, cut the release FROM main (creates the tag + triggers release.yml):
gh release create vX.Y.Z --target main --title "vX.Y.Z" --generate-notes
```

`release.yml` then builds + publishes to PyPI and builds the conda package. The
release is refused if the tag isn't `v*` or its commit isn't on `main`.

## Local setup

This repo uses [pixi](https://pixi.sh):

```bash
pixi install
pixi run pytest tests/ -v
pixi run fixcode    # ruff lint + import sort
pixi run formatcode
```

Maintainers also keep disruptive/experimental work in a separate private R&D repo
and push feature branches here only when they're ready for review.
