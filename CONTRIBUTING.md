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

```bash
gh pr create --base main --head develop --title "Release vX.Y.Z"
# after merge:
git tag vX.Y.Z   # bump version in pyproject.toml first
git push origin vX.Y.Z          # release.yml builds + publishes to PyPI and conda
```

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
