# conda-forge recipe for pyvae

`recipe.yaml` here is the recipe for distributing **pyvae** through
[conda-forge](https://conda-forge.org/). It uses the modern **v1 recipe format**
(`recipe.yaml`, [CEP-13](https://conda-forge.org/docs/maintainer/adding_pkgs/));
the older v0 `meta.yaml` is still accepted but v1 is preferred for new recipes.

conda-forge is *not* a `pixi publish` target — packages are hosted by a
per-package *feedstock* repository whose CI builds and uploads to the
`conda-forge` channel. Getting there is a one-time submission, after which
updates are largely automated.

## One-time submission to conda-forge

1. **Publish to PyPI first.** conda-forge builds from the PyPI sdist. Release to
   PyPI (see the repo README / `release.yml`) so `pyvae-<version>.tar.gz` exists.

2. **Fill in the sdist hash.** Get the sha256 of the published sdist and put it
   in `source.sha256`, and confirm the handle under `extra.recipe-maintainers`:

   ```bash
   # from the built artifact
   pixi run -e publish build-dist
   shasum -a 256 dist/pyvae-*.tar.gz

   # …or straight from PyPI
   curl -sL https://pypi.org/pypi/pyvae/json | python -c \
     'import sys,json; r=json.load(sys.stdin)["urls"]; \
      print(next(u["digests"]["sha256"] for u in r if u["packagetype"]=="sdist"))'
   ```

3. **Submit to staged-recipes.** Fork
   [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes),
   copy this file to `recipes/pyvae/recipe.yaml`, and open a PR. CI lints and
   test-builds it (the `noarch` build plus the `python` imports + `pip check`
   test); a conda-forge member reviews and merges.

4. **Feedstock is created automatically.** Merging spins up
   `conda-forge/pyvae-feedstock`, which builds the `noarch` package and uploads
   it to the `conda-forge` channel. Users then install with:

   ```bash
   conda install -c conda-forge pyvae   # or: pixi add pyvae
   ```

## Ongoing updates

Once the feedstock exists you normally **don't touch this file again**. When a
new version lands on PyPI, the `regro-cf-autotick-bot` opens a version-bump PR
on the feedstock automatically. For dependency/recipe changes, edit
`recipe/recipe.yaml` *in the feedstock* and open a PR there. Keep this copy in
sync as the canonical source if you prefer to manage the recipe from the main
repo.

## Validating the build locally

The same `noarch` package conda-forge will build is produced by pixi:

```bash
pixi publish --target-dir ./conda-dist   # build + copy artifact, no upload
```

To lint/render the recipe itself the way staged-recipes CI does (optional, needs
the tools installed):

```bash
rattler-build build --recipe conda-recipe/recipe.yaml --render-only
```
