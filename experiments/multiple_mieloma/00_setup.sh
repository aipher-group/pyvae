#!/usr/bin/env bash
#
# Check the inputs are in place, fetch the ones that can be fetched, and say
# exactly how to get the ones that cannot.
#
#   ./00_setup.sh
#
# The two MSigDB collections ARE scriptable: data.broadinstitute.org serves them
# without a login, and this script downloads them. The atlas is not -- Zenodo
# serves that record behind a browser check -- so for the atlas this prints
# instructions rather than running curl, because a setup script that silently
# produces a zero-byte file is worse than one that tells you what to fetch.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ok=0
need() {
    local path="$1" bytes="$2" what="$3"
    if [ -f "$path" ]; then
        local have
        have=$(stat -c%s "$path")
        local pretty
        pretty=$(numfmt --to=iec --suffix=B "$have" 2>/dev/null || echo "${have}B")
        if [ "$have" -lt "$bytes" ]; then
            echo "  [TRUNCATED] $path is $pretty, expected at least" \
                 "$(numfmt --to=iec --suffix=B "$bytes" 2>/dev/null || echo "${bytes}B")"
            ok=1
        else
            printf '  [ok]        %-46s %8s\n' "$path" "$pretty"
        fi
    else
        echo "  [MISSING]   $path"
        echo "              $what"
        ok=1
    fi
}

echo "Inputs:"
need data/Tcell.h5ad 1000000000 \
    "Foster & Yong single-cell atlas of myeloma disease evolution, Tcell.h5ad
              (~1.9 GB), from https://zenodo.org/records/13646014  (CC-BY 4.0).
              That record serves Tcell.h5ad directly; the newer version
              https://zenodo.org/records/17418275 serves Tcell.h5ad.zip instead.
              Cite Foster et al. 2026, Nat Commun 17:2400."
need data/metadata-donor.csv 10000 \
    "metadata-donor.csv, same Zenodo record"
# The two MSigDB collections do different jobs -- the Reactome hierarchy is the
# encoder mask, Hallmark is the read-out panel -- and both are pinned to v7.5.1
# so set membership cannot drift between them. resources/PROVENANCE.md explains
# which is which and why they differ.
MSIGDB=https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.5.1
fetch() {
    local path="$1" url="$2" what="$3"
    if [ ! -f "$path" ]; then
        echo "  [fetching]  $path"
        if ! curl -fsS -L -m 300 --create-dirs -o "$path.part" "$url"; then
            rm -f "$path.part"
            echo "  [MISSING]   $path"
            echo "              $what"
            echo "              automatic download from $url failed"
            ok=1
            return
        fi
        mv "$path.part" "$path"
    fi
    need "$path" "$4" "$what"
}
fetch resources/c2.cp.reactome.v7.5.1.symbols.gmt \
    "$MSIGDB/c2.cp.reactome.v7.5.1.symbols.gmt" \
    "MSigDB C2:CP:REACTOME v7.5.1 symbols GMT (~770 KB) -- the encoder mask" \
    500000
fetch resources/h.all.v7.5.1.symbols.gmt \
    "$MSIGDB/h.all.v7.5.1.symbols.gmt" \
    "MSigDB H hallmark v7.5.1 symbols GMT (~48 KB) -- the read-out panel" \
    40000

echo
echo "Environment:"
if command -v pixi >/dev/null 2>&1; then
    printf '  [ok]        pixi %s\n' "$(pixi --version | awk '{print $2}')"
else
    echo "  [MISSING]   pixi — https://pixi.sh, then: pixi install"
    ok=1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    n=$(nvidia-smi --list-gpus | wc -l)
    printf '  [ok]        %s GPU(s), driver CUDA %s\n' "$n" \
        "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    if [ "$n" -ne 3 ]; then
        echo "              config.yaml assumes n_gpus: 3 — edit it, or pass"
        echo "              --resources gpu=$n to snakemake"
    fi
else
    echo "  [note]      no nvidia-smi. The audit (step 1) runs on CPU in ~2 min;"
    echo "              the fits need a GPU."
fi

echo
if [ "$ok" -eq 0 ]; then
    echo "Ready. Next:"
    echo "    pixi install        # once"
    echo "    pixi run plan       # what would run"
    echo "    pixi run all        # the whole experiment, ~1.5 h on 3x A100"
else
    echo "Fetch the inputs above, then re-run ./00_setup.sh"
fi
exit "$ok"
