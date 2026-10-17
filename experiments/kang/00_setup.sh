#!/usr/bin/env bash
#
# Fetch everything this experiment needs, and say exactly what is missing if a
# fetch fails.
#
#   ./00_setup.sh
#
# Both inputs ARE scriptable here, which was not always clear. The README used
# to say the Kang counts needed a browser; that described the old
# WAF-protected figshare page. `pyvae.load_kang` downloads from the figshare
# **API** endpoint, which serves the file to curl without a login, so this
# script fetches it directly and a GPU job never waits on 38 MB.
#
# MSigDB C2:CP:REACTOME is the encoder mask and defines the gene panel. It is
# pinned to v7.5.1, the same release `experiments/multiple_mieloma` uses, so set
# membership cannot drift between the two experiments.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ok=0
need() {
    local path="$1" bytes="$2" what="$3"
    if [ -f "$path" ]; then
        local have pretty
        have=$(stat -c%s "$path")
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

fetch() {
    local path="$1" url="$2" what="$3" bytes="$4"
    if [ ! -f "$path" ]; then
        echo "  [fetching]  $path"
        # Written to .part and moved only on success, so an interrupted
        # download never leaves a short file sitting there looking authoritative.
        if ! curl -fsS -L -m 600 --create-dirs -o "$path.part" "$url"; then
            rm -f "$path.part"
            echo "  [MISSING]   $path"
            echo "              $what"
            echo "              automatic download from $url failed"
            ok=1
            return
        fi
        mv "$path.part" "$path"
    fi
    need "$path" "$bytes" "$what"
}

echo "Inputs:"
fetch data/kang_counts_25k.h5ad \
    "https://api.figshare.com/v2/file/download/34464122" \
    "Kang et al. 2018 PBMC IFN-beta dataset, ~25k cells (~38 MB).
              The same URL pyvae.load_kang uses, so fetching it here only
              moves the download off the critical path." \
    20000000

MSIGDB=https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.5.1
fetch resources/c2.cp.reactome.v7.5.1.symbols.gmt \
    "$MSIGDB/c2.cp.reactome.v7.5.1.symbols.gmt" \
    "MSigDB C2:CP:REACTOME v7.5.1 symbols GMT (~770 KB) -- the encoder mask" \
    500000

echo
if [ "$ok" -eq 0 ]; then
    echo "All inputs present. Next:"
    echo "  pixi run all          # or: snakemake -j 42 --resources gpu=2"
else
    echo "Some inputs are missing -- see above. Nothing else will run until"
    echo "they are in place."
fi
exit "$ok"
