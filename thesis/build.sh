#!/bin/sh
# Compile from any directory with a standard TeX distribution (e.g. MacTeX).
set -eu
# Prefer the matching tools provided by BasicTeX or MacTeX.
if [ -d /Library/TeX/texbin ]; then
    PATH="/Library/TeX/texbin:$PATH"
    export PATH
fi
cd "$(dirname "$0")"
for tool in latexmk xelatex biber; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing $tool. Install the required BasicTeX packages, then run this script again." >&2
        exit 1
    fi
done
# Keep working files cached and expose only the finished PDF.
mkdir -p build/.aux
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 -outdir=build/.aux main.tex
cp build/.aux/main.pdf build/main.pdf
