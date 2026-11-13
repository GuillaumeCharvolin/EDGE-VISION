#!/usr/bin/env bash
# Download the sample video used to test the video pipeline into samples/.
#   vtest.avi: people walking, from the OpenCV repository (samples/data/vtest.avi).
#
# Usage: ./scripts/download_sample.sh

set -euo pipefail

cd "$(dirname "$0")/.."

url="https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
dest="samples/vtest.avi"

if [[ -f "$dest" ]]; then
    echo "$dest already exists"
    exit 0
fi

mkdir -p samples
curl -fL --retry 3 -o "$dest.part" "$url"
mv "$dest.part" "$dest"
echo "Saved $dest"
