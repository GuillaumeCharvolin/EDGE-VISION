#!/bin/sh
# Usage:
#   ./edge-vision.sh           # configure + build
#   ./edge-vision.sh -t        # build, then run tests
#   ./edge-vision.sh -r        # build, then run the program
#   ./edge-vision.sh -d        # build, then generate and open docs
#   ./edge-vision.sh -t -r -d  # build, then each action in the order given
#
# Combined flags like -rtd are not supported. Pass them separately.

set -e

cd "$(dirname "$0")"

cmake -S . -B build
cmake --build build

for arg in "$@"; do
    case "$arg" in
        -t|--test)
            ctest --test-dir build --output-on-failure
            ;;
        -r|--run)
            ./build/edge_vision
            ;;
        -d|--doc)
            cmake --build build --target docs
            firefox docs/api/html/index.html
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Use separate flags: -t, -r, -d" >&2
            exit 1
            ;;
    esac
done
