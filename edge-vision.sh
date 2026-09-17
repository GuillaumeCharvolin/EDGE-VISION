#!/usr/bin/env bash
# Usage:
#   ./edge-vision.sh                              # configure + build
#   ./edge-vision.sh -t                           # build, then run tests
#   ./edge-vision.sh -r                           # build, then run the program
#   ./edge-vision.sh -r --model m.onnx --image a.jpg
#   ./edge-vision.sh -r --model m.onnx --image a.jpg -t
#                                                 # run with those args, then tests
#   ./edge-vision.sh -d                           # build, then generate and open docs
#
# After -r / --run, every following argument that is not a script flag
# (-t/-r/-d or long forms) is forwarded to ./build/edge_vision.
# Combined flags like -rtd are not supported. Pass them separately.

set -euo pipefail

cd "$(dirname "$0")"

cmake -S . -B build
cmake --build build

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--test)
            shift
            ctest --test-dir build --output-on-failure
            ;;
        -d|--doc)
            shift
            cmake --build build --target docs
            firefox docs/api/html/index.html
            ;;
        -r|--run)
            shift
            run_args=()
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    -t|--test|-r|--run|-d|--doc)
                        break
                        ;;
                    *)
                        run_args+=("$1")
                        shift
                        ;;
                esac
            done
            ./build/edge_vision "${run_args[@]}"
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use separate flags: -t, -r, -d" >&2
            exit 1
            ;;
    esac
done
