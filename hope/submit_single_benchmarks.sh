#!/usr/bin/env bash
# Submit one or both distributed single-agent benchmark jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

TARGETS="${TARGETS:-ai2thor procthor}"
for target in $TARGETS; do
    case "$target" in
        ai2thor|procthor)
            config="single_benchmark_${target}.hope"
            ;;
        *)
            echo "Unsupported target: $target" >&2
            exit 2
            ;;
    esac

    if [ ! -f "$config" ]; then
        echo "HOPE config not found: $SCRIPT_DIR/$config" >&2
        exit 1
    fi

    echo "Submitting SpatialWorld $target distributed single-agent benchmark with $config"
    hope run "$config" --m="spatialworld_${target}_single_benchmark"
done
