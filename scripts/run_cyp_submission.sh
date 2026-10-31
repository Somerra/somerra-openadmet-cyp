#!/usr/bin/env bash
set -euo pipefail
build_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$build_repo"
export PYTHONPATH="$build_repo/src:$build_repo/runs/cyp-autoresearch-deps${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$build_repo/runs/neural-env/bin/python" -m somerra_openadmet.cyp_submission "$@"
