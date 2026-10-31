#!/usr/bin/env bash
set -euo pipefail
research_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$research_repo"
export PYTHONPATH="$research_repo/src:$research_repo/runs/cyp-autoresearch-deps${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$research_repo/runs/neural-env/bin/python" -m somerra_openadmet.cyp_research "$@"
