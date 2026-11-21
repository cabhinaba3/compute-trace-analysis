#!/usr/bin/env bash
# Runs the CI checks (.github/workflows/ci.yml) on this machine before you push.
#
#   ./ci-local.sh              lint, format, types and the full test suite (the `test` job)
#   ./ci-local.sh --fast       lint, format and types only (a few seconds)
#   ./ci-local.sh --core       also the `core-install` job: a clean venv, no extras
#   ./ci-local.sh --linux      the `test` job in an ubuntu:24.04 container (needs Docker running)
#   ./ci-local.sh --install-hook   run this before every `git push` (set CTA_SKIP_TESTS=1 to run --fast)
#
# It uses .venv/bin/python when there is one, else python3 on PATH, and needs the dev extras
# (pip install -e ".[dev]"). A step that fails stops the run, as in Actions.
set -u -o pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"

MODE=test
case "${1:-}" in
  "" ) ;;
  --fast) MODE=fast ;;
  --core) MODE=core ;;
  --linux) MODE=linux ;;
  --install-hook)
    git config core.hooksPath .githooks && echo "pre-push hook installed (.githooks/pre-push)"; exit $? ;;
  -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown option: $1 (see --help)"; exit 2 ;;
esac

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
# GitHub sets these, and rich then colours CLI output, which has broken tests that matched plain text.
export CI=true GITHUB_ACTIONS=true

step() {  # step NAME CMD...
  local name="$1"; shift
  printf '\n\033[1m== %s\033[0m\n$ %s\n' "$name" "$*"
  "$@" || { printf '\n\033[31mFAILED: %s\033[0m\n' "$name"; exit 1; }
}

# The commands below are the `run:` lines of the `test` job; tests/unit/test_ci_local.py checks
# that they still appear in ci.yml, so the two cannot drift apart unnoticed.
lint_and_types() {
  step "Lint (Ruff)"      "$PY" -m ruff check src tests examples
  step "Format (Ruff)"    "$PY" -m ruff format --check src tests examples
  step "Typecheck (Mypy)" "$PY" -m mypy src tests examples
}

case "$MODE" in
  fast)
    lint_and_types ;;
  test)
    lint_and_types
    step "Unit & Integration Tests (Pytest)" "$PY" -m pytest --cov=cta --cov-report=term-missing ;;
  core)
    lint_and_types
    step "Unit & Integration Tests (Pytest)" "$PY" -m pytest --cov=cta --cov-report=term-missing
    # The `core-install` job: the default install, with none of the optional packages.
    TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
    step "Check this machine with configure.sh" ./configure.sh
    step "Create a clean venv" "${PYTHON:-python3}" -m venv "$TMP/venv"
    step "Install without extras" env CTA_BUILD_DIR="$TMP/build" "$TMP/venv/bin/python" -m pip install -q -e . pytest
    step "Optional packages are absent" "$TMP/venv/bin/python" - <<'PY'
import importlib.util as u

for name in ("duckdb", "chakra", "google.protobuf", "plotly", "pandas"):
    try:
        found = u.find_spec(name) is not None
    except ModuleNotFoundError:  # the parent package is absent
        found = False
    assert not found, f"{name} is installed"
PY
    step "Core features and install hints" "$TMP/venv/bin/python" -m pytest tests/unit/test_install_extras.py ;;
  linux)
    command -v docker >/dev/null || { echo "docker is not installed"; exit 2; }
    docker info >/dev/null 2>&1 || { echo "the Docker daemon is not running"; exit 2; }
    # A copy of the tree without the macOS build products, like a fresh checkout.
    step "Linux container" docker run --rm -v "$PWD":/src:ro ubuntu:24.04 bash -c '
      set -euo pipefail
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq && apt-get install -y -qq g++ cmake zlib1g-dev libsqlite3-dev libprotobuf-dev \
        protobuf-compiler libsimdjson-dev python3 python3-venv python3-dev git rsync > /dev/null
      rsync -a --exclude .venv --exclude build --exclude "*.so" --exclude .git /src/ /work/
      cd /work
      python3 -m venv /venv && . /venv/bin/activate
      pip install -q -e ".[dev]"
      ruff check src tests examples
      ruff format --check src tests examples
      mypy src tests examples
      pytest --cov=cta --cov-report=term-missing
    ' ;;
esac
printf '\n\033[32mAll checks passed (%s).\033[0m\n' "$MODE"
