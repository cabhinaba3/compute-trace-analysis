#!/usr/bin/env bash
# Checks what this machine has for building cta and lists ways to install it,
# easiest first. It changes nothing unless you pass --run.
#
#   ./configure.sh                 report and options
#   ./configure.sh --run conda     create the conda environment (environment.yml)
#   ./configure.sh --run system    install the missing system packages with sudo
#   ./configure.sh --run system --yes   do not ask before running
set -u

RUN=""
ASSUME_YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN="${2:-}"; shift 2 ;;
    --yes) ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
done

OS="$(uname -s)"
if [ "$OS" != "Linux" ] && [ "$OS" != "Darwin" ]; then
  echo "cta builds its C++ extension on Linux and macOS; this is $OS. Nothing to check."
  exit 0
fi

# The repository: the script's directory, else the current one.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
ROOT="$HERE"; [ -f "$ROOT/pyproject.toml" ] || ROOT="$PWD"

have() { command -v "$1" >/dev/null 2>&1; }
status() { printf '  %-22s %-8s %s\n' "$1" "$2" "${3:-}"; }
MISSING=()      # things the build needs and this machine lacks
PACKAGES=()     # system packages that would provide them (apt names)
need() { MISSING+=("$1"); PACKAGES+=("$2"); }

echo "cta build check for $ROOT"
echo

# --- python -----------------------------------------------------------------
PY=""
for candidate in python3.13 python3.12 python3; do
  if have "$candidate" && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
    PY="$candidate"; break
  fi
done
echo "Python and compiler"
if [ -n "$PY" ]; then
  status "python 3.12 or newer" ok "$($PY --version 2>&1)"
  if "$PY" -c 'import sysconfig, os, sys; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' 2>/dev/null; then
    status "Python.h" ok
  else
    status "Python.h" missing "python development headers"; need "Python.h" "python3-dev"
  fi
  if "$PY" -c 'import venv, ensurepip' 2>/dev/null; then status "venv module" ok; else status "venv module" missing "needed for python -m venv"; need "venv" "python3-venv"; fi
else
  status "python 3.12 or newer" missing "$(python3 --version 2>&1 || echo 'no python3')"; need "python" "python3"
fi

CXX_BIN="${CXX:-}"; [ -n "$CXX_BIN" ] && have "${CXX_BIN%% *}" || CXX_BIN=""
[ -n "$CXX_BIN" ] || for c in g++ c++ clang++; do have "$c" && { CXX_BIN="$c"; break; }; done
if [ -n "$CXX_BIN" ]; then
  if printf 'int main(){return 0;}\n' | $CXX_BIN -std=c++20 -x c++ - -o /dev/null 2>/dev/null; then
    status "C++20 compiler" ok "$($CXX_BIN --version | head -1) (tested with g++ 13 and 15, and Apple clang)"
  else
    status "C++20 compiler" missing "$CXX_BIN cannot compile -std=c++20"; need "c++20" "g++"
  fi
else
  status "C++20 compiler" missing; need "c++20" "g++"
  [ "$OS" = Darwin ] && echo "  (macOS: install the compiler with: xcode-select --install)"
fi

# CMake and Ninja: pip downloads its own for a normal install, so they only matter without build isolation.
cmake_note="pip fetches its own unless you use --no-build-isolation"
if have cmake; then status "cmake" ok "$(cmake --version | head -1); $cmake_note"; else status "cmake" optional "$cmake_note"; fi
if have ninja; then status "ninja" ok "$(ninja --version)"; else status "ninja" optional "$cmake_note"; fi

# --- sources ------------------------------------------------------------------
echo; echo "Sources"
if [ -f "$ROOT/third_party/chakra/schema/protobuf/et_def.proto" ]; then
  status "git submodules" ok "third_party/chakra"
else
  status "git submodules" missing "run: git submodule update --init"; MISSING+=("submodules")
fi

# --- native libraries ---------------------------------------------------------
echo; echo "Native libraries"
PREFIXES="${CMAKE_PREFIX_PATH:-}"
# Homebrew keeps its libraries outside the default search paths.
if [ "$OS" = Darwin ]; then
  BREW_PREFIX="$( (have brew && brew --prefix) 2>/dev/null || true)"
  [ -n "$BREW_PREFIX" ] && PREFIXES="${PREFIXES:+$PREFIXES:}$BREW_PREFIX"
fi
[ -n "${CONDA_PREFIX:-}" ] && PREFIXES="${CONDA_PREFIX}${PREFIXES:+:$PREFIXES}"
PROBE_OUT=""
if have cmake; then
  PROBE_DIR="$(mktemp -d)"
  cat > "$PROBE_DIR/CMakeLists.txt" <<'CM'
cmake_minimum_required(VERSION 3.18)
project(probe LANGUAGES CXX)
find_package(ZLIB QUIET)
find_package(SQLite3 QUIET)
find_package(simdjson CONFIG QUIET)
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND AND NOT protobuf_FOUND)
  find_package(Protobuf QUIET)
endif()
message(STATUS "PROBE zlib=${ZLIB_FOUND} sqlite=${SQLite3_FOUND} simdjson=${simdjson_FOUND} protobuf=${Protobuf_FOUND} protoc=${Protobuf_PROTOC_EXECUTABLE}")
CM
  PROBE_OUT="$(cmake -S "$PROBE_DIR" -B "$PROBE_DIR/b" ${PREFIXES:+-DCMAKE_PREFIX_PATH="$PREFIXES"} 2>&1 | grep 'PROBE' | head -1)"
  rm -rf "$PROBE_DIR"
fi
probe() { printf '%s' "$PROBE_OUT" | sed -n "s/.*PROBE.* $1=\([^ ]*\).*/\1/p"; }
header_found() {  # fallback when cmake is absent: look for a header
  for dir in /usr/include /usr/local/include ${PREFIXES//:/ }; do
    [ -e "$dir/$1" ] || [ -e "$dir/include/$1" ] && return 0
  done; return 1
}
check_lib() {  # name, probe key, header, apt package, cmake-find hint
  local found
  if [ -n "$PROBE_OUT" ]; then
    found="$(probe "$2")"; case "$found" in 1|TRUE|True|ON) found=yes ;; *) found=no ;; esac
  else
    header_found "$3" && found=yes || found=no
  fi
  if [ "$found" = yes ]; then status "$1" ok; else status "$1" missing "header $3"; need "$1" "$4"; fi
}
if [ -z "$PROBE_OUT" ]; then
  if have cmake; then echo "  (the cmake probe failed: looking for headers only)"; else echo "  (no cmake: looking for headers only)"; fi
fi
check_lib "zlib" zlib zlib.h zlib1g-dev
check_lib "sqlite3" sqlite sqlite3.h libsqlite3-dev
check_lib "simdjson" simdjson simdjson.h libsimdjson-dev
check_lib "protobuf" protobuf google/protobuf/message.h libprotobuf-dev
if have protoc || [ -x "${CONDA_PREFIX:-/nonexistent}/bin/protoc" ]; then
  status "protoc" ok "$( (have protoc && protoc --version) || "${CONDA_PREFIX}/bin/protoc --version")"
else
  status "protoc" missing "protobuf compiler"; need "protoc" "protobuf-compiler"
fi

# --- environment --------------------------------------------------------------
echo; echo "Environment"
if "${PY:-python3}" - <<'PYEOF' 2>/dev/null
import urllib.request, sys
try:
    urllib.request.urlopen("https://pypi.org/simple/", timeout=6)
except Exception:
    sys.exit(1)
PYEOF
then status "network (pypi.org)" ok; else status "network (pypi.org)" missing "pip cannot download build tools or dependencies"; fi
CACHE="$ROOT/build/native/CMakeCache.txt"
if [ -f "$CACHE" ]; then
  gone=0
  for key in CMAKE_MAKE_PROGRAM CMAKE_COMMAND NB_DIR; do
    value="$(sed -n "s/^$key:[A-Z]*=//p" "$CACHE" | head -1)"
    [ -n "$value" ] && [ ! -e "$value" ] && gone=1
  done
  if [ "$gone" = 1 ]; then status "build/ cache" stale "an earlier build left paths that are gone; the next install resets it"; else status "build/ cache" ok; fi
fi
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

CAN_SUDO=0
if [ "$(id -u)" = 0 ]; then CAN_SUDO=1; elif have sudo && sudo -n true 2>/dev/null; then CAN_SUDO=1; fi
CONDA_BIN=""; for c in mamba micromamba conda; do have "$c" && { CONDA_BIN="$c"; break; }; done
PM=""; for c in apt-get dnf zypper brew; do have "$c" && { PM="$c"; break; }; done

status "root or sudo" "$([ $CAN_SUDO = 1 ] && echo ok || echo none)" "$([ $CAN_SUDO = 1 ] || echo 'needed only for system packages')"
status "conda / mamba" "$([ -n "$CONDA_BIN" ] && echo ok || echo none)" "${CONDA_BIN}"

# --- result -------------------------------------------------------------------
echo
if [ ${#MISSING[@]} -eq 0 ]; then
  echo "Everything the build needs is here. Install with:"
  echo
  echo "    python3 -m venv .venv && . .venv/bin/activate"
  echo "    CTA_BUILD_JOBS=$JOBS pip install -e \".[dev]\""
  echo
  echo "Smaller install: pip install -e .   (extras: sql, chakra, viz, pandas, all)"
  exit 0
fi

echo "Missing: ${MISSING[*]}"
echo
system_cmd=""
if [ "$PM" = apt-get ]; then
  system_cmd="sudo apt-get install -y $(printf '%s\n' "${PACKAGES[@]}" | sort -u | tr '\n' ' ')"
elif [ "$PM" = dnf ]; then
  system_cmd="sudo dnf install -y gcc-c++ zlib-devel sqlite-devel simdjson-devel protobuf-devel protobuf-compiler python3-devel"
elif [ "$PM" = zypper ]; then
  system_cmd="sudo zypper install -y gcc-c++ zlib-devel sqlite3-devel simdjson-devel protobuf-devel python3-devel"
elif [ "$PM" = brew ]; then
  # zlib and sqlite come with the macOS SDK; the compiler comes from `xcode-select --install`.
  system_cmd="brew install cmake simdjson protobuf"
fi

echo "Options, easiest first for this machine:"
echo
n=1
conda_option() {
  echo "  $n. conda environment (no root, changes nothing outside the environment)"
  if [ -n "$CONDA_BIN" ]; then
    echo "     $CONDA_BIN is installed. It provides the compiler, cmake, ninja and every library."
    echo "       $CONDA_BIN env create -f environment.yml    (or ./configure.sh --run conda)"
  else
    echo "     No conda here. micromamba is one small download and needs no root:"
    case "$OS-$(uname -m)" in
      Darwin-arm64) mamba_platform=osx-arm64 ;;
      Darwin-*) mamba_platform=osx-64 ;;
      *-aarch64) mamba_platform=linux-aarch64 ;;
      *) mamba_platform=linux-64 ;;
    esac
    echo "       curl -Ls https://micro.mamba.pm/api/micromamba/$mamba_platform/latest | tar -xvj bin/micromamba"
    echo "       ./bin/micromamba create -y -f environment.yml"
  fi
  echo "       conda activate cta     (micromamba: micromamba activate cta)"
  echo "       CTA_BUILD_JOBS=$JOBS pip install --no-build-isolation -e \".[dev]\""
  echo
  n=$((n + 1))
}
system_option() {
  if [ -z "$system_cmd" ]; then
    echo "  $n. system packages: this distribution is not covered; install a C++20 compiler and the development"
    echo "     packages for zlib, sqlite3, simdjson, protobuf (with protoc) and Python."
  else
    if [ "$PM" = brew ]; then
      echo "  $n. Homebrew packages (no root)"
    else
      echo "  $n. system packages (needs root$([ $CAN_SUDO = 1 ] || echo ", which you do not have"))"
    fi
    echo "       $system_cmd    (or ./configure.sh --run system)"
    echo "       CTA_BUILD_JOBS=$JOBS pip install -e \".[dev]\""
  fi
  echo
  n=$((n + 1))
}
if [ -n "$CONDA_BIN" ] || { [ $CAN_SUDO = 0 ] && [ "$PM" != brew ]; }; then conda_option; system_option; else system_option; conda_option; fi
echo "  $n. ask an administrator, or on a cluster load the modules (gcc, cmake, zlib, sqlite, simdjson, protobuf),"
echo "     then run ./configure.sh again. Libraries in a non-standard place: CTA_PREFIX_PATH=/path/a:/path/b"
echo
echo "Build settings: CTA_BUILD_JOBS, CTA_PREFIX_PATH, CTA_GENERATOR, CTA_BUILD_TYPE, CTA_BUILD_DIR, CTA_CMAKE_ARGS, CTA_FRESH"
echo "Optional features: pip install -e \".[sql]\"  (SQL plots), \".[chakra]\"  (write .et files), \".[viz]\", \".[all]\""

# --- --run --------------------------------------------------------------------
[ -n "$RUN" ] || exit 0
confirm() {
  [ "$ASSUME_YES" = 1 ] && return 0
  printf 'Run: %s\nContinue? [y/N] ' "$1"; read -r answer; [ "$answer" = y ] || [ "$answer" = Y ]
}
case "$RUN" in
  conda)
    [ -n "$CONDA_BIN" ] || { echo "no conda, mamba or micromamba found; see the options above"; exit 1; }
    cmd="$CONDA_BIN env create -f $ROOT/environment.yml"
    confirm "$cmd" && $cmd ;;
  system)
    [ -n "$system_cmd" ] || { echo "no known package manager"; exit 1; }
    [ "$PM" = brew ] || [ $CAN_SUDO = 1 ] || { echo "no root or passwordless sudo"; exit 1; }
    confirm "$system_cmd" && $system_cmd ;;
  *) echo "--run takes conda or system"; exit 2 ;;
esac
