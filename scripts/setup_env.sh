#!/usr/bin/env bash
# Create the project virtual environment(s) on Linux, including the Jetson Orin.
#
#   bash scripts/setup_env.sh
#
#   x86-64 Linux   .venv         Python 3.13.9 + CUDA PyTorch + requirements.txt
#   Jetson Orin    .venv         Python 3.13.9 + requirements.txt (no PyTorch)
#                  .venv-torch   Python 3.10   + NVIDIA JetPack 6 PyTorch + requirements.txt
#
# NVIDIA ships JetPack 6 PyTorch wheels for Python 3.10 only, so on the Jetson
# the CNN notebooks (10, 11, 13, 16) run in .venv-torch and everything else in
# .venv - the same split as the original Jetson run.  requirements.txt carries
# the Python 3.10 pins as environment markers.
#
# Overrides: PYTHON_VERSION, TORCH_VERSION, CUDA (x86-64), JETSON_TORCH_INDEX
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_VERSION="${PYTHON_VERSION:-3.13.9}"     # same interpreter as the Windows .venv
CUDA="${CUDA:-cu126}"
JETSON_TORCH_INDEX="${JETSON_TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

venv() {   # venv <dir> <python-version>: create once, reuse afterwards
    [ -x "$1/bin/python" ] && return
    rm -rf "$1"                                   # remains of an interrupted run
    # uv can report an error creating its minor-version alias although the
    # interpreter installed fine, so look the executable up instead.
    uv python install "$2" || true
    uv venv "$1" --python "$(uv python find "$2" --system --python-preference only-managed)"
}

if [ -f /etc/nv_tegra_release ]; then
    venv .venv "$PYTHON_VERSION"
    uv pip install --python .venv/bin/python -r requirements.txt

    venv .venv-torch 3.10
    uv pip install --python .venv-torch/bin/python \
        "torch${TORCH_VERSION:+==$TORCH_VERSION}" --index-url "$JETSON_TORCH_INDEX"
    uv pip install --python .venv-torch/bin/python -r requirements.txt

    .venv/bin/python scripts/check_env.py
    .venv-torch/bin/python scripts/check_env.py --require-gpu
else
    venv .venv "$PYTHON_VERSION"
    uv pip install --python .venv/bin/python \
        "torch==${TORCH_VERSION:-2.14.1}" --index-url "https://download.pytorch.org/whl/$CUDA"
    uv pip install --python .venv/bin/python -r requirements.txt
    .venv/bin/python scripts/check_env.py --require-gpu
fi
