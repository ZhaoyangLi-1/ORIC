#!/usr/bin/env bash
set -e

pip install -e ".[dev]"
echo "Installing flash-attn..."
pip install -v flash-attn --no-build-isolation