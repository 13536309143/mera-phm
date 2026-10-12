#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
exec python3 scripts/launch.py --mode local "$@"
