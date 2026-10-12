#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
if [[ ! -f artifacts/main-model/model-bank.joblib ]]; then
  if [[ ! -f artifacts/demo-model/model-bank.joblib ]]; then
    .venv/bin/python main.py demo
  fi
  exec .venv/bin/python main.py serve --data data/demo --model artifacts/demo-model/model-bank.joblib --open "$@"
fi
exec .venv/bin/python main.py serve --open "$@"
