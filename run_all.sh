#!/usr/bin/env bash
# Run after src/prepare.py. Extra arguments are passed to train.py.
set -e
cd "$(dirname "$0")/src"
if [ -x ../.venv/Scripts/python.exe ]; then PY=../.venv/Scripts/python.exe; else PY=../.venv/bin/python; fi
$PY shortcut_check.py
for v in baseline camguide camguide_aug ours; do
  $PY train.py --variant "$v" "$@"
done
$PY evaluate.py
