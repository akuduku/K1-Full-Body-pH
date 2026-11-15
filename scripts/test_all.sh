#!/bin/bash
# All tests of this repository on the CPU (about a minute), then the physics checks.
#   scripts/test_all.sh            PYTHON=/path/to/python scripts/test_all.sh
# Needs the built extension (cd ph_k1_cuda && python setup.py build_ext --inplace), torch, mujoco, rsl-rl-lib 5.5.2,
# tensordict.  The local environment of the thesis: ~/Dokumente/Robotics/venv.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$HOME/Dokumente/Robotics/venv/bin/python}
[ -x "$PY" ] || PY=python3
"$PY" -m pytest ph_k1_cuda/tests rl -q -p no:cacheprovider "$@"
"$PY" validation/checks.py --impl native | tail -1
