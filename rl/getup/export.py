r"""rsl-rl checkpoint -> the standup_ph agent's policy file (EigenMLPHist .bin, 117 inputs -> 66 outputs).

The byte format is the firmware's (rl/common/export.py); the agent that reads it is specified in
rl/getup/README.md.

    python rl/getup/export.py logs/<run>/model_1200.pt          # -> logs/<run>/deploy_1200.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # run as a script
    sys.path.insert(0, str(REPO))

from rl.common import export as E  # noqa: E402
from rl.getup import contract as C  # noqa: E402


def export_bin(checkpoint, out_path: str | Path) -> Path:
    return E.export_bin(checkpoint, out_path, C.OBS_DIM, C.NUM_ACTIONS, C.HIDDEN_DIMS, agent="the standup_ph")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out", nargs="?", default=None, help="default: deploy_<iteration>.bin next to the checkpoint")
    a = ap.parse_args()
    ck = Path(a.checkpoint)
    out = Path(a.out) if a.out else ck.with_name(ck.stem.replace("model", "deploy") + ".bin")
    print(f"wrote {export_bin(ck, out)}")


if __name__ == "__main__":
    main()
