r"""rsl-rl checkpoint -> the kick_walk agent's policy file (EigenMLPHist .bin, 165 or 166 inputs -> 22 targets).

    python rl/kick/export.py logs/<run>/model_1200.pt          # -> logs/<run>/deploy_1200.bin

165 inputs: the firmware's kick_walk agent reads the file as it is.  166 inputs (the request clock, contract.py):
the agent has to append the seconds since the request first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # run as a script
    sys.path.insert(0, str(REPO))

from rl.common import export as E  # noqa: E402
from rl.kick import contract as C  # noqa: E402


def export_bin(checkpoint, out_path: str | Path, obs_dim: int | None = None) -> Path:
    tensors, _, _, _ = E.actor_tensors(checkpoint)
    n = int(tensors["actor.0.weight"].shape[1]) if obs_dim is None else obs_dim
    if n not in (C.OBS_DIM, C.OBS_DIM + 1):
        raise ValueError(f"{n} inputs: the kick_walk agent has {C.OBS_DIM} (or {C.OBS_DIM + 1} with the request clock)")
    return E.export_bin(checkpoint, out_path, n, C.NUM_ACTIONS, C.HIDDEN_DIMS, agent="the kick_walk")


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
