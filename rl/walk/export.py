r"""rsl-rl checkpoint -> the firmware sprint agent's policy file (EigenMLPHist .bin, 75 -> 22).

The byte format, the reader and the numpy forward pass are rl/common/export.py (RL_Core export_eigen_bin, read by
K1_Deploy firmware src/policy_mlp.cpp).  The sprint agent (src/agents/sprint.cpp) loads its policy with that reader
(default /home/booster/etc/policies/sprint.bin); this module fixes its sizes (rl/walk/contract.py).

    python rl/walk/export.py logs/<run>/model_100.pt logs/<run>/deploy_100.bin
    # on the robot:  k1_motion_test --task sprint --policy deploy_100.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # run as a script
    sys.path.insert(0, str(REPO))

from rl.common import export as E  # noqa: E402
from rl.common.export import actor_tensors, read_bin  # noqa: E402,F401  (re-exported for rl/walk users)
from rl.walk import contract as C  # noqa: E402

HIDDEN_DIMS = E.HIDDEN_DIMS


AGENTS = {C.OBS_DIM: "the sprint", C.OBS_DIM + 2: "the walk_clock"}     # 77: the 75 and the gait clock (W5c)


def check_shapes(tensors, obs_dim: int = C.OBS_DIM) -> None:
    """The firmware loads only the standard 512-256-128 stack on the agent's observation: the sprint's 75, or 77
    for the walk_clock agent."""
    E.check_shapes(tensors, obs_dim, C.NJ, HIDDEN_DIMS, agent=AGENTS[obs_dim])


def export_bin(checkpoint, out_path: str | Path, obs_dim: int = C.OBS_DIM) -> Path:
    return E.export_bin(checkpoint, out_path, obs_dim, C.NJ, HIDDEN_DIMS, agent=AGENTS[obs_dim])


def bin_policy(mats):
    """The firmware's forward pass (policy_mlp.cpp) in numpy: obs (n,75) -> actions (n,22)."""
    return E.bin_policy(mats, E.NORMALIZER_EPS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out", nargs="?", default=None, help="default: deploy_<iteration>.bin next to the checkpoint")
    ap.add_argument("--gait-clock", action="store_true", help="a walk_clock policy (W5c): 77 inputs")
    a = ap.parse_args()
    ck = Path(a.checkpoint)
    out = Path(a.out) if a.out else ck.with_name(ck.stem.replace("model", "deploy") + ".bin")
    print(f"wrote {export_bin(ck, out, C.OBS_DIM + (2 if a.gait_clock else 0))}")


if __name__ == "__main__":
    main()
