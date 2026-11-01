r"""Train the stand policy on the pH plant with rsl-rl PPO (Kaggle/Colab GPU; a CPU smoke run works too).

    python rl/stand/train.py --num-envs 4096 --iterations 3000 --run-name stand-ph-v1          # GPU
    python rl/stand/train.py --device cpu --num-envs 64 --iterations 20 --no-wandb             # smoke test

Long runs: --max-minutes stops cleanly before a session limit (final checkpoint + deploy.bin + artifact, the W&B
run finishes normally); --resume-wandb RUN continues from the newest checkpoint of an earlier W&B run (then
--iterations counts the additional iterations; the iteration numbers continue).  The harness is
rl/common/harness.py.

W&B (unless --no-wandb; needs WANDB_API_KEY): entity WANDB_ENTITY (default karl-jahnel-universit-t-leipzig),
project "Akos pH Stand", run = --run-name.  Logged: every iteration's losses/rewards/terminations/energy, a video
of the current policy every --video-every iterations (headless MuJoCo drawing, MUJOCO_GL=egl), every checkpoint
model_<it>.pt with its firmware file deploy_<it>.bin, and at the end the artifact "k1-stand-ph".
Output folder: logs/<run-name>/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco is imported (ph_k1 imports it)
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rl.common import harness as H  # noqa: E402
from rl.stand import contract as C  # noqa: E402
from rl.stand.env import REWARDS, StandConfig, StandEnv  # noqa: E402
from rl.stand.export import export_bin  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    H.add_arguments(ap, run_prefix="stand-ph", wandb_project="Akos pH Stand", iterations=3000, save_every=250,
                    video_every=500)
    ap.add_argument("--no-slew", action="store_true", help="train for an agent without the firmware slew limiter")
    ap.add_argument("--delay-ms", default="10,40", help="transport delay range, ms")
    ap.add_argument("--push-dv", type=float, default=0.3)
    args = ap.parse_args()

    lo, hi = (float(x) for x in args.delay_ms.split(","))
    cfg = StandConfig(num_envs=args.num_envs, device=args.device, dtype=args.dtype, seed=args.seed,
                      slew=not args.no_slew, delay_ms=(lo, hi), push_dv=args.push_dv, rewards=dict(REWARDS))
    env = StandEnv(cfg)
    print(f"[stand] {args.num_envs} robots on {args.device}, stand height {env.stand_height:.3f} m, "
          f"{env.substeps} plant steps per policy step, logs -> {Path(args.log_root) / args.run_name}", flush=True)
    video = H.VideoMaker(lambda: StandEnv(StandConfig(**{**cfg.__dict__, "num_envs": 1, "seed": cfg.seed + 1})),
                         args.video_seconds, cfg.policy_dt, Path(args.log_root) / args.run_name / "videos", "stand")
    H.train(env, args, tag="stand", hidden_dims=C.HIDDEN_DIMS, export=export_bin, video=video,
            artifact="k1-stand-ph", artifact_meta={"agent": "stand", "obs_dim": C.OBS_DIM})


if __name__ == "__main__":
    main()
