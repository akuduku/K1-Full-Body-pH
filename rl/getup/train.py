r"""Train the get-up policy (get up from lying, stay standing) on the pH plant with rsl-rl PPO, by wall-clock time.

    python rl/getup/train.py --hours 2.7 --run-name getup-ph-v1                 # GPU (Colab / Kaggle T4)
    python rl/getup/train.py --device cpu --num-envs 64 --hours 0.02 --no-wandb  # smoke test
    python rl/getup/train.py --no-tank ...                                       # ablation: no energy tank
    python rl/getup/train.py --reward battery=0 ...                              # ablation: no energy reward

Designed for GPU sessions that end: training runs until --hours of training are used (or SIGINT, e.g. from
colab-train's session limit), then stops cleanly (checkpoint + deploy.bin uploaded, W&B run finished).  A
checkpoint every --save-every-min minutes (a disconnect loses at most that), a video every --video-every-min.
--resume model_<it>.pt / --resume-wandb RUN continue the run AND its curriculum (saved in every checkpoint).

W&B: project "Akos pH GetUp" (entity WANDB_ENTITY), run = --run-name.  Logged: losses, reward terms, success per
start kind, time to stand, battery energy per get-up and its efficiency (potential energy gained / battery
energy), standing power, tank level and use, curriculum, videos (get-ups from the back and the front, alternating,
without the assist), every checkpoint with its firmware file deploy_<it>.bin, at the end the artifact
"k1-getup-ph".  The harness is rl/common/harness.py, the task rl/getup/env.py.
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

import torch  # noqa: E402

from rl.common import harness as H  # noqa: E402
from rl.getup import contract as C  # noqa: E402
from rl.getup.env import REWARDS, GetUpConfig, GetUpEnv  # noqa: E402
from rl.getup.export import export_bin  # noqa: E402


def parse_rewards(items) -> dict:
    w = dict(REWARDS)
    for item in items or []:
        k, _, v = item.partition("=")
        if k not in REWARDS or not v:
            raise SystemExit(f"--reward {item!r}: use NAME=WEIGHT with NAME in {sorted(REWARDS)}")
        w[k] = float(v)
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    H.add_arguments(ap, run_prefix="getup-ph", wandb_project="Akos pH GetUp", iterations=100000, save_every=0,
                    video_every=0, save_every_min=10.0, video_every_min=30.0, init_std=0.8)
    ap.add_argument("--no-tank", action="store_true", help="ablation: variable gains without the energy tank")
    ap.add_argument("--tank-capacity", type=float, default=150.0, help="J")
    ap.add_argument("--tank-refill", type=float, default=1.0)
    ap.add_argument("--assist-n", type=float, default=100.0, help="upward assist on lying starts, fading (0: off)")
    ap.add_argument("--lying-kinds", default="supine,prone", help="comma list of supine, prone, side")
    ap.add_argument("--delay-ms", default="10,40", help="transport delay range, ms")
    ap.add_argument("--push-dv", type=float, default=0.3)
    ap.add_argument("--library-rounds", type=int, default=1)
    ap.add_argument("--reward", action="append", help="NAME=WEIGHT, overrides a reward weight (repeatable)")
    ap.set_defaults(video_seconds=10.0)
    args = ap.parse_args()

    lo, hi = (float(x) for x in args.delay_ms.split(","))
    cfg = GetUpConfig(num_envs=args.num_envs, device=args.device, dtype=args.dtype, seed=args.seed,
                      steps_per_env=args.steps_per_env, delay_ms=(lo, hi), tank=not args.no_tank,
                      tank_capacity=args.tank_capacity, tank_refill=args.tank_refill, assist_force=args.assist_n,
                      lying_kinds=tuple(k.strip() for k in args.lying_kinds.split(",") if k.strip()),
                      push_dv=args.push_dv, library_rounds=args.library_rounds, rewards=parse_rewards(args.reward))
    env = GetUpEnv(cfg)
    print(f"[getup] {args.num_envs} robots on {args.device}, stand trunk z {env.stand_z:.3f} m, CoM {env.com_low:.3f} "
          f"-> {env.com_stand:.3f} m, tank {'off' if args.no_tank else f'{args.tank_capacity:g} J'}, assist "
          f"{args.assist_n:g} N, budget {max(args.hours, args.max_minutes / 60):.2f} h, "
          f"logs -> {Path(args.log_root) / args.run_name}", flush=True)

    def video_env():
        return GetUpEnv(GetUpConfig(**{**cfg.__dict__, "num_envs": 1, "seed": cfg.seed + 1, "library_size": 32,
                                       "assist_force": 0.0, "push_dv": 0.0, "obs_noise": {}}))

    kinds = cfg.lying_kinds

    def prepare(venv, it):  # alternate the lying starts (back, front, ...); the assist stays off
        venv.cfg.start = kinds[prepare.n % len(kinds)]
        prepare.n += 1
        venv._reset_idx(torch.arange(1, device=venv.device))
    prepare.n = 0

    video = H.VideoMaker(video_env, args.video_seconds, cfg.policy_dt, Path(args.log_root) / args.run_name / "videos",
                         "getup", prepare)
    H.train(env, args, tag="getup", hidden_dims=C.HIDDEN_DIMS, export=export_bin, video=video,
            artifact="k1-getup-ph", artifact_meta={"agent": "standup_ph", "obs_dim": C.OBS_DIM,
                                                   "num_actions": C.NUM_ACTIONS})


if __name__ == "__main__":
    main()
