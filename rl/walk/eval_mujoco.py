r"""A walk policy in MuJoCo, on the laptop: the sim-to-sim check of a policy trained on the pH plant.

The policy file is what the robot would load, run through the firmware's loop (rl/walk/backends.py: Walker with the
twist ramp, the safety filter, the delay line and the motor driver).  Only the physics engine differs.

    python rl/walk/eval_mujoco.py --policy walk_v9
    python rl/walk/eval_mujoco.py --policy logs/x/deploy_100.bin --command 1.0 0 0 --seconds 15 --push 0.3
    python rl/walk/eval_mujoco.py --policy walk_v9 --physics ph        # the same loop on the pH plant, one robot

Per command: whether it fell, the speed it reached after the first 3 s (the firmware ramps the twist) and the error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rl.walk import contract as C  # noqa: E402
from rl.walk.backends import MuJoCoBackend, PHBackend, Walker, load_walk_policy  # noqa: E402
from rl.walk.evaluate import COMMANDS  # noqa: E402


def run(walker: Walker, command, seconds: float, settle: float = 3.0, push: float = 0.0) -> dict:
    """One robot follows `command` (vx, vy, wz) from standing.  -> fell, the mean velocity in the heading frame after
    `settle` seconds and its error against the (clipped) command."""
    walker.reset()
    steps, first, every = round(seconds / C.POLICY_DT), round(settle / C.POLICY_DT), round(4.0 / C.POLICY_DT)
    want = np.clip(np.asarray(command, dtype=np.float64), C.TWIST_LOW, C.TWIST_HIGH)
    last, vel, yaw_rate = walker.reading, [], []
    for i in range(steps):
        shove = None
        if push and i > 0 and i % every == 0:
            shove = np.array((0.0, push * (1 if (i // every) % 2 else -1), 0.0))
        r = walker.tick(command, shove)
        if walker.fell:
            return {"fell": True, "at_s": i * C.POLICY_DT, "speed": np.full(2, np.nan), "error": float("nan"),
                    "yaw_rate": float("nan")}
        if i >= first:
            yaw = C.yaw_of(last.quat)
            d = (r.pos[:2] - last.pos[:2]) / C.POLICY_DT
            vel.append((np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]))
            yaw_rate.append(np.arctan2(np.sin(C.yaw_of(r.quat) - yaw), np.cos(C.yaw_of(r.quat) - yaw)) / C.POLICY_DT)
        last = r
    speed = np.mean(vel, axis=0)
    return {"fell": False, "at_s": seconds, "speed": speed, "error": float(np.linalg.norm(speed - want[:2])),
            "yaw_rate": float(np.mean(yaw_rate))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="a file (.bin, checkpoint, TorchScript) or a name in k1_sprint/checkpoints")
    ap.add_argument("--physics", choices=("mujoco", "ph"), default="mujoco")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--delay-ms", type=float, default=20.0)
    ap.add_argument("--push", type=float, default=0.0, help="sideways trunk velocity change every 4 s [m/s]")
    ap.add_argument("--command", type=float, nargs=3, action="append", metavar=("VX", "VY", "WZ"))
    ap.add_argument("--gait-clock", action="store_true",
                    help="the walk_clock agent (W5c): the policy takes 77 inputs, the 75 and the gait clock's sin, cos")
    args = ap.parse_args()
    backend = (MuJoCoBackend if args.physics == "mujoco" else PHBackend)(delay_ms=args.delay_ms)
    walker = Walker(backend, load_walk_policy(args.policy, C.OBS_DIM + (2 if args.gait_clock else 0)), clock=args.gait_clock)
    print(f"{Path(args.policy).name} in {backend.name} (one robot, {args.seconds:g} s per command, delay {args.delay_ms:g} ms, "
          f"push {args.push:g} m/s)")
    fell = 0
    for command in (args.command or COMMANDS):
        r = run(walker, command, args.seconds, push=args.push)
        fell += r["fell"]
        text = (f"FELL at {r['at_s']:.2f} s" if r["fell"] else
                f"walked: vx {r['speed'][0]:+.2f} vy {r['speed'][1]:+.2f} m/s, yaw rate {r['yaw_rate']:+.2f}/s, speed error {r['error']:.3f} m/s")
        print(f"  command {command[0]:+.2f} {command[1]:+.2f} {command[2]:+.2f}: {text}", flush=True)
    return 1 if fell else 0


if __name__ == "__main__":
    sys.exit(main())
