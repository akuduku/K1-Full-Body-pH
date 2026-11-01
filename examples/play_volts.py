#!/usr/bin/env python3
"""Put voltages WE choose on the 22 coils and watch the pH model move.  No RL.

Scenarios
  joints   welded trunk; one joint group after the other gets a voltage
           ramp + hold + reverse (like the arm's play_volts.py)
  drop     free robot 5 cm above the floor, all coils at 0 V (= shorted):
           it lands, the shorted motors brake, it folds and lies down
  custom   your own table: edit SCHEDULE below

    cd "RL Policy for pH/Full Body"
    ../../Presentation/.venv/bin/python examples/play_volts.py joints --view
    ../../Presentation/.venv/bin/python examples/play_volts.py drop --seconds 3

Writes outputs/<scenario>.csv and outputs/<scenario>.png (energy tanks).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]  # the repo root (this script lives one folder down)
sys.path.insert(0, str(HERE))

from ph_k1.model import NJ  # noqa: E402
from ph_k1.plant import K1PHPlant  # noqa: E402
from ph_k1.render import Recorder, Viewer  # noqa: E402

GROUPS = {
    "head": ("AAHead_yaw", "Head_pitch"),
    "left arm": ("ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw"),
    "right arm": ("ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"),
    "left leg": ("Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll"),
    "right leg": ("Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"),
}

# custom: (t_start, t_end, joint name, volts)
SCHEDULE = [
    (0.0, 1.0, "ARight_Shoulder_Pitch", -1.0),
    (0.0, 1.0, "Right_Elbow_Pitch", 0.8),
    (1.0, 2.0, "Left_Knee_Pitch", 1.5),
]


def joints_volts(t: float, names: list[str]) -> torch.Tensor:
    """Each group gets 2 s: 0.5 s ramp to +V, 0.5 s hold, 0.5 s to -V, 0.5 s 0."""
    v = torch.zeros(NJ, dtype=torch.float64)
    k, tau = divmod(t, 2.0)
    groups = list(GROUPS.values())
    if int(k) >= len(groups):
        return v
    level = {"head": 1.5}.get(list(GROUPS)[int(k)], 1.0)
    shape = min(tau / 0.5, 1.0) if tau < 1.0 else (-1.0 if tau < 1.5 else 0.0)
    for name in groups[int(k)]:
        v[names.index(name)] = level * shape
    return v


def custom_volts(t: float, names: list[str]) -> torch.Tensor:
    v = torch.zeros(NJ, dtype=torch.float64)
    for t0, t1, name, volts in SCHEDULE:
        if t0 <= t < t1:
            v[names.index(name)] = volts
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=("joints", "drop", "custom"))
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer (drawing only)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    fixed = args.scenario == "joints"
    plant = K1PHPlant(fixed_base=fixed, timestep=args.dt)
    names = plant.model.joint_names
    if args.scenario == "drop":
        plant.reset(clearance=0.05)
    elif fixed:
        plant.reset(pose={})  # MJCF zero pose, trunk welded in the air
    else:
        plant.reset()
    seconds = args.seconds or {"joints": 2.0 * len(GROUPS), "drop": 3.0, "custom": 3.0}[args.scenario]
    volts_of = {"joints": joints_volts, "custom": custom_volts}.get(args.scenario, lambda t, n: torch.zeros(NJ, dtype=torch.float64))

    viewer = Viewer(plant) if args.view else None
    rec = Recorder(plant)
    steps = int(round(seconds / args.dt))
    log_every = max(1, int(0.01 / args.dt))
    wall = time.time()
    for s in range(steps):
        info = plant.step(volts_of(plant.time, names)[None])
        if s % log_every == 0:
            rec.log(info)
        if viewer is not None and s % max(1, int(1 / 60 / args.dt)) == 0:
            if not viewer.is_running():
                break
            viewer.sync()
    rec.log(info)
    out = HERE / "outputs"
    rec.save_csv(out / f"{args.scenario}.csv")
    rec.plot(out / f"{args.scenario}.png", title=f"play_volts {args.scenario}: energy tanks (pH model, no mj_step)")
    last = rec.rows[-1]
    print(f"{seconds:.1f} s simulated in {time.time() - wall:.1f} s wall | H-H0={last['H'] - rec.H0:+.2f} J "
          f"supplied={last['supplied']:.2f} J dissipated={last['dissipated']:.2f} J residual={last['residual']:+.3f} J")
    print(f"wrote {out / (args.scenario + '.png')} and .csv")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
