#!/usr/bin/env python3
"""The classical servo (ph_k1/drive.py) holds the stand pose on pH physics.

This is a PHYSICS check, not the RL task: target angles are fixed, a PD law
turns them into volts, and the pH plant does the rest.  It shows that the
contact, gravity and motor models together can carry the robot.

With the firmware gains (--gain 1) the robot is a soft inverted pendulum
(ankle stiffness 2 x 50 N m/rad against m g h = 85 N m/rad, minus the give
of knees and hips) and it slowly tips over -- physically correct, and
exactly what an RL policy will have to fix.  --gain 3 holds it.

    ../../Presentation/.venv/bin/python examples/hold_pose.py --gain 3 --seconds 10
    ../../Presentation/.venv/bin/python examples/hold_pose.py --gain 1 --view
    ../../Presentation/.venv/bin/python examples/hold_pose.py --gain 3 --push 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]  # the repo root (this script lives one folder down)
sys.path.insert(0, str(HERE))

from ph_k1.drive import K1Drive  # noqa: E402
from ph_k1.plant import K1PHPlant, Ledger  # noqa: E402
from ph_k1.render import Recorder, Viewer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gain", type=float, default=3.0, help="servo gain scale (1 = K1 firmware gains)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--push", type=float, default=0.0,
                    help="knock the trunk sideways with this speed change [cm/s] at t=2 s")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)

    plant = K1PHPlant(timestep=args.dt)
    plant.reset()
    drive = K1Drive(plant, gain_scale=args.gain)
    drive.reset()
    target = plant.theta.clone()
    viewer = Viewer(plant) if args.view else None
    rec = Recorder(plant)
    steps = int(round(args.seconds / args.dt))
    wall = time.time()
    for s in range(steps):
        if args.push and abs(plant.time - 2.0) < 0.5 * args.dt:
            # impulse on the trunk: change its sideways momentum (world y)
            kin = plant.kinematics()
            nu = plant.velocity(kin)
            nu[:, 3:6] += (kin.R[:, 0].transpose(1, 2) @ torch.tensor([0.0, args.push / 100, 0.0], dtype=nu.dtype))
            plant.p = (kin.M @ nu[..., None]).squeeze(-1)
            # the push is an external impulse, not a port: restart the ledger
            plant.ledger = Ledger.zeros(plant.B, plant.dtype, plant.device)
            rec.H0 = float(plant.energy()["total"][0])
        info = plant.step(drive.volts(target))
        if s % 10 == 0:
            rec.log(info)
        if viewer is not None and s % max(1, int(1 / 60 / args.dt)) == 0:
            if not viewer.is_running():
                break
            viewer.sync()
    rec.log(info)
    last = rec.rows[-1]
    upright = last["tilt_deg"] < 10 and last["trunk_z"] > 0.4
    out = HERE / "outputs"
    tag = f"hold_gain{args.gain:g}" + (f"_push{args.push:g}" if args.push else "")
    rec.save_csv(out / f"{tag}.csv")
    rec.plot(out / f"{tag}.png", title=f"servo hold, gain x{args.gain:g}: energy tanks (pH model)")
    print(f"{'UPRIGHT' if upright else 'FELL'} after {plant.time:.1f} s: trunk z={last['trunk_z']:.3f} m, "
          f"tilt={last['tilt_deg']:.1f} deg, contact points={last.get('contact_points')}, "
          f"supplied={last['supplied']:.1f} J (copper {last['copper']:.1f} J), residual={last['residual']:+.3f} J, "
          f"{time.time() - wall:.0f} s wall")
    print(f"wrote {out / (tag + '.png')} and .csv")
    if viewer is not None:
        viewer.wait_until_closed()
        viewer.close()


if __name__ == "__main__":
    main()
