r"""Watch a stand policy on the laptop, in MuJoCo: the sim-to-sim check of a policy trained on the pH plant.

The policy file is what the robot would load (deploy_<it>.bin from training, or a model_<it>.pt checkpoint), run
through the same command path as on the robot (rl/stand/contract.py observation + decode, ph_k1/firmware.py filter,
delay line and motor driver).  Only the physics engine differs: MuJoCo's rigid bodies, contacts and integrator
instead of the pH plant.  For a like-for-like comparison MuJoCo gets the plant's gearbox damping and its end stops
(the MJCF range widened to what the real robot reached); the motors are torque sources limited by the same
torque-speed envelope (the pH servo makes the coil current follow the demand, so torque = demand within limits).

    python rl/stand/eval_mujoco.py --policy logs/stand-ph-v1/deploy_3000.bin --viewer
    python rl/stand/eval_mujoco.py --policy model_3000.pt --seconds 20 --push 0.3 --delay-ms 20
    python rl/stand/eval_mujoco.py --policy deploy.bin --record outputs/stand_mujoco.mp4      # needs MUJOCO_GL=egl

Prints how long it stood, the largest tilt and whether the firmware's abort (|roll| or |pitch| > 0.6) fired.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ph_k1.firmware import FirmwareConfig, gains, safety_filter  # noqa: E402
from ph_k1.model import apply_servo_table, load_k1  # noqa: E402
from ph_k1.motors import build_motors  # noqa: E402
from rl.stand import contract as C  # noqa: E402
from rl.stand.export import actor_tensors, bin_policy, read_bin  # noqa: E402


def load_policy(path: str):
    p = Path(path)
    if p.suffix == ".bin":
        return bin_policy(read_bin(p))
    tensors, mean, std, _ = actor_tensors(p)
    mats = {k: v.numpy() for k, v in tensors.items()}
    mats["obs_normalizer._mean"], mats["obs_normalizer._std"] = mean.numpy()[None], std.numpy()[None]
    return bin_policy(mats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="deploy_<it>.bin or model_<it>.pt")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--delay-ms", type=float, default=20.0, help="transport delay (the robot: about 20 ms)")
    ap.add_argument("--no-slew", action="store_true")
    ap.add_argument("--push", type=float, default=0.0, help="sideways trunk velocity change every 4 s [m/s]")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--record", default=None, help="MP4 path (offscreen rendering)")
    args = ap.parse_args()

    model = load_k1()  # float64 CPU: tables, ranges, motor constants
    mot = build_motors(model)
    C.check_joint_order(model.joint_names)
    con = C.Contract(torch.float64)
    cfg = FirmwareConfig(gains=C.GAINS, slew=not args.no_slew)
    kp, kd = (g[None] for g in gains(model, C.GAINS))
    policy = load_policy(args.policy)

    mj = apply_servo_table(mujoco.MjModel.from_xml_path(str(model.xml_path)))  # k1_skill_cfg armature + torque
    mj.jnt_range[1:, 0] = model.stop_low.numpy()   # the plant's end stops (wider than the command range)
    mj.jnt_range[1:, 1] = model.stop_high.numpy()
    mj.dof_damping[6:] = mot.damping.numpy()       # the plant's gearbox damping
    d = mujoco.MjData(mj)
    h = mj.opt.timestep
    substeps = round(cfg.policy_dt / h)
    lag = round(args.delay_ms * 1e-3 / h)

    # start as on the robot after Prepare + countdown: standing still in the start pose, soles on the floor
    d.qpos[3:7] = (1, 0, 0, 0)
    d.qpos[7:] = C.DEFAULT_QPOS
    d.qpos[2] = 1.0
    mujoco.mj_forward(mj, d)
    feet = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_foot_link", "right_foot_link")]
    lowest = min(d.geom_xpos[g][2] - mj.geom_size[g][2] for g in range(mj.ngeom)
                 if mj.geom_bodyid[g] in feet and mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and mj.geom_contype[g])
    d.qpos[2] -= lowest
    mujoco.mj_forward(mj, d)

    T = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))[None]  # noqa: E731
    last_target = T(C.DEFAULT_QPOS)
    # the delay line: after each append, ring[0] is the target sent `lag` plant steps ago
    ring = deque([np.array(C.DEFAULT_QPOS)] * lag, maxlen=lag + 1)
    last_action = torch.zeros(1, C.NUM_ACTIONS, dtype=torch.float64)
    k, R, vmax, tau_max = mot.k.numpy(), mot.R.numpy(), float(mot.v_max), mot.tau_max.numpy()
    kp_np, kd_np = kp[0].numpy(), kd[0].numpy()

    viewer = mujoco.viewer.launch_passive(mj, d) if args.viewer else None
    renderer = mujoco.Renderer(mj, 480, 640) if args.record else None
    frames, max_tilt, fell_at = [], 0.0, None
    steps = round(args.seconds / cfg.policy_dt)
    wall = time.time()
    for it in range(steps):
        t = it * cfg.policy_dt
        if args.push and it > 0 and it % round(4.0 / cfg.policy_dt) == 0:
            d.qvel[1] += args.push * (1 if (it // round(4.0 / cfg.policy_dt)) % 2 else -1)  # world y
        theta, qd = T(d.qpos[7:]), T(d.qvel[6:])
        gyro, quat = T(d.qvel[3:6]), T(d.qpos[3:7])  # MuJoCo free joint: angular velocity in the body frame
        obs = con.observe(theta, qd, gyro, quat, last_action)
        action = torch.as_tensor(policy(obs.numpy()))
        last_action = action
        target = safety_filter(model, cfg, con.decode(action), last_target, theta, qd, kp, kd)
        last_target = target
        for _ in range(substeps):
            ring.append(target[0].numpy())
            sent = ring[0]
            q, v = d.qpos[7:], d.qvel[6:]
            tau = kp_np * (sent - q) - kd_np * v
            avail = np.maximum((vmax - k * v * np.sign(tau)) * k / R, 0.0)
            cap = np.minimum(avail, tau_max)
            d.ctrl[:] = np.clip(tau, -cap, cap)
            mujoco.mj_step(mj, d)
        rp = con.roll_pitch(T(d.qpos[3:7]))[0].abs()
        max_tilt = max(max_tilt, float(rp.max()))
        if fell_at is None and (float(rp.max()) > C.TILT_ABORT or d.qpos[2] < 0.28):
            fell_at = t
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, cfg.policy_dt - (time.time() - wall) % cfg.policy_dt))
        if renderer is not None:
            renderer.update_scene(d)
            frames.append(renderer.render().copy())
    print(f"{'FELL at %.2f s' % fell_at if fell_at is not None else 'STOOD %.1f s' % args.seconds}: "
          f"max tilt {math.degrees(max_tilt):.1f} deg, trunk z {d.qpos[2]:.3f} m, delay {args.delay_ms:g} ms, "
          f"slew {'off' if args.no_slew else 'on'}, push {args.push:g} m/s")
    if renderer is not None and frames:
        import imageio

        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.record, frames, fps=round(1 / cfg.policy_dt), macro_block_size=8)
        print(f"video {args.record}")
    if viewer is not None:
        viewer.close()
    return 0 if fell_at is None else 1


if __name__ == "__main__":
    sys.exit(main())
