r"""Watch a get-up policy on the laptop, in MuJoCo: the sim-to-sim check of a policy trained on the pH plant.

The policy file is what the robot would load (deploy_<it>.bin, or a model_<it>.pt checkpoint), run through the
same agent as in training: the 117 observations, targets + stiffness + damping (rl/getup/contract.py), the energy
tank (impedance.py), the firmware filter, the delay line (the gains travel with the targets) and the motor
driver's PD inside the motor's torque-speed envelope.  Only the physics engine differs, and MuJoCo HAS
self-collision (the pH plant has not): limbs passing through each other show up here as robot-robot contacts.
For a like-for-like comparison MuJoCo gets the plant's gearbox damping and end stops.

The robot is dropped onto its back or front at HOME, held 0.4 s, then compliant until it lies still (as the
training's lying library), then the policy runs.

    python rl/getup/eval_mujoco.py --policy logs/getup-ph-v1/deploy_1200.bin --start supine --viewer
    python rl/getup/eval_mujoco.py --policy model_1200.pt --start prone --seconds 12 --push 0.5
    python rl/getup/eval_mujoco.py --policy deploy.bin --record outputs/getup_mujoco.mp4      # MUJOCO_GL=egl

Prints when it stood up, how long it stayed up, the battery energy to stand up (estimated with the plant's motor
constants: electrical power tau qdot + R i^2 per motor, i = tau / k, summed, drawn when positive), the potential
energy gained, the tank, and the self-contacts.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ph_k1.firmware import FirmwareConfig, safety_filter  # noqa: E402
from ph_k1.model import apply_servo_table, load_k1  # noqa: E402
from ph_k1.motors import build_motors  # noqa: E402
from rl.common.export import load_policy  # noqa: E402
from rl.getup import contract as C  # noqa: E402
from rl.getup.impedance import EnergyTank  # noqa: E402
from rl.getup.lying import ROLL_PITCH, KINDS, quat_from_rpy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="deploy_<it>.bin or model_<it>.pt")
    ap.add_argument("--start", default="supine", choices=("supine", "prone", "side", "standing"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--delay-ms", type=float, default=20.0, help="transport delay (the robot: about 20 ms)")
    ap.add_argument("--tank-capacity", type=float, default=150.0)
    ap.add_argument("--no-tank", action="store_true")
    ap.add_argument("--push", type=float, default=0.0, help="sideways trunk velocity change every 4 s once standing [m/s]")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--record", default=None, help="MP4 path (offscreen rendering)")
    ap.add_argument("--gain-scale", type=float, default=None,
                    help="the gain curriculum the policy was trained to (default: from a model_<it>.pt, else 1)")
    args = ap.parse_args()
    gain_scale = args.gain_scale
    if gain_scale is None:
        gain_scale = 1.0
        if args.policy.endswith(".pt"):
            try:
                gain_scale = float(torch.load(args.policy, map_location="cpu", weights_only=False)["infos"]
                                   ["curriculum"].get("gain_scale", 1.0))
            except Exception:  # noqa: BLE001  (an older checkpoint: the full gain range)
                pass
    print(f"gain scale {gain_scale:g} (0: the agent's fixed gains, 1: x2^+-{C.GAIN_RANGE:g})")

    model = load_k1()  # float64 CPU: tables, ranges, motor constants
    mot = build_motors(model)
    C.check_joint_order(model.joint_names)
    con = C.Contract(torch.float64)
    cfg = FirmwareConfig(gains=C.GAINS, slew=C.SLEW, delay_gains=True)
    policy = load_policy(args.policy)

    mj = apply_servo_table(mujoco.MjModel.from_xml_path(str(model.xml_path)))  # k1_skill_cfg armature + torque
    mj.jnt_range[1:, 0] = model.stop_low.numpy()   # the plant's end stops (wider than the command range)
    mj.jnt_range[1:, 1] = model.stop_high.numpy()
    mj.dof_damping[6:] = mot.damping.numpy()       # the plant's gearbox damping
    d = mujoco.MjData(mj)
    h = mj.opt.timestep
    substeps = round(cfg.policy_dt / h)
    lag = round(args.delay_ms * 1e-3 / h)
    home = np.array(C.HOME)
    kp_ref, kd_ref = np.array(C.KP_REF), np.array(C.KD_REF)
    k, R, vmax, tau_max = mot.k.numpy(), mot.R.numpy(), float(mot.v_max), mot.tau_max.numpy()
    mass = float(model.mass.sum())

    def pd_step(target, kp, kd):
        q, v = d.qpos[7:], d.qvel[6:]
        tau = kp * (target - q) - kd * v
        avail = np.maximum((vmax - k * v * np.sign(tau)) * k / R, 0.0)
        cap = np.minimum(avail, tau_max)
        d.ctrl[:] = np.clip(tau, -cap, cap)
        mujoco.mj_step(mj, d)
        return d.ctrl.copy()

    # the start: standing at HOME, or dropped onto the back / front / side and settled
    d.qpos[7:] = home
    if args.start == "standing":
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[2] = 1.0
        mujoco.mj_forward(mj, d)
        feet = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_foot_link", "right_foot_link")]
        lowest = min(d.geom_xpos[g][2] - mj.geom_size[g][2] for g in range(mj.ngeom)
                     if mj.geom_bodyid[g] in feet and mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and mj.geom_contype[g])
        d.qpos[2] -= lowest
        mujoco.mj_forward(mj, d)
    else:
        r0, p0 = ROLL_PITCH[KINDS[args.start]]
        t0 = torch.tensor([0.0], dtype=torch.float64)
        d.qpos[3:7] = quat_from_rpy(t0 + r0, t0 + p0, t0)[0].numpy()
        d.qpos[2] = 0.35
        mujoco.mj_forward(mj, d)
        for _ in range(round(0.4 / h)):
            pd_step(home, kp_ref, kd_ref)
        for _ in range(round(1.6 / h)):  # compliant: hold where the joints are
            pd_step(d.qpos[7:].copy(), kp_ref, kd_ref)
    mujoco.mj_forward(mj, d)

    world = 0
    floor_geoms = {g for g in range(mj.ngeom) if mj.geom_bodyid[g] == world}
    feet = {mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_foot_link", "right_foot_link")}

    def com_z():
        return float((mj.body_mass[1:] * d.xipos[1:, 2]).sum() / mj.body_mass[1:].sum())

    self_pairs = Counter()  # body pairs that touch, in MuJoCo physics steps

    def contacts():
        foot = {b: False for b in feet}
        other = robot_robot = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = mj.geom_bodyid[c.geom1], mj.geom_bodyid[c.geom2]
            if c.geom1 in floor_geoms or c.geom2 in floor_geoms:
                b = b2 if c.geom1 in floor_geoms else b1
                if b in foot:
                    foot[b] = True
                else:
                    other = True
            elif b1 != world and b2 != world:
                robot_robot = True
                pair_names = sorted(mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in (b1, b2))
                self_pairs[" - ".join(pair_names)] += 1
        return all(foot.values()), other, robot_robot

    T = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))[None]  # noqa: E731
    tank = EnergyTank(1, C.NJ, torch.float64, "cpu", args.tank_capacity, enabled=not args.no_tank)
    tank.reset(torch.arange(1), T(d.qpos[7:]), T(kp_ref), T(kd_ref))
    last_target = T(d.qpos[7:])
    line = deque([(d.qpos[7:].copy(), kp_ref.copy(), kd_ref.copy())] * lag, maxlen=lag + 1)
    last_action = torch.zeros(1, C.NUM_ACTIONS, dtype=torch.float64)
    qd_prev = T(d.qvel[6:])

    viewer = mujoco.viewer.launch_passive(mj, d) if args.viewer else None
    renderer = mujoco.Renderer(mj, 480, 640) if args.record else None
    frames = []
    com0 = com_z()
    stand_z = 0.518 - 0.04
    stood_at, up_time, battery, battery_at_stand, com_at_stand = None, 0.0, 0.0, None, None
    self_contact_steps, injected, limited = 0, 0.0, 0
    steps = round(args.seconds / cfg.policy_dt)
    push_every = round(4.0 / cfg.policy_dt)
    wall = time.time()
    for it in range(steps):
        t = it * cfg.policy_dt
        if args.push and stood_at is not None and it % push_every == 0:
            d.qvel[1] += args.push * (1 if (it // push_every) % 2 else -1)  # world y
        theta, qd = T(d.qpos[7:]), T(d.qvel[6:])
        gyro, quat = T(d.qvel[3:6]), T(d.qpos[3:7])  # MuJoCo free joint: angular velocity in the body frame
        g = con.projected_gravity(quat)
        obs = con.observe(gyro, g, theta, qd, last_action, tank.fraction())
        action = torch.as_tensor(policy(obs.numpy()))
        last_action = action
        target, kp, kd = con.decode(action, gain_scale)
        filt = lambda tt, kk, dd: safety_filter(model, cfg, tt, last_target, theta, qd, kk, dd)  # noqa: E731
        target, kp, kd, info = tank.step(theta, qd, qd_prev, cfg.policy_dt, target, kp, kd, filt)
        injected += float(info["injected"].clamp_min(0).sum())
        limited += int(info["limited"].sum())
        last_target, qd_prev = target, qd
        cmd = (target[0].numpy(), kp[0].numpy(), kd[0].numpy())
        robot_robot_any = False
        for _ in range(substeps):
            line.append(cmd)
            sent_t, sent_kp, sent_kd = line[0]
            tau = pd_step(sent_t, sent_kp, sent_kd)
            i_m = tau / k
            p_el = float((tau * d.qvel[6:] + R * i_m * i_m).sum())
            battery += max(p_el, 0.0) * h
            robot_robot_any = robot_robot_any or contacts()[2]
        self_contact_steps += int(robot_robot_any)
        feet_down, other, _ = contacts()
        up = -float(con.projected_gravity(T(d.qpos[3:7]))[0, 2]) > math.cos(math.radians(15.0))
        standing = d.qpos[2] > stand_z and up and feet_down and not other
        if standing:
            up_time += cfg.policy_dt
            if stood_at is None:
                stood_at, battery_at_stand, com_at_stand = t + cfg.policy_dt, battery, com_z()
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, cfg.policy_dt - (time.time() - wall) % cfg.policy_dt))
        if renderer is not None:
            renderer.update_scene(d, camera=-1)
            frames.append(renderer.render().copy())
    if stood_at is None:
        print(f"DID NOT STAND UP from {args.start} in {args.seconds:g} s: trunk z {d.qpos[2]:.3f} m, CoM {com_z():.3f} m")
    else:
        dv = mass * 9.81 * (com_at_stand - com0)
        if args.start == "standing":
            print(f"STANDING start: standing {up_time:.1f} of {args.seconds:g} s, average battery power "
                  f"{battery / args.seconds:.1f} W (est.)")
        else:
            print(f"STOOD UP from {args.start} at {stood_at:.2f} s, standing {up_time:.1f} of the "
                  f"{args.seconds - stood_at:.1f} s after; battery to stand up {battery_at_stand:.0f} J (est.), "
                  f"potential energy gained {dv:.0f} J (efficiency {dv / max(battery_at_stand, 1e-6):.0%})")
    print(f"battery over the run {battery:.0f} J (est., the plant's motor constants), controller injection {injected:.0f} J, "
          f"tank limited {limited} of {steps} steps, self-contact in {self_contact_steps} of {steps} policy steps, "
          f"delay {args.delay_ms:g} ms, tank {'off' if args.no_tank else f'{args.tank_capacity:g} J'}")
    if self_pairs:
        print("touching body pairs (MuJoCo steps): " + ", ".join(f"{k} {v}" for k, v in self_pairs.most_common(12)))
    if renderer is not None and frames:
        import imageio

        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.record, frames, fps=round(1 / cfg.policy_dt), macro_block_size=8)
        print(f"video {args.record}")
    if viewer is not None:
        viewer.close()
    return 0 if stood_at is not None else 1


if __name__ == "__main__":
    sys.exit(main())
