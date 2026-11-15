r"""The pH plant's ball against MuJoCo's: the same ball, the same command path, the same leg swing.

    python validation/ball_vs_mujoco.py [--plot out.png]

Three cases, each on both engines:
  drop    the ball falls 0.4 m: height of the first bounce
  roll    the ball rolls at 3 m/s: speed after 1 s
  kick    the trunk is held 3 cm above its standing height (pH: welded trunk; MuJoCo: trunk put back after every
          step), the right leg swings (targets through the firmware filter, 20 ms delay, motor driver), ball 4 cm in
          front of the right foot's toes: the fastest ball velocity within 0.2 s after the touch, three swing speeds
MuJoCo's ball is K1_Deploy's twin ball (k1_kick/assets/ball.xml: r 0.075 m, 0.2 kg, hollow, condim 6, friction
0.6 / 0.005 / 0.01, default solref of the robot's MJCF, integrator implicitfast and pyramidal cone as the twin).  Nothing is fitted here: the numbers show how far the two
contact models are apart before any tuning.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ph_k1.ball import BallParams  # noqa: E402
from ph_k1.firmware import FirmwareConfig, gains, safety_filter  # noqa: E402
from ph_k1 import servo  # noqa: E402
from ph_k1.model import _STANDARD_BODY, _STANDARD_JOINT, load_k1  # noqa: E402
from ph_k1.motors import build_motors  # noqa: E402
from ph_k1.native import K1PHPlantNative  # noqa: E402
from rl.walk import contract as C  # noqa: E402

DELAY_MS = 20.0
LIFT = 0.03   # m: the held trunk above its standing height, so the swinging foot clears the floor
BALL = BallParams(enabled=True)


def mujoco_with_ball(model):
    """The robot's MJCF with the plant's names and servo table, plus the ball after the robot (qpos[29:36])."""
    spec = mujoco.MjSpec.from_file(str(model.xml_path))
    for b in spec.bodies:
        b.name = _STANDARD_BODY.get(b.name, b.name)
    for j in spec.joints:
        j.name = _STANDARD_JOINT.get(j.name, j.name)
    for a in spec.actuators:
        a.name = _STANDARD_JOINT.get(a.name, a.name)
        a.target = _STANDARD_JOINT.get(a.target, a.target)
    for x in spec.sensors:
        x.objname = _STANDARD_BODY.get(x.objname, _STANDARD_JOINT.get(x.objname, x.objname))
    body = spec.worldbody.add_body(name="ball", pos=[50.0, 0.0, BALL.radius])
    body.add_freejoint(name="ball_joint")
    geom = body.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL.radius, 0, 0], condim=6,
                         friction=[0.6, 0.005, 0.01])
    geom.mass, geom.typeinertia = BALL.mass, mujoco.mjtGeomInertia.mjINERTIA_SHELL   # as the twin's ball
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST   # as K1_Deploy's twin (twin/model.py)
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    m = spec.compile()
    m.dof_armature[6:28] = servo.ARMATURE          # apply_servo_table, for the robot's 22 joints
    for a in range(m.nu):
        j = int(m.actuator_trnid[a, 0]) - 1
        m.actuator_forcerange[a] = (-servo.EFFORT[j], servo.EFFORT[j])
    mot = build_motors(model)
    m.jnt_range[1:23, 0], m.jnt_range[1:23, 1] = model.stop_low.numpy(), model.stop_high.numpy()
    m.dof_damping[6:28] = mot.damping.numpy()
    return m, mot


class Mj:
    """MuJoCo with the plant's command path (rl/walk/backends.py MuJoCoBackend) and a ball."""

    def __init__(self, model, hold=False):
        self.model, self.hold = model, hold
        self.m, mot = mujoco_with_ball(model)
        self.d = mujoco.MjData(self.m)
        self.fw = FirmwareConfig(gains=C.GAINS, slew=C.SLEW, delay_ms=(DELAY_MS, DELAY_MS))
        self.kp, self.kd = (g[None] for g in gains(model, C.GAINS))
        self.k, self.R, self.v_max, self.tau_max = mot.k.numpy(), mot.R.numpy(), float(mot.v_max), mot.tau_max.numpy()
        self.sub = round(self.fw.policy_dt / self.m.opt.timestep)
        self.lag = round(DELAY_MS * 1e-3 / self.m.opt.timestep)
        feet = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_foot_link", "right_foot_link")]
        self.soles = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in feet
                      and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and self.m.geom_contype[g]]

    def reset(self, pose):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:3], d.qpos[3:7], d.qpos[7:29] = (0, 0, 1.0), (1, 0, 0, 0), pose
        d.qpos[29:36] = (50.0, 0, BALL.radius, 1, 0, 0, 0)
        mujoco.mj_forward(m, d)
        d.qpos[2] -= min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in self.soles) - (LIFT if self.hold else 0.0)
        mujoco.mj_forward(m, d)
        self.base = d.qpos[:7].copy()
        self.ring = deque([np.array(pose)] * self.lag, maxlen=self.lag + 1)
        self.last = torch.as_tensor(pose, dtype=torch.float64)[None]

    def set_ball(self, xyz, vel=(0, 0, 0), roll=False):
        d = self.d
        d.qpos[29:32], d.qpos[32:36] = (xyz[0], xyz[1], max(xyz[2], BALL.radius)), (1, 0, 0, 0)   # no overlap at the start
        d.qvel[28:31] = vel
        d.qvel[31:34] = (-vel[1] / BALL.radius, vel[0] / BALL.radius, 0) if roll else (0, 0, 0)
        mujoco.mj_forward(self.m, d)

    def step(self, target):
        d = self.d
        T = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))[None]  # noqa: E731
        f = safety_filter(self.model, self.fw, T(target), self.last, T(d.qpos[7:29]), T(d.qvel[6:28]), self.kp, self.kd)
        self.last = f
        for _ in range(self.sub):
            self.ring.append(f[0].numpy())
            v = d.qvel[6:28]
            tau = self.kp[0].numpy() * (self.ring[0] - d.qpos[7:29]) - self.kd[0].numpy() * v
            cap = np.minimum(np.maximum((self.v_max - self.k * v * np.sign(tau)) * self.k / self.R, 0.0), self.tau_max)
            d.ctrl[:] = np.clip(tau, -cap, cap)
            mujoco.mj_step(self.m, d)
            if self.hold:
                d.qpos[:7], d.qvel[:6] = self.base, 0.0

    def ball(self):
        return np.concatenate((self.d.qpos[29:32], self.d.qvel[28:31]))

    def toes(self):
        g = self.soles[1]
        return self.d.geom_xpos[g] + self.d.geom_xmat[g].reshape(3, 3) @ np.array([self.m.geom_size[g][0], 0, 0])


class Ph:
    def __init__(self, model, hold=False):
        self.fw = FirmwareConfig(gains=C.GAINS, slew=C.SLEW, delay_ms=(DELAY_MS, DELAY_MS))
        self.hold = hold
        self.plant = K1PHPlantNative(model=model, batch=1, firmware=self.fw, ball=BALL, fixed_base=hold)
        self.foot = model.body_names.index("right_foot_link")

    def reset(self, pose):
        pose = torch.as_tensor(pose, dtype=torch.float64)
        if not self.hold:
            self.plant.reset(pose=pose)
            return
        free = K1PHPlantNative(model=self.plant.model, batch=1)   # the standing height of this pose
        free.reset(pose=pose)
        self.plant.reset(pose=pose, base_pos=(0.0, 0.0, float(free.q[0, 2]) + LIFT))

    def set_ball(self, xyz, vel=(0, 0, 0), roll=False):
        self.plant.set_ball(xyz[:2], vel if roll else None)
        self.plant.ball[0, 2] = xyz[2]
        if not roll:
            self.plant.ball[0, 3:6] = torch.as_tensor(vel, dtype=torch.float64)

    def step(self, target):
        self.plant.policy_step(torch.as_tensor(target, dtype=torch.float64)[None])

    def ball(self):
        return self.plant.ball[0, :6].numpy().copy()

    def toes(self):
        from ph_k1.ball import build_boxes
        kin, bx = self.plant.kinematics(), build_boxes(self.plant.model)
        R = (kin.R[0, self.foot] @ bx.rot[1]).numpy()
        return kin.pos[0, self.foot].numpy() + kin.R[0, self.foot].numpy() @ bx.pos[1].numpy() + R @ np.array(
            [float(bx.half[1, 0]), 0, 0])


def swing(t, speed):
    """Right leg: lean back for 0.3 s, then swing through (hip pitch forward, knee stretching) in 0.3 s / speed."""
    q = C.HOME.copy()
    hip, knee = servo.JOINTS.index("Right_Hip_Pitch"), servo.JOINTS.index("Right_Knee_Pitch")
    if t < 0.3:
        s = t / 0.3
        q[hip] += 0.35 * s
        q[knee] += 0.5 * s
    else:
        s = min((t - 0.3) * speed / 0.3, 1.0)
        q[hip] += 0.35 - 1.25 * s
        q[knee] += 0.5 - 1.1 * s
    return q


def run_kick(eng, speed):
    eng.reset(C.HOME)
    for _ in range(25):                      # settle 0.5 s
        eng.step(C.HOME)
    toes = eng.toes()
    eng.set_ball((toes[0] + BALL.radius + 0.04, toes[1], BALL.rest_height()))
    best, touch = np.zeros(3), None
    for k in range(60):
        eng.step(swing(k * 0.02, speed))
        b = eng.ball()
        if touch is None and np.linalg.norm(b[3:5]) > 0.3:
            touch = k
        if touch is not None and k - touch <= 10 and np.linalg.norm(b[3:5]) > np.linalg.norm(best[:2]):
            best = b[3:6].copy()
    return best, touch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    model = load_k1()
    rows = []
    for name, cls in (("pH", Ph), ("MuJoCo", Mj)):
        eng, held = cls(model), cls(model, hold=True)
        eng.reset(C.HOME)
        eng.set_ball((2.0, 0.0, 0.4 + BALL.radius))
        z, top, up = [], 0.0, False
        for _ in range(60):
            eng.step(C.HOME)
            b = eng.ball()
            up = up or b[5] > 0.05
            if up:
                top = max(top, b[2])
        eng.reset(C.HOME)
        eng.set_ball((2.0, 0.0, BALL.rest_height()), (3.0, 0, 0), roll=True)
        for _ in range(50):
            eng.step(C.HOME)
        rows.append((name, "drop 0.4 m: first bounce (centre above rest)", f"{top - BALL.radius:.3f} m"))
        rows.append((name, "roll 3 m/s: speed after 1 s", f"{eng.ball()[3]:.2f} m/s"))
        for speed in (1.0, 2.0, 3.0):
            v, touch = run_kick(held, speed)
            sp, ang = np.linalg.norm(v[:2]), np.degrees(np.arctan2(v[1], v[0]))
            rows.append((name, f"kick, swing x{speed:.0f}", f"{sp:.2f} m/s, {ang:+.1f} deg, loft {v[2]:+.2f} m/s, "
                                                             f"touch at {touch}"))
    w = max(len(r[1]) for r in rows)
    for name, what, val in rows:
        print(f"{name:7s} {what:{w}s}  {val}")


if __name__ == "__main__":
    main()
