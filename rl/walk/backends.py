r"""One walking policy on two physics engines, behind one interface.

PHBackend steps the pH plant (K1PHPlantNative.policy_step).  MuJoCoBackend steps MuJoCo with the like-for-like setup of
rl/stand/eval_mujoco.py: the same firmware filter, delay and PD law, the plant's servo table, gearbox damping, end stops
and torque-speed limit.  Both read the same quantities at the end of a policy step (StepInfo holds the last plant step
only), so a difference between them comes from contact, the motor model or the integrator, not from the bookkeeping.

Walker is the firmware's loop around a backend: ramp the twist, observe, run the policy, decode, step.

Energy is booked at every 1 ms plant step on both (backend.energy_book(), cumulative joules since reset()):
    bus      the battery's energy, sum_j v_j i_j (joints share the bus: a braking joint feeds the others)
    copper   sum_j R_j i_j^2
    motor    the motors' mechanical work sum_j tau_j theta_dot_j (net)
pH: from the plant's own ledger (supplied, copper; motor = supplied - copper - change of the coils' magnetic energy).
MuJoCo: an ideal DC motor behind each torque (i = tau / k, R / k^2 = w_max / tau_stall, ph_k1/motors.py), summed
inside its 1 ms loop.  Sampling the torque once per 20 ms policy step instead misses the current's transients (the
pH ledger's copper loss was 10-21 % above such samples while walking; Model Based RL docs/EXPERIMENTS.md, P1).
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ph_k1.firmware import FirmwareConfig, gains, safety_filter  # noqa: E402
from ph_k1.model import NB, K1Model, apply_servo_table, load_k1, load_mj  # noqa: E402
from ph_k1.motors import build_motors  # noqa: E402
from ph_k1.native import K1PHPlantNative  # noqa: E402
from rl.common.export import load_policy  # noqa: E402
from rl.walk import contract as C  # noqa: E402

CHECKPOINTS = REPO.parent / "k1_sprint" / "checkpoints"
FEET = ("left_foot_link", "right_foot_link")


@dataclass
class Reading:
    """The robot at the end of a policy step."""

    qpos: np.ndarray        # (29) trunk position, quaternion (w, x, y, z), joint angles
    theta: np.ndarray       # (22) joint angles
    qd: np.ndarray          # (22) joint velocities
    gyro: np.ndarray        # (3) trunk angular velocity, trunk frame
    torque: np.ndarray      # (22) joint torques of the last plant step
    foot_force: np.ndarray  # (2) ground force along the floor normal, left, right [N]
    foot_pos: np.ndarray    # (2,3) foot body origins, world [m]
    foot_rot: np.ndarray    # (2,3,3) foot body rotations, foot frame to world

    @property
    def pos(self) -> np.ndarray:
        return self.qpos[:3]

    @property
    def quat(self) -> np.ndarray:
        return self.qpos[3:7]


def resolve_policy(name: str | Path) -> Path:
    """A path, or a name in k1_sprint/checkpoints such as walk_v9 or lean-B."""
    p = Path(name)
    if p.exists():
        return p
    for candidate in (CHECKPOINTS / f"{name}.pt", CHECKPOINTS / f"{name}.policy.pt"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no policy {name!r}: not a file, and not in {CHECKPOINTS}")


def load_walk_policy(name: str | Path, obs_dim: int = C.OBS_DIM):
    """A walking policy as a function of one observation (obs_dim, float32: the sprint's 75, or 77 with the gait
    clock) to one action (22): TorchScript, .bin or rsl-rl checkpoint."""
    path = resolve_policy(name)
    fn = None
    if path.suffix == ".pt":
        try:
            net = torch.jit.load(str(path), map_location="cpu").eval()
        except RuntimeError:  # an rsl-rl checkpoint, not TorchScript
            net = None
        if net is not None:
            def fn(obs):
                with torch.no_grad():
                    return net(torch.from_numpy(obs)[None])[0].numpy()
    if fn is None:
        loaded = load_policy(path)

        def fn(obs):
            return np.asarray(loaded(obs[None]))[0]
    try:
        out = fn(np.zeros(obs_dim, dtype=np.float32))
    except RuntimeError as exc:
        raise ValueError(f"{path.name} does not take {obs_dim} inputs") from exc
    if out.shape != (C.NJ,):
        raise ValueError(f"{path.name} gives {out.shape} outputs, the sprint needs ({C.NJ},)")
    return fn


class PHBackend:
    name = "pH"

    def __init__(self, model: K1Model | None = None, delay_ms: float = 20.0, gain_preset: str = C.GAINS,
                 slew: bool = C.SLEW, home: np.ndarray = C.HOME):
        self.model = model or load_k1()
        self.home = np.asarray(home, dtype=np.float64)
        C.check_joint_order(self.model.joint_names)
        self.firmware = FirmwareConfig(gains=gain_preset, slew=slew, delay_ms=(delay_ms, delay_ms))
        self.plant = K1PHPlantNative(model=self.model, batch=1, firmware=self.firmware)
        m = self.model
        self.feet = [m.body_names.index(n) for n in FEET]
        self.sole = [m.contact_is_sole & (m.contact_body == b) for b in self.feet]
        self.reset()

    def reset(self, pose: np.ndarray | None = None) -> None:
        pose = self.home if pose is None else pose
        self.plant.reset(pose=torch.as_tensor(pose, dtype=torch.float64))
        self.qd, self.gyro, self.torque = np.zeros(C.NJ), np.zeros(3), np.zeros(C.NJ)
        self.foot_force = np.zeros(2)

    def read(self) -> Reading:
        kin = self.plant.kinematics()
        qpos = self.plant.q[0].numpy().copy()
        return Reading(qpos=qpos, theta=qpos[7:], qd=self.qd, gyro=self.gyro, torque=self.torque,
                       foot_force=self.foot_force, foot_pos=kin.pos[0, self.feet].numpy().copy(),
                       foot_rot=kin.R[0, self.feet].numpy().copy())

    def step(self, target: np.ndarray, push: np.ndarray | None = None) -> None:
        """One policy step. push (3): change of the trunk's world velocity at its start [m/s]."""
        dv = None if push is None else torch.as_tensor(push, dtype=torch.float64)[None]
        info = self.plant.policy_step(torch.as_tensor(target, dtype=torch.float64)[None], push_dv=dv)
        self.qd = info.extra["nu"][0, NB:].numpy().copy()
        self.gyro = info.extra["gyro"][0].numpy().copy()
        self.torque = info.torque[0].numpy().copy()
        fz = info.contact_force[0, :, 2]
        self.foot_force = np.array([float(fz[s].sum()) for s in self.sole])

    def healthy(self) -> bool:
        return bool(torch.isfinite(self.plant.q).all() and torch.isfinite(self.plant.p).all())

    def energy_book(self) -> dict:
        """Cumulative since reset() [J]: bus, copper, motor (see the module's docstring), from the ledger."""
        led = self.plant.ledger
        sup, cu = float(led.supplied[0]), float(led.copper[0])
        coil = float(self.plant.energy()["coil"][0])
        return {"bus": sup, "copper": cu, "motor": sup - cu - coil}


class MuJoCoBackend:
    name = "MuJoCo"

    def __init__(self, model: K1Model | None = None, delay_ms: float = 20.0, gain_preset: str = C.GAINS,
                 slew: bool = C.SLEW, home: np.ndarray = C.HOME):
        self.model = model or load_k1()
        self.home = np.asarray(home, dtype=np.float64)
        C.check_joint_order(self.model.joint_names)
        self.firmware = FirmwareConfig(gains=gain_preset, slew=slew, delay_ms=(delay_ms, delay_ms))
        self.kp, self.kd = (g[None] for g in gains(self.model, gain_preset))
        mot = build_motors(self.model)
        self.m = apply_servo_table(load_mj(self.model.xml_path))
        self.m.jnt_range[1:, 0] = self.model.stop_low.numpy()
        self.m.jnt_range[1:, 1] = self.model.stop_high.numpy()
        self.m.dof_damping[6:] = mot.damping.numpy()
        self.d = mujoco.MjData(self.m)
        h = self.m.opt.timestep
        self.substeps = round(self.firmware.policy_dt / h)
        self.lag = round(delay_ms * 1e-3 / h)
        self.k, self.R, self.v_max = mot.k.numpy(), mot.R.numpy(), float(mot.v_max)
        self.r_over_k2 = self.R / self.k ** 2
        self.tau_max = mot.tau_max.numpy()
        self.kp_np, self.kd_np = self.kp[0].numpy(), self.kd[0].numpy()
        self.feet = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in FEET]
        self.side = {b: i for i, b in enumerate(self.feet)}
        self.sole_geoms = [g for g in range(self.m.ngeom)
                           if self.m.geom_bodyid[g] in self.feet and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                           and self.m.geom_contype[g]]
        self.reset()

    def reset(self, pose: np.ndarray | None = None) -> None:
        pose = self.home if pose is None else pose
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = (0.0, 0.0, 1.0)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[7:] = pose
        mujoco.mj_forward(m, d)
        d.qpos[2] -= min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in self.sole_geoms)  # soles on the floor
        mujoco.mj_forward(m, d)
        # after each append, ring[0] is the target sent `lag` plant steps ago
        self.ring = deque([np.array(pose)] * self.lag, maxlen=self.lag + 1)
        self.last_target = torch.as_tensor(pose, dtype=torch.float64)[None]
        self.torque = np.zeros(C.NJ)
        self.book = np.zeros(3)          # bus, copper, motor [J], summed every plant step

    def energy_book(self) -> dict:
        """Cumulative since reset() [J]: bus, copper, motor (see the module's docstring)."""
        return dict(zip(("bus", "copper", "motor"), map(float, self.book)))

    def _foot_force(self) -> np.ndarray:
        m, d = self.m, self.d
        out, f = np.zeros(2), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            side = self.side.get(b1, self.side.get(b2))
            if side is not None:
                mujoco.mj_contactForce(m, d, i, f)
                out[side] += f[0]  # along the contact normal
        return out

    def read(self) -> Reading:
        d = self.d
        qpos = d.qpos.copy()
        return Reading(qpos=qpos, theta=qpos[7:], qd=d.qvel[6:].copy(), gyro=d.qvel[3:6].copy(), torque=self.torque,
                       foot_force=self._foot_force(), foot_pos=d.xpos[self.feet].copy(),
                       foot_rot=d.xmat[self.feet].reshape(2, 3, 3).copy())

    def step(self, target: np.ndarray, push: np.ndarray | None = None) -> None:
        """One policy step. push (3): change of the trunk's world velocity at its start [m/s]."""
        d = self.d
        if push is not None:
            d.qvel[:3] += push
        T = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))[None]  # noqa: E731
        filtered = safety_filter(self.model, self.firmware, T(target), self.last_target, T(d.qpos[7:]), T(d.qvel[6:]),
                                 self.kp, self.kd)
        self.last_target = filtered
        sent_now = filtered[0].numpy()
        for _ in range(self.substeps):
            self.ring.append(sent_now)
            sent, q, v = self.ring[0], d.qpos[7:], d.qvel[6:]
            tau = self.kp_np * (sent - q) - self.kd_np * v
            available = np.maximum((self.v_max - self.k * v * np.sign(tau)) * self.k / self.R, 0.0)
            cap = np.minimum(available, self.tau_max)
            d.ctrl[:] = np.clip(tau, -cap, cap)
            mujoco.mj_step(self.m, d)
            # this step's work: tau . (q_new - q_old) = h tau . v_new (semi-implicit Euler)
            p = d.ctrl * d.qvel[6:]
            cu = d.ctrl ** 2 * self.r_over_k2
            self.book += self.m.opt.timestep * np.array([(p + cu).sum(), cu.sum(), p.sum()])
        self.torque = d.ctrl.copy()

    def healthy(self) -> bool:
        return bool(np.isfinite(self.d.qpos).all() and np.isfinite(self.d.qvel).all())


class Walker:
    """The firmware's loop for one robot: ramp the twist, observe, run the policy, decode, step.  clock: the
    walk_clock agent (W5c), the gait clock's sin and cos after the 75 from a step counter, as its firmware draft."""

    def __init__(self, backend, policy, clock: bool = False):
        self.backend, self.policy, self.clock = backend, policy, clock
        from rl.walk.task import WalkParams

        self.clock_params = WalkParams()
        self.reset()

    def reset(self, pose: np.ndarray | None = None) -> None:
        self.backend.reset(pose)
        self.twist = np.zeros(3)
        self.last_action = np.zeros(C.NJ, dtype=np.float32)
        self.target = self.backend.home.copy() if pose is None else np.array(pose)
        self.reading = self.backend.read()
        self.fell = False
        self.clock_steps = 0                # policy steps since the gait clock started (walk_clock's start())

    def clock_inputs(self, obs: np.ndarray) -> np.ndarray:
        """walk_clock's two inputs for this step, from the twist the policy sees (obs 72..74), and the counter on."""
        p = self.clock_params
        on = float(np.hypot(obs[72], obs[73]) + abs(obs[74])) > p.clock_gate
        phase = (self.clock_steps * C.POLICY_DT / p.gait_period) % 1.0
        self.clock_steps = self.clock_steps + 1 if on else 0
        return np.array((np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)) if on else (0.0, 0.0), dtype=np.float32)

    def tick(self, want, push: np.ndarray | None = None) -> Reading:
        """One 20 ms policy step towards the requested twist `want` (3). Once fallen, the targets freeze."""
        r = self.reading
        if not self.fell:
            self.twist = C.ramp_twist(self.twist, want)
            obs = C.observe(r.gyro, C.projected_gravity(r.quat), r.theta, r.qd, self.last_action, self.twist)
            if self.clock:
                obs = np.concatenate((obs, self.clock_inputs(obs)))
            self.last_action = self.policy(obs)
            self.target = C.decode(self.last_action)
        self.backend.step(self.target, push)
        if self.backend.healthy():
            self.reading = self.backend.read()
            self.fell = self.fell or C.fallen(float(self.reading.pos[2]), self.reading.quat)
        else:
            self.fell = True
        return self.reading
