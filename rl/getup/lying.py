r"""Start states for the get-up task: a library of lying poses that the pH physics settled (as RL_Core's
K1StandUpEnv does it): every robot is dropped from a noisy lying pose (on the back, the front or a side), the
pose is held 0.4 s at the standup gains, then the driver goes compliant (target = the current pose) until the
robot lies still.  The settled states are kept; a reset picks one and turns it to a random heading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ph_k1.dynamics import body_frames, com_height
from ph_k1.firmware import FirmwareConfig
from ph_k1.model import NJ
from ph_k1.native import K1PHPlantNative
from ph_k1.spatial import quat_mul

STANDING, SUPINE, PRONE, SIDE = 0, 1, 2, 3
KIND_NAMES = {STANDING: "standing", SUPINE: "supine", PRONE: "prone", SIDE: "side"}
KINDS = {v: k for k, v in KIND_NAMES.items()}
# (roll, pitch) of the drop: pitch +pi/2 turns the trunk's x axis (forward) to -z: face down
ROLL_PITCH = {SUPINE: (0.0, -math.pi / 2), PRONE: (0.0, math.pi / 2), SIDE: (math.pi / 2, 0.0)}


def quat_from_rpy(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack((cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
                        cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr), -1)


def turn(quat: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """The same orientation turned about the world's vertical axis by yaw."""
    z = torch.zeros_like(yaw)
    return quat_mul(quat_from_rpy(z, z, yaw), quat)


@dataclass
class LyingLibrary:
    q: torch.Tensor      # (L, NQ) settled states: trunk position, quaternion, 22 joint angles
    kind: torch.Tensor   # (L,) SUPINE / PRONE / SIDE (the kind of the drop)
    com_z: torch.Tensor  # (L,) centre-of-mass height [m]

    def index(self, kind: int) -> torch.Tensor:
        return (self.kind == kind).nonzero(as_tuple=False).squeeze(-1)


def build_library(model, n: int, kinds, home: torch.Tensor, *, rounds: int = 1, joint_noise: float = 0.35,
                  tilt: float = 0.35, drop_z: float = 0.45, hold_s: float = 0.4, settle_s: float = 1.6,
                  policy_dt: float = 0.02, plant_dt: float = 1e-3, generator: torch.Generator | None = None,
                  log=print) -> LyingLibrary:
    """`rounds` x n drops, the lying kinds in turn; joint noise in units of a third of each joint's range."""
    dev, dt = model.device, model.dtype
    fw = FirmwareConfig(gains="standup", slew=False, delay_ms=(0.0, 0.0), policy_dt=policy_dt)
    plant = K1PHPlantNative(model=model, batch=n, timestep=plant_dt, firmware=fw)
    third = (model.jnt_high - model.jnt_low) / 3.0
    kinds_t = torch.as_tensor([KINDS[k] if isinstance(k, str) else int(k) for k in kinds], device=dev)
    rnd = lambda *s: torch.rand(*s, generator=generator, dtype=dt, device=dev)  # noqa: E731
    qs, ks = [], []
    hold, settle = round(hold_s / policy_dt), round(settle_s / policy_dt)
    for r in range(rounds):
        kind = kinds_t[torch.arange(n, device=dev) % len(kinds_t)]
        q0 = home + torch.randn(n, NJ, generator=generator, dtype=dt, device=dev) * joint_noise * third
        q0 = torch.minimum(torch.maximum(q0, model.jnt_low), model.jnt_high)
        roll, pitch = torch.zeros(n, dtype=dt, device=dev), torch.zeros(n, dtype=dt, device=dev)
        for k, (r0, p0) in ROLL_PITCH.items():
            roll[kind == k], pitch[kind == k] = r0, p0
        side = torch.where(rnd(n) < 0.5, -1.0, 1.0).to(dt)
        roll = torch.where(kind == SIDE, roll * side, roll)
        quat = quat_from_rpy(roll + tilt * (2 * rnd(n) - 1), pitch + tilt * (2 * rnd(n) - 1), torch.zeros_like(roll))
        plant.reset(pose=q0, base_pos=(0.0, 0.0, drop_z), base_quat=quat)
        for _ in range(hold):
            info = plant.policy_step(q0)
        for _ in range(settle):  # compliant: the driver holds where the joints are
            info = plant.policy_step(plant.theta.clone())
        q = plant.q.clone()
        speed = info.extra["nu"][:, 3:6].norm(dim=-1)
        ok = torch.isfinite(q).all(-1) & (q[:, 2] < 0.3) & (speed < 0.2)
        qs.append(q[ok])
        ks.append(kind[ok])
        log(f"[getup] lying library round {r + 1}: kept {int(ok.sum())}/{n} settled poses")
    q = torch.cat(qs)
    kind = torch.cat(ks)
    com_z = com_height(model, body_frames(model, q)[2])
    lib = LyingLibrary(q=q, kind=kind, com_z=com_z)
    for k in kinds_t.tolist():
        idx = lib.index(k)
        if len(idx) == 0:
            raise RuntimeError(f"no settled {KIND_NAMES[k]} pose; check the drop settings")
        z = q[idx, 2]
        log(f"[getup] {KIND_NAMES[k]}: {len(idx)} poses, trunk z {float(z.min()):.3f}-{float(z.max()):.3f} m, "
            f"CoM z {float(com_z[idx].mean()):.3f} m")
    return lib
