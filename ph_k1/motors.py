"""22 DC motors, one per hinge -- the electrical half of the pH system.

    L dI/dt = v - R i - s k theta_dot        (flux row:  lambda = L i)
    tau     = s k i                          (gyrator, same s both ways)

Every motor is the equivalent DC motor whose full-voltage torque-speed line
passes through the corners of k1_skill_cfg's torque-speed curve (servo.py):

    effort E up to the knee speed w_k, then a straight line to 0 at w_max.
    unclipped stall  tau0 = E w_max / (w_max - w_k)
    k  = V_NOM / w_max            [N m / A = V s / rad]
    R  = k V_NOM / tau0           [ohm]
    L  = ELECTRICAL_TIME * R      [H]

This is exactly the derivation of the arm plant (Right Hand Model/plant.py),
so the four right-arm motors here are identical to the arm's R14 motors.

The torque ceiling s = min(1, tau_max / |k i|) is k1_skill_cfg's effort
(servo.EFFORT: knee 112, hip pitch 68, hip roll 76, hip yaw / ankles 38.3,
arms 14, head 6 N m), not the forcerange of our MJCF (knee 40, hip pitch 30).
Passive joint damping as k1_skill_cfg: 0.2 on the ankles, 0 elsewhere.

PLACEHOLDERS (ask Booster for real values): V_NOM, ELECTRICAL_TIME.  Volts/amperes are this model's internal units; what matches
the robot is torque against joint speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import servo
from .model import K1Model

V_NOM = 7.0  # V, full-scale command (same as the arm plant)
ELECTRICAL_TIME = 0.05  # s, L/R (same as the arm plant)
JOINT_DAMPING = servo.JOINT_DAMPING  # N m s/rad passive damping of the non-ankle joints (k1_skill_cfg: none)

ANKLE_DAMPING = servo.ANKLE_PASSIVE_DAMPING  # N m s/rad passive ankle damping (k1_skill_cfg)

# (effort N m, no-load speed rad/s, knee speed rad/s): k1_skill_cfg's torque-speed curve
DATASHEET = dict(servo.MOTOR_CURVE)


def motor_type(joint_name: str) -> str:
    n = joint_name.lower()
    if "head" in n:
        return "HT4438"
    if "shoulder" in n or "elbow" in n:
        return "R14"
    if "hip_pitch" in n:
        return "E6408"
    if "hip_roll" in n:
        return "E4315"
    if "hip_yaw" in n or "ankle" in n:
        return "E4310"
    if "knee" in n:
        return "E6416"
    raise KeyError(joint_name)


@dataclass
class MotorBank:
    k: torch.Tensor  # (22,)
    R: torch.Tensor
    L: torch.Tensor
    tau_max: torch.Tensor
    damping: torch.Tensor
    v_max: float
    types: list[str]

    def gyrator_scale(self, i: torch.Tensor) -> torch.Tensor:
        """s in (0,1]: torque ceiling that keeps tau.theta_dot = e.i exact."""
        ki = (self.k * i).abs()
        return torch.where(ki > self.tau_max, self.tau_max / ki.clamp_min(1e-300), torch.ones_like(ki))

    def torque(self, i: torch.Tensor) -> torch.Tensor:
        return self.gyrator_scale(i) * self.k * i


def build_motors(model: K1Model, v_nom: float = V_NOM, electrical_time: float = ELECTRICAL_TIME,
                 joint_damping: float = JOINT_DAMPING) -> MotorBank:
    ks, Rs, types = [], [], []
    for name in model.joint_names:
        t = motor_type(name)
        effort, w_max, w_knee = DATASHEET[t]
        # HT4438: its knee is at no-load speed, so the envelope is just
        # the straight DC line from `effort` at stall to 0 at w_max.
        stall = effort * w_max / (w_max - w_knee) if w_knee < w_max else effort
        k = v_nom / w_max
        ks.append(k)
        Rs.append(k * v_nom / stall)
        types.append(t)
    T = lambda a: torch.as_tensor(a, dtype=model.dtype, device=model.device)  # noqa: E731
    k, R = T(ks), T(Rs)
    return MotorBank(
        k=k,
        R=R,
        L=electrical_time * R,
        tau_max=model.effort.clone(),
        damping=T([ANKLE_DAMPING if "ankle" in n.lower() else joint_damping for n in model.joint_names]),
        v_max=v_nom,
        types=types,
    )
