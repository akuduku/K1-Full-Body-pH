"""Classical servo: target angles -> VOLTS for the pH plant.  Not RL.

Used to command the motors by hand while we validate the physics (and later
as a baseline / teacher for RL).  It is the arm's drive.py for 22 joints:

    tau*  = kp (q* - q) - kd q_dot              joint PD (K1 firmware style)
    tau*  clipped to what the motor can give at this speed
    i*    = tau* / k
    v     = L (i* - i)/h + R i* + s k q_dot     one-step current loop,
                                                matched to the plant's
                                                implicit electrical row
    v     clipped to +-V_NOM

Gains: rl_core "contract" profile (plant/motors.py): legs 80/4, ankles
50/(5.6 pitch, 3.0 roll), arms 10/1, head 4/0.25.

This servo is for the physics checks (validation/checks.py, examples/hold_pose.py).  It is NOT the
robot's command path: the robot filters a policy's target once per 20 ms
policy step (per-joint slew, target clamp, overspeed guard), holds it, and
delays it before the motor driver -- see ph_k1/firmware.py, which RL uses.
"""

from __future__ import annotations

import torch

from .model import NJ
from .motors import motor_type
from .plant import K1PHPlant

GAINS = {  # (kp N m/rad, kd N m s/rad)
    "head": (4.0, 0.25),
    "arm": (10.0, 1.0),
    "leg": (80.0, 4.0),
    "ankle_pitch": (50.0, 5.625),
    "ankle_roll": (50.0, 3.007),
}
SLEW = 5.0  # rad/s, firmware max_target_velocity (legs)


def _group(name: str) -> str:
    n = name.lower()
    if "ankle_pitch" in n:
        return "ankle_pitch"
    if "ankle_roll" in n:
        return "ankle_roll"
    t = motor_type(name)
    return {"HT4438": "head", "R14": "arm"}.get(t, "leg")


class K1Drive:
    def __init__(self, plant: K1PHPlant, gain_scale: float = 1.0):
        self.plant = plant
        m = plant.model
        kp, kd = zip(*(GAINS[_group(n)] for n in m.joint_names))
        T = lambda a: torch.as_tensor(a, dtype=m.dtype, device=m.device)  # noqa: E731
        self.kp = gain_scale * T(kp)
        self.kd = gain_scale * T(kd)
        self.q_star: torch.Tensor | None = None

    def reset(self) -> None:
        self.q_star = self.plant.theta.clone()

    def volts(self, q_command: torch.Tensor) -> torch.Tensor:
        pl, mot = self.plant, self.plant.motors
        if self.q_star is None:
            self.reset()
        m = pl.model
        cmd = torch.minimum(torch.maximum(torch.as_tensor(q_command, dtype=m.dtype), m.jnt_low), m.jnt_high)
        step = SLEW * pl.h
        self.q_star = self.q_star + (cmd - self.q_star).clamp(-step, step)

        q = pl.theta
        qd = pl.velocity()[:, -NJ:]
        i = pl.current
        tau = self.kp * (self.q_star - q) - self.kd * qd
        # what the motor can deliver at this speed with full voltage
        avail = ((mot.v_max - mot.k * qd * tau.sign()) * mot.k / mot.R).clamp_min(0)
        cap = torch.minimum(avail, mot.tau_max)
        tau = torch.maximum(torch.minimum(tau, cap), -cap)
        i_star = tau / mot.k
        s = mot.gyrator_scale(i)
        v = mot.L * (i_star - i) / pl.h + mot.R * i_star + s * mot.k * qd
        return v.clamp(-mot.v_max, mot.v_max)
