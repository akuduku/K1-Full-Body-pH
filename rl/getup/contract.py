r"""The get-up agent with a port-Hamiltonian controller ("standup_ph"): what the policy sees, and how its 66 outputs
become the motor driver's targets AND gains.

It is the firmware's StandupAgent (K1_Deploy firmware src/agents/sprint.cpp, the rl_core stand-up contract the
robot runs today: `k1_motion_test --task standup`) extended by the gains:

    sprint.cpp     kHome, kKp, kKd, kEffort; firmware joint order (no remap); raw gyro and joint velocity;
                   targets kHome + 0.25 * kEffort / kKp * a; no slew; StandupAgent: no tilt abort (it lies on the
                   floor), obs = the sprint's first 72 values, it never ends (gets up again after a fall)
    new            22 stiffness and 22 damping actions: kp = kKp * 2^(1.5 tanh a), kd = kKd * 2^(1.5 tanh a)
                   (x0.35 .. x2.8 of the firmware's gains, always > 0); the energy tank of rl/getup/impedance.py
                   pays for every energy injection of the controller; its level is the 117th input

Observation (117): gyro (3) | projected gravity (3) | q - kHome (22) | qdot (22) | last action (66) | tank (1),
joints in firmware order = MJCF order = the plant's theta.  The robot measures all of it (the tank is computed
from q, qdot and the commands).  The target clamp of apply_safety uses k1::effort_limit (the motors' limits, =
the MJCF forcerange): the new agent must not override Agent::effort() with sprint.cpp's larger kEffort (that one
only sets the action scale here).  What the firmware needs for this agent: rl/getup/README.md.
"""

from __future__ import annotations

import torch

from rl.stand.contract import JOINT_NAMES, check_joint_order  # noqa: F401  (the same MJCF / firmware order)

# src/agents/sprint.cpp
HOME = (0.0, 0.0,
        0.0, -1.4, 0.0, -0.4,
        0.0, 1.4, 0.0, 0.4,
        -0.4, 0.0, 0.0, 0.8, -0.4, 0.0,
        -0.4, 0.0, 0.0, 0.8, -0.4, 0.0)
KP_REF = (4.0, 4.0) + (10.0,) * 8 + (80.0, 80.0, 80.0, 80.0, 50.0, 50.0) * 2
KD_REF = (0.25, 0.25) + (1.0,) * 8 + (4.0, 4.0, 4.0, 4.0, 5.625, 3.007) * 2  # k1_skill_cfg (robot: 5.6 / 3.0)
EFFORT_DECODE = (6.0, 6.0) + (14.0,) * 8 + (68.0, 76.0, 38.3, 112.0, 38.3, 38.3) * 2
ACTION_SCALE = tuple(0.25 * e / k for e, k in zip(EFFORT_DECODE, KP_REF))
GAINS = "standup"      # ph_k1.firmware.GAIN_PRESETS key = (KP_REF, KD_REF)
SLEW = False           # SprintAgent::slew()
# the gains of the new agent
GAIN_RANGE = 1.5       # log2 half-range: x 2^-1.5 .. 2^1.5 = 0.35 .. 2.83
NJ = 22
NUM_ACTIONS = 3 * NJ   # targets | stiffness | damping
OBS_DIM = 3 + 3 + NJ + NJ + NUM_ACTIONS + 1
HIDDEN_DIMS = (512, 256, 128)
NORMALIZER_EPS = 0.01


class Contract:
    """The agent's observation and action, batched, on one device/dtype."""

    def __init__(self, dtype=torch.float32, device="cpu"):
        T = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
        self.home = T(HOME)
        self.kp_ref = T(KP_REF)
        self.kd_ref = T(KD_REF)
        self.action_scale = T(ACTION_SCALE)

    @staticmethod
    def projected_gravity(quat: torch.Tensor) -> torch.Tensor:
        """Unit gravity in the trunk frame, R^T (0, 0, -1), from the trunk quaternion (w, x, y, z)
        (lying on the back: (-1, 0, 0); on the front: (+1, 0, 0); upright: (0, 0, -1))."""
        w, x, y, z = quat.unbind(-1)
        return torch.stack((-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))), -1)

    def observe(self, gyro, gravity, theta, qd, last_action, tank_fraction, gains_locked: bool = False) -> torch.Tensor:
        """117 inputs: gyro | gravity | theta - HOME | qd | last action (66, raw policy output) | tank level / capacity.

        gains_locked: the 44 gain outputs read 0 in the last action (they act on nothing while gain_scale is 0).
        A policy trained with GetUpConfig.mask_locked_gains at gain_scale 0 must be observed this way, also on the
        robot; older policies (getup-mb-v1d and before) saw the raw outputs."""
        if gains_locked:
            last_action = torch.cat((last_action[:, :NJ], torch.zeros_like(last_action[:, NJ:])), -1)
        return torch.cat((gyro, gravity, theta - self.home, qd, last_action, tank_fraction[:, None]), -1)

    def decode(self, actions: torch.Tensor, gain_scale: float = 1.0):
        """(B,66) raw policy output -> joint targets, stiffness kp, damping kd, each (B,22) in firmware order.

        gain_scale in [0, 1] (the env's gain curriculum): 0 = the agent's fixed gains kKp / kKd whatever the gain
        outputs say (the stock StandupAgent), 1 = the full range x 2^(+-GAIN_RANGE).  A trained policy is decoded with
        the gain_scale it ended its training with (saved in the checkpoint's curriculum state)."""
        a_t, a_k, a_d = actions[:, :NJ], actions[:, NJ:2 * NJ], actions[:, 2 * NJ:]
        target = self.home + self.action_scale * a_t
        r = GAIN_RANGE * float(gain_scale)
        kp = self.kp_ref * torch.exp2(r * torch.tanh(a_k))
        kd = self.kd_ref * torch.exp2(r * torch.tanh(a_d))
        return target, kp, kd
