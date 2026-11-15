r"""The firmware's sprint agent: what a walking policy sees, and how its output and the twist command are used.

Copied from the robot's software, K1_Deploy firmware (the binary `k1_motion_test`) and k1_deploy/agents/sprint.py:

    src/agents/sprint.cpp   SprintAgent: kHome, kKp, kKd, kEffort; firmware joint order (no remap); raw gyro and joint
                            velocity (no observation scales); targets kHome + 0.25 * kEffort / kKp * a; no slew
    src/main.cpp            the limits of the twist (--max-vx, --max-back, --max-vy, --max-wz) and its ramp

Observation (75): gyro (3) | projected gravity (3) | q - HOME (22) | qdot (22) | last action (22) | twist (3).

The walks of k1_sprint/checkpoints (walk_v9, soccer-rolling-walk, sprint-repro-001, lean-B) follow this contract.
The command path after decode is ph_k1/firmware.py with the gain preset GAINS.
"""

from __future__ import annotations

import numpy as np

from rl.getup.contract import ACTION_SCALE as _ACTION_SCALE
from rl.getup.contract import GAINS, SLEW  # noqa: F401  (sprint.cpp: the same gains and no slew)
from rl.getup.contract import HOME as _HOME
from rl.stand.contract import TILT_ABORT, check_joint_order  # noqa: F401

NJ = 22
OBS_DIM = 75
HOME = np.array(_HOME, dtype=np.float64)
ACTION_SCALE = np.array(_ACTION_SCALE, dtype=np.float64)
POLICY_DT = 0.02  # k1_config.h kDt

# main.cpp defaults: --max-back, --max-vx | --max-vy | --max-wz
TWIST_LOW = np.array((-0.5, -1.0, -2.0))
TWIST_HIGH = np.array((2.5, 1.0, 2.0))
# main.cpp: step = {opt.accel * kDt, 2 * kDt, 4 * kDt}, with --accel 1.5
TWIST_RATE = np.array((1.5, 2.0, 4.0))

FALL_HEIGHT = 0.25  # m, trunk height below which the robot is down


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Unit gravity in the trunk frame, R^T (0, 0, -1), from the trunk quaternion (w, x, y, z)."""
    w, x, y, z = quat
    return np.array((-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))))


def roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    """(roll, pitch) of the trunk, ZYX Euler angles as the IMU reports them."""
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def observe(gyro, gravity, theta, qd, last_action, twist) -> np.ndarray:
    """The 75 inputs (sprint.py observe): gyro | gravity | theta - HOME | qd | last action | twist, as float32."""
    twist = np.clip(np.asarray(twist, dtype=np.float64), TWIST_LOW, TWIST_HIGH)
    return np.concatenate((gyro, gravity, theta - HOME, qd, last_action, twist)).astype(np.float32)


def decode(action: np.ndarray) -> np.ndarray:
    """Raw policy output (22) to joint targets (22) in firmware order."""
    return HOME + ACTION_SCALE * np.asarray(action, dtype=np.float64)


def ramp_twist(twist: np.ndarray, want: np.ndarray, dt: float = POLICY_DT) -> np.ndarray:
    """main.cpp: twist[i] += clamp(want[i] - twist[i], -step[i], step[i]), the requested twist limited first."""
    want = np.clip(np.asarray(want, dtype=np.float64), TWIST_LOW, TWIST_HIGH)
    step = TWIST_RATE * dt
    return twist + np.clip(want - twist, -step, step)


def fallen(trunk_z: float, quat: np.ndarray) -> bool:
    """The firmware's tilt abort, |roll| or |pitch| beyond TILT_ABORT, or the trunk near the floor."""
    roll, pitch = roll_pitch(quat)
    return trunk_z < FALL_HEIGHT or abs(roll) > TILT_ABORT or abs(pitch) > TILT_ABORT
