r"""The firmware's stand agent: what a stand policy sees, and how its output is used on the real K1.

Copied from the robot's software, K1_Deploy firmware (the binary `k1_motion_test`):

    src/agents/stand.cpp   obs_dim 72; kp/kd = k1::stiffness / k1::damping; start pose default_qpos;
                           slew on; abort when |roll| or |pitch| > 0.6 rad (agent.h tilt_abort, main.cpp)
    include/obs.h          fill_proprio (the observation) and decode_stand (the action)
    include/k1_config.h    default_qpos, action_scale, real2sim / sim2real, the observation scales
    src/policy_mlp.cpp     the network: 512-256-128 ELU MLP, (x - mean) / (std + 0.01) in front

A policy trained against this module runs on the robot as  k1_motion_test --task stand --policy <file>.bin
(file from rl/stand/export.py).  The command path after decode (clamp, slew, target clamp, delay, motor driver) is
ph_k1/firmware.py.

Joint orders.  "real" = firmware order = MJCF order = the plant's theta.  The network sees and outputs the
"sim" order: observation joint k is real joint REAL2SIM[k]; action entry SIM2REAL[r] drives real joint r.
"""

from __future__ import annotations

import torch

# MJCF / firmware joint order (K1_Deploy k1_config.h "firmware joint order")
JOINT_NAMES = (
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
# k1_config.h
DEFAULT_QPOS = (0.0, 0.0, 0.0, -1.3, 0.0, 0.0, 0.0, 1.3, 0.0, 0.0) + (0.0,) * 12
ACTION_SCALE = (0.379954, 0.379954, 0.886570, 0.886570, 0.886570, 0.886570, 0.886570, 0.886570, 0.886570, 0.886570,
                0.562895, 0.885865, 0.536534, 0.463561, 0.268267, 0.268267,
                0.562895, 0.885865, 0.536534, 0.463561, 0.268267, 0.268267)
REAL2SIM = (0, 2, 6, 10, 16, 1, 3, 7, 11, 17, 4, 8, 12, 18, 5, 9, 13, 19, 14, 20, 15, 21)
SIM2REAL = (0, 5, 1, 6, 10, 14, 2, 7, 11, 15, 3, 8, 12, 16, 18, 20, 4, 9, 13, 17, 19, 21)
ANG_VEL_SCALE = 0.25   # kAngVelScale
DOF_POS_SCALE = 1.0    # kDofPosScale
DOF_VEL_SCALE = 0.05   # kDofVelScale
NORMALIZER_EPS = 0.01  # kNormalizerEps (policy_mlp.cpp)
# agent.h / stand.cpp
TILT_ABORT = 0.6       # rad, |roll| or |pitch| beyond it: the firmware stops the policy (fall)
GAINS = "firmware"     # ph_k1.firmware.GAIN_PRESETS key: k1::stiffness / k1::damping
SLEW = True            # Agent::slew() default
ACTION_SCALE_CLI = 1.0  # k1_motion_test --scale default
# policy_mlp.cpp
HIDDEN_DIMS = (512, 256, 128)
OBS_DIM = 72
NUM_ACTIONS = 22


def check_joint_order(joint_names) -> None:
    if tuple(joint_names) != JOINT_NAMES:
        raise ValueError(f"the MJCF joint order differs from the firmware's: {list(joint_names)}")
    if sorted(REAL2SIM) != list(range(22)) or any(SIM2REAL[REAL2SIM[k]] != k for k in range(22)):
        raise ValueError("real2sim / sim2real are not inverse permutations")


class Contract:
    """The stand agent's observation and action, batched, on one device/dtype."""

    def __init__(self, dtype=torch.float32, device="cpu"):
        T = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
        self.default_qpos = T(DEFAULT_QPOS)
        self.action_scale = T(ACTION_SCALE)
        self.real2sim = torch.as_tensor(REAL2SIM, device=device)
        self.sim2real = torch.as_tensor(SIM2REAL, device=device)

    @staticmethod
    def projected_gravity(quat: torch.Tensor) -> torch.Tensor:
        """Unit gravity in the trunk frame, R^T (0, 0, -1), from the trunk quaternion (w, x, y, z).
        The firmware computes the same vector from the IMU's roll/pitch/yaw (math.h projected_gravity)."""
        w, x, y, z = quat.unbind(-1)
        return torch.stack((-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))), -1)

    @staticmethod
    def roll_pitch(quat: torch.Tensor) -> torch.Tensor:
        """(roll, pitch) of the trunk, ZYX Euler angles as the IMU reports them (the firmware's tilt abort)."""
        w, x, y, z = quat.unbind(-1)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
        return torch.stack((roll, pitch), -1)

    def observe(self, theta, qd, gyro, quat, last_action, gravity=None) -> torch.Tensor:
        """obs.h fill_proprio: gyro*0.25 | projected gravity | (q - default)[real2sim] | qd*0.05 [real2sim] | last action.
        theta, qd (B,22) real order; gyro (B,3) trunk frame; quat (B,4); last_action (B,22) sim order (raw policy output)."""
        g = self.projected_gravity(quat) if gravity is None else gravity
        return torch.cat((gyro * ANG_VEL_SCALE, g,
                          (theta - self.default_qpos)[:, self.real2sim] * DOF_POS_SCALE,
                          qd[:, self.real2sim] * DOF_VEL_SCALE,
                          last_action), -1)

    def decode(self, actions: torch.Tensor, scale: float = ACTION_SCALE_CLI) -> torch.Tensor:
        """obs.h decode_stand: target[r] = default_qpos[r] + scale * action_scale[r] * a[sim2real[r]] (real order)."""
        return self.default_qpos + scale * self.action_scale * actions[:, self.sim2real]
