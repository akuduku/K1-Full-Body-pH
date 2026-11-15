r"""The walk-and-kick agent ("kick_walk"): what the policy sees and how its 22 outputs are used.

It is the agent Eric Plass's k1_kick trains with `--variant speed --camera --walk` (kick-014 and later; on the robot
`k1_motion_test --task kick_walk`), a member of the sprint family (K1_Deploy firmware src/agents/sprint.cpp):

    sprint.cpp   kHome, kKp, kKd, kEffort; firmware joint order; raw gyro and joint velocity;
                 targets kHome + 0.25 * kEffort / kKp * a; no slew; the firmware clamps the targets to its range
    k1_kick      env.py get_observations / actor_layout: the actor's 165 inputs

Observation (165), joints in firmware order = MJCF order = the plant's theta:

    frame    72  gyro (3) | projected gravity (3) | q - kHome (22) | qdot (22) | last action (22)
    command   3  vx, vy [m/s] in the heading frame, yaw rate [rad/s]; zero during a request
    request   1  1 while a kick request runs, else 0.  Everything below is zero while no request runs.
    ball      6  the camera's held report of the ball relative to the trunk (x, y) and its age [s, at most 1];
                 the velocity fitted through the reports of the last 0.3 s (x, y, 0)
    target    2  the kick direction in the heading frame (cos, sin)
    speed     1  (requested ball speed - 3) / 2
    reports  80  the last 40 held reports (x, y pairs, oldest first)

With `phase` a 166th input follows: the seconds since the request started (0 while walking).  The robot's firmware
does not have it yet; it is the clock a policy needs to follow a planned motion (Model Based RL mbrl/kick).
"""

from __future__ import annotations

import torch

from rl.getup.contract import ACTION_SCALE, GAINS, HOME, SLEW  # noqa: F401  (sprint.cpp: kHome, kKp/kKd, no slew)
from rl.stand.contract import JOINT_NAMES, check_joint_order  # noqa: F401

NJ = 22
FRAME = 72
NUM_ACTIONS = NJ
REPORTS = 40
HIDDEN_DIMS = (512, 256, 128)
WALK_DIM = FRAME + 3  # the deployed walk's inputs: the frame and the command


def layout(phase: bool = False) -> tuple[dict, int]:
    """The actor's input blocks, name -> (first column, width), and the width (k1_kick env.py actor_layout with
    one frame, the speed request, the camera and the walk)."""
    blocks, at = {}, 0
    for name, width in (("frame", FRAME), ("command", 3), ("request", 1), ("ball", 6), ("target", 2), ("speed", 1),
                        ("reports", 2 * REPORTS), ("phase", int(phase))):
        blocks[name] = (at, width)
        at += width
    return blocks, at


OBS_DIM = layout()[1]  # 165


# ------------------------------------------------------------------ frames (rl_core/envs/k1_skill.py, the same)
def rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors (..., 3) by quaternions (..., 4), w x y z."""
    w, xyz = quat[..., :1], quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


def rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return rotate(quat * quat.new_tensor([1.0, -1.0, -1.0, -1.0]), vec)


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def to_heading(yaw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """World x-y(-z) vectors (N, ..., 2|3) into the heading frame of each robot (yaw (N,)), i.e. turned by -yaw."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    shape = (-1,) + (1,) * (vec.dim() - 2)
    c, s = c.view(shape), s.view(shape)
    x, y = vec[..., 0], vec[..., 1]
    out = vec.clone()
    out[..., 0], out[..., 1] = c * x + s * y, -s * x + c * y
    return out


def wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def direction_error(velocity: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Angle [rad] between the planar velocity and the target direction."""
    planar = velocity[:, :2]
    cos = (planar * target[:, :2]).sum(dim=1) / torch.linalg.vector_norm(planar, dim=1).clamp(min=1e-6)
    return torch.acos(cos.clamp(-1.0, 1.0))


class Contract:
    """The agent's observation and action, batched, on one device/dtype."""

    def __init__(self, dtype=torch.float32, device="cpu", phase: bool = False):
        T = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
        self.home = T(HOME)
        self.action_scale = T(ACTION_SCALE)
        self.phase = phase
        self.blocks, self.obs_dim = layout(phase)

    @staticmethod
    def projected_gravity(quat: torch.Tensor) -> torch.Tensor:
        """Unit gravity in the trunk frame, R^T (0, 0, -1), from the trunk quaternion (w, x, y, z)."""
        w, x, y, z = quat.unbind(-1)
        return torch.stack((-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))), -1)

    def frame(self, gyro, gravity, theta, qd, last_action) -> torch.Tensor:
        """The sprint's first 72 inputs."""
        return torch.cat((gyro, gravity, theta - self.home, qd, last_action), -1)

    def observe(self, frame, command, request, seen, age, velocity, target, speed, reports, phase_s=None) -> torch.Tensor:
        """The 165 (166) inputs.  request (B,) 0/1; seen (B,2) the held report; age (B,) [s]; velocity (B,2) the
        fitted ball velocity; target (B,2) cos/sin in the heading frame; speed (B,) the requested ball speed [m/s];
        reports (B,40,2); phase_s (B,) seconds since the request.  The request's blocks are zeroed without one."""
        on = request[:, None]
        zero = torch.zeros_like(on)
        parts = [frame, command, on, seen * on, age.clamp(max=1.0)[:, None] * on, velocity * on, zero, target * on,
                 (speed[:, None] - 3.0) / 2.0 * on, reports.flatten(1) * on]
        if self.phase:
            parts.append(phase_s[:, None] * on)
        return torch.cat(parts, -1)

    def decode(self, actions: torch.Tensor) -> torch.Tensor:
        """(B,22) raw policy output -> joint targets kHome + 0.25 * kEffort / kKp * a (the firmware's range clamp
        follows in the plant's command path)."""
        return self.home + self.action_scale * actions
