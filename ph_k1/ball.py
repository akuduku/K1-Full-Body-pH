r"""A ball as port-Hamiltonian elements next to the robot: one free sphere per robot.

    state    ball = (x [3], u [3], omega [3])     centre, linear velocity, angular velocity (world frame)
    tanks    1/2 m u^2 + 1/2 I omega^2            kinetic (hollow sphere: I = 2/3 m r^2)
             m g z                                gravity
             1/2 k delta^2 per touched box        robot-ball springs
             1/2 k_g delta_g^2                    ball-ground spring

Elements:
  * robot-ball: the ball against the collision boxes of both feet and both shanks.  The closest point c of a box to
    the ball's centre, overlap delta = r - |x - c| > 0, normal n = (x - c) / |x - c| (robot -> ball):
        f = max(k delta - c_d g, 0),  g = n . (u - v_c)  (separation speed);  +f n on the ball, -f n on the robot at c.
    Unilateral (pushes, never pulls).  Its Coulomb friction, |f_t| <= mu_robot f, acts at the contact point on
    the ball (u and omega) and with the opposite sign on the robot.
  * ball-ground: the same normal element along z with (k_g, c_g).
  * ground friction: Coulomb at the ball's lowest point, |f_t| <= mu N, acting on u and omega (slip turns into
    rolling); rolling resistance: a torque mu_r N r against the roll.

Integration (ph_k1/plant.py step, ph_k1_cuda step.hpp): the two normal elements are stiff against the ball's small
mass, so they are solved implicitly together with the robot: the ball's 3 linear velocities are eliminated from the
robot's linear system (a Schur complement of a 3x3 block),

    [A + h c_e J^T J      -C  ] [nu_1]   [p + h F - h J^T k delta      ]         c_e = c_d + h k
    [     -C^T           A_bb ] [u_1 ] = [m u_0 + h (m g + n k delta ...)]        C = h c_e sum J_p^T n_p^T

with J_p = n_p^T J_c (1 x 28), A_bb = m 1 + h c_e sum n n^T + h c_ge z z^T.  Then the unilateral check and one
corrected solve, as for the robot's other contacts.  Friction and rolling resistance act on the ball alone: after
the solve, as an impulse that can only remove kinetic energy.  All losses go to the plant's contact ledger: per
step, the work of the robot on the ball's elements minus what the ball's tanks gained (so the damping that the
implicit step itself adds to a 10 ms impact of a 0.2 kg ball is booked too).

Off by default (K1PHPlant(ball=BallParams(enabled=True))): with it off the plant is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from .dynamics import GRAVITY, Kinematics, point_jacobian
from .model import K1Model, _quat_mat

BOX_BODIES = ("left_foot_link", "right_foot_link", "Left_Shank", "Right_Shank")
NBALL = 9  # x, u, omega
PARKED = (50.0, 0.0)  # where a ball lies that is not in play


@dataclass
class BallParams:
    radius: float = 0.075     # m   (k1_kick/assets/ball.xml)
    mass: float = 0.2         # kg  (hollow: I = 2/3 m r^2)
    k: float = 2.0e4          # N/m    robot-ball spring
    c: float = 60.0           # N s/m  robot-ball damper: ball speed of a leg swing within ~10 % of MuJoCo's twin ball
    k_ground: float = 2.0e4   # N/m
    c_ground: float = 45.0    # N s/m  (first bounce of a 0.4 m drop ~3 cm, as MuJoCo's twin ball)
    mu: float = 0.6           # ball-ground Coulomb friction
    mu_robot: float = 0.6     # robot-ball Coulomb friction (a rising foot lifts the ball)
    roll: float = 0.095       # rolling resistance: torque = roll * N * r (0.56 m/s^2, as MuJoCo's twin ball)
    enabled: bool = False

    @property
    def inertia(self) -> float:
        return 2.0 / 3.0 * self.mass * self.radius**2

    def rest_height(self, g: float = 9.81) -> float:
        """Centre height of a ball at rest on the ground (the ground spring carries its weight)."""
        return self.radius - self.mass * g / self.k_ground


@dataclass
class BallBoxes:
    body: torch.Tensor   # (nb,) body of every box
    pos: torch.Tensor    # (nb,3) centre in the body frame
    rot: torch.Tensor    # (nb,3,3) box axes in the body frame
    half: torch.Tensor   # (nb,3) half sizes

    @property
    def nbox(self) -> int:
        return int(self.body.shape[0])


def build_boxes(model: K1Model) -> BallBoxes:
    """The collision boxes of the feet and the shanks (cached on the model)."""
    cached = getattr(model, "_ball_boxes", None)
    if cached is not None:
        return cached
    m = model.mj
    body, pos, rot, half = [], [], [], []
    for name in BOX_BODIES:
        b = model.body_names.index(name)
        for g in range(m.ngeom):
            if (m.geom_bodyid[g] == b + 1 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                    and (m.geom_contype[g] or m.geom_conaffinity[g])):
                body.append(b)
                pos.append(np.array(m.geom_pos[g]))
                rot.append(_quat_mat(m.geom_quat[g]))
                half.append(np.array(m.geom_size[g]))
    T = lambda a: torch.as_tensor(np.stack(a), dtype=model.dtype, device=model.device)  # noqa: E731
    boxes = BallBoxes(torch.as_tensor(body, device=model.device), T(pos), T(rot), T(half))
    model._ball_boxes = boxes
    return boxes


@dataclass
class PairState:
    active: torch.Tensor  # (B,nb) bool: the ball overlaps the box
    delta: torch.Tensor   # (B,nb) overlap (0 where apart)
    n: torch.Tensor       # (B,nb,3) normal, robot -> ball
    point: torch.Tensor   # (B,nb,3) the box's closest point (world)


def pair_state(boxes: BallBoxes, kin: Kinematics, x: torch.Tensor, radius: float) -> PairState:
    """The ball (centre x, (B,3)) against every box at this configuration."""
    Rw = kin.R[:, boxes.body] @ boxes.rot[None]                                           # (B,nb,3,3)
    cw = kin.pos[:, boxes.body] + (kin.R[:, boxes.body] @ boxes.pos[None, :, :, None]).squeeze(-1)
    local = (Rw.transpose(-1, -2) @ (x[:, None, :] - cw)[..., None]).squeeze(-1)          # ball centre, box frame
    half = boxes.half[None]
    cl = torch.minimum(torch.maximum(local, -half), half)
    diff = local - cl
    d = torch.linalg.norm(diff, dim=-1)
    inside = d < 1e-9  # centre inside the box: out through the nearest face
    room = half - local.abs()
    axis = room.argmin(-1, keepdim=True)
    sign = torch.where(local.gather(-1, axis) >= 0, 1.0, -1.0).to(local.dtype)
    n_in = torch.zeros_like(local).scatter(-1, axis, sign)
    cl_in = local.scatter(-1, axis, sign * half.expand_as(local).gather(-1, axis))
    n_loc = torch.where(inside[..., None], n_in, diff / d.clamp_min(1e-9)[..., None])
    cl = torch.where(inside[..., None], cl_in, cl)
    delta = torch.where(inside, radius + room.gather(-1, axis).squeeze(-1), radius - d)
    active = delta > 0
    n = (Rw @ n_loc[..., None]).squeeze(-1)
    point = cw + (Rw @ cl[..., None]).squeeze(-1)
    return PairState(active, delta * active, n, point)


def jacobian(boxes: BallBoxes, kin: Kinematics, s: PairState) -> torch.Tensor:
    """J_p = n^T J_c (B,nb,28): n . (velocity of the robot's point)."""
    Jc = point_jacobian(kin, boxes.body, s.point)  # (B,nb,3,28)
    return torch.einsum("bpi,bpid->bpd", s.n, Jc)


def ground_overlap(p: BallParams, x: torch.Tensor) -> torch.Tensor:
    return (p.radius - x[:, 2]).clamp_min(0)


def gravity_vector(dtype, device) -> torch.Tensor:
    return torch.tensor(GRAVITY, dtype=dtype, device=device)


def energy(p: BallParams, boxes: BallBoxes, kin: Kinematics, ball: torch.Tensor, gravity: bool) -> torch.Tensor:
    """The ball's tanks: kinetic, gravity, both springs.  (B,)"""
    x, u, om = ball[:, 0:3], ball[:, 3:6], ball[:, 6:9]
    e = 0.5 * p.mass * (u * u).sum(-1) + 0.5 * p.inertia * (om * om).sum(-1)
    if gravity:
        e = e - p.mass * (x @ gravity_vector(ball.dtype, ball.device))
    e = e + 0.5 * p.k_ground * ground_overlap(p, x) ** 2
    s = pair_state(boxes, kin, x, p.radius)
    return e + 0.5 * p.k * (s.delta**2).sum(-1)


def friction(p: BallParams, u: torch.Tensor, om: torch.Tensor, normal: torch.Tensor, h: float):
    """Rolling resistance, then Coulomb friction at the ball's lowest point, for one step: (u, omega, energy lost).

    normal (B,): the ground's normal force (0 in the air).  Both are impulses that can only remove kinetic energy:
    the torque stops at omega = 0, the friction impulse stops at zero slip (|J| <= mu N h)."""
    m, I, r = p.mass, p.inertia, p.radius
    ke0 = 0.5 * m * (u * u).sum(-1) + 0.5 * I * (om * om).sum(-1)
    # rolling resistance on the horizontal spin
    oh = om[:, :2]
    on = torch.linalg.norm(oh, dim=-1)
    cut = torch.minimum(on, h * p.roll * normal * r / I)
    oh = oh - (cut / on.clamp_min(1e-12))[:, None] * oh
    om = torch.cat((oh, om[:, 2:3]), -1)
    # slip of the contact point r_c = (0, 0, -r):  s = u_t + (omega x r_c)_t = (u_x - r omega_y, u_y + r omega_x)
    s = torch.stack((u[:, 0] - r * om[:, 1], u[:, 1] + r * om[:, 0]), -1)
    sn = torch.linalg.norm(s, dim=-1)
    m_eff = 1.0 / (1.0 / m + r * r / I)
    Jn = torch.minimum(sn * m_eff, p.mu * normal * h)
    J = -(Jn / sn.clamp_min(1e-12))[:, None] * s                       # (B,2) impulse on the ball at the contact
    u = torch.cat((u[:, :2] + J / m, u[:, 2:3]), -1)
    # d omega = (r_c x J) / I = (r J_y, -r J_x, 0) / I
    om = torch.stack((om[:, 0] + r * J[:, 1] / I, om[:, 1] - r * J[:, 0] / I, om[:, 2]), -1)
    ke1 = 0.5 * m * (u * u).sum(-1) + 0.5 * I * (om * om).sum(-1)
    return u, om, ke0 - ke1


def pair_friction(p: BallParams, u, om, n, normal, vc, h: float):
    """Coulomb friction of every touched box on the ball for one step: (u, omega, J, energy lost).

    n (B,nb,3), normal (B,nb) the boxes' normal forces, vc (B,nb,3) the velocities of the robot's contact points.
    Per box an impulse J on the ball at its contact point -r n, against the slip there, at most mu_robot N h and at
    most what stops the slip; -J goes to the robot (plant.step adds J_c^T (-J) to its momentum).  Lost energy:
    what the ball lost minus what the robot got (J . v_c)."""
    m, I, r = p.mass, p.inertia, p.radius
    m_eff = 1.0 / (1.0 / m + r * r / I)
    lost = torch.zeros_like(normal[:, 0])
    Js = []
    for k in range(n.shape[1]):
        nk, rc = n[:, k], -r * n[:, k]
        rel = u + torch.linalg.cross(om, rc) - vc[:, k]
        rel_t = rel - (rel * nk).sum(-1, keepdim=True) * nk
        sn = torch.linalg.norm(rel_t, dim=-1)
        Jn = torch.minimum(sn * m_eff, p.mu_robot * normal[:, k] * h)
        J = -(Jn / sn.clamp_min(1e-12))[:, None] * rel_t
        ke0 = 0.5 * m * (u * u).sum(-1) + 0.5 * I * (om * om).sum(-1)
        u = u + J / m
        om = om + torch.linalg.cross(rc, J) / I
        ke1 = 0.5 * m * (u * u).sum(-1) + 0.5 * I * (om * om).sum(-1)
        lost = lost + ke0 - ke1 + (J * vc[:, k]).sum(-1)
        Js.append(J)
    return u, om, torch.stack(Js, 1), lost


def rest_state(p: BallParams, n: int, dtype, device, xy=PARKED) -> torch.Tensor:
    """(n, 9): a ball at rest on the ground at xy."""
    b = torch.zeros(n, NBALL, dtype=dtype, device=device)
    b[:, 0], b[:, 1] = xy
    b[:, 2] = p.rest_height()
    return b
