"""Spatial-vector and quaternion arithmetic (batched, PyTorch).

Conventions (Featherstone, everything in the WORLD frame, about the world
origin O):

    motion vector (twist)   V = (omega, v_O)      v_O = velocity of the body
                                                  point that is at O right now
    force vector  (wrench)  W = (n_O, f)          n_O = moment about O
    pairing                 W . V = n.omega + f.v  [W]

A hinge with world axis a through world point o has the motion axis
S = (a, o x a).  Quaternions are MuJoCo's (w, x, y, z).
"""

from __future__ import annotations

import torch


def skew(v: torch.Tensor) -> torch.Tensor:
    """[v]x : (...,3) -> (...,3,3) with skew(a) @ b = a x b."""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((o, -z, y), -1),
            torch.stack((z, o, -x), -1),
            torch.stack((-y, x, o), -1),
        ),
        -2,
    )


def cross_motion(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a x_m b for twists (...,6): (w1 x w2, w1 x v2 + v1 x w2)."""
    w1, v1 = a[..., :3], a[..., 3:]
    w2, v2 = b[..., :3], b[..., 3:]
    return torch.cat(
        (torch.cross(w1, w2, dim=-1), torch.cross(w1, v2, dim=-1) + torch.cross(v1, w2, dim=-1)),
        -1,
    )


def cross_force(a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """a x* f for a twist a and a wrench f: (w x n + v x F, w x F)."""
    w, v = a[..., :3], a[..., 3:]
    n, F = f[..., :3], f[..., 3:]
    return torch.cat(
        (torch.cross(w, n, dim=-1) + torch.cross(v, F, dim=-1), torch.cross(w, F, dim=-1)), -1
    )


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(...,4) unit quaternion (w,x,y,z) -> (...,3,3) rotation matrix."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        -2,
    )


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a (x) b."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def quat_exp(rotvec: torch.Tensor) -> torch.Tensor:
    """Unit quaternion of the rotation vector (...,3) (exponential map)."""
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(half)/angle, with its limit 1/2 at angle -> 0
    k = torch.where(angle > 1e-12, torch.sin(half) / angle.clamp_min(1e-300), 0.5 - angle**2 / 48.0)
    return torch.cat((torch.cos(half), k * rotvec), -1)


def quat_rate_matrix(q: torch.Tensor) -> torch.Tensor:
    """E(q) (...,4,3) with  q_dot = E(q) omega_body  ( = 1/2 q (x) (0, omega) )."""
    w, x, y, z = q.unbind(-1)
    return 0.5 * torch.stack(
        (
            torch.stack((-x, -y, -z), -1),
            torch.stack((w, -z, y), -1),
            torch.stack((z, w, -x), -1),
            torch.stack((-y, x, w), -1),
        ),
        -2,
    )


def axis_angle_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: fixed unit axis (3,), angles (B,) -> (B,3,3)."""
    K = skew(axis)
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    s = torch.sin(angle)[:, None, None]
    c = torch.cos(angle)[:, None, None]
    return eye + s * K + (1 - c) * (K @ K)


def spatial_inertia(mass: torch.Tensor, com: torch.Tensor, I_com: torch.Tensor) -> torch.Tensor:
    """6x6 spatial inertia about the world origin (Featherstone order).

    mass (...,), com (...,3) world COM, I_com (...,3,3) world rotational
    inertia about the COM.  h = I V gives (angular momentum about O,
    linear momentum).
    """
    C = skew(com)
    m = mass[..., None, None]
    eye = torch.eye(3, dtype=com.dtype, device=com.device).expand_as(I_com)
    top = torch.cat((I_com - m * (C @ C), m * C), -1)
    bot = torch.cat((m * C.transpose(-1, -2), m * eye), -1)
    return torch.cat((top, bot), -2)
