r"""Rigid-body part of the K1 Hamiltonian -- our own code, no MuJoCo.

    H_mech(q, p) = 1/2 p^T M(q)^-1 p + V_grav(q),        p = M(q) nu

M(q) is assembled body by body (pH.pdf, "Inertia Matrix"):

    M = sum_b J_b^T I_b J_b + diag(armature)

with J_b the 6x28 spatial Jacobian of body b and I_b its 6x6 spatial inertia,
both in the world frame about the origin.

The momentum equation is the port-Hamiltonian one,  p_dot = -dH/dq + ...,
written in the quasi-velocities nu (Boltzmann-Hamel form, needed because the
trunk orientation lives on SO(3), see Literatur/"PH_modeling_rigid_MBS"):

    p_dot[base]  = -nu_B x* p_B           (gyroscopic block J_gyro(p), skew)
                   + Ad^T W_ext            (gravity, contact on all bodies)
    p_dot[theta] = dT/dtheta|_nu           (= -dH/dtheta|_p for the kinetic part)
                   + S_k^T W_subtree(k)    (gravity, contact)
                   + motor + damping       (plant.py)

dT/dtheta is ANALYTIC (no finite differences, unlike the arm plant):

    dT/dtheta_k = - H_k . (S_k x_m V_parent(k)),
    H_k = sum of spatial momenta I_b V_b of all bodies at/after hinge k.

Derivation: dS_j/dtheta_k = S_k x_m S_j for k before j, and
dI_b/dtheta_k = S_k x* I_b - I_b S_k x_m; the two contributions cancel down
to the line above.  validation/checks.py compares it with autograd and with MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import NB, NJ, NV, K1Model
from .spatial import cross_motion, quat_to_mat, skew, spatial_inertia

GRAVITY = (0.0, 0.0, -9.81)


@dataclass
class Kinematics:
    R: torch.Tensor  # (B,nb,3,3) body orientation
    pos: torch.Tensor  # (B,nb,3) body origin
    com: torch.Tensor  # (B,nb,3)
    S: torch.Tensor  # (B,6,28) world motion axis of every dof
    J: torch.Tensor  # (B,nb,6,28) body Jacobians
    I: torch.Tensor  # (B,nb,6,6) spatial inertias about O
    M: torch.Tensor  # (B,28,28)


def _mv(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(B,3,3) @ (3,) or (B,3) -> (B,3)."""
    if v.dim() == 1:
        return R @ v
    return (R @ v[..., None]).squeeze(-1)


def _hinge_cache(model: K1Model) -> dict:
    """Constant per-hinge tensors, built once per model."""
    c = getattr(model, "_fk_cache", None)
    if c is None:
        K = skew(model.jnt_axis)  # (22,3,3)
        jb = model.jnt_body
        c = {
            "K": K,
            "K2": K @ K,
            "rot": model.body_rot[jb],  # (22,3,3)
            "pos": model.body_pos[jb],  # (22,3)
            "jpos": model.jnt_pos,
            "anchored": bool(model.jnt_pos.abs().max() > 0),
            "jb": torch.as_tensor(jb, device=model.device),
        }
        model._fk_cache = c  # type: ignore[attr-defined]
    return c


def body_frames(model: K1Model, q: torch.Tensor):
    """q (B,29) -> body orientations R (B,nb,3,3), origins P (B,nb,3) and centres of mass com (B,nb,3): the cheap
    part of forward_kinematics (no Jacobians, no M), e.g. for rewards."""
    B = q.shape[0]
    r, quat, theta = q[:, :3], q[:, 3:7], q[:, 7:]
    nb = model.nbody
    c = _hinge_cache(model)
    s_, c_ = torch.sin(theta)[..., None, None], torch.cos(theta)[..., None, None]
    Rh = torch.eye(3, dtype=q.dtype, device=q.device) + s_ * c["K"] + (1 - c_) * c["K2"]  # (B,22,3,3)
    R_loc = c["rot"] @ Rh
    P_loc = c["pos"].expand(B, NJ, 3)
    if c["anchored"]:
        jp = c["jpos"]
        P_loc = P_loc + (c["rot"] @ (jp - (Rh @ jp[..., None]).squeeze(-1))[..., None]).squeeze(-1)
    Rs: list[torch.Tensor] = [None] * nb  # type: ignore[list-item]
    Ps: list[torch.Tensor] = [None] * nb  # type: ignore[list-item]
    for b in range(nb):
        j = model.body_joint[b]
        if j == -2:  # trunk: free joint, pose = (r, Q)
            Rs[b], Ps[b] = quat_to_mat(quat), r
            continue
        pR, pP = Rs[model.parent[b]], Ps[model.parent[b]]
        if j >= 0:
            Rs[b] = pR @ R_loc[:, j]
            Ps[b] = pP + _mv(pR, P_loc[:, j])
        else:
            Rs[b] = pR @ model.body_rot[b]
            Ps[b] = pP + _mv(pR, model.body_pos[b])
    R = torch.stack(Rs, 1)
    P = torch.stack(Ps, 1)
    com = P + (R @ model.ipos[None, :, :, None]).squeeze(-1)
    return R, P, com


def com_height(model: K1Model, com: torch.Tensor) -> torch.Tensor:
    """Height of the whole robot's centre of mass (B,) from the body CoMs (B,nb,3): V_grav / (m g)."""
    return (model.mass[None, :] * com[..., 2]).sum(-1) / model.mass.sum()


def forward_kinematics(model: K1Model, q: torch.Tensor) -> Kinematics:
    """q (B,29) -> frames, Jacobians and M(q)."""
    B = q.shape[0]
    r, quat, theta = q[:, :3], q[:, 3:7], q[:, 7:]
    nb = model.nbody
    c = _hinge_cache(model)
    # all 22 hinge rotations at once (Rodrigues), then the local transforms
    s_, c_ = torch.sin(theta)[..., None, None], torch.cos(theta)[..., None, None]
    Rh = torch.eye(3, dtype=q.dtype, device=q.device) + s_ * c["K"] + (1 - c_) * c["K2"]  # (B,22,3,3)
    R_loc = c["rot"] @ Rh
    P_loc = c["pos"].expand(B, NJ, 3)
    if c["anchored"]:
        jp = c["jpos"]
        P_loc = P_loc + (c["rot"] @ (jp - (Rh @ jp[..., None]).squeeze(-1))[..., None]).squeeze(-1)

    Rs: list[torch.Tensor] = [None] * nb  # type: ignore[list-item]
    Ps: list[torch.Tensor] = [None] * nb  # type: ignore[list-item]
    for b in range(nb):
        j = model.body_joint[b]
        if j == -2:  # trunk: free joint, pose = (r, Q)
            Rs[b], Ps[b] = quat_to_mat(quat), r
            continue
        pR, pP = Rs[model.parent[b]], Ps[model.parent[b]]
        if j >= 0:
            Rs[b] = pR @ R_loc[:, j]
            Ps[b] = pP + _mv(pR, P_loc[:, j])
        else:
            Rs[b] = pR @ model.body_rot[b]
            Ps[b] = pP + _mv(pR, model.body_pos[b])
    R = torch.stack(Rs, 1)
    P = torch.stack(Ps, 1)

    # hinge motion axes S_k = (a, o x a), a = R_body axis, o = anchor
    Rj, Pj = R[:, c["jb"]], P[:, c["jb"]]
    a = (Rj @ model.jnt_axis[..., None]).squeeze(-1)
    o = Pj + (Rj @ c["jpos"][..., None]).squeeze(-1)
    S_joint = torch.cat((a, torch.cross(o, a, dim=-1)), -1)  # (B,22,6)

    # base columns: body twist (omega_B, v_B) -> world twist (Ad_T)
    Rt = R[:, 0]
    S_w = torch.cat((Rt, skew(r) @ Rt), 1)  # (B,6,3)
    S_v = torch.cat((torch.zeros_like(Rt), Rt), 1)
    S = torch.cat((S_w, S_v, S_joint.transpose(1, 2)), -1)  # (B,6,28)
    J = S[:, None] * model.support[None, :, None, :]

    com = P + (R @ model.ipos[None, :, :, None]).squeeze(-1)
    I_com = R @ model.I_body[None] @ R.transpose(-1, -2)
    I = spatial_inertia(model.mass.expand(B, nb), com, I_com)
    M = torch.einsum("bnid,bnij,bnje->bde", J, I, J)
    M = M.clone()
    idx = torch.arange(NB, NV, device=q.device)
    M[:, idx, idx] += model.armature
    return Kinematics(R=R, pos=P, com=com, S=S, J=J, I=I, M=M)


def body_twists(kin: Kinematics, nu: torch.Tensor) -> torch.Tensor:
    """(B,nb,6) world twists V_b = J_b nu."""
    return torch.einsum("bnid,bd->bni", kin.J, nu)


def kinetic_gradient(model: K1Model, kin: Kinematics, nu: torch.Tensor) -> torch.Tensor:
    """dT/dtheta at fixed nu, (B,22).  Equals -dH_kin/dtheta at fixed p."""
    V = body_twists(kin, nu)
    h = torch.einsum("bnij,bnj->bni", kin.I, V)  # spatial momenta
    Hk = torch.einsum("kn,bni->bki", model.subtree, h)
    Vpar = V[:, model.jnt_parent_body]  # (B,22,6)
    Sk = kin.S[:, :, NB:].transpose(1, 2)  # (B,22,6)
    return -(Hk * cross_motion(Sk, Vpar)).sum(-1)


def base_gyroscopic(p: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """-nu_B x* p_B for the trunk rows, (B,6).  Equals J_gyro(p) nu."""
    w, v = nu[:, :3], nu[:, 3:6]
    n, f = p[:, :3], p[:, 3:6]
    return torch.cat(
        (torch.cross(n, w, dim=-1) + torch.cross(f, v, dim=-1), torch.cross(f, w, dim=-1)), -1
    )


def gyro_matrix(p: torch.Tensor) -> torch.Tensor:
    """J_gyro(p) (B,6,6), skew-symmetric, with J_gyro(p) nu_B = -nu_B x* p_B."""
    n, f = skew(p[:, :3]), skew(p[:, 3:6])
    return torch.cat((torch.cat((n, f), -1), torch.cat((f, torch.zeros_like(f)), -1)), -2)


def gravity_wrenches(model: K1Model, kin: Kinematics) -> torch.Tensor:
    """(B,nb,6) world wrench of gravity on every body, about O."""
    g = torch.tensor(GRAVITY, dtype=kin.com.dtype, device=kin.com.device)
    f = (model.mass[None, :, None] * g).expand_as(kin.com)
    return torch.cat((torch.cross(kin.com, f, dim=-1), f), -1)


def generalized_force(kin: Kinematics, wrenches: torch.Tensor) -> torch.Tensor:
    """sum_b J_b^T W_b  (B,28)."""
    return torch.einsum("bnid,bni->bd", kin.J, wrenches)


def gravity_potential(model: K1Model, kin: Kinematics) -> torch.Tensor:
    g = torch.tensor(GRAVITY, dtype=kin.com.dtype, device=kin.com.device)
    return -(model.mass[None, :] * (kin.com @ g)).sum(-1)


def point_jacobian(kin: Kinematics, body: torch.Tensor, point_world: torch.Tensor) -> torch.Tensor:
    """Linear-velocity Jacobian of world points fixed to bodies: (B,nc,3,28)."""
    Jb = kin.J[:, body]  # (B,nc,6,28)
    return Jb[:, :, 3:] - skew(point_world) @ Jb[:, :, :3]
