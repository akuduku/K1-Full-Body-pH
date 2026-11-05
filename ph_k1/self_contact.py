r"""Self-collision as port-Hamiltonian elements: the robot's own limbs cannot pass through each other.

Every body with collision geometry in the MJCF gets a chain of spheres along the longest side of the box that
encloses its collision geoms (the same box as its ground-contact corners, ph_k1/model.py).  Two spheres of
different bodies with overlap delta = r_a + r_b - |x_a - x_b| > 0 are joined by one element:

    normal tank    V_s = 1/2 k delta^2                  (energy storage, in H: the plant's "contact" tank)
    normal damper  f_d = -c v_n, only while delta > 0   (in D, >= 0)
    unilateral     f = max(k delta - c v_n, 0) along n = (x_a - x_b) / |x_a - x_b|: it pushes, never pulls

The force acts with equal and opposite sign on the two bodies at the middle of the overlap, so it is internal: it
moves energy between the tanks and the damper removes some, and the momentum of the free robot does not change.
Integrated like the ground contact (ph_k1/plant.py): the spring force at x_n, the damper and the linearised stiffness
implicit in the (nu, i) system (A += h (c + h k) J^T J with J = n . (J_a - J_b) of the pair), then the unilateral check
and one corrected solve.  An explicit damper is not an option: several pairs on a light hand kick it at h = 1 ms
(49 rad/s against 15 rad/s at a fine step, and 16 of 58 J missing in the ledger, from an 87 mm overlap).
The contact ledger gets (work on the element) - (change of V_s), as for the ground.

Pairs without an element: the same body, bodies at most 2 links apart in the tree (they overlap by construction),
and body pairs whose spheres overlap in a rest pose (the proxies are larger than the parts there).

Off by default (K1PHPlant(self_contact=SelfContactParams(enabled=True))): with it off the plant is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from .dynamics import forward_kinematics, point_jacobian
from .model import HOME_POSE, NJ, NQ, K1Model, _quat_mat

# rest poses of the skills: no element between bodies that overlap here (get-up / sprint kHome)
AGENT_HOME = (0.0, 0.0, 0.0, -1.4, 0.0, -0.4, 0.0, 1.4, 0.0, 0.4,
              -0.4, 0.0, 0.0, 0.8, -0.4, 0.0, -0.4, 0.0, 0.0, 0.8, -0.4, 0.0)
MAX_LINKS_APART = 2


@dataclass
class SelfContactParams:
    k: float = 2.0e4   # N/m
    c: float = 200.0   # N s/m
    enabled: bool = False


@dataclass
class SelfGeometry:
    body: torch.Tensor    # (S,) body of every sphere
    pos: torch.Tensor     # (S,3) centre in the body frame
    radius: torch.Tensor  # (S,)
    pair_a: torch.Tensor  # (P,) sphere indices
    pair_b: torch.Tensor  # (P,)

    @property
    def nsphere(self) -> int:
        return int(self.body.shape[0])

    @property
    def npair(self) -> int:
        return int(self.pair_a.shape[0])


def _body_box(m: mujoco.MjModel, mb: int):
    """(lo, hi) of the box in the frame of MuJoCo body mb around its collision geoms, or None."""
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != mb or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        ctr, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        pts.append(m.geom_pos[g] + (ctr + corners * half) @ _quat_mat(m.geom_quat[g]).T)
    if not pts:
        return None
    pts = np.concatenate(pts)
    return pts.min(0), pts.max(0)


def _chain(lo: np.ndarray, hi: np.ndarray):
    """Spheres along the longest side of a box: radius = mean of the two shorter half sides, spaced at most r."""
    ctr, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    axis = int(np.argmax(half))
    r = float(np.mean(np.delete(half, axis)))
    reach = max(float(half[axis]) - r, 0.0)
    n = 1 if reach <= 0 else int(np.ceil(2 * reach / r)) + 1
    out = []
    for t in np.linspace(-reach, reach, n):
        p = ctr.copy()
        p[axis] += t
        out.append((p, r))
    return out


def _links_apart(parent: list[int], a: int, b: int) -> int:
    """Number of joints on the tree path between two bodies."""
    path = {}
    x, n = a, 0
    while x >= 0:
        path[x] = n
        x, n = parent[x], n + 1
    x, n = b, 0
    while x not in path:
        x, n = parent[x], n + 1
    return n + path[x]


def _sphere_centres(geom_body, geom_pos, kin) -> torch.Tensor:
    return kin.pos[:, geom_body] + (kin.R[:, geom_body] @ geom_pos[None, :, :, None]).squeeze(-1)


def build_geometry(model: K1Model) -> SelfGeometry:
    """The spheres and the pairs of the K1 (cached on the model)."""
    cached = getattr(model, "_self_geometry", None)
    if cached is not None:
        return cached
    dev, dt = model.device, model.dtype
    bodies, centres, radii = [], [], []
    for b in range(model.nbody):
        box = _body_box(model.mj, b + 1)
        if box is None:
            continue
        for p, r in _chain(*box):
            bodies.append(b)
            centres.append(p)
            radii.append(r)
    body = torch.as_tensor(bodies, device=dev)
    pos = torch.as_tensor(np.stack(centres), dtype=dt, device=dev)
    radius = torch.as_tensor(radii, dtype=dt, device=dev)
    # rest poses: the trunk upright at the origin
    rest = torch.stack((torch.zeros(NJ, dtype=dt, device=dev), model.pose_vector(HOME_POSE),
                        torch.as_tensor(AGENT_HOME, dtype=dt, device=dev)))
    q = torch.zeros(len(rest), NQ, dtype=dt, device=dev)
    q[:, 3] = 1.0
    q[:, 7:] = rest
    x = _sphere_centres(body, pos, forward_kinematics(model, q))
    S = len(bodies)
    apart = torch.cdist(x, x)                                     # (rest, S, S)
    touching = (apart < radius[None, :, None] + radius[None, None, :]).any(0).cpu().numpy()
    nb = model.nbody
    rest_overlap = np.zeros((nb, nb), bool)
    for i in range(S):
        for j in range(S):
            if touching[i, j]:
                rest_overlap[bodies[i], bodies[j]] = True
    pair_a, pair_b = [], []
    for i in range(S):
        for j in range(i + 1, S):
            a, b = bodies[i], bodies[j]
            if a == b or rest_overlap[a, b] or _links_apart(model.parent, a, b) <= MAX_LINKS_APART:
                continue
            pair_a.append(i)
            pair_b.append(j)
    geom = SelfGeometry(body, pos, radius, torch.as_tensor(pair_a, device=dev), torch.as_tensor(pair_b, device=dev))
    model._self_geometry = geom
    return geom


@dataclass
class PairState:
    active: torch.Tensor  # (B,P) overlapping
    delta: torch.Tensor   # (B,P) overlap, 0 if apart
    n: torch.Tensor       # (B,P,3) unit normal, from b to a
    x: torch.Tensor       # (B,P,3) force point: the middle of the overlap


def pair_state(geom: SelfGeometry, kin) -> PairState:
    x = _sphere_centres(geom.body, geom.pos, kin)
    xa, xb = x[:, geom.pair_a], x[:, geom.pair_b]
    ra, rb = geom.radius[geom.pair_a], geom.radius[geom.pair_b]
    dv = xa - xb
    d = torch.linalg.norm(dv, dim=-1)
    active = d < ra + rb
    delta = (ra + rb - d) * active
    n = dv / d.clamp_min(1e-9)[..., None]
    return PairState(active, delta, n, xb + n * (rb - 0.5 * delta)[..., None])


def potential(params: SelfContactParams, geom: SelfGeometry, kin) -> torch.Tensor:
    """V_s = sum 1/2 k delta^2 per robot."""
    return 0.5 * params.k * (pair_state(geom, kin).delta ** 2).sum(-1)


def jacobian(geom: SelfGeometry, kin, st: PairState) -> torch.Tensor:
    """(B,P,28) J = n . (J_a - J_b) of every pair at its force point: J nu = separation speed."""
    ja = point_jacobian(kin, geom.body[geom.pair_a], st.x)
    jb = point_jacobian(kin, geom.body[geom.pair_b], st.x)
    return torch.einsum("bpi,bpid->bpd", st.n, ja - jb)


def dissipated(params: SelfContactParams, st: PairState, f: torch.Tensor, vn: torch.Tensor, h: float) -> torch.Tensor:
    """Ledger of one step (B,): work the bodies did on the elements minus what the tanks stored.
    f (B,P) the forces of the step, vn (B,P) the separation speeds at nu_{n+1}."""
    work_in = -h * (f * vn).sum(-1)
    d_new = (st.delta - h * vn).clamp_min(0.0)
    return work_in - (0.5 * params.k * (d_new ** 2 - st.delta ** 2) * st.active).sum(-1)


__all__ = ["SelfContactParams", "SelfGeometry", "build_geometry", "dissipated", "jacobian", "pair_state", "potential"]
