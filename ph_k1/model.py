"""The K1 as numbers: read the MJCF ONCE, then never ask MuJoCo again.

MuJoCo is used here only as an XML parser: masses, inertia tensors, link
offsets, hinge axes, armature, joint ranges, foot boxes.  Stepping,
contact and forces are all ours (dynamics.py, contact.py, plant.py).

Generalised velocity (28):
    nu = (omega_B, v_B, theta_dot)
         omega_B, v_B : trunk angular / linear velocity in the TRUNK frame
                        (body twist; v_B = R^T r_dot)
         theta_dot    : the 22 hinges in MJCF order
Configuration (29):
    q  = (r [3], Q [4, w x y z], theta [22])   -- same layout as MuJoCo qpos
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
# Bundled copy (assets/K1, with meshes); falls back to the thesis folder layout.
_LOCAL_XML = HERE.parent / "assets" / "K1" / "K1_22dof.xml"
_THESIS_XML = HERE.parents[2] / "Presentation" / "assets_K1" / "K1_22dof.xml"
DEFAULT_XML = _LOCAL_XML if _LOCAL_XML.exists() else _THESIS_XML

NB = 6  # base dofs
NJ = 22  # hinges
NV = NB + NJ
NQ = 7 + NJ

FOOT_BODIES = ("left_foot_link", "right_foot_link")

# RL_Core HOME_KEYFRAME (rl_core/plant/k1_constants.py): bent-knee stand.
HOME_POSE = {
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Yaw": 0.4,
    "Left_Hip_Pitch": -0.4,
    "Left_Knee_Pitch": 0.8,
    "Left_Ankle_Pitch": -0.4,
    "Right_Hip_Pitch": -0.4,
    "Right_Knee_Pitch": 0.8,
    "Right_Ankle_Pitch": -0.4,
}


def _quat_mat(q) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=float))
    return out.reshape(3, 3)


@dataclass
class K1Model:
    """Tree topology + mass properties as torch tensors.

    Bodies are indexed 0..nbody-1 = MuJoCo bodies 1..25 (world dropped).
    """

    mj: mujoco.MjModel
    xml_path: Path
    dtype: torch.dtype
    device: torch.device
    body_names: list[str]
    joint_names: list[str]
    parent: list[int]  # parent body index, -1 for the trunk
    body_joint: list[int]  # hinge index 0..21 moving this body, -1 = welded, -2 = trunk (free)
    body_pos: torch.Tensor  # (nb,3) offset in parent frame
    body_rot: torch.Tensor  # (nb,3,3) body_quat as a matrix
    jnt_axis: torch.Tensor  # (22,3) local
    jnt_pos: torch.Tensor  # (22,3) local anchor
    jnt_body: list[int]  # body of each hinge
    jnt_parent_body: list[int]  # parent of that body
    mass: torch.Tensor  # (nb,)
    ipos: torch.Tensor  # (nb,3) COM in body frame
    I_body: torch.Tensor  # (nb,3,3) rotational inertia about COM, body frame
    armature: torch.Tensor  # (22,)
    jnt_low: torch.Tensor  # (22,)
    jnt_high: torch.Tensor  # (22,)
    forcerange: torch.Tensor  # (22,) |tau| ceiling of the MJCF <motor>
    support: torch.Tensor  # (nb,28) 1 if dof d moves body b
    subtree: torch.Tensor  # (22,nb) 1 if body b is at/after hinge k
    contact_body: torch.Tensor  # (nc,) body index of every sole point
    contact_local: torch.Tensor  # (nc,3) contact points in their body frame
    contact_is_sole: torch.Tensor  # (nc,) True for the 8 foot-sole points
    total_mass: float

    @property
    def nbody(self) -> int:
        return len(self.body_names)

    @property
    def ncontact(self) -> int:
        return int(self.contact_body.shape[0])

    def joint_index(self, name: str) -> int:
        return self.joint_names.index(name)

    def pose_vector(self, pose: dict[str, float] | None = None) -> torch.Tensor:
        theta = torch.zeros(NJ, dtype=self.dtype, device=self.device)
        for name, val in (pose or {}).items():
            theta[self.joint_index(name)] = val
        return theta


def load_k1(
    xml_path: Path | str = DEFAULT_XML,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    body_contacts: bool = True,
) -> K1Model:
    """body_contacts=False: only the 8 sole points touch the ground."""
    xml_path = Path(xml_path)
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    dev = torch.device(device)
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=dev)  # noqa: E731

    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free trunk joint"
    nb = m.nbody - 1
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b + 1) for b in range(nb)]
    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]

    parent, body_joint = [], []
    for b in range(nb):
        mb = b + 1
        parent.append(int(m.body_parentid[mb]) - 1)
        if m.body_jntnum[mb] == 0:
            body_joint.append(-1)
        else:
            assert m.body_jntnum[mb] == 1, "one joint per body expected"
            j = int(m.body_jntadr[mb])
            body_joint.append(-2 if j == 0 else j - 1)
    jnt_body = [int(m.jnt_bodyid[j + 1]) - 1 for j in range(NJ)]

    support = np.zeros((nb, NV))
    for b in range(nb):  # MuJoCo bodies are topologically ordered
        if parent[b] >= 0:
            support[b] = support[parent[b]]
        if body_joint[b] == -2:
            support[b, :NB] = 1
        elif body_joint[b] >= 0:
            support[b, NB + body_joint[b]] = 1
    subtree = np.zeros((NJ, nb))
    for k in range(NJ):
        subtree[k] = support[:, NB + k]

    I_body = np.zeros((nb, 3, 3))
    for b in range(nb):
        Ri = _quat_mat(m.body_iquat[b + 1])
        I_body[b] = Ri @ np.diag(m.body_inertia[b + 1]) @ Ri.T

    # Sole points: the 4 bottom corners of each foot collision box.
    cb, cl, sole = [], [], []
    for fname in FOOT_BODIES:
        fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fname)
        boxes = [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == fb and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_contype[g]
        ]
        assert len(boxes) == 1, f"{fname}: expected one collision box"
        g = boxes[0]
        Rg, pg, hs = _quat_mat(m.geom_quat[g]), m.geom_pos[g], m.geom_size[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                cb.append(fb - 1)
                cl.append(pg + Rg @ np.array([sx * hs[0], sy * hs[1], -hs[2]]))
                sole.append(True)
    if body_contacts:
        # Every other body with collision geometry: the 8 corners of the box
        # (in its own frame) that encloses all its collision geoms.  Only
        # against the ground -- so a fallen robot lies ON the floor.
        foot_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in FOOT_BODIES}
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        for mb in range(1, m.nbody):
            if mb in foot_ids:
                continue
            pts = []
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != mb or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                    continue
                ctr, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                Rg = _quat_mat(m.geom_quat[g])
                pts.append(m.geom_pos[g] + (ctr + corners * half) @ Rg.T)
            if not pts:
                continue
            pts = np.concatenate(pts)
            lo, hi = pts.min(0), pts.max(0)
            for k in corners:
                cb.append(mb - 1)
                cl.append(np.where(k > 0, hi, lo))
                sole.append(False)

    act_range = np.zeros(NJ)
    for a in range(m.nu):
        j = int(m.actuator_trnid[a, 0]) - 1
        act_range[j] = m.actuator_forcerange[a, 1] if m.actuator_forcelimited[a] else np.inf

    rng = m.jnt_range[1:]
    return K1Model(
        mj=m,
        xml_path=xml_path,
        dtype=dtype,
        device=dev,
        body_names=names,
        joint_names=jnames,
        parent=parent,
        body_joint=body_joint,
        body_pos=T(m.body_pos[1:]),
        body_rot=T(np.stack([_quat_mat(m.body_quat[b + 1]) for b in range(nb)])),
        jnt_axis=T(m.jnt_axis[1:]),
        jnt_pos=T(m.jnt_pos[1:]),
        jnt_body=jnt_body,
        jnt_parent_body=[parent[b] for b in jnt_body],
        mass=T(m.body_mass[1:]),
        ipos=T(m.body_ipos[1:]),
        I_body=T(I_body),
        armature=T(m.dof_armature[NB:]),
        jnt_low=T(rng[:, 0]),
        jnt_high=T(rng[:, 1]),
        forcerange=T(act_range),
        support=T(support),
        subtree=T(subtree),
        contact_body=torch.as_tensor(cb, device=dev),
        contact_local=T(np.stack(cl)),
        contact_is_sole=torch.as_tensor(sole, device=dev),
        total_mass=float(m.body_subtreemass[1]),
    )
