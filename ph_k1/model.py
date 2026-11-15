"""The K1 as numbers: read the MJCF ONCE, then never ask MuJoCo again.

MuJoCo is used here only as an XML parser: masses, inertia tensors, link
offsets, hinge axes, joint ranges, foot boxes.  Stepping, contact and forces
are all ours (dynamics.py, contact.py, plant.py).

The servo numbers do NOT come from the MJCF: armature and torque ceiling
are Eric Plass's k1_skill_cfg.py values (servo.py); `apply_servo_table`
overwrites the MJCF's in every MjModel used next to the plant (load_mj).

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

from . import servo

HERE = Path(__file__).resolve().parent
# The robot model: Booster's standard K1_22dof.xml as K1_Deploy's twin installs it (assets/K1_standard, with its
# manifest: the upstream revision and file hashes).  assets/K1/K1_22dof.xml is an older variant with convex-decomposed
# hand collision meshes, hand-tip bodies and other joint ranges; the plant used it until 15 Nov 2026 (HAND_XML, for
# reproducing runs from before).  Masses, inertias, link offsets and the joint order are the same in both.
STANDARD_XML = HERE.parent / "assets" / "K1_standard" / "K1_22dof.xml"
HAND_XML = HERE.parent / "assets" / "K1" / "K1_22dof.xml"
_THESIS_XML = HERE.parents[2] / "Presentation" / "assets_K1" / "K1_22dof.xml"
DEFAULT_XML = next((x for x in (STANDARD_XML, HAND_XML) if x.exists()), _THESIS_XML)

# The plant's own names (those of the older MJCF; code, tests and contracts use them).  The standard MJCF's names are
# mapped onto them when it is loaded (load_mj).
JOINT_NAMES = (
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
_STANDARD_JOINT = {"aahead_yaw_joint": "AAHead_yaw", "aahead_pitch_joint": "Head_pitch",
                   **{("aa" if n[0] == "A" else "") + n.lstrip("A").lower() + "_joint": n for n in JOINT_NAMES[2:]}}
_STANDARD_BODY = {
    "trunk": "Trunk", "aahead_yaw_link": "Head_1", "aahead_pitch_link": "Head_2",
    **{f"{a}{s}_{part}_link": f"{S}_{name}"
       for s, S in (("left", "Left"), ("right", "Right"))
       for a, part, name in (("aa", "shoulder_pitch", "Arm_1"), ("", "shoulder_roll", "Arm_2"),
                             ("", "elbow_pitch", "Arm_3"), ("", "hip_pitch", "Hip_Pitch"),
                             ("", "hip_roll", "Hip_Roll"), ("", "hip_yaw", "Hip_Yaw"), ("", "knee_pitch", "Shank"),
                             ("", "ankle_pitch", "Ankle_Cross"))},
    "left_elbow_yaw_link": "left_hand_link", "right_elbow_yaw_link": "right_hand_link",   # the forearm: no hand
    "left_ankle_roll_link": "left_foot_link", "right_ankle_roll_link": "right_foot_link",
}

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

# Joint angles the real K1 (Phobos) reached in its two get-ups, min/max over getup_2236.csv and
# getup_front_2250.csv (Karls Repo/Logs/30102026, 25 Oct 2026, 500 Hz).  Several lie outside the MJCF range
# of the older MJCF and of the firmware's command clamp (K1_Deploy k1_config.h joint_pos_lower/upper, the same
# numbers as the older MJCF's ranges); the standard MJCF's ranges are wider at the elbows and the knee.
MEASURED_RANGE = {
    "AAHead_yaw": (-0.718, 1.343), "Head_pitch": (-0.413, 0.753),
    "ALeft_Shoulder_Pitch": (-2.005, 0.490), "Left_Shoulder_Roll": (-1.804, -0.148),
    "Left_Elbow_Pitch": (-0.858, 1.403), "Left_Elbow_Yaw": (-2.237, 0.221),
    "ARight_Shoulder_Pitch": (-1.093, 1.110), "Right_Shoulder_Roll": (0.343, 1.772),
    "Right_Elbow_Pitch": (-1.595, 0.447), "Right_Elbow_Yaw": (-0.169, 2.066),
    "Left_Hip_Pitch": (-1.111, 2.168), "Left_Hip_Roll": (-0.511, 1.240), "Left_Hip_Yaw": (-0.774, 1.053),
    "Left_Knee_Pitch": (-0.010, 1.520), "Left_Ankle_Pitch": (-0.843, 0.141), "Left_Ankle_Roll": (-0.445, 0.348),
    "Right_Hip_Pitch": (-1.691, 0.644), "Right_Hip_Roll": (-1.521, 0.455), "Right_Hip_Yaw": (-0.719, 0.786),
    "Right_Knee_Pitch": (-0.028, 1.943), "Right_Ankle_Pitch": (-0.793, 0.331), "Right_Ankle_Roll": (-0.353, 0.503),
}
# Three ranges per joint:
#   jnt_low/high   the firmware's command clamp (k1_config.h): what a policy can ask for
#   mjcf_low/high  the range written in the MJCF
#   stop_low/high  the mechanical end stops of the plant (the pH end-stop springs): the wider of the two above,
#                  widened by STOP_TOLERANCE (the firmware accepts readings up to 0.15 rad past its range, K1_Deploy
#                  main.cpp range check), and further to MEASURED_RANGE + STOP_MARGIN where the robot went beyond.
STOP_TOLERANCE = 0.15
STOP_MARGIN = 0.05


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
    armature: torch.Tensor  # (22,) servo.ARMATURE (reflected motor inertia, k1_skill_cfg)
    jnt_low: torch.Tensor  # (22,) the firmware's command clamp (k1_config.h joint_pos_lower/upper)
    jnt_high: torch.Tensor  # (22,)
    mjcf_low: torch.Tensor  # (22,) the range written in the MJCF
    mjcf_high: torch.Tensor  # (22,)
    stop_low: torch.Tensor  # (22,) mechanical end stops (end-stop springs), wider: see MEASURED_RANGE
    stop_high: torch.Tensor  # (22,)
    forcerange: torch.Tensor  # (22,) |tau| ceiling of this MJCF's <motor> as authored
    effort: torch.Tensor  # (22,) servo.EFFORT: the torque ceiling of k1_skill_cfg (the plant's)
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


def _check_table_order(joint_names: list[str]) -> None:
    """servo.JOINTS and the MJCF hinges must be the same 22 joints in the same order."""
    norm = lambda n: n.lstrip("A").lower()  # noqa: E731  MJCF prefixes A/AA for the sort order
    if [norm(n) for n in joint_names] != [n.lower() for n in servo.JOINTS]:
        raise ValueError(f"MJCF joint order {joint_names} differs from servo.JOINTS {servo.JOINTS}")


def apply_servo_table(m: mujoco.MjModel) -> mujoco.MjModel:
    """Overwrite the MJCF's servo numbers with k1_skill_cfg's (servo.py): armature and force range."""
    _check_table_order([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)])
    m.dof_armature[NB:] = servo.ARMATURE  # (the ankles' damping 0.2 is the plant's motor damping, motors.py)
    for a in range(m.nu):
        j = int(m.actuator_trnid[a, 0]) - 1
        m.actuator_forcerange[a] = (-servo.EFFORT[j], servo.EFFORT[j])
    return m


def load_mj(xml_path: Path | str = DEFAULT_XML) -> mujoco.MjModel:
    """The MJCF as an MjModel with the plant's body and joint names (the standard MJCF's names are renamed; the
    older MJCF has them already).  Everything that needs a MuJoCo model of the robot loads it here."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    if not any(j.name in _STANDARD_JOINT for j in spec.joints):
        return spec.compile()
    for b in spec.bodies:
        b.name = _STANDARD_BODY.get(b.name, b.name)
    for j in spec.joints:
        j.name = _STANDARD_JOINT.get(j.name, j.name)
    for a in spec.actuators:
        a.name = _STANDARD_JOINT.get(a.name, a.name)
        a.target = _STANDARD_JOINT.get(a.target, a.target)
    for x in spec.sensors:
        x.objname = _STANDARD_BODY.get(x.objname, _STANDARD_JOINT.get(x.objname, x.objname))
    return spec.compile()


def load_k1(
    xml_path: Path | str = DEFAULT_XML,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    body_contacts: bool = True,
) -> K1Model:
    """body_contacts=False: only the 8 sole points touch the ground."""
    xml_path = Path(xml_path)
    m = load_mj(xml_path)
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
    apply_servo_table(m)  # after reading the authored force range

    from . import firmware  # here: firmware imports this module

    rng = m.jnt_range[1:]
    clamp_lo, clamp_hi = np.asarray(firmware.JOINT_POS_LOWER), np.asarray(firmware.JOINT_POS_UPPER)
    stop_lo = np.minimum(rng[:, 0], clamp_lo) - STOP_TOLERANCE
    stop_hi = np.maximum(rng[:, 1], clamp_hi) + STOP_TOLERANCE
    for j, name in enumerate(jnames):
        if name in MEASURED_RANGE:
            lo, hi = MEASURED_RANGE[name]
            stop_lo[j] = min(stop_lo[j], lo - STOP_MARGIN)
            stop_hi[j] = max(stop_hi[j], hi + STOP_MARGIN)
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
        jnt_low=T(clamp_lo),
        jnt_high=T(clamp_hi),
        mjcf_low=T(rng[:, 0]),
        mjcf_high=T(rng[:, 1]),
        stop_low=T(stop_lo),
        stop_high=T(stop_hi),
        forcerange=T(act_range),
        effort=T(servo.EFFORT),
        support=T(support),
        subtree=T(subtree),
        contact_body=torch.as_tensor(cb, device=dev),
        contact_local=T(np.stack(cl)),
        contact_is_sole=torch.as_tensor(sole, device=dev),
        total_mass=float(m.body_subtreemass[1]),
    )
