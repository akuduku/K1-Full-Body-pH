r"""The walk task, written once: what a walking robot is paid for, and how its velocity commands are drawn.

The rewards and the commands are Eric Plass's walk task (RL_Core rl_core/envs/k1_skill.py and k1_skill_cfg.py, with
k1_sprint/env.py's tracking terms; its "base" variant, the task walk_v9 was trained on), computed from the
port-Hamiltonian plant's quantities.  Nothing else is taken from that work: no policy, no imitation.  Added from
the pH plant: the electrical power drawn from the battery (`energy`).

No plant lives in this file.  Both environments use it:

    rl/walk/env.py                          the C++ plant (K1PHPlantNative): scoring, PPO, the planner
    Model Based RL mbrl/grad/walk_env.py    the Python plant: autograd through plant and reward

so "the same task on both plants" holds by construction.  Everything is plain torch and, where a term has a
derivative, autograd passes through it.

    task = WalkTask(model)
    mem  = task.new_memory(B, rand)                              # commands and the rewards' memory, a dict of tensors
    sig  = task.signals(q, nu, prev_qd, contact_active, contact_force)
    terms, mem = task.terms(sig, mem, a, a_last, a_before, power_w, terminated)     # this step's rewards
    reward     = task.reward(terms)                              # weighted sum x policy step
    mem  = task.update_commands(mem, sig.yaw, dt, rand)          # after resets: timers, new requests, the ramp

`rand(*shape)` gives uniform numbers in [0, 1) (each environment brings its own generator).  The memory is never
written in place: every call returns a new dict, so a state can be kept and put back.

Different from Eric's environment, and why:
  * self_collisions is left out: the pH plant has no contact between the robot's own bodies.
  * joint acceleration is the change of the joint velocity over one policy step (as rl/stand), not MuJoCo's qacc.
  * the foot height is the lowest point of his 5 x 3 sole grid, the foot contact comes from the plant's 4 sole
    points per foot; the hand tips are 0.203 m along the hand link (his body left_hand_end_ball).
  * his top-speed curriculum gate is not here: `top_speed` and `command_scale` are set by the training.

Added for a gradient learner, off unless asked for (STAND_INS): `air_time_smooth`, a smooth stand-in for his
`air_time`.  `air_time` counts feet in the air (yes/no), so its derivative is zero and a learner that follows
derivatives never sees that lifting a foot pays.  The stand-in pays for the weight a foot has taken off the ground
and kept off: his 0.05 s minimum (the smallest unload of the last 3 control steps), his 0.5 s limit (counted in
seconds of unload since the foot last carried nearly its half) and his request gate.  Eric's own terms, and so the
full reward that is logged and scored, stay as they are.

The gait clock (W5c, also not Eric's; SHAPING): a phase that runs while a move is asked for, as Booster Gym's gait
cycle.  `clock(mem)` gives the network's two extra inputs (sin and cos, 0 while standing), and the term `gait_clock`
pays each foot for unloading while the clock says it swings and for carrying while it says it stands.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch

from ph_k1 import firmware as fw
from ph_k1.dynamics import GRAVITY, Kinematics, body_twists, forward_kinematics
from ph_k1.model import K1Model
from rl.kick.contract import rotate, to_heading, wrap, yaw_of

from . import contract as C

# k1_skill_cfg.py joint_names: the firmware's order
JOINTS = ("Head_Yaw", "Head_Pitch",
          "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
          "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
          "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
          "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll")

# weight per reward term (k1_skill_reward_cfg reward_scales); the sum is multiplied by the policy step
REWARDS = {
    "track_linear_velocity": 4.0,
    "track_angular_velocity": 2.0,
    "upright": 1.0,
    "body_ang_vel": -0.01,
    "angular_momentum": -0.005,
    "dof_pos_limits": -1.0,
    "action_rate": -0.1,
    "air_time": 0.1,
    "foot_clearance": -2.0,
    "foot_swing_height": -0.25,
    "foot_slip": -0.2,
    "soft_landing": -0.01,
    "upper_body_posture": -0.1,
    "standing_pose": -0.5,
    "toe_clearance": -10.0,
    "leg_spacing": -20.0,
    "feet_parallel": -2.0,
    "overspeed": -0.05,
    "arm_swing": -0.5,
    "knee_depth": -10.0,
    "yaw_rate_fine": 2.0,
    "action_jerk": -0.1,
    "leg_acc": -2e-6,
    "trunk_bounce": -2.5,
    "velocity_fine": 2.0,
    "heading_hold": -30.0,
    "terminated": -100.0,
    "foot_touchdown": -1.0,
    "foot_spacing": -150.0,
    "knees_out": -10.0,
    "heel_walking": -10.0,
    "reference_lag": -1.0,
    "energy": -0.002,            # per watt of electrical power drawn from the battery (pH ledger); not in Eric's task
}
LEFT_OUT = {"self_collisions": -1.0}                 # Eric's terms the pH plant cannot report
OPTIONAL = ("forward_lean", "backward_lean", "air_time_smooth", "gait_clock")   # off unless asked for: k1_sprint's
                                                 # "lean" variant (weights -20 and as given); air_time's stand-in; the clock
COUNTS = ("air_time", "terminated")                  # terms that only count events: no derivative at all
# smooth stand-ins of counting terms for a gradient learner (Model Based RL mbrl/grad): the same idea with a slope.
# Not Eric's terms: a learner that adds one still logs, and is scored on, Eric's full reward.
STAND_INS = {"air_time_smooth": "air_time"}
# every term added for a gradient learner: never part of Eric's full reward
SHAPING = ("air_time_smooth", "gait_clock")
FLOAT_MEMORY = ("swing_peak", "yaw_rate_avg", "velocity_avg", "heading_reference",   # memory a derivative flows through
                "unload_history")


@dataclass
class WalkParams:
    """The numbers of the rewards (k1_skill_reward_cfg) and of the commands (k1_skill_command_cfg)."""

    # tracking
    linear_std: float = 0.5
    linear_std_at_rest: float = 0.1
    linear_relative_std: float = 0.75
    linear_progress_weight: float = 0.5
    angular_std: float = 0.7071
    fine_std: float = 0.25
    yaw_rate_tau: float = 0.15
    velocity_tau: float = 0.25
    heading_frame: bool = False          # k1_sprint --heading_frame: track in the yaw frame, not the trunk frame
    start_grace_s: float = 0.0           # reference_lag waits this long after a request changed ("lean": 0.3)
    # posture: std per regime (standing < 0.05, walking, running >= 1.5 / 1.0 m/s)
    upright_std: tuple = (0.4472, 0.5, 0.6325)
    upper_body_std: dict = field(default_factory=lambda: {
        "standing": {"Head_.*": 0.05, ".*_Shoulder_.*": 0.05, ".*_Elbow_.*": 0.05},
        "walking": {"Head_.*": 0.05, ".*_Shoulder_.*": 0.15, ".*_Elbow_.*": 0.15},
        "running": {"Head_.*": 0.05, ".*_Shoulder_Pitch": 0.5, ".*_Shoulder_Roll": 0.2, ".*_Elbow_.*": 0.35},
    })
    lean_rest: float = 0.05
    lean_per_mps: float = 0.06
    lean_back: float = 0.035
    soft_limit_factor: float = 0.9       # dof_pos_limits: this share of the MJCF range around its middle
    # feet
    air_time_range: tuple = (0.05, 0.5)
    unload_reset: float = 0.1            # air_time_smooth (not Eric's): a foot carrying 90 % of its half again ends an unload
    # the gait clock (not Eric's; W5c), as Booster Gym's gait cycle: a phase that says which foot should be in the air.
    # Period and swing share are walk_v9's own on the pH plant at 0.5 m/s (a cycle of 0.46 s, each foot 36 % of it in
    # the air).  It runs while the request asks for a move (speed + |turn rate| above clock_gate), else it is at 0.
    gait_period: float = 0.46
    gait_swing: float = 0.36
    gait_ramp: float = 0.04              # the swing windows' edges, as a share of the cycle
    clock_gate: float = 0.2              # Eric's request gate of air_time
    toe: tuple = (0.115, 0.0, -0.03)     # toe tip in the foot frame
    toe_clearance: float = 0.04
    touchdown_height: float = 0.08
    foot_spacing: tuple = (0.23, 0.33)
    foot_spacing_fast: float = 0.19
    min_feet_spacing: float = 0.12
    min_knee_spacing: float = 0.15
    knees_out_margin: float = 0.03
    knee_depth: float = 0.9
    arm_swing: dict = field(default_factory=lambda: {"front": 0.12, "back": 0.12, "width": 0.22})
    overspeed_fraction: float = 0.8
    sole_site: tuple = (0.027364, 0.001141, -0.03)   # k1.xml left_foot_sole in the foot frame (right: y mirrored)
    hand_tip: float = 0.203                          # k1.xml left_hand_end_ball: +y of the hand link (right: -y)
    # commands
    lin_vel_x: tuple = (-1.5, 2.0)
    lin_vel_y: tuple = (-1.0, 1.0)
    ang_vel_z: tuple = (-2.0, 2.0)
    resampling_time_s: tuple = (1.0, 4.0)
    rel_standing: float = 0.1
    rel_heading: float = 0.3             # yaw rate steers to a random heading
    heading_stiffness: float = 0.5
    rel_forward: float = 0.1
    rel_sprint: float = 0.3              # straight runs at 60-100 % of the top speed
    rel_overspeed: float = 0.15          # of the sprints: requests above the top speed
    rel_turn: float = 0.1
    rel_sidestep: float = 0.2
    brake_prob: float = 0.5              # sprints that end in a stand
    accel: float = 2.0                   # [m/s^2] the rewards track a reference ramping toward the request
    sprint_yaw: float = 0.5
    top_speed: tuple = (2.0, 3.6)        # the sprints' top speed: start, maximum


@dataclass
class Signals:
    """What the rewards read, of B robots.  Feet, shanks and hands: index 0 left, 1 right."""

    base_pos: torch.Tensor        # (B,3)
    base_quat: torch.Tensor       # (B,4)
    yaw: torch.Tensor             # (B,)
    gravity: torch.Tensor         # (B,3) unit gravity in the trunk frame
    base_lin_vel: torch.Tensor    # (B,3) trunk frame
    base_ang_vel: torch.Tensor    # (B,3) trunk frame
    lin_vel_w: torch.Tensor       # (B,3) world
    ang_vel_w: torch.Tensor       # (B,3) world
    heading_vel: torch.Tensor     # (B,3) world velocity turned into the yaw frame
    theta: torch.Tensor           # (B,22)
    qd: torch.Tensor              # (B,22)
    dof_acc: torch.Tensor         # (B,22) change of qd over the policy step
    foot_pos: torch.Tensor        # (B,2,3) foot link origins (Eric's sites left_foot / right_foot)
    foot_rot: torch.Tensor        # (B,2,3,3)
    foot_vel: torch.Tensor        # (B,2,3) world velocity of the foot link origin
    foot_height: torch.Tensor     # (B,2) lowest point of the sole grid, in [0, 1]
    toe: torch.Tensor             # (B,2,3) toe tips
    shank_pos: torch.Tensor       # (B,2,3)
    hand_pos: torch.Tensor        # (B,2,3) hand tips
    com: torch.Tensor             # (B,3) whole-body centre of mass
    angular_momentum: torch.Tensor  # (B,3) about the centre of mass, world frame
    contact: torch.Tensor         # (B,2) bool: a sole point of the foot is on the ground
    contact_force: torch.Tensor   # (B,2,3) net ground force on each foot
    contact_pos: torch.Tensor     # (B,2,3) where it acts (centre of the normal forces; the foot origin without contact)
    ground: torch.Tensor          # (B,) bool: another body than a foot touches the ground
    roll_pitch: torch.Tensor      # (B,2)


class WalkTask:
    def __init__(self, model: K1Model, policy_dt: float = C.POLICY_DT, params: WalkParams | None = None,
                 rewards: dict | None = None):
        self.model, self.dt, self.p = model, policy_dt, params or WalkParams()
        self.rewards = dict(REWARDS if rewards is None else rewards)
        unknown = set(self.rewards) - set(REWARDS) - set(OPTIONAL)
        if unknown:
            raise KeyError(f"unknown reward terms {sorted(unknown)}; known: {sorted(REWARDS)} and {OPTIONAL}")
        self.dtype, self.device = model.dtype, model.device
        self.contract = C.Contract(self.dtype, self.device)
        self.top_speed = self.p.top_speed[0]
        self.command_scale = 1.0     # the training's knob: 0 = requests of zero (stand), 1 = Eric's ranges
        p, m = self.p, model
        T = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        ids = lambda *pats: torch.tensor([i for i, n in enumerate(JOINTS) if any(re.fullmatch(q, n) for q in pats)],  # noqa: E731
                                         device=self.device)

        def per_joint(values: dict) -> torch.Tensor:
            out = torch.zeros(len(JOINTS), dtype=self.dtype, device=self.device)
            for pattern, value in values.items():
                out[ids(pattern)] = value
            return out

        self.home = self.contract.home
        self.upper_body = ids("Head_.*", ".*_Shoulder_.*", ".*_Elbow_.*")
        self.upper_body_std = {k: per_joint(v)[self.upper_body] for k, v in p.upper_body_std.items()}
        self.legs = ids(".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*")
        self.knees = ids(".*_Knee_Pitch")
        self.no_load_speed = fw.firmware_table(m, fw.MAX_JOINT_VELOCITY)    # k1_skill_cfg actuators "speed"
        mid, half = 0.5 * (m.mjcf_low + m.mjcf_high), 0.5 * (m.mjcf_high - m.mjcf_low)
        self.soft_low, self.soft_high = mid - p.soft_limit_factor * half, mid + p.soft_limit_factor * half
        self.upright_std = T(p.upright_std)
        self.half_weight = -0.5 * GRAVITY[2] * float(m.mass.sum())       # [N] what each foot carries in a quiet stance
        # air_time pays from the control step its clock passes the range's start (0.05 s: the 3rd step in the air);
        # air_time_smooth pays the unload a foot has kept that many steps
        self.hold_steps = int(p.air_time_range[0] / self.dt + 1e-9) + 1

        names = m.body_names
        self.feet = [names.index("left_foot_link"), names.index("right_foot_link")]
        self.shanks = [names.index("Left_Shank"), names.index("Right_Shank")]
        self.hands = [names.index("left_hand_link"), names.index("right_hand_link")]
        self.hand_tip = T([[0.0, p.hand_tip, 0.0], [0.0, -p.hand_tip, 0.0]])
        self.toe_local = T(p.toe)
        # the sole grid for the foot height: 5 x 3 points over 13 x 6 cm around the sole site (k1_sprint env.py)
        x, y = torch.meshgrid(torch.linspace(-0.065, 0.065, 5), torch.linspace(-0.03, 0.03, 3), indexing="ij")
        grid = torch.stack((x.flatten(), y.flatten(), torch.zeros(15)), 1).to(self.dtype).to(self.device)
        site = T([p.sole_site, (p.sole_site[0], -p.sole_site[1], p.sole_site[2])])
        self.sole_grid = site[:, None, :] + grid[None]                       # (2,15,3) in the foot frames
        sole, cb = m.contact_is_sole, m.contact_body
        self.sole = torch.stack([(sole & (cb == b)) for b in self.feet]).to(self.dtype)      # (2,nc)
        self.not_sole = (~sole).to(self.dtype)
        self.sole_ids = [(sole & (cb == b)).nonzero(as_tuple=False).squeeze(-1) for b in self.feet]
        self.sole_local = torch.stack([m.contact_local[i] for i in self.sole_ids])           # (2,4,3)

    # ------------------------------------------------------------------ what the rewards read
    def signals(self, q, nu, prev_qd, contact_active, contact_force, kin: Kinematics | None = None) -> Signals:
        """q (B,29), nu (B,28) = (trunk angular, trunk linear: trunk frame | joint velocities), prev_qd (B,22) the
        joint velocities one policy step ago, contact_active (B,nc), contact_force (B,nc,3): the plant's report."""
        m = self.model
        kin = kin or forward_kinematics(m, q)
        V = body_twists(kin, nu)                                  # (B,nb,6): omega, velocity of the point at the origin
        quat, R0 = q[:, 3:7], kin.R[:, 0]
        ang_b, lin_b = nu[:, :3], nu[:, 3:6]
        lin_w = (R0 @ lin_b[..., None]).squeeze(-1)
        ang_w = (R0 @ ang_b[..., None]).squeeze(-1)
        yaw = yaw_of(quat)

        def at(body: list, local: torch.Tensor | None = None):
            """World position and velocity of body origins (local None) or of points fixed to them."""
            P, R = kin.pos[:, body], kin.R[:, body]
            if local is not None:
                P = P + (R @ local[None, :, :, None]).squeeze(-1)
            vel = V[:, body, 3:] + torch.cross(V[:, body, :3], P, dim=-1)
            return P, R, vel

        foot_pos, foot_rot, foot_vel = at(self.feet)
        grid = foot_pos[:, :, None, :] + (foot_rot[:, :, None] @ self.sole_grid[None, :, :, :, None]).squeeze(-1)   # (B,2,15,3)
        toe = foot_pos + (foot_rot @ self.toe_local[None, None, :, None]).squeeze(-1)
        hand_pos = at(self.hands, self.hand_tip)[0]

        mass = m.mass[None, :, None]
        com = (mass * kin.com).sum(1) / m.mass.sum()
        h = torch.einsum("bnij,bnj->bni", kin.I, V).sum(1)        # spatial momentum about the origin: (angular, linear)
        angular_momentum = h[:, :3] - torch.cross(com, h[:, 3:], dim=-1)

        active = contact_active.to(self.dtype)
        force = contact_force * active[..., None]
        foot_force = torch.einsum("bnc,fn->bfc", force, self.sole)
        points = torch.stack([kin.pos[:, b, None, :] + (kin.R[:, b, None] @ self.sole_local[i, None, :, :, None]).squeeze(-1)
                              for i, b in enumerate(self.feet)], 1)                         # (B,2,4,3)
        normal = torch.stack([force[:, ids, 2] for ids in self.sole_ids], 1).clamp(min=0.0)  # (B,2,4)
        total = normal.sum(-1, keepdim=True)
        centre = (normal[..., None] * points).sum(2) / total.clamp(min=1e-9)
        on = (active @ self.sole.T) > 0
        return Signals(
            base_pos=q[:, :3], base_quat=quat, yaw=yaw, gravity=self.contract.projected_gravity(quat),
            base_lin_vel=lin_b, base_ang_vel=ang_b, lin_vel_w=lin_w, ang_vel_w=ang_w, heading_vel=to_heading(yaw, lin_w),
            theta=q[:, 7:], qd=nu[:, 6:], dof_acc=(nu[:, 6:] - prev_qd) / self.dt,
            foot_pos=foot_pos, foot_rot=foot_rot, foot_vel=foot_vel, foot_height=grid[..., 2].amin(-1).clamp(0.0, 1.0),
            toe=toe, shank_pos=kin.pos[:, self.shanks], hand_pos=hand_pos, com=com, angular_momentum=angular_momentum,
            contact=on, contact_force=foot_force, contact_pos=torch.where(total > 0, centre, foot_pos),
            ground=(active @ self.not_sole) > 0, roll_pitch=self.contract.roll_pitch(quat))

    # ------------------------------------------------------------------ memory and commands
    def new_memory(self, B: int, rand) -> dict:
        z = lambda *s: torch.zeros(B, *s, dtype=self.dtype, device=self.device)        # noqa: E731
        no = lambda: torch.zeros(B, dtype=torch.bool, device=self.device)              # noqa: E731
        mem = {"commands": z(3), "target": z(3), "heading_target": z(), "is_heading": no(), "is_standing": no(),
               "is_sprint": no(), "is_overspeed": no(), "sprint_time": z(), "command_time_left": z(),
               "command_counter": torch.zeros(B, dtype=torch.long, device=self.device), "command_age": z(),
               "swing_peak": z(2), "yaw_rate_avg": z(), "velocity_avg": z(2), "heading_reference": z(),
               "last_target": torch.full((B, 3), math.nan, dtype=self.dtype, device=self.device),
               "air_time": z(2), "contact_time": z(2), "unloaded_time": z(2), "unload_history": z(2, self.hold_steps),
               "gait_phase": z()}
        return self.resample(mem, torch.arange(B, device=self.device), rand)

    def reset_memory(self, mem: dict, ids: torch.Tensor, rand) -> dict:
        """Robots `ids` start a new episode: the rewards' memory cleared, a first request drawn."""
        if ids.numel() == 0:
            return mem
        mem = {k: v.clone() for k, v in mem.items()}
        for k in ("swing_peak", "yaw_rate_avg", "velocity_avg", "air_time", "contact_time", "unloaded_time",
                  "unload_history", "gait_phase"):
            mem[k][ids] = 0.0
        mem["last_target"][ids] = math.nan
        mem["command_counter"][ids] = 0
        return self.resample(mem, ids, rand, copy=False)

    def resample(self, mem: dict, ids: torch.Tensor, rand, copy: bool = True) -> dict:
        """New velocity requests for the robots `ids` (k1_sprint _resample_commands): uniform in the ranges, with
        shares of heading control, standing and forward-only; then a third become straight sprints near the top
        speed, and pure turns and sidesteps.  Half of the sprints that end brake to a stand."""
        c, n = self.p, int(ids.numel())
        if n == 0:
            return mem
        if copy:
            mem = {k: v.clone() for k, v in mem.items()}
        uniform = lambda lo=0.0, hi=1.0: lo + (hi - lo) * rand(n)        # noqa: E731
        mem["command_time_left"][ids] = uniform(*c.resampling_time_s)
        was_sprinting = mem["is_sprint"][ids]
        mem["target"][ids[mem["command_counter"][ids] == 0]] = 0.0

        cmd = torch.stack((uniform(*c.lin_vel_x), uniform(*c.lin_vel_y), uniform(*c.ang_vel_z)), 1)
        mem["heading_target"][ids] = uniform(-math.pi, math.pi)
        is_heading = uniform() <= c.rel_heading
        is_standing = uniform() <= c.rel_standing
        forward = uniform() <= c.rel_forward
        cmd[forward, 0] = cmd[forward, 0].abs().clamp(min=0.3)
        cmd[forward, 1:] = 0.0

        r4 = rand(n, 4)
        sprint, over = r4[:, 0] < c.rel_sprint, r4[:, 1] < c.rel_overspeed
        top, max_top = self.top_speed, c.top_speed[1]
        low = torch.where(over, torch.full_like(r4[:, 0], top), torch.full_like(r4[:, 0], 0.6 * top))
        high = torch.where(over, torch.full_like(low, min(top + max_top, max_top)), torch.full_like(low, top))
        cmd[sprint, 0] = (low + (high - low) * r4[:, 2])[sprint]
        cmd[sprint, 1] = 0.0
        cmd[sprint, 2] = ((2.0 * r4[:, 3] - 1.0) * c.sprint_yaw)[sprint]

        kind, r = uniform(), uniform(-1.0, 1.0)
        turn = ~sprint & (kind < c.rel_turn)
        side = ~sprint & (kind >= c.rel_turn) & (kind < c.rel_turn + c.rel_sidestep)
        cmd[turn] = 0.0
        cmd[turn, 2] = r[turn] * c.ang_vel_z[1]
        cmd[side] = 0.0
        cmd[side, 1] = r[side] * c.lin_vel_y[1]
        plain = sprint | turn | side
        brake = was_sprinting & ~sprint & (r4[:, 1] < c.brake_prob)
        cmd[brake] = 0.0

        mem["commands"][ids] = cmd * self.command_scale
        mem["is_heading"][ids] = is_heading & ~plain & ~brake
        mem["is_standing"][ids] = (is_standing & ~plain) | brake
        mem["is_sprint"][ids] = sprint
        mem["is_overspeed"][ids] = sprint & over
        mem["sprint_time"][ids] = 0.0
        mem["command_age"][ids] = 0.0
        mem["command_counter"][ids] += 1
        return mem

    def update_commands(self, mem: dict, yaw: torch.Tensor, dt: torch.Tensor, rand) -> dict:
        """After a step (and its resets): timers advance by `dt` (B,; 0 for robots that just reset), expired
        requests are redrawn, heading robots steer, and the reference moves toward the request (accel)."""
        c = self.p
        mem = {k: v.clone() for k, v in mem.items()}
        mem["sprint_time"] += self.dt
        mem["command_age"] += dt
        mem["command_time_left"] -= dt
        expired = (mem["command_time_left"] <= 0.0).nonzero(as_tuple=False).flatten()
        mem = self.resample(mem, expired, rand, copy=False)

        steer = (c.heading_stiffness * wrap(mem["heading_target"] - yaw.detach())).clamp(*c.ang_vel_z) * self.command_scale
        mem["commands"][:, 2] = torch.where(mem["is_heading"], steer, mem["commands"][:, 2])
        mem["commands"][mem["is_standing"]] = 0.0

        goal = mem["commands"].clone()
        goal[:, 0] = goal[:, 0].clamp(max=self.top_speed)
        step = c.accel * self.dt
        mem["target"][:, :2] += (goal[:, :2] - mem["target"][:, :2]).clamp(-step, step)
        mem["target"][:, 2] = goal[:, 2]
        return mem

    def set_command(self, mem: dict, twist) -> dict:
        """A fixed request for every robot (evaluation): request and reference set, no redraw for a long time."""
        mem = {k: v.clone() for k, v in mem.items()}
        twist = torch.as_tensor(twist, dtype=self.dtype, device=self.device).expand_as(mem["commands"])
        mem["commands"], mem["target"] = twist.clone(), twist.clone()
        mem["command_time_left"] = torch.full_like(mem["command_time_left"], 1e9)
        for k in ("is_heading", "is_standing", "is_sprint", "is_overspeed"):
            mem[k] = torch.zeros_like(mem[k])
        return mem

    # ------------------------------------------------------------------ the gait clock (W5c, not Eric's)
    def clock_on(self, mem: dict) -> torch.Tensor:
        """(B,) bool: the clock runs while the request (what the network is told) asks for a move, its speed plus
        |turn rate| above clock_gate.  Asked to stand, the clock's inputs are 0 and its phase is back at 0 (Booster Gym:
        "for the standing gait, the gait cycle is set to zero")."""
        c = torch.minimum(torch.maximum(mem["commands"], self.contract.twist_low), self.contract.twist_high)   # as observed
        return (torch.linalg.vector_norm(c[:, :2], dim=1) + c[:, 2].abs()) > self.p.clock_gate

    def clock(self, mem: dict) -> torch.Tensor:
        """(B,2) the network's two clock inputs: sin and cos of 2 pi x phase while the clock runs, (0, 0) while it
        stands.  The firmware's walk_clock agent computes the same from its own tick counter."""
        angle = 2.0 * math.pi * mem["gait_phase"]
        return torch.stack((torch.sin(angle), torch.cos(angle)), -1) * self.clock_on(mem)[:, None].to(self.dtype)

    def next_phase(self, mem: dict) -> torch.Tensor:
        """(B,) the clock's phase at the next control step: one step of its period on while it runs, back to 0 while
        the robot is asked to stand (the firmware: a step counter, set to 0 while standing)."""
        phase = mem["gait_phase"]
        return torch.where(self.clock_on(mem), torch.remainder(phase + self.dt / self.p.gait_period, 1.0),
                           torch.zeros_like(phase))

    def swing(self, phase: torch.Tensor) -> torch.Tensor:
        """(B,2) how much of each foot's swing window this phase is in: 1 inside, 0 outside, a ramp gait_ramp wide at
        the edges.  The left foot swings in [0, gait_swing) of the cycle, the right foot half a cycle later."""
        p, h = self.p, 0.5 * self.p.gait_swing
        centre = torch.tensor([h, 0.5 + h], dtype=self.dtype, device=self.device)
        d = torch.remainder(phase[:, None] - centre + 0.5, 1.0) - 0.5           # signed distance on the circle
        return ((h - d.abs()) / p.gait_ramp + 0.5).clamp(0.0, 1.0)

    # ------------------------------------------------------------------ rewards
    def _regime(self, total, standing, walking, running, running_threshold: float = 1.5):
        if standing.dim() > 0 and standing.shape[-1] > 1:     # per-joint values
            total = total[:, None]
        return torch.where(total < 0.05, standing, torch.where(total < running_threshold, walking, running))

    def terms(self, s: Signals, mem: dict, a, a_last, a_before, power_w, terminated) -> tuple[dict, dict]:
        """This step's reward terms, unweighted, and the memory after it.

        a, a_last, a_before (B,22): this step's raw policy output and the two before; power_w (B,) electrical power
        drawn [W]; terminated (B,) bool.  -> ({name: (B,)} for the terms in self.rewards, new memory)."""
        p, dt, f = self.p, self.dt, lambda b: b.to(self.dtype)
        target, commands = mem["target"], mem["commands"]
        speed = torch.linalg.vector_norm(target[:, :2], dim=1)
        total = speed + target[:, 2].abs()
        track = s.heading_vel if p.heading_frame else s.base_lin_vel
        theta, contact = s.theta, s.contact
        on, off = f(contact), f(~contact)

        # the feet's clocks (mjlab's contact sensor): time in the air, time on the ground, the step a foot landed
        air_time = torch.where(contact, torch.zeros_like(mem["air_time"]), mem["air_time"] + dt)
        contact_time = torch.where(contact, mem["contact_time"] + dt, torch.zeros_like(mem["contact_time"]))
        landed = contact & (contact_time < dt + 1e-6)
        heading_y = lambda pos: to_heading(s.yaw, pos[:, 0] - pos[:, 1])[:, 1]       # noqa: E731  left minus right
        feet_y, knees_y = heading_y(s.foot_pos), heading_y(s.shank_pos)
        foot_speed = torch.linalg.vector_norm(s.foot_vel[..., :2], dim=-1)
        swing_target = (0.06 + 0.02 * (speed - 1.0).clamp(min=0.0)).clamp(max=0.10)   # 6 cm walking, 10 cm at 3 m/s
        new = dict(mem, air_time=air_time, contact_time=contact_time)
        want = self.rewards
        t: dict = {}

        if "track_linear_velocity" in want:
            error = (target[:, :2] - track[:, :2]).square().sum(1)
            sigma = (p.linear_relative_std * speed).clamp(p.linear_std_at_rest, p.linear_std)
            tracking = torch.exp(-(error + track[:, 2].square()) / sigma.square())
            progress = (1.0 - (error + 1e-16).sqrt() / speed.clamp(min=p.linear_std_at_rest)).clamp(0.0, 1.0)
            t["track_linear_velocity"] = (1.0 - p.linear_progress_weight) * tracking + p.linear_progress_weight * progress
        if "track_angular_velocity" in want:
            error = (target[:, 2] - s.base_ang_vel[:, 2]).square() + s.base_ang_vel[:, :2].square().sum(1)
            t["track_angular_velocity"] = torch.exp(-error / p.angular_std**2)
        if "upright" in want:
            std = self._regime(total, *self.upright_std)
            t["upright"] = torch.exp(-s.gravity[:, :2].square().sum(1) / std.square())
        if "body_ang_vel" in want:
            t["body_ang_vel"] = s.ang_vel_w[:, :2].square().sum(1)
        if "angular_momentum" in want:
            t["angular_momentum"] = s.angular_momentum.square().sum(1)
        if "dof_pos_limits" in want:
            t["dof_pos_limits"] = ((self.soft_low - theta).clamp(min=0.0) + (theta - self.soft_high).clamp(min=0.0)).sum(1)
        if "action_rate" in want:
            t["action_rate"] = (a - a_last).square().sum(1) / (1.0 + 0.5 * speed)
        if "air_time" in want:
            low, high = p.air_time_range
            t["air_time"] = f((air_time > low) & (air_time < high)).sum(1) * f(total > 0.2)
        if "air_time_smooth" in want:       # air_time's stand-in (STAND_INS), per foot:
            # unload = the weight the foot has taken off the ground: 0 while it carries half the robot or more, 1 when
            # the ground carries none of it.  The ground force has a derivative (the plant's contact spring), so the
            # first part of a step, the weight moving onto the other foot, pays a little more at every newton.
            # Paid is the unload the foot has KEPT over the last hold_steps control steps (their smallest): air_time's
            # 0.05 s minimum, so rocking the weight from foot to foot pays nothing.  (W5b's first version paid the
            # unload of the moment, and the optimiser collected it by rocking about 12 times a second.)
            # air_time's 0.5 s limit counts seconds of unload (unload x dt) since the foot last carried nearly its half
            # (unload below unload_reset): a swing (unload 1) gets 0.5 s, as in air_time; neither leaning on one foot
            # nor resting a weightless foot on the floor is paid for ever; a quiet stance keeps its slope.
            # And air_time's request gate.
            unload = (1.0 - s.contact_force[..., 2] / self.half_weight).clamp(0.0, 1.0)
            held = torch.cat((mem["unload_history"][..., 1:], unload[..., None]), -1)    # the last hold_steps unloads
            new["unload_history"] = held
            u = unload.detach()
            spent = torch.where(u < p.unload_reset, torch.zeros_like(u), mem["unloaded_time"] + u * dt)
            new["unloaded_time"] = spent
            t["air_time_smooth"] = (held.amin(-1) * f(spent < p.air_time_range[1])).sum(1) * f(total > 0.2)
        if "gait_clock" in want:            # step in time with the clock (W5c; Booster Gym's feet swing, with a slope):
            # per foot, unload while the clock says it swings, carry while it says it stands, both measured by the
            # weight on the foot (the stand-in's unload), averaged over the feet; 1 = in time.  Paid at the phase the
            # network saw for this step, only while the clock runs.  Its slope: weight off for the foot whose turn it
            # is, weight on for the other.  The clock is time, so the derivative goes through the loads only.
            unload = (1.0 - s.contact_force[..., 2] / self.half_weight).clamp(0.0, 1.0)
            sw = self.swing(mem["gait_phase"])
            t["gait_clock"] = (sw * unload + (1.0 - sw) * (1.0 - unload)).mean(1) * f(self.clock_on(mem))
        if "foot_clearance" in want:
            cost = ((s.foot_height - swing_target[:, None]).abs() * foot_speed).sum(1)
            t["foot_clearance"] = cost * f(speed > 0.05) / speed.clamp(min=1.0)
        if "foot_swing_height" in want:     # peak swing height at touchdown against the target
            peak = torch.where(contact, mem["swing_peak"], torch.maximum(mem["swing_peak"], s.foot_height))
            error = peak / swing_target[:, None] - 1.0
            t["foot_swing_height"] = (error.square() * f(landed)).sum(1) * f(speed > 0.05)
            new["swing_peak"] = torch.where(landed, torch.zeros_like(peak), peak)
        if "foot_slip" in want:
            t["foot_slip"] = (foot_speed.square() * on).sum(1) * f(total > 0.05)
        if "soft_landing" in want:
            force = torch.linalg.vector_norm(s.contact_force, dim=-1)
            t["soft_landing"] = (force * f(landed)).sum(1) * f(speed > 0.05) / (1.0 + speed)
        if "upper_body_posture" in want:
            std = self._regime(total, *(self.upper_body_std[k][None] for k in ("standing", "walking", "running")),
                               running_threshold=1.0)
            t["upper_body_posture"] = ((theta - self.home)[:, self.upper_body].square() / std.square()).mean(1)
        if "standing_pose" in want:
            t["standing_pose"] = (theta - self.home).abs().mean(1) * (1.0 - (total / 0.05).clamp(max=1.0))
        if "toe_clearance" in want:         # toe tips of swinging feet below 4 cm, weighted by foot speed
            t["toe_clearance"] = ((p.toe_clearance - s.toe[..., 2]).clamp(min=0.0) * foot_speed * off).sum(1)
        if "leg_spacing" in want:
            t["leg_spacing"] = ((p.min_knee_spacing - knees_y).clamp(min=0.0) + (p.min_feet_spacing - feet_y).clamp(min=0.0)
                                + (feet_y - knees_y).clamp(min=0.0))
        if "feet_parallel" in want:
            forward = to_heading(s.yaw, s.foot_rot[..., :, 0])          # the feet's x axes in the heading frame
            foot_yaw = torch.atan2(forward[..., 1], forward[..., 0])
            t["feet_parallel"] = foot_yaw.square().sum(1) * f(target[:, 2].abs() <= 0.3)
        if "overspeed" in want:
            t["overspeed"] = (s.qd.abs() - p.overspeed_fraction * self.no_load_speed).clamp(min=0.0).square().sum(1)
        if "arm_swing" in want:             # hand tips relative to the centre of mass, heading frame
            arm = p.arm_swing
            tips = to_heading(s.yaw, s.hand_pos - s.com[:, None])
            centre = tips[..., 0].mean(1)
            ahead = (tips[..., 0] - arm["front"]).clamp(min=0.0).square().sum(1)
            behind = (-tips[..., 0] - arm["back"]).clamp(min=0.0).square().sum(1)
            wide = (tips[..., 1].abs() - arm["width"]).clamp(min=0.0).square().sum(1)
            t["arm_swing"] = (centre.square() + ahead + behind + wide) / 0.05**2
        if "knee_depth" in want:
            t["knee_depth"] = ((theta[:, self.knees] - p.knee_depth).clamp(min=0.0).square() * on).sum(1)
        if "yaw_rate_fine" in want:
            avg = mem["yaw_rate_avg"] + min(dt / p.yaw_rate_tau, 1.0) * (s.ang_vel_w[:, 2] - mem["yaw_rate_avg"])
            t["yaw_rate_fine"] = torch.exp(-((target[:, 2] - avg) / p.fine_std).square())
            new["yaw_rate_avg"] = avg
        if "action_jerk" in want:
            t["action_jerk"] = (a - 2.0 * a_last + a_before).square().sum(1) / (1.0 + 0.5 * speed)
        if "leg_acc" in want:
            t["leg_acc"] = s.dof_acc[:, self.legs].square().sum(1) / (1.0 + 0.5 * speed)
        if "trunk_bounce" in want:
            t["trunk_bounce"] = s.lin_vel_w[:, 2].square() / (1.0 + speed)
        if "velocity_fine" in want:
            avg = mem["velocity_avg"] + min(dt / p.velocity_tau, 1.0) * (track[:, :2] - mem["velocity_avg"])
            t["velocity_fine"] = torch.exp(-(target[:, :2] - avg).square().sum(1) / p.fine_std**2)
            new["velocity_avg"] = avg
        if "heading_hold" in want:          # heading change since the command was set, while no turn is requested
            still = target[:, 2].abs() < 0.05
            changed = ~((target - mem["last_target"]).abs().amax(1) < 1e-4) | ~still
            reference = torch.where(changed, s.yaw, mem["heading_reference"])
            t["heading_hold"] = wrap(s.yaw - reference).square() * f(still)
            new["heading_reference"], new["last_target"] = reference, target.clone()
        if "terminated" in want:
            t["terminated"] = f(terminated)
        if "foot_touchdown" in want:        # downward speed of a foot about to land (in the air, below 8 cm)
            low = f(s.foot_pos[..., 2] < p.touchdown_height)
            t["foot_touchdown"] = ((-s.foot_vel[..., 2]).clamp(min=0.0) * low * off).sum(1)
        if "foot_spacing" in want:          # feet outside the stance band, not while turning
            low, high = p.foot_spacing
            fast = ((target[:, 0].abs() - 1.5) / 1.5).clamp(0.0, 1.0)
            low = low + (p.foot_spacing_fast - low) * fast
            cost = ((low - feet_y).clamp(min=0.0) + (feet_y - high).clamp(min=0.0)).square()
            t["foot_spacing"] = cost * f(target[:, 2].abs() <= 0.3)
        if "knees_out" in want:
            t["knees_out"] = (feet_y + p.knees_out_margin - knees_y).clamp(min=0.0)
        if "heel_walking" in want:          # walking backwards: the stance feet's force centre behind the ankle
            along = (s.foot_rot.transpose(-1, -2) @ (s.contact_pos - s.foot_pos)[..., None]).squeeze(-1)[..., 0]
            t["heel_walking"] = ((-along).clamp(min=0.0) * on).sum(1) * f(target[:, 0] < -0.2)
        if "reference_lag" in want:         # speed short of the reference while it still ramps toward the request
            ramping = (target[:, :2] - commands[:, :2]).abs().amax(1) > 1e-3
            settled = mem["command_age"] >= p.start_grace_s
            t["reference_lag"] = torch.linalg.vector_norm(target[:, :2] - track[:, :2], dim=1) * f(ramping & settled)
        if "forward_lean" in want:
            allow = p.lean_rest + p.lean_per_mps * target[:, 0].clamp(min=0.0)
            t["forward_lean"] = (s.gravity[:, 0] - allow).clamp(min=0.0).square()
        if "backward_lean" in want:
            t["backward_lean"] = (-s.gravity[:, 0] - p.lean_back).clamp(min=0.0).square()
        if "energy" in want:
            t["energy"] = power_w.clamp(min=0.0)
        new["gait_phase"] = self.next_phase(mem)       # the gait clock moves on (read only by gait_clock and clock())
        return t, new

    def reward(self, terms: dict, only: tuple | None = None) -> torch.Tensor:
        """Weighted sum x policy step (as RL_Core does), of all terms or of the names in `only`."""
        names = [k for k in self.rewards if only is None or k in only]
        return sum(self.rewards[k] * terms[k] for k in names) * self.dt

    def critic_extra(self, s: Signals, mem: dict) -> torch.Tensor:
        """What Eric's critic sees beyond the actor's 75 (15): trunk velocity, foot heights, air times, contacts, forces."""
        force = s.contact_force.flatten(1)
        return torch.cat((s.base_lin_vel, s.foot_height, mem["air_time"], s.contact.to(self.dtype),
                          torch.sign(force) * torch.log1p(force.abs())), -1)

    def fallen(self, s: Signals) -> torch.Tensor:
        """The episode ends: the firmware's tilt abort, a body other than a foot on the ground, or the trunk down."""
        return (s.roll_pitch.abs() > C.TILT_ABORT).any(-1) | s.ground | (s.base_pos[:, 2] < C.FALL_HEIGHT)
