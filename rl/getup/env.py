r"""Get up from lying and stay standing on the port-Hamiltonian K1: an rsl-rl VecEnv on K1PHPlantNative.

The physics: the pH plant (motor electrics, contact elements, end stops, energy ledger) behind the robot's command
path (ph_k1/firmware.py: firmware filter, 10-40 ms delay -- the gains travel with the targets --, motor driver).

The controller (rl/getup/contract.py, impedance.py): the policy sets targets AND stiffness and damping of every
joint every 20 ms; an energy tank makes the closed loop passive (every energy injection of the controller is paid
from a bounded tank, refilled by its own damping).

The task follows RL_Core's K1StandUpEnv (rl_core/envs/k1_standup.py, the recipe of the get-up policy the K1 runs):
a library of lying poses settled by the physics, a growing share of lying starts, episodes that end only on the
10 s timeout (it gets up, stays up and gets up again after a push), always-on progress rewards, a standing bonus,
head penalties and strong pushes once it gets up, an optional upward assist that fades out with the success.
The rewards are taken from the Hamiltonian and the energy ledger where the pH plant can measure the quantity:
  height       centre-of-mass height = the gravity tank V_grav / (m g) of H, from lying to standing (0..1)
  stillness    kinetic energy T of H while standing (one mass-weighted term instead of velocity terms)
  battery      electrical power drawn from the battery, max(d supplied / dt, 0) [W]: work + heat, not only heat
  impacts      power dissipated in the contact elements (impacts, sliding) [W]
  end_stops    power dissipated in the end stops [W]
  voltage_margin  the steady-state coil voltage R tau / k + k qdot beyond 80 % of the supply: how far out on its
               torque-speed line the motor works (not the driver's L di/dt, which spikes on every torque step)
The energy term is weighted by a curriculum that grows with the get-up success (get up first, then cheaply).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from tensordict import TensorDict

from ph_k1 import firmware as fw
from ph_k1.dynamics import body_frames, com_height
from ph_k1.model import NB, NJ, load_k1
from ph_k1.native import K1PHPlantNative
from ph_k1.self_contact import SelfContactParams

from . import contract as C
from .impedance import EnergyTank
from .lying import KIND_NAMES, KINDS, STANDING, build_library, turn

try:
    from rsl_rl.env import VecEnv
except ImportError:  # the parts can be imported without rsl-rl
    VecEnv = object

# weight per term (unweighted values below); the sum is multiplied by the policy step (as RL_Core does)
REWARDS = {
    # progress, always on
    "height": 3.0,               # centre-of-mass height from lying (0) to standing (1), from the gravity tank of H
    "upright": 2.0,              # ((1 - g_z) / 2)^2 of the trunk
    # standing (gated: high, upright, both feet down, nothing else touching)
    "standing": 6.0,
    "stillness": -2.0,           # kinetic energy T [J] while standing
    # energy (x the energy curriculum)
    "battery": -0.03,            # electrical power drawn from the battery [W]
    # the hardware
    "impacts": -0.02,            # power dissipated in contacts [W]
    "end_stops": -0.1,           # power dissipated in the end stops [W]
    "voltage_margin": -1.0,      # sum (|R tau / k + k qdot| / v_max - 0.8)+: the motor's torque-speed line used
    "torque_margin": -2.0,       # sum (|tau| / tau_max - 0.8)+
    "soft_joint_limits": -2.0,   # rad beyond 90 % of the joint range
    "target_beyond_limits": -0.5,  # rad of decoded target beyond the joint range
    "self_proximity": -2.0,      # the plant has no self-collision: limb pairs that never touch, closer than 9 cm
    # smoothness
    "action_rate": -0.01,        # the 22 targets, + the 44 gain outputs x the gain curriculum
    "joint_acc": -2.5e-7,
    # head and neck (the neck breaks on head strikes; ramp in once it gets up)
    "head_contact": -1.0,
    "head_impact": -2.0,
    "neck_torque": -2.0,
    # NaN / explosion, the only non-timeout end
    "exploded": -200.0,
}
# costs of moving: weighted by the energy curriculum, which starts at 0 (get up first, then cheaply). With a
# penalty from the start the policy learnt to go limp lying down (stiffness x0.4, 350 -> 16 W) and never got up
# (runs getup-ph-v1, getup-ph-notank-v1).  impacts = contact dissipation, which pushing up from the floor causes.
ENERGY_TERMS = ("battery", "impacts")
HEAD_TERMS = ("head_contact", "head_impact", "neck_torque")
# the get-up example's self-contact proxy (Stand Up humanoid_standup/robot_config.py): link origins, pairs of
# non-adjacent limbs that should never press together (tuck-safe: no hand/foot/knee - trunk pairs)
SELF_LINKS = ("left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link", "Left_Shank", "Right_Shank",
              "Left_Arm_3", "Right_Arm_3", "Head_2")
SELF_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 5), (1, 4), (0, 3), (1, 2), (0, 8), (1, 8))
HEAD_BODIES = ("Head_1", "Head_2")


@dataclass
class GetUpConfig:
    num_envs: int = 4096
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 1
    episode_s: float = 10.0
    policy_dt: float = 0.02
    plant_dt: float = 1e-3
    steps_per_env: int = 24           # PPO rollout length: the curriculum counts iterations with it
    delay_ms: tuple = (10.0, 70.0)    # k1_skill_cfg: 2..14 steps of 5 ms
    delay_hold_prob: float = 0.3      # k1_skill_cfg: the lag is kept for a policy step with probability 0.3
    gain_dr: float = 0.10             # the driver's real gains = commanded x U(1 -+ gain_dr), per robot and episode
    obs_noise: dict = field(default_factory=lambda: {"gyro": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 1.0})
    action_guard: float = 100.0
    # limbs cannot pass through each other (ph_k1/self_contact.py): without it the policy tucked its hand through
    # the shank, which works on the plant and not on the robot
    self_collision: bool = True
    # the controller
    tank: bool = True
    tank_capacity: float = 150.0      # J
    tank_refill: float = 1.0          # share of the controller's damping dissipation that goes back into the tank
    # starts
    lying_kinds: tuple = ("supine", "prone")
    lying_share: tuple = (0.5, 0.85)  # grows with the lying success
    start: str | None = None          # force a start kind ("standing", "supine", ...): videos, tests
    reset_joint_noise: float = 0.02
    library_size: int | None = None   # drops per round (default num_envs)
    library_rounds: int = 1
    # standing: trunk above the stand height - margin, upright within the angle, both feet down, nothing else
    standing_height_margin: float = 0.04
    standing_tilt_deg: float = 15.0
    success_ema: float = 0.01
    # assist: an upward force on lying starts that are not standing, fading with their success (0 = off)
    assist_force: float = 100.0       # N (the K1 weighs 193 N)
    assist_off_ema: float = 0.5
    # head penalties start once both starts succeed this often, then ramp in over these iterations
    head_penalty_ema: float = 0.5
    head_ramp_iters: int = 500
    # energy curriculum: the energy terms' weight x lo .. 1 as the success EMA goes from ema_lo to ema_hi
    energy_scale_lo: float = 0.0
    energy_ema: tuple = (0.5, 0.9)
    # gain curriculum: the 44 gain outputs act only once both lying starts succeed this often (None: from the
    # start), then ramp in over gain_ramp_iters; before that the driver gets the agent's fixed gains, as the stock
    # StandupAgent (RL_Core's get-up recipe): learn to get up first, then variable impedance
    gain_free_ema: float | None = 0.5
    gain_ramp_iters: int = 500
    # pushes on standing robots; strong pushes (x strong_push_scale) ramp in once the head penalties are full
    push_interval_s: tuple = (3.0, 6.0)
    push_dv: float = 0.3
    strong_push_scale: float = 2.5
    strong_push_ramp_iters: int = 500
    # NaN / explosion
    max_joint_vel: float = 100.0
    max_base_lin_vel: float = 20.0
    rewards: dict = field(default_factory=lambda: dict(REWARDS))


class GetUpEnv(VecEnv):
    def __init__(self, cfg: GetUpConfig | None = None):
        self.cfg = cfg = cfg or GetUpConfig()
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        self.num_envs = B = cfg.num_envs
        self.num_actions = C.NUM_ACTIONS
        self.max_episode_length = int(round(cfg.episode_s / cfg.policy_dt))
        self.substeps = int(round(cfg.policy_dt / cfg.plant_dt))
        unknown = set(cfg.rewards) - set(REWARDS)
        if unknown:
            raise KeyError(f"unknown reward terms {sorted(unknown)}; known: {sorted(REWARDS)}")

        model = load_k1(dtype=self.dtype, device=self.device)
        C.check_joint_order(model.joint_names)
        self.model = model
        self.fw = fw.FirmwareConfig(gains=C.GAINS, slew=C.SLEW, delay_ms=tuple(cfg.delay_ms), policy_dt=cfg.policy_dt,
                                    delay_hold_prob=cfg.delay_hold_prob, delay_gains=True)
        self.self_contact = SelfContactParams(enabled=cfg.self_collision)
        self.plant = K1PHPlantNative(model=model, batch=B, timestep=cfg.plant_dt, firmware=self.fw,
                                     self_contact=self.self_contact)
        self.contract = C.Contract(self.dtype, self.device)
        self.kp0, self.kd0 = (g.clone() for g in self.plant.gains())  # the agent's gains before its first command
        self.tank = EnergyTank(B, NJ, self.dtype, self.device, cfg.tank_capacity, cfg.tank_refill, enabled=cfg.tank)
        mot = self.plant.motors
        self.v_max, self.tau_max = float(mot.v_max), mot.tau_max.clone()
        self.mot_R, self.mot_k = mot.R.clone(), mot.k.clone()

        # contact bookkeeping: the 8 sole points, the head points, every other point
        sole, cb = model.contact_is_sole, model.contact_body
        left, right = model.body_names.index("left_foot_link"), model.body_names.index("right_foot_link")
        self.left_sole = (sole & (cb == left)).to(self.dtype)
        self.right_sole = (sole & (cb == right)).to(self.dtype)
        self.other_point = ~sole
        heads = [model.body_names.index(n) for n in HEAD_BODIES]
        self.head_point = torch.stack([cb == h for h in heads])  # (2, nc)
        self.self_links = torch.as_tensor([model.body_names.index(n) for n in SELF_LINKS], device=self.device)
        self.self_pairs = torch.as_tensor(SELF_PAIRS, device=self.device)
        self.neck = torch.as_tensor([0, 1], device=self.device)
        span = model.jnt_high - model.jnt_low
        mid = 0.5 * (model.jnt_high + model.jnt_low)
        self.soft_lo, self.soft_hi = mid - 0.45 * span, mid + 0.45 * span

        z = lambda *s: torch.zeros(*s, dtype=self.dtype, device=self.device)  # noqa: E731
        zb = lambda *s: torch.zeros(*s, dtype=torch.bool, device=self.device)  # noqa: E731
        self.actions, self.prev_actions = z(B, C.NUM_ACTIONS), z(B, C.NUM_ACTIONS)
        self.qd, self.prev_qd, self.gyro, self.v_body = z(B, NJ), z(B, NJ), z(B, 3), z(B, 3)
        self.contact = zb(B, model.ncontact)
        self.gain_err_kp, self.gain_err_kd = z(B, NJ) + 1, z(B, NJ) + 1
        self.start_type = torch.zeros(B, dtype=torch.long, device=self.device)
        self.standing, self.exploded, self.up = zb(B), zb(B), zb(B)
        self.time_to_stand = z(B) + float("nan")
        self.push_timer = z(B)
        self.episode_length_buf = torch.zeros(B, dtype=torch.long, device=self.device)
        self.episode_sums = {k: z(B) for k in cfg.rewards}
        self.ep_battery, self.ep_to_stand, self.ep_stand_energy, self.ep_stand_time = z(B), z(B) + float("nan"), z(B), z(B)
        self.ep_dv = z(B) + float("nan")  # potential energy gained from the start to the first standing [J]
        self.mg = float(model.mass.sum()) * 9.81
        self.com_start, self.com_z, self.kinetic, self.battery_w = z(B), z(B), z(B), z(B)
        self.assist_now = z(B)
        self.common_step_counter = 0
        self.extra_log = {}

        # curriculum
        names = list(cfg.lying_kinds)
        if cfg.start not in (None, "standing") and cfg.start not in names:
            names.append(cfg.start)
        self.kinds = tuple(KINDS[k] for k in names)
        self.lying_share = float(cfg.lying_share[0])
        self.success_ema = {k: 0.0 for k in self.kinds}
        self.assist_scale = {k: 1.0 for k in self.kinds}
        self.head_start_iter, self.head_scale = None, 0.0
        self.push_start_iter, self.push_scale = None, 1.0
        self.energy_scale = cfg.energy_scale_lo
        self.gain_start_iter = None
        self.gain_scale = 1.0 if cfg.gain_free_ema is None else 0.0

        # the stand reference: HOME with the soles on the floor
        self.plant.reset(pose=self.contract.home)
        self.stand_z = float(self.plant.q[:, 2].mean())
        self.com_stand = float(com_height(model, body_frames(model, self.plant.q)[2]).mean())
        if cfg.start != "standing":
            self.library = build_library(model, cfg.library_size or B, self.kinds, self.contract.home,
                                         rounds=cfg.library_rounds, policy_dt=cfg.policy_dt, plant_dt=cfg.plant_dt,
                                         self_contact=self.self_contact)
            self.lib_index = {k: self.library.index(k) for k in self.kinds}
            self.com_low = float(self.library.com_z.mean())
        else:
            self.library, self.lib_index, self.com_low = None, {}, 0.1
        self._reset_idx(torch.arange(B, device=self.device))
        self.extras = {"log": {}}

    # ------------------------------------------------------------------ utils
    def _uniform(self, lo, hi, shape):
        return lo + (hi - lo) * torch.rand(shape, dtype=self.dtype, device=self.device)

    @property
    def iteration(self) -> float:
        return self.common_step_counter / self.cfg.steps_per_env

    # ------------------------------------------------------------------ reset
    def _choose_starts(self, n):
        c = self.cfg
        if c.start is not None:
            return torch.full((n,), KINDS[c.start], dtype=torch.long, device=self.device)
        lying = torch.rand(n, device=self.device) < self.lying_share
        kinds = torch.as_tensor(self.kinds, device=self.device)
        pick = kinds[torch.randint(len(kinds), (n,), device=self.device)]
        return torch.where(lying, pick, torch.full_like(pick, STANDING))

    def _reset_idx(self, ids: torch.Tensor) -> None:
        n = int(ids.numel())
        if n == 0:
            return
        c, m = self.cfg, self.model
        self._update_curriculum(ids)
        kind = self._choose_starts(n)
        self.start_type[ids] = kind
        # standing starts: HOME, soles on the floor
        stand = (kind == STANDING).nonzero(as_tuple=False).squeeze(-1)
        if len(stand):
            pose = self.contract.home + self._uniform(-c.reset_joint_noise, c.reset_joint_noise, (len(stand), NJ))
            pose = torch.minimum(torch.maximum(pose, m.jnt_low), m.jnt_high)
            self.plant.reset(pose=pose, env_ids=ids[stand])
        # lying starts: a settled library pose at a random heading
        for k in self.kinds:
            rows = (kind == k).nonzero(as_tuple=False).squeeze(-1)
            if len(rows) == 0:
                continue
            idx = self.lib_index[k]
            pick = idx[torch.randint(len(idx), (len(rows),), device=self.device)]
            ql = self.library.q[pick]
            quat = turn(ql[:, 3:7], self._uniform(-math.pi, math.pi, (len(rows),)))
            self.plant.reset(pose=ql[:, 7:], base_pos=(0.0, 0.0, ql[:, 2]), base_quat=quat, env_ids=ids[rows])
        self.tank.reset(ids, self.plant.theta, self.kp0, self.kd0)
        self.gain_err_kp[ids] = self._uniform(1 - c.gain_dr, 1 + c.gain_dr, (n, NJ))
        self.gain_err_kd[ids] = self._uniform(1 - c.gain_dr, 1 + c.gain_dr, (n, NJ))
        for buf in (self.actions, self.prev_actions, self.qd, self.prev_qd, self.gyro, self.v_body):
            buf[ids] = 0.0
        self.contact[ids] = False
        self.standing[ids], self.exploded[ids], self.up[ids] = False, False, False
        self.time_to_stand[ids] = float("nan")
        self.push_timer[ids] = self._uniform(*c.push_interval_s, (n,))
        self.episode_length_buf[ids] = 0
        for v in self.episode_sums.values():
            v[ids] = 0.0
        self.ep_battery[ids], self.ep_to_stand[ids], self.ep_dv[ids] = 0.0, float("nan"), float("nan")
        self.ep_stand_energy[ids], self.ep_stand_time[ids] = 0.0, 0.0
        self.com_start[ids] = com_height(m, body_frames(m, self.plant.q[ids])[2])

    # ------------------------------------------------------------ curriculum
    def _update_curriculum(self, ids):
        c = self.cfg
        ran = self.episode_length_buf[ids] > 0  # not the initial reset
        log = self.extra_log
        for k in self.kinds:
            mask = (self.start_type[ids] == k) & ran
            if bool(mask.any()):
                rate = float(self.standing[ids][mask].to(self.dtype).mean())
                self.success_ema[k] += c.success_ema * (rate - self.success_ema[k])
                log[f"Metrics/success_{KIND_NAMES[k]}"] = rate
                tts = self.time_to_stand[ids][mask]
                tts = tts[torch.isfinite(tts)]
                if len(tts):
                    log[f"Metrics/time_to_stand_{KIND_NAMES[k]}"] = float(tts.mean())
                e = self.ep_to_stand[ids][mask]
                stood = torch.isfinite(e)
                if bool(stood.any()):  # the thesis numbers: battery energy from lying to standing, and its efficiency
                    log[f"Energy/getup_J_{KIND_NAMES[k]}"] = float(e[stood].mean())
                    dv = self.ep_dv[ids][mask][stood]
                    log[f"Energy/getup_efficiency_{KIND_NAMES[k]}"] = float((dv / e[stood].clamp_min(1e-6)).mean())
        mask = (self.start_type[ids] == STANDING) & ran
        if bool(mask.any()):
            log["Metrics/success_standing"] = float(self.standing[ids][mask].to(self.dtype).mean())
        st = self.ep_stand_time[ids] * ran
        if bool((st > 1.0).any()):
            sel = st > 1.0
            log["Energy/standing_W"] = float((self.ep_stand_energy[ids][sel] / st[sel]).mean())
        if not self.kinds:
            return
        ema = min(self.success_ema.values())
        lo, hi = c.lying_share
        self.lying_share = lo + (hi - lo) * min(1.0, ema / 0.8)
        assisted = False
        if c.assist_force > 0:
            for k in self.kinds:
                self.assist_scale[k] = min(self.assist_scale[k], max(0.0, 1.0 - self.success_ema[k] / c.assist_off_ema))
            assisted = max(self.assist_scale.values()) > 0.0
        it = self.iteration
        if self.head_start_iter is None and not assisted and ema >= c.head_penalty_ema:
            self.head_start_iter = it
            print(f"[getup] head penalties ramp in from iteration {it:.0f}", flush=True)
        if self.head_start_iter is not None:
            self.head_scale = min(1.0, (it - self.head_start_iter) / max(1, c.head_ramp_iters))
        if c.strong_push_scale != 1.0 and self.push_start_iter is None and self.head_scale >= 1.0:
            self.push_start_iter = it
            print(f"[getup] strong pushes ramp in from iteration {it:.0f}", flush=True)
        if self.push_start_iter is not None:
            ramp = min(1.0, (it - self.push_start_iter) / max(1, c.strong_push_ramp_iters))
            self.push_scale = 1.0 + (c.strong_push_scale - 1.0) * ramp
        if c.gain_free_ema is not None:
            if self.gain_start_iter is None and ema >= c.gain_free_ema:
                self.gain_start_iter = it
                print(f"[getup] variable gains ramp in from iteration {it:.0f}", flush=True)
            if self.gain_start_iter is not None:
                self.gain_scale = min(1.0, (it - self.gain_start_iter) / max(1, c.gain_ramp_iters))
        e_lo, e_hi = c.energy_ema
        self.energy_scale = c.energy_scale_lo + (1.0 - c.energy_scale_lo) * min(1.0, max(0.0, (ema - e_lo) / (e_hi - e_lo)))
        log["Curriculum/lying_share"] = self.lying_share
        log["Curriculum/success_ema"] = ema
        log["Curriculum/head_scale"] = self.head_scale
        log["Curriculum/push_scale"] = self.push_scale
        log["Curriculum/energy_scale"] = self.energy_scale
        log["Curriculum/gain_scale"] = self.gain_scale
        for k in self.kinds:
            log[f"Curriculum/assist_{KIND_NAMES[k]}_N"] = c.assist_force * self.assist_scale[k]

    def curriculum_state(self) -> dict:
        """What a resumed run needs to continue the curriculum (saved inside every checkpoint)."""
        return {"success_ema": {KIND_NAMES[k]: v for k, v in self.success_ema.items()},
                "assist_scale": {KIND_NAMES[k]: v for k, v in self.assist_scale.items()},
                "lying_share": self.lying_share, "head_start_iter": self.head_start_iter, "head_scale": self.head_scale,
                "push_start_iter": self.push_start_iter, "push_scale": self.push_scale,
                "energy_scale": self.energy_scale, "gain_start_iter": self.gain_start_iter,
                "gain_scale": self.gain_scale, "common_step_counter": self.common_step_counter}

    def load_curriculum_state(self, s: dict) -> None:
        for k in self.kinds:
            self.success_ema[k] = float(s["success_ema"].get(KIND_NAMES[k], 0.0))
            self.assist_scale[k] = float(s["assist_scale"].get(KIND_NAMES[k], 1.0))
        self.lying_share = float(s["lying_share"])
        self.head_start_iter, self.head_scale = s["head_start_iter"], float(s["head_scale"])
        self.push_start_iter, self.push_scale = s["push_start_iter"], float(s["push_scale"])
        self.energy_scale = float(s["energy_scale"])
        self.gain_start_iter = s.get("gain_start_iter")
        self.gain_scale = float(s.get("gain_scale", 1.0))  # checkpoints from before the gain curriculum: full range
        self.common_step_counter = int(s["common_step_counter"])

    # ------------------------------------------------------------ external inputs
    def _pushes(self):
        c = self.cfg
        if c.push_dv <= 0:
            return None
        self.push_timer -= c.policy_dt
        fire = (self.push_timer <= 0) & self.standing
        self.push_timer = torch.where(self.push_timer <= 0, self._uniform(*c.push_interval_s, (self.num_envs,)),
                                      self.push_timer)
        if not bool(fire.any()):
            return None
        ids = fire.nonzero(as_tuple=False).squeeze(-1)
        n = int(ids.numel())
        angle = self._uniform(0.0, 2 * math.pi, (n,))
        speed = self._uniform(0.0, c.push_dv * self.push_scale, (n,))
        dv = torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        dv[ids, 0], dv[ids, 1] = speed * torch.cos(angle), speed * torch.sin(angle)
        return dv

    def _assist(self):
        c = self.cfg
        self.assist_now.zero_()
        if c.assist_force <= 0 or max(self.assist_scale.values(), default=0.0) <= 0:
            return None
        for k in self.kinds:
            self.assist_now[self.start_type == k] = c.assist_force * self.assist_scale[k]
        self.assist_now *= (~self.standing).to(self.dtype)
        f = torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        f[:, 2] = self.assist_now
        return f

    # ----------------------------------------------------------------- terms
    def _terms(self, info, target_raw, P, d_supplied, d_contact, d_stops) -> dict:
        m, c, q = self.model, self.cfg, self.plant.q
        dt = c.policy_dt
        theta = q[:, 7:]
        g = self.contract.projected_gravity(q[:, 3:7])
        sp = self.self_links
        links = P[:, sp]
        d = (links[:, self.self_pairs[:, 0]] - links[:, self.self_pairs[:, 1]]).norm(dim=-1)
        f = info.contact_force.norm(dim=-1)  # (B, nc)
        head_f = torch.stack([(f * hp).amax(-1) for hp in self.head_point.to(self.dtype)], -1)  # (B, 2)
        head_touch = (self.contact[:, None, :] & self.head_point[None]).any(-1).any(-1)
        excess = target_raw - torch.minimum(torch.maximum(target_raw, m.jnt_low), m.jnt_high)
        standing = self.standing.to(self.dtype)
        return {
            "height": ((self.com_z - self.com_low) / (self.com_stand - self.com_low)).clamp(0.0, 1.0),
            "upright": ((1.0 - g[:, 2]) / 2.0).square(),
            "standing": standing,
            "stillness": standing * self.kinetic,
            "battery": d_supplied.clamp_min(0.0) / dt,
            "impacts": d_contact.clamp_min(0.0) / dt,
            "end_stops": d_stops.clamp_min(0.0) / dt,
            "voltage_margin": ((self.mot_R * info.torque / self.mot_k + self.mot_k * self.qd).abs() / self.v_max
                               - 0.8).clamp_min(0.0).sum(-1),
            "torque_margin": (info.torque.abs() / self.tau_max - 0.8).clamp_min(0.0).sum(-1),
            "soft_joint_limits": ((self.soft_lo - theta).clamp_min(0) + (theta - self.soft_hi).clamp_min(0)).sum(-1),
            "target_beyond_limits": excess.abs().sum(-1),
            "self_proximity": (1.0 - d / 0.09).clamp(0.0, 1.0).sum(-1),
            "action_rate": ((self.actions - self.prev_actions).square()
                            * torch.cat((torch.ones(NJ, dtype=self.dtype, device=self.device),
                                         torch.full((2 * NJ,), self.gain_scale, dtype=self.dtype,
                                                    device=self.device)))).sum(-1),
            "joint_acc": ((self.qd - self.prev_qd) / dt).square().sum(-1),
            "head_contact": head_touch.to(self.dtype),
            "head_impact": ((head_f - 50.0).clamp_min(0.0) / 100.0).clamp(max=10.0).sum(-1),
            "neck_torque": (info.torque[:, self.neck].abs() / self.tau_max[self.neck] - 0.5).clamp_min(0.0).sum(-1),
            "exploded": self.exploded.to(self.dtype),
        }

    # --------------------------------------------------------- observations
    def _noise(self, x, key):
        s = self.cfg.obs_noise.get(key, 0.0)
        return x if s <= 0 else x + s * (2 * torch.rand_like(x) - 1)

    def get_observations(self) -> TensorDict:
        q = self.plant.q
        theta = q[:, 7:]
        g = self.contract.projected_gravity(q[:, 3:7])
        tank = self.tank.fraction()
        clean = self.contract.observe(self.gyro, g, theta, self.qd, self.actions, tank)
        actor = self.contract.observe(self._noise(self.gyro, "gyro"), self._noise(g, "gravity"),
                                      self._noise(theta, "joint_pos"), self._noise(self.qd, "joint_vel"),
                                      self.actions, tank)
        active = self.contact.to(self.dtype)
        feet = torch.stack(((active @ self.left_sole) > 0, (active @ self.right_sole) > 0), -1).to(self.dtype)
        others = (self.contact & self.other_point).to(self.dtype).mean(-1, keepdim=True)
        critic = torch.cat((clean, self.v_body, q[:, 2:3] - self.stand_z, (self.com_z - self.com_stand)[:, None], feet,
                            others, self.standing.to(self.dtype)[:, None], self.battery_w[:, None] / 100.0,
                            self.kinetic[:, None] / 10.0, self.assist_now[:, None] / 100.0), -1)
        return TensorDict({"policy": torch.nan_to_num(actor), "critic": torch.nan_to_num(critic)},
                          batch_size=[self.num_envs], device=self.device)

    # ----------------------------------------------------------------- step
    def step(self, actions: torch.Tensor):
        c, m = self.cfg, self.model
        dt = c.policy_dt
        a = torch.nan_to_num(actions.to(self.device, self.dtype)).clamp(-c.action_guard, c.action_guard)
        self.prev_actions = self.actions
        self.actions = a
        target_raw, kp, kd = self.contract.decode(a, self.gain_scale)
        kp, kd = kp * self.gain_err_kp, kd * self.gain_err_kd  # what the driver really applies
        theta = self.plant.theta
        filt = lambda t, k, d: fw.safety_filter(m, self.fw, t, self.plant.last_target, theta, self.qd, k, d)  # noqa: E731
        target, kp, kd, tank = self.tank.step(theta, self.qd, self.prev_qd, dt, target_raw, kp, kd, filt)

        led0 = self.plant.ledger
        # (the ledger fields are views of the plant's buffer, which the step updates in place)
        supplied0, contact0, stops0, copper0 = (led0.supplied.clone(), led0.contact.clone(), led0.stops.clone(),
                                                led0.copper.clone())
        info = self.plant.policy_step(target, self.substeps, kp=kp, kd=kd, push_dv=self._pushes(),
                                      trunk_force=self._assist())
        self.common_step_counter += 1
        nu = info.extra["nu"]
        self.prev_qd = self.qd
        self.qd, self.v_body, self.gyro = nu[:, NB:].clone(), nu[:, 3:6].clone(), info.extra["gyro"].clone()
        self.contact = info.contact_active.clone()
        led = self.plant.ledger
        d_supplied, d_contact, d_stops = led.supplied - supplied0, led.contact - contact0, led.stops - stops0
        self.battery_w = d_supplied.clamp_min(0.0) / dt
        self.kinetic = 0.5 * (self.plant.p * nu).sum(-1)
        q = self.plant.q
        _, P, com = body_frames(m, q)
        self.com_z = com_height(m, com)
        self.episode_length_buf += 1

        # standing: high, upright, both feet down, nothing else touching
        g = self.contract.projected_gravity(q[:, 3:7])
        state = torch.cat((q, nu), -1)
        self.exploded = (~torch.isfinite(state).all(-1) | (self.qd.abs().amax(-1) > c.max_joint_vel)
                         | (self.v_body.norm(dim=-1) > c.max_base_lin_vel))
        active = self.contact.to(self.dtype)
        feet = ((active @ self.left_sole) > 0) & ((active @ self.right_sole) > 0)
        others = (self.contact & self.other_point).any(-1)
        high = q[:, 2] > self.stand_z - c.standing_height_margin
        upright = -g[:, 2] > math.cos(math.radians(c.standing_tilt_deg))
        self.standing = high & upright & feet & ~others & ~self.exploded
        self.up = high & (self.up | self.standing)
        t_now = self.episode_length_buf.to(self.dtype) * dt
        first = self.standing & torch.isnan(self.time_to_stand)
        self.time_to_stand = torch.where(first, t_now, self.time_to_stand)
        self.ep_battery += self.battery_w * dt
        self.ep_to_stand = torch.where(first, self.ep_battery, self.ep_to_stand)
        self.ep_dv = torch.where(first, self.mg * (self.com_z - self.com_start), self.ep_dv)
        self.ep_stand_energy += self.battery_w * dt * self.standing.to(self.dtype)
        self.ep_stand_time += dt * self.standing.to(self.dtype)

        terms = self._terms(info, target_raw, P, d_supplied, d_contact, d_stops)
        reward = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        for k, w in c.rewards.items():
            scale = self.energy_scale if k in ENERGY_TERMS else (self.head_scale if k in HEAD_TERMS else 1.0)
            v = torch.nan_to_num(w * scale * terms[k] * dt, nan=0.0, posinf=0.0, neginf=0.0)
            self.episode_sums[k] += v
            reward += v

        timeout = (self.episode_length_buf >= self.max_episode_length) & ~self.exploded
        dones = self.exploded | timeout
        log = {
            "Energy/battery_W": self.battery_w.mean(),
            "Energy/copper_W": ((led.copper - copper0) / dt).mean(),
            "Energy/contact_W": (d_contact / dt).mean(),
            "Energy/tank_J": tank["level"].mean(),
            "Energy/tank_limited": tank["limited"].to(self.dtype).mean(),
            "Energy/tank_injected_W": (tank["injected"].clamp_min(0.0) / dt).mean(),
            "Energy/tank_violation_J": tank["violation"].mean(),
            "Metrics/standing": self.standing.to(self.dtype).mean(),
            "Metrics/com_height_m": self.com_z.mean(),
            "Metrics/stiffness_scale": (kp / (self.contract.kp_ref * self.gain_err_kp)).mean(),
            "Metrics/damping_scale": (kd / (self.contract.kd_ref * self.gain_err_kd)).mean(),
        }
        ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            for k, v in self.episode_sums.items():
                log[f"Episode_Reward/{k}"] = v[ids].mean()
            log["Episode_Termination/exploded"] = self.exploded[ids].to(self.dtype).mean()
            log["Episode_Termination/timeout"] = timeout[ids].to(self.dtype).mean()
            self.extra_log = {}
            self._reset_idx(ids)
            log.update(self.extra_log)
        self.extras = {"time_outs": timeout, "log": log}
        return self.get_observations(), reward, dones, self.extras

    def config_dict(self) -> dict:
        d = asdict(self.cfg)
        d["contract"] = {"agent": "standup_ph", "obs_dim": C.OBS_DIM, "num_actions": C.NUM_ACTIONS,
                         "hidden_dims": list(C.HIDDEN_DIMS), "gain_range_log2": C.GAIN_RANGE, "gains": C.GAINS}
        d["stand_trunk_z_m"], d["com_stand_m"], d["com_lying_m"] = self.stand_z, self.com_stand, self.com_low
        return d
