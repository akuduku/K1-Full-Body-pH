r"""Stand still on the port-Hamiltonian K1: an rsl-rl VecEnv on K1PHPlantNative.

The task is the firmware's `stand` agent (rl/stand/contract.py): 72 observations the robot can measure, 22 actions
decoded around the firmware's default pose, the firmware's gains and command path (ph_k1/firmware.py: clamp,
slew, target clamp, overspeed guard, 10-40 ms delay, motor driver), and the firmware's abort (|roll| or
|pitch| > 0.6 rad).  The physics underneath is the pH plant: coil currents, torque limits, contact bristles.

Episodes start as on the robot after Prepare + countdown: standing still in the start pose, soles on the floor
(no assist; the untrained policy falls slowly, early termination keeps the samples near upright).  Rewards are
RL_Core's K1StandEnv terms (rl_core/envs/k1_stand.py) plus the pH plant's copper loss R*i^2 (heat in the coils).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from tensordict import TensorDict

from ph_k1.firmware import FirmwareConfig
from ph_k1.model import NJ, load_k1
from ph_k1.native import K1PHPlantNative

from . import contract as C

try:
    from rsl_rl.env import VecEnv
except ImportError:  # the contract/env can be imported without rsl-rl (tests of the parts)
    VecEnv = object

# weights per reward term (unweighted terms below); the sum is multiplied by the policy step (as RL_Core does)
REWARDS = {
    "orientation": -5.0,
    "base_height": -20.0,
    "base_lin_vel": -2.0,
    "base_ang_vel": -0.2,
    "joint_vel": -2.0e-3,
    "joint_acc": -2.5e-7,
    "action_rate": -0.1,
    "similar_to_default": -0.2,
    "joint_limits": -5.0,
    "undesired_contacts": -1.0,
    "feet_grounded": 0.5,
    "alive": 1.5,
    "action_excess": -0.5,
    "copper_loss": -0.005,  # per watt of coil heat: the pH plant's energy term
    "termination": -50.0,   # once per fall (x policy_dt = -1): falling must never pay, even while exploring
}


@dataclass
class StandConfig:
    num_envs: int = 4096
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 1
    episode_s: float = 10.0
    policy_dt: float = 0.02          # 50 Hz, the firmware's kDt
    plant_dt: float = 1e-3
    joint_noise: float = 0.02        # rad, start pose noise
    gain_dr: float = 0.10            # kp, kd x U(1 - gain_dr, 1 + gain_dr) per robot and joint, per episode
    delay_ms: tuple = (10.0, 40.0)   # transport delay, drawn per robot per episode
    slew: bool = C.SLEW
    push_interval_s: tuple = (4.0, 8.0)
    push_dv: float = 0.3             # max horizontal trunk velocity change of a push [m/s]; 0 = no pushes
    min_trunk_height: float = 0.28   # m (RL_Core K1StandEnv termination)
    action_guard: float = 100.0      # numerical guard only (the firmware does not clip); never reached by a sane policy
    obs_noise: dict = field(default_factory=lambda: {"gyro": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 1.0})
    rewards: dict = field(default_factory=lambda: dict(REWARDS))


class StandEnv(VecEnv):
    def __init__(self, cfg: StandConfig | None = None):
        self.cfg = cfg = cfg or StandConfig()
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
        fw = FirmwareConfig(gains=C.GAINS, slew=cfg.slew, delay_ms=tuple(cfg.delay_ms), policy_dt=cfg.policy_dt)
        self.plant = K1PHPlantNative(model=model, batch=B, timestep=cfg.plant_dt, firmware=fw)
        self.contract = C.Contract(self.dtype, self.device)
        self.base_kp, self.base_kd = (g.clone() for g in self.plant.gains())

        # contact bookkeeping: the 8 sole points (4 per foot) and the bodies of all other points
        sole = model.contact_is_sole
        cb = model.contact_body
        left, right = model.body_names.index("left_foot_link"), model.body_names.index("right_foot_link")
        self.left_sole = (sole & (cb == left)).to(self.dtype)
        self.right_sole = (sole & (cb == right)).to(self.dtype)
        self.other_body = torch.nn.functional.one_hot(cb, model.nbody).to(self.dtype) * (~sole)[:, None].to(self.dtype)

        z = lambda *s: torch.zeros(*s, dtype=self.dtype, device=self.device)  # noqa: E731
        self.actions, self.prev_actions = z(B, NJ), z(B, NJ)
        self.qd, self.prev_qd, self.gyro, self.v_body = z(B, NJ), z(B, NJ), z(B, 3), z(B, 3)
        self.contact = torch.zeros(B, model.ncontact, dtype=torch.bool, device=self.device)
        self.kp, self.kd = self.base_kp.clone(), self.base_kd.clone()
        self.push_timer = z(B)
        self.episode_length_buf = torch.zeros(B, dtype=torch.long, device=self.device)
        self.episode_sums = {k: z(B) for k in cfg.rewards}

        # the stand height: the start pose without noise, soles on the floor
        self.plant.reset(pose=self.contract.default_qpos)
        self.stand_height = float(self.plant.q[:, 2].mean())
        self._reset_idx(torch.arange(B, device=self.device))
        self.extras = {"log": {}}

    # ------------------------------------------------------------------ reset
    def _uniform(self, lo, hi, shape):
        return lo + (hi - lo) * torch.rand(shape, dtype=self.dtype, device=self.device)

    def _reset_idx(self, ids: torch.Tensor) -> None:
        n = int(ids.numel())
        if n == 0:
            return
        c, m = self.cfg, self.model
        pose = self.contract.default_qpos + self._uniform(-c.joint_noise, c.joint_noise, (n, NJ))
        pose = torch.minimum(torch.maximum(pose, m.jnt_low), m.jnt_high)
        self.plant.reset(pose=pose, env_ids=ids)  # soles on the floor, at rest; firmware state, delay line, delays
        self.kp[ids] = self.base_kp[ids] * self._uniform(1 - c.gain_dr, 1 + c.gain_dr, (n, NJ))
        self.kd[ids] = self.base_kd[ids] * self._uniform(1 - c.gain_dr, 1 + c.gain_dr, (n, NJ))
        for buf in (self.actions, self.prev_actions, self.qd, self.prev_qd, self.gyro, self.v_body):
            buf[ids] = 0.0
        self.contact[ids] = False
        self.push_timer[ids] = self._uniform(*c.push_interval_s, (n,))
        self.episode_length_buf[ids] = 0
        for v in self.episode_sums.values():
            v[ids] = 0.0

    # --------------------------------------------------------------- pushes
    def _pushes(self):
        c = self.cfg
        if c.push_dv <= 0:
            return None
        self.push_timer -= c.policy_dt
        fire = self.push_timer <= 0
        if not bool(fire.any()):
            return None
        ids = fire.nonzero(as_tuple=False).squeeze(-1)
        n = int(ids.numel())
        angle = self._uniform(0.0, 2 * math.pi, (n,))
        speed = self._uniform(0.0, c.push_dv, (n,))
        dv = torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        dv[ids, 0], dv[ids, 1] = speed * torch.cos(angle), speed * torch.sin(angle)
        self.push_timer[ids] = self._uniform(*c.push_interval_s, (n,))
        return dv

    # ------------------------------------------------------------- rewards
    def _terms(self, copper_w: torch.Tensor, terminated: torch.Tensor) -> dict:
        m, q = self.model, self.plant.q
        theta, quat = q[:, 7:], q[:, 3:7]
        g = self.contract.projected_gravity(quat)
        active = self.contact.to(self.dtype)
        both_feet = ((active @ self.left_sole) > 0) & ((active @ self.right_sole) > 0)
        others = ((active @ self.other_body) > 0).sum(-1).to(self.dtype)
        return {
            "orientation": g[:, :2].square().sum(-1),
            "base_height": (q[:, 2] - self.stand_height).square(),
            "base_lin_vel": self.v_body.square().sum(-1),
            "base_ang_vel": self.gyro.square().sum(-1),
            "joint_vel": self.qd.square().sum(-1),
            "joint_acc": ((self.qd - self.prev_qd) / self.cfg.policy_dt).square().sum(-1),
            "action_rate": (self.actions - self.prev_actions).square().sum(-1),
            "similar_to_default": (theta - self.contract.default_qpos).abs().sum(-1),
            "joint_limits": ((m.jnt_low - theta).clamp_min(0) + (theta - m.jnt_high).clamp_min(0)).sum(-1),
            "undesired_contacts": others,
            "feet_grounded": both_feet.to(self.dtype),
            "alive": torch.ones_like(copper_w),
            "action_excess": ((self.actions.abs() - 1.0).clamp_min(0).sum(-1)).clamp(max=10.0),
            "copper_loss": copper_w,
            "termination": terminated.to(self.dtype),
        }

    # --------------------------------------------------------- observations
    def _noise(self, x, key):
        s = self.cfg.obs_noise.get(key, 0.0)
        return x if s <= 0 else x + s * (2 * torch.rand_like(x) - 1)

    def get_observations(self) -> TensorDict:
        q = self.plant.q
        theta, quat = q[:, 7:], q[:, 3:7]
        clean = self.contract.observe(theta, self.qd, self.gyro, quat, self.actions)
        g = self.contract.projected_gravity(quat)
        actor = self.contract.observe(self._noise(theta, "joint_pos"), self._noise(self.qd, "joint_vel"),
                                      self._noise(self.gyro, "gyro"), quat, self.actions, gravity=self._noise(g, "gravity"))
        active = self.contact.to(self.dtype)
        feet = torch.stack(((active @ self.left_sole) > 0, (active @ self.right_sole) > 0), -1).to(self.dtype)
        critic = torch.cat((clean, self.v_body, (q[:, 2:3] - self.stand_height), feet), -1)
        return TensorDict({"policy": actor, "critic": critic}, batch_size=[self.num_envs], device=self.device)

    # ----------------------------------------------------------------- step
    def step(self, actions: torch.Tensor):
        c = self.cfg
        a = torch.nan_to_num(actions.to(self.device, self.dtype)).clamp(-c.action_guard, c.action_guard)
        self.prev_actions = self.actions
        self.actions = a
        copper0 = self.plant.ledger.copper.clone()
        supplied0 = self.plant.ledger.supplied.clone()
        info = self.plant.policy_step(self.contract.decode(a), self.substeps, kp=self.kp, kd=self.kd,
                                      push_dv=self._pushes())
        nu = info.extra["nu"]
        self.prev_qd = self.qd
        self.qd, self.v_body, self.gyro = nu[:, 6:].clone(), nu[:, 3:6].clone(), info.extra["gyro"].clone()
        self.contact = info.contact_active.clone()
        led = self.plant.ledger
        copper_w = (led.copper - copper0) / c.policy_dt
        supplied_w = (led.supplied - supplied0) / c.policy_dt
        self.episode_length_buf += 1

        rp = self.contract.roll_pitch(self.plant.q[:, 3:7])
        fell = (rp.abs() > C.TILT_ABORT).any(-1)
        low = self.plant.q[:, 2] < c.min_trunk_height
        terminated = fell | low

        terms = self._terms(copper_w, terminated)
        reward = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        for k, w in c.rewards.items():
            v = w * terms[k] * c.policy_dt
            self.episode_sums[k] += v
            reward += v

        timeout = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        dones = terminated | timeout

        log = {
            "Metrics/copper_W": copper_w.mean(),
            "Metrics/supplied_W": supplied_w.mean(),
            "Metrics/tilt_deg": torch.rad2deg(rp.abs().amax(-1)).mean(),
            "Metrics/trunk_height_m": self.plant.q[:, 2].mean(),
        }
        ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            for k, v in self.episode_sums.items():
                log[f"Episode_Reward/{k}"] = v[ids].mean()
            log["Episode_Termination/fell"] = fell[ids].to(self.dtype).mean()
            log["Episode_Termination/too_low"] = low[ids].to(self.dtype).mean()
            log["Episode_Termination/timeout"] = timeout[ids].to(self.dtype).mean()
            log["Episode/length_s"] = self.episode_length_buf[ids].to(self.dtype).mean() * c.policy_dt
            self._reset_idx(ids)
        self.extras = {"time_outs": timeout, "log": log}
        return self.get_observations(), reward, dones, self.extras

    def config_dict(self) -> dict:
        d = asdict(self.cfg)
        d["contract"] = {"agent": "stand", "gains": C.GAINS, "obs_dim": C.OBS_DIM, "hidden_dims": list(C.HIDDEN_DIMS),
                         "tilt_abort": C.TILT_ABORT, "action_scale_cli": C.ACTION_SCALE_CLI}
        d["stand_height_m"] = self.stand_height
        return d
