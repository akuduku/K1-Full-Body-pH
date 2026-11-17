r"""Walk on velocity commands on the port-Hamiltonian K1: an rsl-rl VecEnv on K1PHPlantNative (the C++ plant).

The agent is the firmware's sprint agent (rl/walk/contract.py: 75 inputs, 22 joint targets, the sprint's gains and
command path through ph_k1/firmware.py), so a policy trained here is a file the robot loads.  The task (rewards,
commands) is rl/walk/task.py: Eric Plass's walk task on the pH plant's quantities.  The same task on the
differentiable Python plant: Model Based RL mbrl/grad/walk_env.py.

An episode starts standing in the sprint's home pose with a random heading, soles on the floor, and lasts 20 s or
until the robot falls (the firmware's tilt abort, a body other than a foot on the ground, the trunk below 0.25 m).

Randomised as in k1_skill_cfg.py: start pose, PD gains, encoder bias, transport delay, sensor noise, pushes.  Not
randomised (the pH plant's own parameters are used): masses, centres of mass, ground stiffness, foot friction.

Step order (mjlab's, as RL_Core): action -> physics -> fall check -> rewards -> resets -> commands -> observation.
A push is drawn before the physics of the step it acts in.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from tensordict import TensorDict

from ph_k1.firmware import FirmwareConfig
from ph_k1.model import NJ, load_k1
from ph_k1.native import K1PHPlantNative
from rl.kick.contract import yaw_of

from . import contract as C
from .task import REWARDS, WalkParams, WalkTask

try:
    from rsl_rl.env import VecEnv
except ImportError:  # the env can be imported without rsl-rl (tests, evaluation)
    VecEnv = object


@dataclass
class WalkConfig:
    num_envs: int = 4096
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 1
    episode_s: float = 20.0
    policy_dt: float = C.POLICY_DT       # 50 Hz, the firmware's kDt
    plant_dt: float = 1e-3
    joint_noise: float = 0.1             # rad, start pose (k1_skill_cfg reset_joint_offset)
    random_yaw: bool = True              # start heading uniform in (-pi, pi)
    kp_scale: tuple = (0.6, 1.3)         # per robot and joint, per episode
    kd_scale: tuple = (0.6, 1.3)
    encoder_bias: float = 0.015          # rad, per robot and joint, per episode
    delay_ms: tuple = (10.0, 70.0)
    delay_hold_prob: float = 0.3
    slew: bool = C.SLEW
    push_interval_s: tuple = (1.5, 4.0)
    push_dv: float = 0.5                 # horizontal trunk velocity change per axis, uniform in +-push_dv [m/s]; 0 = none
    action_clip: float = 20.0            # k1_sprint: gaits stay below 10; stops a runaway policy feeding itself
    observe_target: bool = False         # the actor sees the ramped reference instead of the request
    command_scale: float = 1.0           # x the drawn requests: 0 = stand, 1 = Eric's ranges (the training's curriculum)
    top_speed: float | None = None       # the sprints' top speed; None = WalkParams.top_speed[0]
    fixed_command: tuple | None = None   # (vx, vy, wz): every robot gets this request and nothing else (evaluation)
    obs_noise: dict = field(default_factory=lambda: {"gyro": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 1.5})
    rewards: dict = field(default_factory=lambda: dict(REWARDS))
    params: WalkParams = field(default_factory=WalkParams)


def quiet(cfg: WalkConfig, **changes) -> WalkConfig:
    """No pushes, no sensor noise, nominal gains, no encoder bias, 20 ms delay, the home pose facing +x."""
    from dataclasses import replace

    return replace(cfg, obs_noise={}, push_dv=0.0, kp_scale=(1.0, 1.0), kd_scale=(1.0, 1.0), encoder_bias=0.0,
                   delay_ms=(20.0, 20.0), joint_noise=0.0, random_yaw=False, **changes)


class WalkEnv(VecEnv):
    def __init__(self, cfg: WalkConfig | None = None):
        self.cfg = cfg = cfg or WalkConfig()
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        self.num_envs = B = cfg.num_envs
        self.num_actions = NJ
        self.max_episode_length = int(round(cfg.episode_s / cfg.policy_dt))
        self.substeps = int(round(cfg.policy_dt / cfg.plant_dt))

        model = load_k1(dtype=self.dtype, device=self.device)
        C.check_joint_order(model.joint_names)
        self.model = model
        fw = FirmwareConfig(gains=C.GAINS, slew=cfg.slew, delay_ms=tuple(cfg.delay_ms), policy_dt=cfg.policy_dt,
                            delay_hold_prob=cfg.delay_hold_prob)
        self.plant = K1PHPlantNative(model=model, batch=B, timestep=cfg.plant_dt, firmware=fw)
        self.task = WalkTask(model, cfg.policy_dt, cfg.params, cfg.rewards)
        self.task.command_scale = cfg.command_scale
        if cfg.top_speed is not None:
            self.task.top_speed = cfg.top_speed
        self.contract = self.task.contract
        self.base_kp, self.base_kd = (g.clone() for g in self.plant.gains())

        z = lambda *s: torch.zeros(B, *s, dtype=self.dtype, device=self.device)  # noqa: E731
        self.actions, self.last_actions, self.before_last_actions = z(NJ), z(NJ), z(NJ)
        self.qd, self.prev_qd, self.gyro = z(NJ), z(NJ), z(3)
        self.kp, self.kd, self.encoder_bias = self.base_kp.clone(), self.base_kd.clone(), z(NJ)
        self.critic_extra = z(15)
        self.push_timer = z()
        self.episode_length_buf = torch.zeros(B, dtype=torch.long, device=self.device)
        self.episode_sums = {k: z() for k in cfg.rewards}
        self.mem = self.task.new_memory(B, self._rand)
        self._reset_idx(torch.arange(B, device=self.device))
        self.mem = self.task.update_commands(self.mem, yaw_of(self.plant.q[:, 3:7]), z(), self._rand)
        self.extras = {"log": {}}

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, dtype=self.dtype, device=self.device)

    def _uniform(self, lo, hi, *shape) -> torch.Tensor:
        return lo + (hi - lo) * self._rand(*shape)

    def set_command(self, twist) -> None:
        """Every robot follows this request (vx, vy, wz) from now on, also after a reset."""
        self.cfg.fixed_command = tuple(float(v) for v in twist)
        self.mem = self.task.set_command(self.mem, self.cfg.fixed_command)

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, ids: torch.Tensor) -> None:
        n = int(ids.numel())
        if n == 0:
            return
        c, m = self.cfg, self.model
        pose = self.contract.home + self._uniform(-c.joint_noise, c.joint_noise, n, NJ)
        pose = torch.minimum(torch.maximum(pose, m.jnt_low), m.jnt_high)
        yaw = self._uniform(-math.pi, math.pi, n) if c.random_yaw else torch.zeros(n, dtype=self.dtype, device=self.device)
        quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
        self.plant.reset(pose=pose, base_quat=quat, env_ids=ids)   # soles on the floor, at rest; command path, delays
        self.kp[ids] = self.base_kp[ids] * self._uniform(*c.kp_scale, n, NJ)
        self.kd[ids] = self.base_kd[ids] * self._uniform(*c.kd_scale, n, NJ)
        self.encoder_bias[ids] = self._uniform(-c.encoder_bias, c.encoder_bias, n, NJ)
        for buf in (self.actions, self.last_actions, self.before_last_actions, self.qd, self.prev_qd, self.gyro):
            buf[ids] = 0.0
        self.push_timer[ids] = self._uniform(*c.push_interval_s, n)
        self.episode_length_buf[ids] = 0
        for v in self.episode_sums.values():
            v[ids] = 0.0
        self.mem = self.task.reset_memory(self.mem, ids, self._rand)
        if c.fixed_command is not None:
            self.mem = self.task.set_command(self.mem, c.fixed_command)
        # what the critic sees of a robot that just started: standing still, both feet down, no force yet
        q = self.plant.q[ids]
        zero = torch.zeros(n, self.model.ncontact, dtype=self.dtype, device=self.device)
        rest = self.task.signals(q, torch.zeros(n, 28, dtype=self.dtype, device=self.device), self.qd[ids],
                                 zero + self.task.sole.sum(0), zero[..., None].expand(-1, -1, 3))
        self.critic_extra[ids] = self.task.critic_extra(rest, {"air_time": self.mem["air_time"][ids]})

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
        dv = torch.zeros(self.num_envs, 3, dtype=self.dtype, device=self.device)
        dv[ids, :2] = self._uniform(-c.push_dv, c.push_dv, n, 2)
        self.push_timer[ids] = self._uniform(*c.push_interval_s, n)
        return dv

    # --------------------------------------------------------- observations
    def _noise(self, x, key):
        s = self.cfg.obs_noise.get(key, 0.0)
        return x if s <= 0 else x + s * (2 * torch.rand_like(x) - 1)

    def get_observations(self) -> TensorDict:
        q = self.plant.q
        theta, quat = q[:, 7:], q[:, 3:7]
        g = self.contract.projected_gravity(quat)
        command = self.mem["target"] if self.cfg.observe_target else self.mem["commands"]
        actor = self.contract.observe(self._noise(self.gyro, "gyro"), self._noise(g, "gravity"),
                                      self._noise(theta + self.encoder_bias, "joint_pos"), self._noise(self.qd, "joint_vel"),
                                      self.actions, command)
        clean = self.contract.observe(self.gyro, g, theta, self.qd, self.actions, command)
        critic = torch.cat((clean, self.critic_extra), -1)
        return TensorDict({"policy": actor, "critic": critic}, batch_size=[self.num_envs], device=self.device)

    # ----------------------------------------------------------------- step
    def step(self, actions: torch.Tensor):
        c = self.cfg
        a = torch.nan_to_num(actions.to(self.device, self.dtype)).clamp(-c.action_clip, c.action_clip)
        self.before_last_actions, self.last_actions, self.actions = self.last_actions, self.actions, a
        led = self.plant.ledger
        supplied0, copper0 = led.supplied.clone(), led.copper.clone()
        info = self.plant.policy_step(self.contract.decode(a), self.substeps, kp=self.kp, kd=self.kd,
                                      push_dv=self._pushes())
        nu = info.extra["nu"]
        self.prev_qd = self.qd
        self.qd, self.gyro = nu[:, 6:].clone(), info.extra["gyro"].clone()
        led = self.plant.ledger
        supplied_w = (led.supplied - supplied0) / c.policy_dt
        copper_w = (led.copper - copper0) / c.policy_dt
        self.episode_length_buf += 1

        sig = self.sig = self.task.signals(self.plant.q, nu, self.prev_qd, info.contact_active, info.contact_force)
        terminated = self.task.fallen(sig)
        terms, self.mem = self.task.terms(sig, self.mem, a, self.last_actions, self.before_last_actions, supplied_w,
                                          terminated)
        reward = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        for k, w in c.rewards.items():
            v = torch.nan_to_num(terms[k], nan=0.0, posinf=0.0, neginf=0.0) * (w * c.policy_dt)
            self.episode_sums[k] += v
            reward += v
        self.critic_extra = self.task.critic_extra(sig, self.mem)

        timeout = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        dones = terminated | timeout
        track = sig.heading_vel if c.params.heading_frame else sig.base_lin_vel
        target = self.mem["target"]
        log = {
            "Metrics/speed_error_mps": torch.linalg.vector_norm(target[:, :2] - track[:, :2], dim=1).mean(),
            "Metrics/yaw_rate_error": (target[:, 2] - sig.base_ang_vel[:, 2]).abs().mean(),
            "Metrics/reference_speed_mps": torch.linalg.vector_norm(target[:, :2], dim=1).mean(),
            "Metrics/supplied_W": supplied_w.mean(),
            "Metrics/copper_W": copper_w.mean(),
            "Metrics/feet_in_air": (~sig.contact).to(self.dtype).sum(-1).mean(),
            "Metrics/trunk_height_m": self.plant.q[:, 2].mean(),
        }
        ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            for k, v in self.episode_sums.items():
                log[f"Episode_Reward/{k}"] = v[ids].mean() / c.episode_s
            log["Episode_Termination/fell"] = terminated[ids].to(self.dtype).mean()
            log["Episode_Termination/timeout"] = timeout[ids].to(self.dtype).mean()
            log["Episode/length_s"] = self.episode_length_buf[ids].to(self.dtype).mean() * c.policy_dt
            self._reset_idx(ids)
        dt = torch.full((self.num_envs,), c.policy_dt, dtype=self.dtype, device=self.device)
        dt[ids] = 0.0
        if c.fixed_command is None:
            self.mem = self.task.update_commands(self.mem, yaw_of(self.plant.q[:, 3:7]), dt, self._rand)
        self.extras = {"time_outs": timeout, "log": log, "terms": terms, "terminated": terminated,
                       "supplied_W": supplied_w}
        return self.get_observations(), reward, dones, self.extras

    # ------------------------------------------------------------ snapshots
    STATE = ("actions", "last_actions", "before_last_actions", "qd", "prev_qd", "gyro", "kp", "kd", "encoder_bias",
             "critic_extra", "push_timer", "episode_length_buf")

    def snapshot(self, rows: torch.Tensor | None = None) -> dict:
        """A copy of the robots' state (plant, command path, this env's buffers, the task's memory)."""
        take = lambda t: (t if rows is None else t[rows]).clone()  # noqa: E731
        return {"plant": self.plant.snapshot(rows), "env": {k: take(getattr(self, k)) for k in self.STATE},
                "mem": {k: take(v) for k, v in self.mem.items()}, "sums": {k: take(v) for k, v in self.episode_sums.items()}}

    def restore(self, snap: dict, rows: torch.Tensor | None = None) -> None:
        """Put a snapshot back: row i goes to robot rows[i] (all robots without rows)."""
        self.plant.restore(snap["plant"], rows)

        def put(old, new):
            new = new.to(device=old.device, dtype=old.dtype)
            if rows is None:
                return new.clone()
            old = old.clone()
            old[rows] = new
            return old

        for k, v in snap["env"].items():
            setattr(self, k, put(getattr(self, k), v))
        self.mem = {k: put(self.mem[k], v) for k, v in snap["mem"].items()}
        self.episode_sums = {k: put(self.episode_sums[k], v) for k, v in snap["sums"].items()}

    def config_dict(self) -> dict:
        d = asdict(self.cfg)
        d["contract"] = {"agent": "sprint", "gains": C.GAINS, "obs_dim": C.OBS_DIM, "tilt_abort": C.TILT_ABORT}
        return d
