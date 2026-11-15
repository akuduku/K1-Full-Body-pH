r"""Walk, then kick a ball where and as fast as requested, on the port-Hamiltonian K1: an rsl-rl VecEnv.

The task is Eric Plass's k1_kick with `--variant speed --camera --walk` (kick-014 to kick-016, resting balls), on the
pH plant instead of MuJoCo; the agent is the firmware's kick_walk (rl/kick/contract.py: 165 inputs, 22 targets, the
sprint's gains and command path through ph_k1/firmware.py).

An episode: the robot walks on velocity commands for 1.5-5 s (a share of the robots stop before, the rest are
still walking), then a request starts: a ball is put on the floor ahead of the trunk, with a direction and a ball
speed, and the request runs for 4 s.  A fifth of the fresh robots get their request at once, from the stance.  The
episode ends with the request, or when the robot falls.

The shot (k1_kick's): the touch is the first step the ball rolls faster than 0.5 m/s; the shot is the ball's
velocity at its fastest within 0.3 s after it.

Rewards: k1_kick's task terms (walk: follow the command; kick: launch speed, speed error, direction, loft, back to
the stance and the heading) and its safety terms, without its style terms (which foot, how the leg swings): a
planned motion can supply the style (Model Based RL mbrl/kick adds tracking terms through EXTRA_REWARDS).  From the
pH plant: the electrical energy drawn (ledger).

What the plant adds to the log: the Hamiltonian's tanks (a sample of the robots), the ledger's power flows, the
electrical energy of every kick, the ball's energy at launch and their ratio.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from tensordict import TensorDict

from ph_k1 import firmware as fw
from ph_k1.ball import BallParams
from ph_k1.dynamics import body_frames
from ph_k1.firmware import FirmwareConfig
from ph_k1.model import NJ, load_k1
from ph_k1.native import K1PHPlantNative

from . import contract as C
from .camera import CAMERA, BallCamera

try:
    from rsl_rl.env import VecEnv
except ImportError:  # the contract/env can be imported without rsl-rl (tests of the parts)
    VecEnv = object

# weight per reward term (the sum is multiplied by the policy step, as k1_kick does) and when the term counts
REWARDS = {
    # always
    "upright": 1.0,
    "body_ang_vel": -0.01,
    "joint_limits": -1.0,
    "action_rate": -0.1,
    "overspeed": -1.0,
    "energy": -0.002,            # per watt of electrical power drawn from the battery (pH ledger)
    "termination": -1000.0,      # once per fall (x policy_dt = -20)
    # while walking
    "track_lin_vel": 2.0,
    "track_ang_vel": 1.0,
    "walk_imitation": 4.0,       # only with a walk teacher (KickConfig.walk_teacher)
    # during a request
    "launch_speed": 20.0,
    "launch_speed_error": 40.0,
    "launch_direction": 60.0,
    "launch_loft": -5.0,
    "wait": -1.0,
    "stand": 10.0,
    "stand_pose": 10.0,
    "stand_feet": 5.0,
    "stand_still": -0.01,
    "hold_heading": 10.0,
    "settle_action_rate": -1.0,
}
WALK_TERMS = ("track_lin_vel", "track_ang_vel", "walk_imitation")
SHARED_TERMS = ("upright", "body_ang_vel", "joint_limits", "action_rate", "overspeed", "energy", "termination")
TANKS = ("kinetic", "gravity", "contact", "stops", "coil", "ball", "total")
# one row per finished episode (KickEnv.last_shot): a request ran | the ball was touched | launch speed along the
# direction [m/s] | direction error [deg] | vertical launch speed [m/s] | relative speed error | requested speed [m/s]
# | the robot fell | ball distance at the request [m] | robot speed at the request [m/s] | touch time [s] |
# electrical energy of the kick [J] | the ball's energy at launch [J] | the ball's planar launch speed [m/s]
SHOT = ("requested", "kicked", "along", "direction_deg", "loft", "speed_error", "want", "fell", "range",
        "robot_speed", "touch_time", "energy_J", "ball_J", "speed")


@dataclass
class KickConfig:
    num_envs: int = 4096
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 1
    policy_dt: float = 0.02            # 50 Hz, the firmware's kDt
    plant_dt: float = 1e-3
    phase_input: bool = False          # a 166th actor input: seconds since the request (contract.py)
    # the request
    request_s: float = 4.0
    ball_x: tuple = (0.3, 1.0)         # m ahead of the trunk (heading frame); kick-016
    ball_y: tuple = (-0.12, 0.12)
    ball_per_speed: float = 0.15       # m further ahead per m/s the robot walks forward
    foot_clearance: tuple = (0.21, 0.12)  # the ball at least this far ahead of a foot within this far beside it
    kick_angle: tuple = (-0.6, 0.6)    # rad, the direction relative to the heading
    kick_speed: tuple = (1.0, 6.0)     # m/s, the requested ball speed
    full_power: float = 5.0            # requests from here up: as hard as possible, only a shortfall counts
    kicked_speed: float = 0.5          # m/s planar ball speed: the touch
    launch_window_s: float = 0.3
    settle_s: float = 0.4              # after the touch: the kicking foot is due on the floor, back to the stance
    ball_radius: float = 0.075
    ball_mass: float = 0.2
    # the walk before it
    walk_s: tuple = (1.5, 5.0)
    stop_s: tuple = (0.5, 1.5)         # its last part without a command (the robot stops) ...
    stop_prob: float = 0.5             # ... for this share of the requests; the others come mid-walk
    stand_prob: float = 0.2            # fresh robots with their request at once
    lin_vel_x: tuple = (-0.5, 2.0)
    lin_vel_y: tuple = (-0.6, 0.6)
    ang_vel_z: tuple = (-1.5, 1.5)
    still_prob: float = 0.15           # commands that stand
    resample_s: tuple = (1.0, 3.0)
    walk_teacher: str | None = None    # TorchScript walk policy (75 -> 22): walk_imitation, teacher_actions()
    # start and disturbances
    joint_noise: float = 0.1           # rad, start pose noise
    yaw_noise: float = 0.3             # rad, start heading
    gain_dr: float = 0.10              # kp, kd x U(1 - gain_dr, 1 + gain_dr) per robot and joint, per episode
    delay_ms: tuple = (0.0, 70.0)      # transport delay (k1_kick kick-016: 0-70 ms)
    delay_hold_prob: float = 0.3
    push_interval_s: tuple = (2.0, 5.0)
    push_dv: float = 0.25              # max horizontal trunk velocity change of a push [m/s]; 0 = no pushes
    obs_noise: dict = field(default_factory=lambda: {"gyro": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 1.5})
    camera: dict = field(default_factory=lambda: dict(CAMERA))
    # the end
    termination_tilt_deg: float = 60.0
    action_guard: float = 100.0        # numerical guard only
    # reward parameters (k1_kick's)
    upright_std: float = 0.4472
    track_std: float = 0.5
    teacher_std: float = math.sqrt(0.1)
    launch_speed_cap: float = 10.0
    speed_error_std: tuple = (1.0, 0.15)
    direction_std: tuple = (0.3, 0.08)
    accurate_speed: float = 2.0
    stand_pose_std: tuple = (0.5, 0.15)
    stand_feet_std: tuple = (0.3, 0.06)
    heading_std: tuple = (0.5, 0.15)
    overspeed_fraction: float = 0.8
    energy_log_envs: int = 64          # the Hamiltonian's tanks are logged for this many robots
    rewards: dict = field(default_factory=lambda: dict(REWARDS))


class KickEnv(VecEnv):
    # reward terms a subclass adds (its _terms returns them; EXTRA_PHASE says when each counts: "walk", "kick", "both")
    EXTRA_REWARDS: tuple = ()
    EXTRA_PHASE: dict = {}

    def __init__(self, cfg: KickConfig | None = None):
        self.cfg = cfg = cfg or KickConfig()
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        self.num_envs = B = cfg.num_envs
        self.num_actions = C.NUM_ACTIONS
        self.kick_steps = int(round(cfg.request_s / cfg.policy_dt))
        self.max_episode_length = self.kick_steps + math.ceil(cfg.walk_s[1] / cfg.policy_dt)
        self.substeps = int(round(cfg.policy_dt / cfg.plant_dt))
        unknown = set(cfg.rewards) - set(REWARDS) - set(self.EXTRA_REWARDS)
        if unknown:
            raise KeyError(f"unknown reward terms {sorted(unknown)}; known: {sorted(REWARDS) + list(self.EXTRA_REWARDS)}")

        model = load_k1(dtype=self.dtype, device=self.device)
        C.check_joint_order(model.joint_names)
        self.model = model
        firmware = FirmwareConfig(gains=C.GAINS, slew=C.SLEW, delay_ms=tuple(cfg.delay_ms), policy_dt=cfg.policy_dt,
                                  delay_hold_prob=cfg.delay_hold_prob)
        self.ball_params = BallParams(enabled=True, radius=cfg.ball_radius, mass=cfg.ball_mass)
        self.plant = K1PHPlantNative(model=model, batch=B, timestep=cfg.plant_dt, firmware=firmware,
                                     ball=self.ball_params)
        self.contract = C.Contract(self.dtype, self.device, phase=cfg.phase_input)
        self.base_kp, self.base_kd = (g.clone() for g in self.plant.gains())
        self.v_max = fw.firmware_table(model, fw.MAX_JOINT_VELOCITY)

        T = lambda x: torch.as_tensor(x, dtype=self.dtype, device=self.device)  # noqa: E731
        z = lambda *s: torch.zeros(*s, dtype=self.dtype, device=self.device)  # noqa: E731
        zb = lambda: torch.zeros(B, dtype=torch.bool, device=self.device)  # noqa: E731
        zl = lambda: torch.zeros(B, dtype=torch.long, device=self.device)  # noqa: E731
        self.index = torch.arange(B, device=self.device)

        # bodies and contact points
        names = model.body_names
        self.feet = [names.index("left_foot_link"), names.index("right_foot_link")]
        self.head = names.index("Head_2")
        sole, cb = model.contact_is_sole, model.contact_body
        self.sole = torch.stack([(sole & (cb == b)) for b in self.feet]).to(self.dtype)   # (2, nc)
        self.not_sole = ~sole

        # the walk teacher (the deployed walk: its 75 inputs are the first 75 of this agent's)
        self.teacher = None
        if cfg.walk_teacher:
            self.teacher = torch.jit.load(str(cfg.walk_teacher), map_location=self.device).eval()

        # per-robot state
        self.actions, self.last_actions = z(B, NJ), z(B, NJ)
        self.qd, self.gyro, self.v_body = z(B, NJ), z(B, 3), z(B, 3)
        self.torque, self.current = z(B, NJ), z(B, NJ)
        self.held_target = self.contract.home.expand(B, NJ).clone()   # what the motor driver held in the last plant step
        self.contact = torch.zeros(B, model.ncontact, dtype=torch.bool, device=self.device)
        self.kp, self.kd = self.base_kp.clone(), self.base_kd.clone()
        self.push_timer = z(B)
        self.episode_length_buf = zl()
        self.terminated, self.time_outs, self.fell, self.ground, self.exploded = zb(), zb(), zb(), zb(), zb()
        # the walk
        self.commands, self.command_timer = z(B, 3), z(B)
        self.active = zb()                      # a request runs
        self.request_step, self.stop_step = zl(), zl()
        # the request
        self.target_w = z(B, 3)                 # the kick direction, world frame
        self.target_w[:, 0] = 1.0
        self.target_speed = torch.full((B,), 3.0, dtype=self.dtype, device=self.device)
        self.request_yaw, self.request_range, self.request_speed = z(B), z(B), z(B)
        self.foot_request = zl()                # the foot on the ball's side: 0 left, 1 right
        # the shot
        self.kicked, self.launching, self.touched = zb(), zb(), zb()
        self.kick_time, self.touch_time = z(B), z(B)
        self.launch_vel = z(B, 3)
        self.request_energy = z(B)              # electrical energy since the request started, until the launch is over
        self.episode_sums = {k: z(B) for k in cfg.rewards}
        self.last_shot = z(B, len(SHOT))        # the last finished episode of every robot (SHOT's columns)

        self.camera = BallCamera(B, self.device, **cfg.camera)

        # the stance: standing height and where the feet are relative to the trunk (heading frame), from the start pose
        self.plant.reset(pose=self.contract.home)
        self.stand_height = float(self.plant.q[:, 2].mean())
        _, P, _ = body_frames(model, self.plant.q[:1])
        self.stance = (P[0, self.feet] - P[0, 0])[:, :2].clone()   # (2,2)

        self.phase = {name: "walk" if name in WALK_TERMS else "both" if name in SHARED_TERMS else "kick"
                      for name in REWARDS}
        self.phase.update(self.EXTRA_PHASE)
        if "walk_imitation" in cfg.rewards and self.teacher is None:
            cfg.rewards = {k: v for k, v in cfg.rewards.items() if k != "walk_imitation"}
            self.episode_sums.pop("walk_imitation", None)

        self.actor_obs = None
        self.extras = {"log": {}}
        self._reset_idx(self.index)
        self._refresh()
        self._start_requests()

    # ------------------------------------------------------------------ helpers
    def _uniform(self, lo, hi, shape):
        return lo + (hi - lo) * torch.rand(shape, dtype=self.dtype, device=self.device)

    def _refresh(self) -> None:
        """What the rewards and observations read, from the plant's state and the last step's velocities."""
        q = self.plant.q
        self.base_pos, self.base_quat, self.theta = q[:, :3], q[:, 3:7], q[:, 7:]
        self.yaw = C.yaw_of(self.base_quat)
        self.gravity = self.contract.projected_gravity(self.base_quat)
        self.lin_vel_w = C.rotate(self.base_quat, self.v_body)
        self.ang_vel_w = C.rotate(self.base_quat, self.gyro)
        _, P, _ = body_frames(self.model, q)
        self.foot_pos = P[:, self.feet]                     # (B,2,3)
        self.head_pos = P[:, self.head]
        ball = self.plant.ball
        self.ball_pos, self.ball_vel = ball[:, 0:3], ball[:, 3:6]

    def time_in_request(self) -> torch.Tensor:
        """Seconds since the request started (0 without one)."""
        return (self.episode_length_buf - self.request_step).clamp(min=0).to(self.dtype) * self.cfg.policy_dt * self.active

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, ids: torch.Tensor) -> None:
        n = int(ids.numel())
        if n == 0:
            return
        c, m = self.cfg, self.model
        pose = self.contract.home + self._uniform(-c.joint_noise, c.joint_noise, (n, NJ))
        pose = torch.minimum(torch.maximum(pose, m.jnt_low), m.jnt_high)
        yaw = self._uniform(-c.yaw_noise, c.yaw_noise, (n,))
        quat = torch.zeros(n, 4, dtype=self.dtype, device=self.device)
        quat[:, 0], quat[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        self.plant.reset(pose=pose, base_quat=quat, env_ids=ids)   # soles on the floor, at rest; the ball is parked
        self.kp[ids] = self.base_kp[ids] * self._uniform(1 - c.gain_dr, 1 + c.gain_dr, (n, NJ))
        self.kd[ids] = self.base_kd[ids] * self._uniform(1 - c.gain_dr, 1 + c.gain_dr, (n, NJ))
        for buf in (self.actions, self.last_actions, self.qd, self.gyro, self.v_body, self.torque, self.current):
            buf[ids] = 0.0
        self.held_target[ids] = pose
        self.contact[ids] = False
        self.push_timer[ids] = self._uniform(*c.push_interval_s, (n,))
        self.episode_length_buf[ids] = 0
        for v in self.episode_sums.values():
            v[ids] = 0.0
        # the walk before the request: its length, and from when on there is no command
        steps = (self._uniform(*c.walk_s, (n,)) / c.policy_dt).long()
        steps = torch.where(torch.rand(n, device=self.device) < c.stand_prob, torch.zeros_like(steps), steps)
        stop = (self._uniform(*c.stop_s, (n,)) / c.policy_dt).long()
        stop = torch.where(torch.rand(n, device=self.device) < c.stop_prob, stop, torch.zeros_like(stop))
        self.request_step[ids] = steps
        self.stop_step[ids] = (steps - stop).clamp(min=0)
        self.active[ids] = False
        self._draw_commands(ids)
        self.commands[ids[self.stop_step[ids] == 0]] = 0.0
        # no request yet
        self.kicked[ids], self.launching[ids], self.touched[ids] = False, False, False
        self.kick_time[ids], self.touch_time[ids] = 0.0, 0.0
        self.launch_vel[ids] = 0.0
        self.request_energy[ids] = 0.0
        self.request_yaw[ids] = yaw
        self.request_range[ids], self.request_speed[ids] = 0.0, 0.0

    # ------------------------------------------------------------------- walk
    def _draw_commands(self, ids: torch.Tensor) -> None:
        c, n = self.cfg, int(ids.numel())
        cmd = torch.stack((self._uniform(*c.lin_vel_x, (n,)), self._uniform(*c.lin_vel_y, (n,)),
                           self._uniform(*c.ang_vel_z, (n,))), -1)
        cmd[torch.rand(n, device=self.device) < c.still_prob] = 0.0
        self.commands[ids] = cmd
        self.command_timer[ids] = self._uniform(*c.resample_s, (n,))

    def _update_commands(self) -> None:
        self.command_timer -= self.cfg.policy_dt
        due = (self.command_timer <= 0.0).nonzero(as_tuple=False).squeeze(-1)
        if len(due):
            self._draw_commands(due)
        self.commands[self.active | (self.episode_length_buf >= self.stop_step)] = 0.0

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

    # ---------------------------------------------------------------- request
    def _draw_request(self, ids: torch.Tensor):
        """Where the ball goes (heading frame, relative to the trunk), the direction relative to the heading and
        the ball speed of new requests: (local (n,2), angle (n,), speed (n,)).  A subclass may fix them."""
        c, n = self.cfg, int(ids.numel())
        local = torch.stack((self._uniform(*c.ball_x, (n,)), self._uniform(*c.ball_y, (n,))), -1)
        return local, self._uniform(*c.kick_angle, (n,)), self._uniform(*c.kick_speed, (n,))

    def _start_requests(self) -> torch.Tensor:
        """The requests whose walk is over: the ball goes ahead of the trunk, clear of the feet; the direction is
        relative to the heading.  Returns their robots."""
        due = (~self.active & (self.episode_length_buf >= self.request_step)).nonzero(as_tuple=False).squeeze(-1)
        n = int(due.numel())
        if n == 0:
            return due
        c = self.cfg
        yaw = self.yaw[due]
        local, angle, speed = self._draw_request(due)
        forward = C.to_heading(yaw, self.lin_vel_w[due])[:, 0].clamp(min=0.0)
        local[:, 0] = local[:, 0] + c.ball_per_speed * forward        # a walking robot walks into the kick
        feet = C.to_heading(yaw, self.foot_pos[due] - self.base_pos[due, None])
        ahead, beside = c.foot_clearance
        for foot in range(2):
            near = (local[:, 1] - feet[:, foot, 1]).abs() < beside
            local[:, 0] = torch.where(near, torch.maximum(local[:, 0], feet[:, foot, 0] + ahead), local[:, 0])
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        xy = torch.stack((self.base_pos[due, 0] + cy * local[:, 0] - sy * local[:, 1],
                          self.base_pos[due, 1] + sy * local[:, 0] + cy * local[:, 1]), -1)
        self.plant.set_ball(xy, env_ids=due)
        self.ball_pos, self.ball_vel = self.plant.ball[:, 0:3], self.plant.ball[:, 3:6]
        world = yaw + angle
        self.target_w[due, 0], self.target_w[due, 1] = torch.cos(world), torch.sin(world)
        self.target_speed[due] = speed
        self.foot_request[due] = (local[:, 1] < 0.0).long()
        self.request_range[due] = torch.linalg.vector_norm(local, dim=1)
        self.request_speed[due] = torch.linalg.vector_norm(self.lin_vel_w[due, :2], dim=1)
        self.request_yaw[due] = yaw
        self.kicked[due], self.launching[due], self.touched[due] = False, False, False
        self.kick_time[due], self.touch_time[due] = 0.0, 0.0
        self.launch_vel[due] = 0.0
        self.request_energy[due] = 0.0
        self.request_step[due] = self.episode_length_buf[due]
        self.active[due] = True
        self.commands[due] = 0.0
        self.camera.start(due, xy.float(), self.base_pos[due, :2].float(), yaw.float())
        return due

    def _update_kick(self) -> None:
        c = self.cfg
        speed = torch.linalg.vector_norm(self.ball_vel[:, :2], dim=1)
        touched = self.active & ~self.kicked & (speed > c.kicked_speed)
        self.touched = touched
        self.kicked |= touched
        self.kick_time = torch.where(touched, torch.zeros_like(self.kick_time), self.kick_time + c.policy_dt)
        self.touch_time = torch.where(touched, self.time_in_request(), self.touch_time)
        self.launching = self.kicked & (self.kick_time < c.launch_window_s)
        faster = self.launching & (speed > torch.linalg.vector_norm(self.launch_vel[:, :2], dim=1))
        self.launch_vel = torch.where(faster[:, None], self.ball_vel, self.launch_vel)

    def settling(self) -> torch.Tensor:
        """After the kick: the kicking foot is due on the floor again, back to the stance."""
        return self.kicked & (self.kick_time >= self.cfg.settle_s)

    def speed_error(self, velocity: torch.Tensor) -> torch.Tensor:
        """Relative error of the planar ball speed to the request; from full_power up only a shortfall counts."""
        speed = torch.linalg.vector_norm(velocity[:, :2], dim=1)
        err = torch.where(self.target_speed >= self.cfg.full_power, (self.target_speed - speed).clamp(min=0.0),
                          (speed - self.target_speed).abs())
        return err / self.target_speed

    # ---------------------------------------------------------------- rewards
    @staticmethod
    def _two_gauss(error, stds):
        coarse, fine = stds
        return 0.5 * (torch.exp(-(error / coarse).square()) + torch.exp(-(error / fine).square()))

    def _terms(self, supplied_w: torch.Tensor) -> dict:
        c, m = self.cfg, self.model
        f = lambda x: x.to(self.dtype)  # noqa: E731
        upright = torch.exp(-self.gravity[:, :2].square().sum(-1) / c.upright_std**2)
        action_rate = (self.actions - self.last_actions).square().sum(-1)
        vel_h = C.to_heading(self.yaw, self.lin_vel_w)
        along = (self.ball_vel * self.target_w).sum(-1)
        speed = torch.linalg.vector_norm(self.ball_vel[:, :2], dim=1)
        launching, settling = f(self.launching), f(self.settling())
        full = f(self.target_speed >= c.full_power)
        accurate = (0.5 * self.target_speed).clamp(max=c.accurate_speed)
        rms = (self.theta - self.contract.home).square().mean(-1).sqrt()
        rel = C.to_heading(self.yaw, self.foot_pos - self.base_pos[:, None])[..., :2]
        self.stance_offset = torch.linalg.vector_norm(rel - self.stance, dim=-1).sum(-1)
        terms = {
            "upright": upright,
            "body_ang_vel": self.ang_vel_w[:, :2].square().sum(-1),
            "joint_limits": ((m.jnt_low - self.theta).clamp_min(0) + (self.theta - m.jnt_high).clamp_min(0)).sum(-1),
            "action_rate": action_rate,
            "overspeed": (self.qd.abs() - c.overspeed_fraction * self.v_max).clamp_min(0).square().sum(-1),
            "energy": supplied_w.clamp_min(0),
            "termination": f(self.terminated),
            "track_lin_vel": torch.exp(-(self.commands[:, :2] - vel_h[:, :2]).square().sum(-1) / c.track_std**2),
            "track_ang_vel": torch.exp(-(self.commands[:, 2] - self.ang_vel_w[:, 2]).square() / c.track_std**2),
            "launch_speed": along.clamp(0.0, c.launch_speed_cap) * launching * full,
            "launch_speed_error": self._two_gauss(self.speed_error(self.ball_vel), c.speed_error_std) * launching,
            "launch_direction": (self._two_gauss(C.direction_error(self.ball_vel, self.target_w), c.direction_std)
                                 * (speed / accurate).clamp(max=1.0) * launching),
            "launch_loft": self.ball_vel[:, 2].abs() * launching,
            "wait": f(~self.kicked),
            "stand": upright * settling,
            "stand_pose": self._two_gauss(rms, c.stand_pose_std) * settling,
            "stand_feet": self._two_gauss(self.stance_offset, c.stand_feet_std) * settling,
            "stand_still": self.qd.square().sum(-1) * settling,
            "hold_heading": self._two_gauss(C.wrap(self.yaw - self.request_yaw), c.heading_std) * f(self.kicked),
            "settle_action_rate": action_rate * settling,
        }
        if self.teacher is not None and "walk_imitation" in c.rewards:
            err = (self.actions - self.teacher_actions()).square().mean(-1)
            terms["walk_imitation"] = torch.exp(-err / c.teacher_std**2)
        return terms

    def _reward_scale(self, name: str) -> float:
        return 1.0

    def teacher_actions(self) -> torch.Tensor:
        """What the walk teacher does on what the policy saw (its first 75 inputs)."""
        with torch.no_grad():
            return self.teacher(self.actor_obs[:, :C.WALK_DIM].float()).to(self.dtype)

    # --------------------------------------------------------- observations
    def _noise(self, x, key):
        s = self.cfg.obs_noise.get(key, 0.0)
        return x if s <= 0 else x + s * (2 * torch.rand_like(x) - 1)

    def _critic_extra(self) -> list:
        """More columns for the critic (a subclass: e.g. where the plan is)."""
        return []

    def get_observations(self) -> TensorDict:
        f = lambda x: x.to(self.dtype)  # noqa: E731
        con = self.contract
        on = f(self.active)
        seen, velocity, reports = (f(x) for x in self.camera.reports())
        age = f(self.camera.age)
        target = C.to_heading(self.yaw, self.target_w)[:, :2]
        phase = self.time_in_request()
        noisy = con.frame(self._noise(self.gyro, "gyro"), self._noise(self.gravity, "gravity"),
                          self._noise(self.theta, "joint_pos"), self._noise(self.qd, "joint_vel"), self.actions)
        actor = con.observe(noisy, self.commands, on, seen, age, velocity, target, self.target_speed, reports, phase)
        clean = con.frame(self.gyro, self.gravity, self.theta, self.qd, self.actions)
        active = f(self.contact)
        feet = (active @ self.sole.T > 0).to(self.dtype)
        o = on[:, None]
        critic = torch.cat([
            clean, self.v_body, (self.base_pos[:, 2:3] - self.stand_height), feet,
            C.to_heading(self.yaw, self.ball_pos - self.base_pos) * o, C.to_heading(self.yaw, self.ball_vel) * o,
            target * o, f(self.kicked)[:, None], f(self.launching)[:, None],
            (self.kick_time * self.kicked).clamp(max=3.0)[:, None], phase[:, None],
            (2.0 * self.foot_request.to(self.dtype) - 1.0)[:, None] * o,
            (self.target_speed[:, None] - 3.0) / 2.0 * o, reports.flatten(1) * o, self.commands, o,
            *self._critic_extra(),
        ], -1)
        self.actor_obs = actor
        return TensorDict({"policy": actor, "critic": critic}, batch_size=[self.num_envs], device=self.device)

    # ----------------------------------------------------------------- step
    def step(self, actions: torch.Tensor):
        c = self.cfg
        a = torch.nan_to_num(actions.to(self.device, self.dtype)).clamp(-c.action_guard, c.action_guard)
        self.last_actions = self.actions
        self.actions = a
        led = self.plant.ledger
        before = {k: getattr(led, k).clone() for k in ("supplied", "copper", "gearbox", "stops", "contact")}
        info = self.plant.policy_step(self.contract.decode(a), self.substeps, kp=self.kp, kd=self.kd,
                                      push_dv=self._pushes())
        nu = info.extra["nu"]
        self.qd, self.v_body, self.gyro = nu[:, 6:].clone(), nu[:, 3:6].clone(), info.extra["gyro"].clone()
        self.torque, self.current = info.torque, info.current
        self.held_target = info.extra["target"].clone()
        self.contact = info.contact_active.clone()
        led = self.plant.ledger
        power = {k: (getattr(led, k) - before[k]) / c.policy_dt for k in before}
        self.episode_length_buf += 1
        self._refresh()
        self.camera.update(c.policy_dt, self.ball_pos.float(), self.head_pos.float(), self.base_quat.float(),
                           self.base_pos[:, :2].float(), self.yaw.float())

        # the end of an episode
        end = self.request_step + self.kick_steps
        self.time_outs = self.active & (self.episode_length_buf >= end)
        tilt = torch.acos((-self.gravity[:, 2]).clamp(-1.0, 1.0))
        self.exploded = ~(torch.isfinite(self.plant.q).all(-1) & torch.isfinite(self.plant.p).all(-1)
                          & torch.isfinite(self.plant.ball).all(-1))
        self.fell = (tilt > math.radians(c.termination_tilt_deg)) | self.exploded
        self.ground = (self.contact & self.not_sole).any(-1)
        self.terminated = self.fell | self.ground
        self.time_outs = self.time_outs & ~self.terminated
        dones = self.terminated | self.time_outs

        self._update_kick()
        # the kick's electrical energy: from the request until the launch is over
        counting = self.active & (~self.kicked | self.launching)
        self.request_energy = self.request_energy + power["supplied"] * c.policy_dt * counting

        terms = self._terms(power["supplied"])
        mask = {"kick": self.active.to(self.dtype), "walk": (~self.active).to(self.dtype), "both": 1.0}
        reward = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        for k, w in c.rewards.items():
            v = torch.nan_to_num(w * self._reward_scale(k) * terms[k] * c.policy_dt * mask[self.phase[k]],
                                 nan=0.0, posinf=0.0, neginf=0.0)
            self.episode_sums[k] += v
            reward += v
        self._update_commands()

        log = self._step_log(power, tilt)
        ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            self._episode_log(ids, log)
            if bool(self.exploded.any()):  # a non-finite state: its robots start again, the others go on
                bad = self.exploded.nonzero(as_tuple=False).squeeze(-1)
                reward[bad] = 0.0
            self._reset_idx(ids)
        self._refresh()
        self._start_requests()
        self.extras = {"time_outs": self.time_outs, "log": log}
        return self.get_observations(), reward, dones, self.extras

    # ------------------------------------------------------------------ logs
    def _step_log(self, power: dict, tilt: torch.Tensor) -> dict:
        c = self.cfg
        legs = slice(10, 22)
        tau, tau_max = self.torque[:, legs].abs(), self.plant.motors.tau_max[..., legs]
        walking = ~self.active & (self.commands.abs().sum(-1) > 0)
        vel_h = C.to_heading(self.yaw, self.lin_vel_w)[:, :2]
        log = {
            "Energy/supplied_W": power["supplied"].mean(),
            "Energy/copper_W": power["copper"].mean(),
            "Energy/gearbox_W": power["gearbox"].mean(),
            "Energy/contact_W": power["contact"].mean(),
            "Energy/stops_W": power["stops"].mean(),
            "Motors/leg_torque_mean_Nm": tau.mean(),
            "Motors/leg_torque_peak_Nm": tau.amax(),
            "Motors/leg_current_peak_A": self.current[:, legs].abs().amax(),
            "Motors/at_torque_limit": (tau >= 0.98 * tau_max).to(self.dtype).mean(),
            "Motors/overspeed": (self.qd.abs() > c.overspeed_fraction * self.v_max).to(self.dtype).mean(),
            "Policy/action_abs": self.actions.abs().mean(),
            "Policy/action_rate": (self.actions - self.last_actions).abs().mean(),
            "Metrics/tilt_deg": torch.rad2deg(tilt).mean(),
            "Metrics/trunk_height_m": self.base_pos[:, 2].mean(),
            "Metrics/in_request": self.active.to(self.dtype).mean(),
            "Walk/velocity_error": torch.linalg.vector_norm(self.commands[:, :2] - vel_h, dim=1)[walking],
            "Walk/yaw_rate_error": (self.commands[:, 2] - self.ang_vel_w[:, 2]).abs()[walking],
        }
        n = min(c.energy_log_envs, self.num_envs)
        if n > 0:
            rows = self.index[:n]
            ok = torch.isfinite(self.plant.q[rows]).all(-1)
            if bool(ok.all()):
                tanks = self.plant.energy(rows=rows)
                for k in TANKS:
                    log[f"Energy/H_{k}_J"] = tanks[k].mean()
        return log

    def _episode_log(self, ids: torch.Tensor, log: dict) -> None:
        c = self.cfg
        f = lambda x: x.to(self.dtype)  # noqa: E731
        for k, v in self.episode_sums.items():
            log[f"Episode_Reward/{k}"] = v[ids].mean()
        log["Episode_Termination/fell"] = f(self.fell[ids])
        log["Episode_Termination/ground"] = f(self.ground[ids])
        log["Episode_Termination/timeout"] = f(self.time_outs[ids])
        log["Episode/length_s"] = f(self.episode_length_buf[ids]) * c.policy_dt
        kicked, requested, fell = self.kicked[ids], self.active[ids], self.terminated[ids]
        launch, target, want = self.launch_vel[ids], self.target_w[ids], self.target_speed[ids]
        along = (launch * target).sum(-1)
        derr = torch.rad2deg(C.direction_error(launch, target))
        full = want >= c.full_power
        serr = self.speed_error(self.launch_vel)[ids]
        log["Kick/kick_rate"] = f(kicked[requested])
        log["Kick/launch_speed"] = along[kicked]
        log["Kick/launch_speed_full_power"] = along[kicked & full]
        log["Kick/speed_error"] = serr[kicked & ~full]
        log["Kick/direction_error_deg"] = derr[kicked]
        log["Kick/within_5deg"] = f(derr[kicked] < 5.0)
        log["Kick/within_10deg"] = f(derr[kicked] < 10.0)
        log["Kick/loft"] = launch[kicked, 2].abs()
        log["Kick/touch_time"] = self.touch_time[ids][kicked]
        log["Kick/fell_walking"] = f(fell & ~requested)
        log["Kick/fell_kicking"] = f(fell[requested])
        survived = self.time_outs[ids]
        log["Kick/end_heading_deg"] = torch.rad2deg(C.wrap(self.yaw[ids] - self.request_yaw[ids]).abs())[survived]
        log["Kick/end_pose_deg"] = torch.rad2deg((self.theta[ids] - self.contract.home).abs().mean(-1))[survived]
        log["Kick/end_stance_offset_m"] = self.stance_offset[ids][survived]
        far = requested & (self.request_range[ids] >= 0.6)
        moving = requested & (self.request_speed[ids] >= 0.3)
        log["Kick/kick_rate_far"] = f(kicked[far])
        log["Kick/direction_error_deg_far"] = derr[kicked & far]
        log["Kick/kick_rate_moving"] = f(kicked[moving])
        log["Kick/direction_error_deg_moving"] = derr[kicked & moving]
        log["Kick/launch_speed_moving"] = along[kicked & moving]
        # energy of the kick: what the battery gave from the request to the end of the launch, what the ball got
        spent = self.request_energy[ids]
        ball_j = 0.5 * c.ball_mass * launch.square().sum(-1)
        log["Energy/kick_supplied_J"] = spent[kicked]
        log["Energy/ball_launch_J"] = ball_j[kicked]
        good = kicked & (spent > 1.0)
        log["Energy/kick_efficiency"] = (ball_j / spent.clamp(min=1.0))[good]
        self.last_shot[ids] = torch.stack((
            f(requested), f(kicked), along, derr, launch[:, 2].abs(), serr, want, f(fell), self.request_range[ids],
            self.request_speed[ids], self.touch_time[ids], spent, ball_j, torch.linalg.vector_norm(launch[:, :2], dim=1),
        ), -1)

    def config_dict(self) -> dict:
        d = asdict(self.cfg)
        d["contract"] = {"agent": "kick_walk", "gains": C.GAINS, "obs_dim": self.contract.obs_dim,
                         "hidden_dims": list(C.HIDDEN_DIMS)}
        d["stand_height_m"] = self.stand_height
        d["ball"] = asdict(self.ball_params)
        return d
