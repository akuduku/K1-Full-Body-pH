r"""The head camera's reports of the ball, as the robot's firmware has them (k1_deploy's ball tracker).

`BallCamera` is copied unchanged from Eric Plass's k1_kick env.py (there: "k1_rolling's camera, as of rolling-020"),
so a policy trained there sees the same inputs here.  `CAMERA` is k1_kick train.py's setting for the kick.
"""

from __future__ import annotations

import math

import torch

from .contract import rotate_inverse, to_heading, wrap

# k1_kick train.py CAMERA (k1_rolling's, rolling-020): what the robot's firmware has of the ball
CAMERA = {
    "rate_hz": (8.0, 30.0),      # frames per second, per request
    "dropout": 0.15,             # share of frames without a detection
    "noise": 0.02,               # x-y noise std [m] ...
    "noise_per_m": 0.03,         # ... plus this much per metre of distance
    "history_steps": 40,         # reports the policy sees (one per control step: 0.8 s)
    "velocity_window": 15,       # control steps (0.3 s) of reports the ball velocity is fitted through
    "fov_deg": (90.0, 60.0),     # the head tracks the ball within its joint limits
    "head_yaw": 1.0,             # [rad]
    "head_pitch": (-0.349, 0.855),  # [rad], positive looks down
    "max_range": 8.0,            # [m]
    "latency_steps": (1, 3),     # 20-60 ms, per request
    "frame": "yaw",              # reports relative to the trunk at capture, turned by the IMU's yaw since
    "gyro_bias": 0.02,           # [rad/s] drift of the IMU's yaw, per request
    "gyro_scale": 0.03,          # scale error of the IMU's yaw change, per request
}
# a camera without errors (evaluation without noise): every step a frame, no dropout, no noise, one step late
CAMERA_CLEAN = {**CAMERA, "rate_hz": (50.0, 50.0), "dropout": 0.0, "noise": 0.0, "noise_per_m": 0.0,
                "latency_steps": (1, 1), "gyro_bias": 0.0, "gyro_scale": 0.0}


class BallCamera:
    """
    k1_rolling's camera (as of rolling-020).

    What the head camera reports about the ball: positions at a random frame
    rate, only while the ball is in view, with dropped frames, latency and x-y
    noise growing with distance, held between frames. The velocity estimate is
    the least-squares slope through the frames of the last ``velocity_window``
    steps (held while fewer than two frames are in the window); ``age`` is the
    time since the last frame.

    The head is assumed to track the ball within its joint limits: the ball
    is in view if its direction from the head (trunk frame) is within the yaw
    limit plus half the horizontal field of view, the pitch range plus half
    the vertical one, and closer than ``max_range``. The last
    ``history_steps`` reports (one per control step, oldest first) are kept as
    world points, so the robot's own motion does not smear the ball's path.

    ``frame="robot"`` (since rolling-019) reports what the robot's firmware has
    instead (k1_deploy's ball tracker, as k1_dribble since dribble-006): each
    report is the ball relative to the trunk in the heading frame of the moment
    the frame was taken (the pose ``lag`` steps ago), held between frames and
    never re-projected as the robot moves; the history holds these held reports
    and the velocity is fitted through them (the ball relative to the moving
    robot). The robot has no odometry to do better.

    ``frame="yaw"`` (since rolling-020) adds what the robot's IMU knows: each
    report is kept in the axes of the IMU's yaw at capture and turned into the
    heading frame of now, so the robot's own turn no longer moves the reports
    (its walk still does). The IMU's yaw is the true yaw change times a scale
    error up to ``gyro_scale`` plus a bias drift up to ``gyro_bias`` rad/s, both
    per request.
    """

    def __init__(self, num_envs, device, rate_hz, dropout, noise, noise_per_m, history_steps, velocity_window,
                 fov_deg, head_yaw, head_pitch, max_range, latency_steps, frame="world",
                 gyro_bias=0.0, gyro_scale=0.0):
        self.rate_hz, self.dropout = rate_hz, dropout
        self.noise, self.noise_per_m = noise, noise_per_m
        self.window = velocity_window
        half_h, half_v = math.radians(fov_deg[0]) / 2, math.radians(fov_deg[1]) / 2
        self.azimuth = head_yaw + half_h
        self.elevation = (-(head_pitch[1] + half_v), -head_pitch[0] + half_v)
        self.max_range = max_range
        self.latency_steps = latency_steps
        self.period = torch.ones(num_envs, device=device)  # s between frames
        self.since = torch.zeros(num_envs, device=device)  # s since the last frame
        self.lag = torch.zeros(num_envs, dtype=torch.long, device=device)  # control steps of latency
        self.seen = torch.zeros(num_envs, 2, device=device)  # last reported ball x-y (world)
        self.past = torch.zeros(num_envs, latency_steps[1] + 1, 2, device=device)  # true ball x-y, newest last
        self.history = torch.zeros(num_envs, history_steps, 2, device=device)  # reports, oldest first
        self.fresh = torch.zeros(num_envs, history_steps, dtype=torch.bool, device=device)  # which reports were new frames
        self.age = torch.zeros(num_envs, device=device)  # s since the last frame
        self.vel = torch.zeros(num_envs, 2, device=device)  # velocity estimate (world x-y)
        self.times = torch.arange(velocity_window, device=device, dtype=torch.float) - (velocity_window - 1)  # steps before now
        if frame not in ("world", "robot", "yaw"):
            raise ValueError(f"camera frame {frame!r}: 'world', 'robot' or 'yaw'")
        self.relative = frame != "world"
        self.turn = frame == "yaw"
        # robot and yaw frames: trunk x-y, yaw and the IMU's yaw of the past steps (newest last), the held
        # report and its history (yaw frame: in the IMU's axes)
        self.past_pose = torch.zeros(num_envs, latency_steps[1] + 1, 4, device=device)
        self.seen_rel = torch.zeros(num_envs, 2, device=device)
        self.history_rel = torch.zeros(num_envs, history_steps, 2, device=device)
        # yaw frame: the IMU's yaw, the true yaw it follows, its bias drift (rad/s) and scale per request
        self.gyro_bias, self.gyro_scale = gyro_bias, gyro_scale
        self.imu_yaw = torch.zeros(num_envs, device=device)
        self.last_yaw = torch.zeros(num_envs, device=device)
        self.bias = torch.zeros(num_envs, device=device)
        self.scale = torch.ones(num_envs, device=device)

    def start(self, env_ids, ball_xy, base_xy=None, yaw=None):
        """A new request: new frame rate and latency, the history filled with a first report
        (robot frame: relative to the trunk pose ``base_xy``, ``yaw``)."""
        n = len(env_ids)
        self.period[env_ids] = 1.0 / torch.empty(n, device=ball_xy.device).uniform_(*self.rate_hz)
        self.lag[env_ids] = torch.randint(self.latency_steps[0], self.latency_steps[1] + 1, (n,), device=ball_xy.device)
        self.since[env_ids] = 0.0
        self.seen[env_ids] = ball_xy
        self.past[env_ids] = ball_xy[:, None]
        self.history[env_ids] = ball_xy[:, None]
        self.fresh[env_ids] = False
        self.fresh[env_ids, -1] = True
        self.age[env_ids] = 0.0
        self.vel[env_ids] = 0.0
        if self.relative:
            self.imu_yaw[env_ids] = yaw
            self.last_yaw[env_ids] = yaw
            self.bias[env_ids] = torch.empty(n, device=yaw.device).uniform_(-self.gyro_bias, self.gyro_bias)
            self.scale[env_ids] = 1.0 + torch.empty(n, device=yaw.device).uniform_(-self.gyro_scale, self.gyro_scale)
            self.past_pose[env_ids] = torch.cat([base_xy, yaw[:, None], yaw[:, None]], dim=1)[:, None]
            rel = to_heading(yaw, ball_xy - base_xy)
            if self.turn:  # into the IMU's axes
                rel = to_heading(-yaw, rel)
            self.seen_rel[env_ids] = rel
            self.history_rel[env_ids] = rel[:, None]

    def update(self, dt, ball_w, head_w, trunk_quat_w, base_xy=None, yaw=None):
        """One control step: maybe a camera frame of the ball as it was ``lag``
        steps ago, if in view; then append the held report."""
        self.past = torch.cat([self.past[:, 1:], ball_w[:, None, :2]], dim=1)
        if self.relative:
            self.imu_yaw = self.imu_yaw + self.scale * wrap(yaw - self.last_yaw) + self.bias * dt
            self.last_yaw = yaw.clone()
            pose = torch.cat([base_xy, yaw[:, None], self.imu_yaw[:, None]], dim=1)
            self.past_pose = torch.cat([self.past_pose[:, 1:], pose[:, None]], dim=1)
        self.since += dt
        frame = self.since >= self.period
        self.since[frame] = 0.0
        rel = rotate_inverse(trunk_quat_w, ball_w - head_w)
        azimuth = torch.atan2(rel[:, 1], rel[:, 0]).abs()
        elevation = torch.atan2(rel[:, 2], torch.linalg.vector_norm(rel[:, :2], dim=1))
        visible = ((azimuth <= self.azimuth) & (elevation >= self.elevation[0]) & (elevation <= self.elevation[1])
                   & (torch.linalg.vector_norm(rel, dim=1) <= self.max_range))
        seen = frame & visible & (torch.rand_like(self.since) >= self.dropout)
        ball_xy = self.past[torch.arange(len(self.lag), device=self.lag.device), -1 - self.lag]
        distance = torch.linalg.vector_norm(ball_xy - head_w[:, :2], dim=1, keepdim=True)
        noisy = ball_xy + torch.randn_like(ball_xy) * (self.noise + self.noise_per_m * distance)
        self.seen = torch.where(seen[:, None], noisy, self.seen)
        self.age = torch.where(seen, torch.zeros_like(self.age), self.age + dt)
        self.history = torch.cat([self.history[:, 1:], self.seen[:, None]], dim=1)
        self.fresh = torch.cat([self.fresh[:, 1:], seen[:, None]], dim=1)
        if self.relative:
            then = self.past_pose[torch.arange(len(self.lag), device=self.lag.device), -1 - self.lag]
            rel = to_heading(then[:, 2], noisy - then[:, :2])
            if self.turn:  # into the IMU's axes at capture
                rel = to_heading(-then[:, 3], rel)
            self.seen_rel = torch.where(seen[:, None], rel, self.seen_rel)
            self.history_rel = torch.cat([self.history_rel[:, 1:], self.seen_rel[:, None]], dim=1)
        self._fit_velocity(dt)

    def _fit_velocity(self, dt):
        """Least-squares slope (world x-y; robot and yaw frames: the held reports') through the frames of
        the last ``velocity_window`` steps; the estimate is held while fewer than two frames are
        in the window."""
        weight = self.fresh[:, -self.window:].float()
        points = (self.history_rel if self.relative else self.history)[:, -self.window:]
        times = self.times * dt
        count = weight.sum(dim=1).clamp(min=1.0)
        centred = (times - (weight * times).sum(dim=1, keepdim=True) / count[:, None]) * weight
        mean = (weight[..., None] * points).sum(dim=1) / count[:, None]
        var = centred.square().sum(dim=1)
        cov = (centred[..., None] * (points - mean[:, None])).sum(dim=1)
        valid = (weight.sum(dim=1) >= 2.0) & (var > 0.0)
        self.vel = torch.where(valid[:, None], cov / var.clamp(min=1e-9)[:, None], self.vel)

    def velocity(self):
        """The ball velocity estimate (world x-y; robot frame: relative to the robot, heading frame;
        yaw frame: relative to the robot, the IMU's axes)."""
        return self.vel

    def reports(self):
        """Robot and yaw frames: the held report, the velocity estimate and the history (N, H, 2) in the
        heading frame of now (yaw frame: turned by the IMU's yaw since each capture)."""
        if not self.turn:
            return self.seen_rel, self.vel, self.history_rel
        return (to_heading(self.imu_yaw, self.seen_rel), to_heading(self.imu_yaw, self.vel),
                to_heading(self.imu_yaw, self.history_rel))
