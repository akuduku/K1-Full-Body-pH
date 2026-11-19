r"""A video of a walk policy for W&B, with the Hamiltonian figure of the same episode (rl/walk/hamiltonian.py).

One robot, the quiet world (no pushes, no sensor noise, nominal gains, 20 ms delay), on the C++ pH plant, through
the firmware's command path; MuJoCo only draws (ph_k1/video.py).  The requests change during the episode (SCHEDULE),
and the request is drawn as an arrow on the floor (its length: where the robot would be in 1 s).

    video = WalkVideo(WalkConfig(...), out_dir)
    result = video(policy, it)       # None (no renderer), the MP4's path, or (path, {"Energy/hamiltonian": png})
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from rl.common.harness import VideoMaker
from rl.kick.contract import yaw_of
from rl.walk.env import WalkConfig, WalkEnv, quiet
from rl.walk.hamiltonian import Recorder, figure

# (start [s], request (vx, vy, wz), label)
SCHEDULE = ((0.0, (0.0, 0.0, 0.0), "stand"), (2.0, (0.5, 0.0, 0.0), "forward 0.5 m/s"),
            (6.0, (0.0, 0.0, 0.5), "turn 0.5 rad/s"), (8.0, (0.0, 0.3, 0.0), "sideways 0.3 m/s"))


class WalkVideo(VideoMaker):
    def __init__(self, cfg: WalkConfig, out_dir: Path, seconds: float = 10.0, schedule=SCHEDULE,
                 hamiltonian: bool = True, grid: int = 41):
        super().__init__(lambda: WalkEnv(quiet(cfg, num_envs=1, device="cpu", seed=cfg.seed + 11)), seconds,
                         cfg.policy_dt, Path(out_dir), "walk", prepare=self._prepare)
        self.schedule, self.hamiltonian, self.grid, self.dt = tuple(schedule), hamiltonian, grid, cfg.policy_dt
        self.t, self.ham = 0.0, None

    def _request(self, t: float):
        return [s for s in self.schedule if s[0] <= t + 1e-9][-1][1]

    def _prepare(self, env, it: int) -> None:
        env.set_command(self.schedule[0][1])
        env._reset_idx(torch.arange(1, device=env.device))
        self.t = 0.0
        self.ham = Recorder(env) if self.hamiltonian else None

    def on_step(self, env, rec) -> None:
        self.t += self.dt
        want = self._request(self.t)
        if tuple(float(v) for v in env.mem["commands"][0]) != tuple(want):
            env.set_command(want)
        q = env.plant.q[0].detach().to("cpu", torch.float64).numpy()
        v = np.asarray(want[:2], dtype=np.float64)
        arrow = None
        if np.hypot(*v) > 0.05:                  # the request in the world frame, from under the trunk, 1 s long
            yaw = yaw_of(torch.as_tensor(q[3:7])[None]).item()
            c, s = math.cos(yaw), math.sin(yaw)
            start = np.array((q[0], q[1], 0.01))
            arrow = np.concatenate((start, start + np.array((c * v[0] - s * v[1], s * v[0] + c * v[1], 0.0))))
        rec.capture(arrow)
        if self.ham is not None:
            self.ham.capture(self.t)

    def on_end(self, env, path: Path, it: int):
        if self.ham is None:
            return path
        try:
            png = figure(self.ham, path.with_name(f"hamiltonian_{it:05d}.png"),
                         segments=[(s[0], s[2]) for s in self.schedule], grid=self.grid,
                         title=f"update {it}: one quiet episode on the pH plant ("
                               + ", ".join(f"{s[2]} from {s[0]:g} s" for s in self.schedule) + ")")
        except Exception as exc:  # noqa: BLE001  (the figure must not cost the video)
            print(f"[video] Hamiltonian figure failed: {type(exc).__name__}: {exc}", flush=True)
            png = None
        return (path, {"Energy/hamiltonian": png}) if png else path
