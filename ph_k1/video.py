r"""MP4 videos of the pH plant, e.g. for W&B every 500 training iterations.

MuJoCo only DRAWS here, exactly like render.py's viewer: q has MuJoCo's qpos layout, so
qpos = q[b] and mj_kinematics place the bodies; mj_step is never called.

Headless machines (Kaggle, Colab): set MUJOCO_GL before mujoco is imported (before importing ph_k1),
    MUJOCO_GL=egl         NVIDIA GPU
    MUJOCO_GL=osmesa      software fallback (apt-get install libosmesa6)
and have an MP4 writer:  pip install imageio imageio-ffmpeg

    rec = VideoRecorder(plant, b=0)
    for it in range(250):
        plant.policy_step(target)
        rec.capture()
    path = rec.save("outputs/hold.mp4")
    log_video(path, "video/hold", step=iteration)      # wandb.Video, if a W&B run is active
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


class VideoRecorder:
    """Frames of robot `b` with a camera that follows the trunk."""

    def __init__(self, plant, b: int = 0, width: int = 640, height: int = 480, fps: float = 50.0,
                 distance: float = 2.2, azimuth: float = 135.0, elevation: float = -15.0):
        self.plant, self.b, self.fps = plant, b, fps
        self.mj = plant.model.mj
        self.data = mujoco.MjData(self.mj)
        width = min(width, self.mj.vis.global_.offwidth)
        height = min(height, self.mj.vis.global_.offheight)
        self.renderer = mujoco.Renderer(self.mj, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.frames: list[np.ndarray] = []

    def capture(self) -> None:
        q = self.plant.q[self.b].detach().to("cpu", dtype=self.plant.q.dtype).double().numpy()
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.mj, self.data)
        # follow the trunk at a fixed height (mj_kinematics does not compute what a tracking camera needs)
        self.cam.lookat[:] = (q[0], q[1], max(q[2] - 0.1, 0.3))
        self.renderer.update_scene(self.data, camera=self.cam)
        self.frames.append(self.renderer.render().copy())

    def save(self, path: str | Path) -> Path:
        import imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), self.frames, fps=self.fps, macro_block_size=8)
        return path

    def close(self) -> None:
        self.renderer.close()


def log_video(path: str | Path, key: str = "video", step: int | None = None, fps: float = 50.0) -> bool:
    """Log an MP4 to the active W&B run; returns False when there is none."""
    try:
        import wandb
    except ImportError:
        return False
    if wandb.run is None:
        return False
    wandb.log({key: wandb.Video(str(path), fps=int(fps), format="mp4")}, step=step)
    return True
