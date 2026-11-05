r"""MP4 videos of the pH plant, e.g. for W&B every 500 training iterations.

MuJoCo only DRAWS here, exactly like render.py's viewer: q has MuJoCo's qpos layout, so
qpos = q[b] and mj_kinematics place the bodies; mj_step is never called.

The drawing happens in a child process (ph_k1/render_frames.py), which inherits the environment.
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

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RENDERER = Path(__file__).with_name("render_frames.py")


class VideoRecorder:
    """Frames of robot `b` with a camera that follows the trunk.

    capture() only keeps the pose; save() draws them in a child process (ph_k1/render_frames.py), so no OpenGL context
    ever exists in the calling process: a crash of the renderer raises RuntimeError here instead of killing a training."""

    def __init__(self, plant, b: int = 0, width: int = 640, height: int = 480, fps: float = 50.0,
                 distance: float = 2.2, azimuth: float = 135.0, elevation: float = -15.0):
        self.plant, self.b, self.fps = plant, b, fps
        self.size = (width, height)
        self.view = (distance, azimuth, elevation)
        self.frames: list[np.ndarray] = []

    def capture(self) -> None:
        self.frames.append(self.plant.q[self.b].detach().to("cpu").double().numpy().copy())

    def save(self, path: str | Path, timeout: float = 600.0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "frames.npz"
            np.savez(npz, qpos=np.stack(self.frames), xml=str(self.plant.model.xml_path), fps=self.fps,
                     width=self.size[0], height=self.size[1], distance=self.view[0], azimuth=self.view[1],
                     elevation=self.view[2])
            r = subprocess.run([sys.executable, str(RENDERER), str(npz), str(path)], capture_output=True, text=True,
                               timeout=timeout)
        if r.returncode != 0 or not path.exists():
            raise RuntimeError(f"render process exited with {r.returncode}: {r.stderr[-300:]}")
        return path

    def close(self) -> None:
        pass


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
