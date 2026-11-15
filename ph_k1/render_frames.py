r"""Render recorded robot poses to an MP4, in a process of its own (run by ph_k1/video.py, not imported).

Drawing lives here and not in the training process: creating the EGL context there killed the whole training with
SIGSEGV on Kaggle (mujoco 3.17, inside mujoco.Renderer), although a fresh process renders fine.  A segfault cannot be
caught, a child process that dies only costs the video.  This script needs mujoco, numpy and imageio, not torch.

    python render_frames.py frames.npz out.mp4
"""

import sys

import imageio
import mujoco
import numpy as np


def main(npz: str, out: str) -> None:
    d = np.load(npz)
    mj = mujoco.MjModel.from_xml_path(str(d["xml"]))   # the MJCF's own names: only qpos is set here
    data = mujoco.MjData(mj)
    width = min(int(d["width"]), mj.vis.global_.offwidth)
    height = min(int(d["height"]), mj.vis.global_.offheight)
    renderer = mujoco.Renderer(mj, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = float(d["distance"]), float(d["azimuth"]), float(d["elevation"])
    balls = d["ball"] if "ball" in d else None
    arrows = d["arrow"] if "arrow" in d else None
    radius = float(d["ball_radius"]) if "ball_radius" in d else 0.075
    eye = np.eye(3).flatten()
    frames = []
    for k, q in enumerate(d["qpos"]):
        data.qpos[:] = q
        mujoco.mj_kinematics(mj, data)
        # follow the trunk at a fixed height (mj_kinematics does not compute what a tracking camera needs)
        cam.lookat[:] = (q[0], q[1], max(q[2] - 0.1, 0.3))
        renderer.update_scene(data, camera=cam)
        scn = renderer.scene
        if balls is not None and scn.ngeom < scn.maxgeom and np.all(np.isfinite(balls[k])):  # the plant's ball
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]),
                                balls[k], eye, np.array([1.0, 0.45, 0.05, 1.0], dtype=np.float32))
            scn.ngeom += 1
        if arrows is not None and scn.ngeom < scn.maxgeom and np.all(np.isfinite(arrows[k])):  # e.g. the kick direction
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), eye,
                                np.array([0.1, 0.7, 0.2, 0.9], dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, arrows[k, :3], arrows[k, 3:])
            scn.ngeom += 1
        frames.append(renderer.render().copy())
    renderer.close()
    imageio.mimsave(out, frames, fps=float(d["fps"]), macro_block_size=8)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
