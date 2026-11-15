r"""Drive a trained walking policy yourself, on the pH plant, on MuJoCo, or on both at once.

    python rl/walk/drive.py                                # pH plant, walk_v9
    python rl/walk/drive.py --physics mujoco               # same policy, MuJoCo physics
    python rl/walk/drive.py --physics both                 # two robots, same sliders, side by side
    python rl/walk/drive.py --policy soccer-rolling-walk   # a name in k1_sprint/checkpoints, or a file
    python rl/walk/drive.py --no-viewer --seconds 8 --twist 0.5 0 0   # headless check

Three sliders steer the robot: forward speed vx, sideways speed vy and turn rate wz, with the limits printed at both
ends.  A slider stays where you leave it, so the robot keeps walking at that request until you move it again.  The
command then follows the firmware's ramp (contract.py ramp_twist), as on the robot.

Buttons: Stop (all sliders to 0), Reset robot.  In the MuJoCo window: Home is Stop, End or R is Reset, Q or ESC quits.
The panel shows the requested, ramped and measured twist under the sliders.  (Not in the MuJoCo window: viewer.set_texts
stalls the loop with mujoco 3.10.)
A robot that fell is put back on its feet after 2 s, and the sliders keep their values.

What runs: the sprint agent exactly as on the robot (contract.py), through the firmware's command path, on one physics
engine or both (backends.py).  MuJoCo always draws.  With --physics pH the pH plant moves the robot and MuJoCo's own
dynamics are not used at all.  With --physics both the pH robot has the model's colours, the MuJoCo robot is orange and
is drawn --offset metres to the side; the two do not touch each other.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ph_k1.model import load_k1  # noqa: E402
from rl.walk import contract as C  # noqa: E402
from rl.walk.backends import MuJoCoBackend, PHBackend, Walker, load_walk_policy  # noqa: E402

# label, unit, index in the twist, spacing of the numbers under the slider
SLIDERS = (("Forward speed  vx", "m/s", 0, 0.5), ("Sideways speed  vy", "m/s", 1, 0.5), ("Turn rate  wz", "rad/s", 2, 1.0))
RESOLUTION = 0.05
RESTART_AFTER_S = 2.0
READOUT_EVERY = 5  # policy steps between updates of the panel's numbers
ORANGE = (0.95, 0.55, 0.15, 1.0)
GLFW_ESCAPE, GLFW_HOME, GLFW_END = 256, 268, 269


class TwistPanel:
    """The three sliders.  Runs in the main thread (pump()).  Only Stop changes a slider from the program."""

    def __init__(self, start=(0.0, 0.0, 0.0)):
        import tkinter as tk

        self._error = tk.TclError
        self.root = tk.Tk()
        self.root.title("K1 walk command")
        self.closed = False
        self.reset_pressed = False
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.vars = []
        self.scales = []
        for row, (label, unit, i, tick) in enumerate(SLIDERS):
            lo, hi = float(C.TWIST_LOW[i]), float(C.TWIST_HIGH[i])
            var = tk.DoubleVar(value=float(np.clip(start[i], lo, hi)))
            tk.Label(self.root, text=f"{label}  [{unit}]", font=("TkDefaultFont", 11, "bold"), anchor="w").grid(
                row=2 * row, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 0))
            tk.Label(self.root, text=f"{lo:+g}", font=("TkDefaultFont", 12, "bold"), width=5).grid(row=2 * row + 1, column=0)
            scale = tk.Scale(self.root, variable=var, from_=lo, to=hi, resolution=RESOLUTION, tickinterval=tick, digits=3,
                             orient=tk.HORIZONTAL, length=520, width=20, sliderlength=26)
            scale.grid(row=2 * row + 1, column=1, sticky="ew")
            tk.Label(self.root, text=f"{hi:+g}", font=("TkDefaultFont", 12, "bold"), width=5).grid(row=2 * row + 1, column=2)
            self.vars.append(var)
            self.scales.append(scale)
        buttons = tk.Frame(self.root)
        buttons.grid(row=2 * len(SLIDERS), column=0, columnspan=3, pady=14)
        tk.Button(buttons, text="Stop (all to 0)", width=16, command=self.stop).pack(side=tk.LEFT, padx=8)
        tk.Button(buttons, text="Reset robot", width=16, command=self._press_reset).pack(side=tk.LEFT, padx=8)
        self.readout = tk.Label(self.root, font=("TkFixedFont", 11), justify=tk.LEFT, anchor="w")
        self.readout.grid(row=2 * len(SLIDERS) + 1, column=0, columnspan=3, sticky="w", padx=14, pady=(0, 12))
        self.root.columnconfigure(1, weight=1)

    def _close(self):
        self.closed = True
        self.root.destroy()

    def _press_reset(self):
        self.reset_pressed = True

    def stop(self):
        for var in self.vars:
            var.set(0.0)

    def values(self) -> np.ndarray:
        return np.array([var.get() for var in self.vars])

    def show(self, text: str) -> None:
        self.readout.configure(text=text)

    def take_reset(self) -> bool:
        pressed, self.reset_pressed = self.reset_pressed, False
        return pressed

    def pump(self) -> bool:
        """Serve the window. False once it was closed."""
        if self.closed:
            return False
        try:
            self.root.update()
        except self._error:
            self.closed = True
        return not self.closed


class Meter:
    """Measured twist in the heading frame, from the trunk's motion, smoothed over about 0.4 s."""

    def __init__(self):
        self.value = np.zeros(3)
        self.last = None

    def clear(self):
        self.last = None

    def update(self, reading) -> np.ndarray:
        yaw = C.yaw_of(reading.quat)
        if self.last is not None:
            pos, last_yaw = self.last
            dx, dy = (reading.pos[:2] - pos[:2]) / C.POLICY_DT
            c, s = math.cos(yaw), math.sin(yaw)
            turn = math.remainder(yaw - last_yaw, 2 * math.pi) / C.POLICY_DT
            self.value += 0.05 * (np.array((c * dx + s * dy, -s * dx + c * dy, turn)) - self.value)
        self.last = (reading.pos.copy(), yaw)
        return self.value


def status_text(walkers, meters, want, colours) -> str:
    """The panel's readout: requested, ramped and measured twist."""
    def row(label, v):
        return f"{label:<18}vx {v[0]:+.2f}  vy {v[1]:+.2f}  wz {v[2]:+.2f}"

    lines = [row("Requested", want), row("Ramped", walkers[0].twist)]
    for walker, meter, colour in zip(walkers, meters, colours):
        label = f"{walker.backend.name}{colour}"
        lines.append(f"{label:<18}FELL, restarting" if walker.fell else row(label, meter.value))
    return "\n".join(lines)


def draw_second_robot(viewer, mj, data, offset: float) -> None:
    """The second robot, drawn from its own MjData into the viewer's extra geometry."""
    import mujoco

    scene = viewer.user_scn
    scene.ngeom = 0
    mujoco.mjv_addGeoms(mj, data, viewer.opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
    for i in range(scene.ngeom):
        geom = scene.geoms[i]
        geom.pos[1] += offset
        geom.rgba[:] = ORANGE


def headless(walkers, want, seconds: float) -> None:
    """Hold one request for `seconds` and print how each robot did."""
    for walker in walkers:
        meter = Meter()
        started = time.time()
        steps = round(seconds / C.POLICY_DT)
        for _ in range(steps):
            meter.update(walker.tick(want))
            if walker.fell:
                break
        v = meter.value
        print(f"{walker.backend.name:7s} fell={walker.fell}  measured vx {v[0]:+.2f} vy {v[1]:+.2f} wz {v[2]:+.2f}  "
              f"(computed in {time.time() - started:.1f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--physics", choices=("ph", "mujoco", "both"), default="ph")
    ap.add_argument("--policy", default="walk_v9", help="a name in k1_sprint/checkpoints, or a .pt / .bin file")
    ap.add_argument("--delay-ms", type=float, default=20.0, help="transport delay (the robot: about 20 ms)")
    ap.add_argument("--twist", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "WZ"),
                    help="start value of the sliders")
    ap.add_argument("--offset", type=float, default=0.8, help="--physics both: drawn distance of the MuJoCo robot [m]")
    ap.add_argument("--seconds", type=float, default=0.0, help="0: until a window is closed")
    ap.add_argument("--no-viewer", action="store_true", help="headless: hold --twist for --seconds, print the outcome")
    args = ap.parse_args()

    model = load_k1()
    policy = load_walk_policy(args.policy)
    backends = {"ph": (PHBackend,), "mujoco": (MuJoCoBackend,), "both": (PHBackend, MuJoCoBackend)}[args.physics]
    walkers = [Walker(b(model, delay_ms=args.delay_ms), policy) for b in backends]

    if args.no_viewer:
        headless(walkers, np.array(args.twist), args.seconds or 8.0)
        return

    import glfw

    if os.environ.get("WAYLAND_DISPLAY"):  # as the K1_Deploy twin: without libdecor a Wayland window has no title bar
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    import mujoco
    import mujoco.viewer

    state = {"quit": False, "stop": False, "reset": False}

    def on_key(keycode):  # the viewer's thread: only set flags, the main loop owns the panel
        if keycode in (ord("R"), ord("r"), GLFW_END):
            state["reset"] = True
        elif keycode == GLFW_HOME:
            state["stop"] = True
        elif keycode in (ord("Q"), ord("q"), GLFW_ESCAPE):
            state["quit"] = True

    panel = TwistPanel(args.twist)
    mj = model.mj
    frames = [mujoco.MjData(mj) for _ in walkers]
    meters = [Meter() for _ in walkers]
    fell_at = [None for _ in walkers]
    colours = ("", "") if len(walkers) == 1 else ("  (grey)", "  (orange)")
    with mujoco.viewer.launch_passive(mj, frames[0], key_callback=on_key, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, model.body_names[0])
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.2, -15.0, 135.0
        print("[drive] sliders steer | Stop: all to 0 | Reset robot | in the MuJoCo window: Home stop, R reset, Q quit",
              flush=True)
        t, steps, tick = 0.0, 0, time.monotonic()
        started = tick
        while viewer.is_running() and panel.pump() and not state["quit"]:
            if args.seconds and t >= args.seconds:
                break
            if state["stop"]:
                state["stop"] = False
                panel.stop()
            want = panel.values()
            reset_all = panel.take_reset() or state["reset"]
            state["reset"] = False
            for i, walker in enumerate(walkers):
                if walker.fell and fell_at[i] is None:
                    fell_at[i] = t
                    print(f"[drive] t={t:.1f} s: {walker.backend.name} fell at request {np.round(want, 2)}", flush=True)
                if reset_all or (fell_at[i] is not None and t - fell_at[i] >= RESTART_AFTER_S):
                    walker.reset()
                    meters[i].clear()
                    fell_at[i] = None
                meters[i].update(walker.tick(want))
            with viewer.lock():
                for walker, frame in zip(walkers, frames):
                    frame.qpos[:] = walker.reading.qpos
                    mujoco.mj_kinematics(mj, frame)
                if len(walkers) == 2:
                    draw_second_robot(viewer, mj, frames[1], args.offset)
            viewer.sync()
            if steps % READOUT_EVERY == 0:
                panel.show(status_text(walkers, meters, want, colours))
            steps += 1
            t += C.POLICY_DT
            tick += C.POLICY_DT
            time.sleep(max(0.0, tick - time.monotonic()))
            if time.monotonic() - tick > 0.5:  # fell behind (window dragged, ...): do not catch up
                tick = time.monotonic()
    print(f"[drive] {t:.1f} s walked in {time.monotonic() - started:.1f} s", flush=True)
    if not panel.closed:
        panel.root.destroy()
    # as the K1_Deploy twin: glfw's exit handler hangs after a passive viewer closed
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
