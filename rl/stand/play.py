r"""Play with the trained stand policy on the laptop: the pH plant moves the robot, MuJoCo draws it, you push it
with the mouse, and the energy tanks of H are plotted live.

    python rl/stand/play.py                                   # newest outputs/kaggle/stand-ph-v1/.../deploy_*.bin
    python rl/stand/play.py --policy deploy_2999.bin --delay-ms 30
    python rl/stand/play.py --no-viewer --seconds 12          # headless check: scripted pushes, plot to a PNG

Push: in the MuJoCo window DOUBLE-CLICK a body to select it, then hold CTRL + RIGHT mouse button and drag: a
spring pulls the selected point towards the mouse (MuJoCo's rule: stiffness x the mass hanging on that body;
the window draws the spring).  The slider "push strength" under the plot scales that force (x0 .. x5).
R (in the MuJoCo window) puts the robot back on its feet; it also stands up again by itself 2 s after a fall.

What runs: the stand agent exactly as in training -- rl/stand/contract.py observation and decode, the firmware's
gains, filter (clamp, slew, target clamp, overspeed guard), transport delay and motor driver
(ph_k1/firmware.py, K1PHPlantNative.policy_step) -- on the pH plant.  MuJoCo only draws (mj_kinematics) and
catches the mouse.  The push enters the plant as the generalised force J^T f of the pulled point (the plant's
gen_force port); its work goes into the energy ledger (mech_input), so

    H(t) - H(0) = supplied (battery) + push work - losses        (printed at the end, the residual is small)

The plot: the tanks of H that move -- kinetic energy, gravity (relative to the start) and the magnetic energy in
the coils -- over the last seconds, with the push force, the push work and the battery energy as numbers.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ph_k1.dynamics import point_jacobian  # noqa: E402
from ph_k1.firmware import FirmwareConfig  # noqa: E402
from ph_k1.model import NB, load_k1  # noqa: E402
from ph_k1.native import K1PHPlantNative  # noqa: E402
from rl.common.export import load_policy  # noqa: E402
from rl.stand import contract as C  # noqa: E402

TANKS = ("kinetic", "gravity", "coil")
COLORS = {"kinetic": "tab:blue", "gravity": "tab:green", "coil": "tab:orange"}


def newest_policy() -> Path | None:
    runs = REPO / "outputs" / "kaggle" / "stand-ph-v1" / "runs" / "stand-ph-v1"
    found = sorted(runs.glob("deploy_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    return found[-1] if found else None


def mouse_force(active: int, body: int | None, refpos, localpos, kin, mass_on_body: float, stiffness: float,
                scale: float, nu: torch.Tensor, max_force: float = float("inf")):
    """MuJoCo's mouse spring on the pH plant: world force f on the selected body point and the generalised force
    J^T f (1, 28).  f = scale * m * (k (refpos - x) - 2 sqrt(k) v_x)  (spring + critical damping per unit mass),
    at most scale * max_force [N] (the mass hanging on the trunk is the whole robot: a long drag would otherwise
    ask for kilonewtons, held for a 20 ms policy step).  Returns (f (3,), gen_force (1,28) or None, x (3,))."""
    if body is None or not (active & 1) or scale <= 0:
        return np.zeros(3), None, None
    R, P = kin.R[0, body], kin.pos[0, body]
    lp = torch.as_tensor(np.asarray(localpos, dtype=np.float64), dtype=R.dtype)
    x = P + R @ lp
    J = point_jacobian(kin, torch.tensor([body]), x[None, None])[:, 0]  # (1, 3, 28)
    v = (J @ nu[..., None]).squeeze(-1)[0]
    ref = torch.as_tensor(np.asarray(refpos, dtype=np.float64), dtype=R.dtype)
    f = scale * mass_on_body * (stiffness * (ref - x) - 2.0 * math.sqrt(stiffness) * v)
    n = float(f.norm())
    if n > scale * max_force:
        f = f * (scale * max_force / n)
    gen = (J.transpose(1, 2) @ f[:, None]).squeeze(-1)
    return f.numpy(), gen, x.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=None, help="deploy_<it>.bin or model_<it>.pt (default: the newest stand-ph-v1 file)")
    ap.add_argument("--delay-ms", type=float, default=20.0, help="transport delay (the robot: about 20 ms)")
    ap.add_argument("--seconds", type=float, default=0.0, help="0: until a window is closed")
    ap.add_argument("--window", type=float, default=10.0, help="seconds shown in the plot")
    ap.add_argument("--strength", type=float, default=1.0, help="start value of the push-strength slider")
    ap.add_argument("--max-force", type=float, default=100.0, help="largest push at strength x1 [N]")
    ap.add_argument("--no-viewer", action="store_true", help="headless: scripted pushes on the trunk, plot to a PNG")
    ap.add_argument("--test-force", type=float, default=60.0, help="headless: N of the scripted pushes")
    args = ap.parse_args()

    policy_path = Path(args.policy) if args.policy else newest_policy()
    if policy_path is None or not policy_path.exists():
        raise SystemExit("no policy: give --policy, or fetch the run first: TASK=stand kaggle/fetch.sh stand-ph-v1")
    policy = load_policy(policy_path)

    import matplotlib

    if args.no_viewer:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    model = load_k1()  # float64, CPU
    C.check_joint_order(model.joint_names)
    con = C.Contract(torch.float64)
    d = args.delay_ms
    plant = K1PHPlantNative(model=model, batch=1, firmware=FirmwareConfig(gains=C.GAINS, slew=C.SLEW, delay_ms=(d, d)))
    mj = model.mj
    import mujoco

    subtree = {b: float(mj.body_subtreemass[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n)])
               for b, n in enumerate(model.body_names)}
    stiffness = float(mj.vis.map.stiffness)
    state = {"reset": True, "fell_at": None}

    # MuJoCo window: draws the pH plant's pose, catches the mouse
    viewer, vdata = None, None
    if not args.no_viewer:
        import mujoco.viewer

        def on_key(keycode):
            if keycode in (ord("R"), ord("r")):
                state["reset"] = True

        vdata = mujoco.MjData(mj)
        viewer = mujoco.viewer.launch_passive(mj, vdata, key_callback=on_key)

    # the plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(bottom=0.25)
    n_keep = int(args.window / 0.02)
    hist = {k: deque(maxlen=n_keep) for k in ("t",) + TANKS}
    lines = {k: ax.plot([], [], label=k, color=COLORS[k], lw=1.8)[0] for k in TANKS}
    ax.set_xlabel("time [s]")
    ax.set_ylabel("energy [J]")
    ax.set_title(f"pH plant: tanks of H  ({policy_path.name}, delay {d:g} ms)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    info_text = ax.text(0.99, 0.97, "", transform=ax.transAxes, ha="right", va="top", family="monospace", fontsize=9)
    slider = Slider(fig.add_axes([0.18, 0.08, 0.64, 0.04]), "push strength", 0.0, 5.0, valinit=args.strength)
    if not args.no_viewer:
        plt.show(block=False)

    def do_reset():
        plant.reset(pose=con.default_qpos)
        state.update(V0=float(plant.energy()["gravity"][0]), last_action=torch.zeros(1, C.NUM_ACTIONS, dtype=torch.float64),
                     gyro=torch.zeros(1, 3, dtype=torch.float64), qd=torch.zeros(1, 22, dtype=torch.float64),
                     nu=torch.zeros(1, NB + 22, dtype=torch.float64), H0=float(plant.energy()["total"][0]),
                     reset=False, fell_at=None)

    dt = plant.firmware.policy_dt
    t, it, max_force, pushes = 0.0, 0, 0.0, 0
    wall0 = time.time()
    try:
        while True:
            if state["reset"]:
                do_reset()
                state["t_reset"] = t
            if args.seconds and t >= args.seconds:
                break
            if viewer is not None and not viewer.is_running():
                break
            if not args.no_viewer and not plt.fignum_exists(fig.number):
                break
            q = plant.q
            quat, theta = q[:, 3:7], q[:, 7:]
            obs = con.observe(theta, state["qd"], state["gyro"], quat, state["last_action"])
            action = torch.as_tensor(policy(obs.numpy()))
            state["last_action"] = action
            target = con.decode(action)

            # the push: the mouse spring (viewer), or scripted pulls on the trunk (headless)
            kin = plant.kinematics()
            gen, f = None, np.zeros(3)
            if viewer is not None:
                with viewer.lock():
                    pert = viewer.perturb
                    sel, active = int(pert.select), int(pert.active)
                    refpos, localpos = np.array(pert.refpos), np.array(pert.localpos)
                body = None
                if sel > 0:
                    name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, sel)
                    body = model.body_names.index(name) if name in model.body_names else None
                if (sel, active) != state.get("mouse"):  # tell what the mouse does (select / pull / release)
                    state["mouse"] = (sel, active)
                    what = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, sel) if sel > 0 else "nothing"
                    print(f"[play] t={t:.1f} s mouse: selected {what}, "
                          f"{'PULLING' if active & 1 else ('rotating' if active & 2 else 'not pulling')}", flush=True)
                f, gen, _ = mouse_force(active, body, refpos, localpos, kin, subtree.get(body, 0.0) if body is not None else 0.0,
                                        stiffness, float(slider.val), state["nu"], args.max_force)
            elif t % 3.0 >= 1.5 and t % 3.0 < 1.65:  # every 3 s a 150 ms pull of --test-force N on the trunk
                ang = 1.3 * int(t // 3.0)
                f_w = slider.val * args.test_force * torch.tensor([math.cos(ang), math.sin(ang), 0.0], dtype=torch.float64)
                J = point_jacobian(kin, torch.tensor([0]), kin.pos[:, 0][:, None])[:, 0]
                gen, f = (J.transpose(1, 2) @ f_w[:, None]).squeeze(-1), f_w.numpy()
            if gen is not None:
                pushes += 1
            max_force = max(max_force, float(np.linalg.norm(f)))

            info = plant.policy_step(target, gen_force=gen)
            state["nu"] = info.extra["nu"].clone()
            state["qd"], state["gyro"] = state["nu"][:, NB:].clone(), info.extra["gyro"].clone()
            t += dt
            it += 1

            if not (torch.isfinite(plant.q).all() and torch.isfinite(plant.p).all()):
                print(f"[play] t={t:.1f} s the state blew up (push too hard): reset", flush=True)
                state["reset"] = True
                continue

            rp = con.roll_pitch(plant.q[:, 3:7])[0].abs()
            fallen = float(rp.max()) > C.TILT_ABORT
            if fallen and state["fell_at"] is None:
                state["fell_at"] = t
                print(f"[play] fell at {t:.1f} s (|roll| or |pitch| > {C.TILT_ABORT} rad: the firmware would stop here)")
            if state["fell_at"] is not None and t - state["fell_at"] > 2.0:
                state["reset"] = True

            e = plant.energy()
            hist["t"].append(t)
            hist["kinetic"].append(float(e["kinetic"][0]))
            hist["gravity"].append(float(e["gravity"][0]) - state["V0"])
            hist["coil"].append(float(e["coil"][0]))

            if viewer is not None:
                vdata.qpos[:] = plant.q[0].numpy()
                mujoco.mj_kinematics(mj, vdata)
                viewer.sync()
            if it % 5 == 0 or args.no_viewer and args.seconds and t >= args.seconds - dt / 2:
                ts = list(hist["t"])
                for k in TANKS:
                    lines[k].set_data(ts, list(hist[k]))
                ax.set_xlim(max(0.0, t - args.window), max(args.window, t))
                vals = [v for k in TANKS for v in hist[k] if math.isfinite(v)] or [0.0]
                lo, hi = min(vals), max(vals)
                ax.set_ylim(lo - 0.05 * (hi - lo + 1e-3) - 0.05, hi + 0.15 * (hi - lo + 1e-3) + 0.05)
                led = plant.ledger
                info_text.set_text(f"push {np.linalg.norm(f):6.1f} N   push work {float(led.mech_input[0]):7.2f} J\n"
                                   f"battery {float(led.supplied[0]):7.1f} J   losses {float(led.dissipated[0]):7.1f} J\n"
                                   f"{'FALLEN' if state['fell_at'] is not None else 'standing'}")
                if not args.no_viewer:
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
            if not args.no_viewer:  # real time
                sleep = wall0 + it * dt - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                elif sleep < -0.5:  # far behind (a window was dragged): don't race to catch up
                    wall0 = time.time() - it * dt
    except KeyboardInterrupt:
        pass

    led = plant.ledger
    H = float(plant.energy()["total"][0])
    residual = (H - state["H0"]) - float(led.supplied[0] + led.mech_input[0] - led.dissipated[0])
    print(f"[play] {t - state['t_reset']:.1f} s since the last reset, largest push {max_force:.0f} N, push work {float(led.mech_input[0]):.2f} J, "
          f"battery {float(led.supplied[0]):.1f} J, losses {float(led.dissipated[0]):.1f} J, "
          f"ledger residual {residual:+.3f} J, {'fallen' if state['fell_at'] is not None else 'standing'}")
    if args.no_viewer:
        out = REPO / "outputs" / "stand_play_tanks.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        print(f"[play] plot {out}")
    if viewer is not None:
        viewer.close()
    return 0 if state["fell_at"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
