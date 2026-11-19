r"""The Hamiltonian of a walking robot as a figure: where the energy is, where it goes, and the closed-loop
Hamiltonian over one joint's phase plane.  The walking twin of rl/kick/hamiltonian.py (whose slice_hd it uses).

Sprangers, Lopes, Babuska ("Reinforcement learning for port-Hamiltonian systems", Fig. 3b) draw the desired
closed-loop Hamiltonian H_d over the pendulum's phase plane (q, p) with the trajectory as dots.  The K1 has 28
velocities, 22 coil fluxes and contact bristles: there is no single phase plane.  From one quiet episode of one robot
(rl/walk/video.py records it while it films), with requests that change during the episode:

  (a) the tanks of H over the episode, each as its change since the start: kinetic, gravity, contact + end stops,
      coils, their sum H, and the closed-loop Hamiltonian
          H_d = H + sum_j 1/2 kp_j (theta_j - target_j)^2
      whose second term is the potential the motor law adds: the driver's PD is a virtual spring to the target the
      policy sets (energy shaping) and a virtual damper (damping injection).  The policy moves the targets every
      20 ms, so it reshapes H_d every 20 ms;
  (b) the ledger: the electrical energy supplied since the start, the heat in the coils, the gearboxes and the
      contacts, the energy stored (change of H), and what the ledger does not book: supplied - heat - change of H.
      That is the damping of the plant's implicit Euler step (ph_k1/plant.py): 1/2 L (i1 - i0)^2 in the coils and
      1/2 dnu^T M dnu in the mechanics per 1 ms step (checked 19 Nov to 0.006 J of 24 J).  It grows when the joint
      targets jump (the current jumps within a step: a real motor turns that into heat in a transient faster than
      the step), and it never adds energy: the discrete plant stays passive.  The episode has no pushes, which would
      add energy outside the ledger;
  (c, d) H_d over (theta, p) of the left knee and the left hip pitch, every other coordinate as in the middle of the
      walking request, with that request's trajectory as dots: a gait is a closed orbit, standing still a point.

    rec = Recorder(env)                         # a 1-robot WalkEnv
    ... after every policy step: rec.capture(t)
    figure(rec, "walk_hamiltonian.png", segments=((0.0, "stand"), (2.0, "forward 0.5 m/s"), ...))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rl.kick.hamiltonian import LEDGER, TANKS, slice_hd

JOINTS = ("Left_Knee_Pitch", "Left_Hip_Pitch")


class Recorder:
    """One robot's energy after every policy step (robot `b` of a WalkEnv).  The ledger of the C++ plant counts up
    over the episode; the figure shows changes since the start."""

    def __init__(self, env, b: int = 0):
        self.env, self.b = env, b
        self.rows: list[dict] = []
        self.states: list[dict] = []

    def capture(self, t: float) -> None:
        env, b = self.env, self.b
        pl = env.plant
        tanks = pl.energy(rows=torch.tensor([b], device=env.device))
        target = env.contract.decode(env.actions)[b]             # the joint targets the policy set this step
        shaped = 0.5 * (env.kp[b] * (pl.q[b, 7:] - target) ** 2).sum()
        row = {f"H_{k}": float(tanks[k][0]) for k in TANKS}
        row.update({f"L_{k}": float(getattr(pl.ledger, k)[b]) for k in LEDGER})
        row.update(shaped=float(shaped), t=float(t), step=int(env.episode_length_buf[b]),
                   request=tuple(float(v) for v in env.mem["commands"][b]))
        self.rows.append(row)
        cpu = lambda x: x[b].detach().to("cpu", torch.float64).clone()  # noqa: E731
        self.states.append({"q": cpu(pl.q), "p": cpu(pl.p), "xi": cpu(pl.xi), "lam": cpu(pl.lam), "ball": cpu(pl.ball),
                            "target": target.detach().to("cpu", torch.float64).clone(), "kp": cpu(env.kp)})

    def until_restart(self) -> list[int]:
        """The indices up to the robot's first restart (a fall ends an episode; the environment starts it again)."""
        out = []
        for i, r in enumerate(self.rows):
            if out and r["step"] <= self.rows[out[-1]]["step"]:
                break
            out.append(i)
        return out


def figure(rec: Recorder, path: str | Path, segments=(), grid: int = 41, title: str | None = None) -> Path | None:
    """Draw the four panels; segments: (start time [s], label) of the requests, drawn as lines.  None if the episode
    is too short (the robot fell at once)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = rec.until_restart()
    if len(idx) < 5:
        return None
    rows = [rec.rows[i] for i in idx]
    t = np.array([r["t"] for r in rows])
    A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k.startswith(("H_", "L_")) or k == "shaped"}
    d = lambda x: x - x[0]  # noqa: E731
    fell = len(idx) < len(rec.rows)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    a = ax[0, 0]                                                    # (a) the tanks
    a.plot(t, d(A["H_kinetic"]), color="#d95f02", label="kinetic")
    a.plot(t, d(A["H_gravity"]), color="#7570b3", label="gravity")
    a.plot(t, d(A["H_contact"] + A["H_stops"]), color="#666666", label="contact + end stops")
    a.plot(t, d(A["H_coil"]), color="#1b9e77", label="coils")
    a.plot(t, d(A["H_total"]), color="black", lw=2.2, label="H (sum)")
    a.plot(t, d(A["H_total"] + A["shaped"]), color="black", lw=1.6, ls="--",
           label=r"$H_d$ = H + $\sum \frac{1}{2}k_p(\theta-\theta^*)^2$")
    a.set(xlabel="time [s]", ylabel="change since the start [J]", title="(a) The Hamiltonian's tanks")
    a.legend(fontsize=8, ncol=2)
    b = ax[0, 1]                                                    # (b) the ledger
    heat = d(A["L_copper"]) + d(A["L_gearbox"]) + d(A["L_stops"]) + d(A["L_contact"])
    b.plot(t, d(A["L_supplied"]), color="#1b9e77", lw=2.2, label="electrical energy supplied")
    b.plot(t, d(A["L_copper"]), color="#d95f02", label="heat in the coils")
    b.plot(t, d(A["L_gearbox"]), color="#7570b3", label="heat in the gearboxes")
    b.plot(t, d(A["L_contact"]) + d(A["L_stops"]), color="#666666", label="lost in contacts and end stops")
    b.plot(t, d(A["H_total"]), color="black", lw=1.6, label="stored: change of H")
    b.plot(t, d(A["L_supplied"]) - heat - d(A["H_total"]), color="red", lw=1.2, ls=":",
           label="not booked: the integrator's damping of fast current changes")
    b.set(xlabel="time [s]", ylabel="energy since the start [J]",
          title="(b) Ledger: supplied = heat + change of H + not booked")
    b.legend(fontsize=8)
    for axis in (a, b):
        axis.grid(alpha=0.3)
        for start, label in segments:
            if start <= t[-1]:
                axis.axvline(start, color="#1f78b4", lw=0.7, alpha=0.6)
                axis.text(start, axis.get_ylim()[1], f" {label}", color="#1f78b4", va="top", fontsize=7, rotation=90)
        if fell:
            axis.axvline(t[-1], color="red", lw=0.9)
            axis.text(t[-1], axis.get_ylim()[0], " fell", color="red", va="bottom", fontsize=8)

    # (c, d) H_d over (theta, p) during the first walking request (else the whole episode)
    walking = [i for i in idx if np.hypot(*rec.rows[i]["request"][:2]) > 0.05]
    part = walking if len(walking) >= 10 else idx
    mid = part[len(part) // 2]
    state = rec.states[mid]
    names = rec.env.model.joint_names
    for axis, joint_name, tag in ((ax[1, 0], JOINTS[0], "(c)"), (ax[1, 1], JOINTS[1], "(d)")):
        j = names.index(joint_name)
        qs = np.array([float(rec.states[i]["q"][7 + j]) for i in part])
        ps = np.array([float(rec.states[i]["p"][6 + j]) for i in part])
        mq, mp = 0.25 * max(np.ptp(qs), 0.2), 0.25 * max(np.ptp(ps), 0.2)
        q_grid = np.linspace(qs.min() - mq, qs.max() + mq, grid)
        p_grid = np.linspace(ps.min() - mp, ps.max() + mp, grid)
        Hd = slice_hd(state, j, q_grid, p_grid, rec.env.plant.ball_params)
        Hd = Hd - Hd.min()
        top = float(np.percentile(Hd, 85))                          # walls (a foot in the floor) saturate
        levels = np.linspace(0.0, max(top, 1e-6), 15)
        cs = axis.contourf(q_grid, p_grid, np.minimum(Hd, levels[-1]), levels=levels, cmap="Greys_r", extend="max")
        axis.contour(q_grid, p_grid, np.minimum(Hd, levels[-1]), levels=levels[::2], colors="k", linewidths=0.4)
        fig.colorbar(cs, ax=axis, label=r"$H_d - \min H_d$ [J] (cut at the 85th percentile)")
        axis.scatter(qs, ps, c=np.arange(len(qs)), cmap="viridis", s=9, zorder=3)
        axis.plot(qs[0], ps[0], "o", mfc="white", mec="black", ms=7, label="first step of the part")
        axis.plot(qs[part.index(mid)], ps[part.index(mid)], "o", color="red", ms=7, label="the slice's state")
        axis.plot(float(state["target"][j]), 0.0, "x", color="#1b9e77", ms=9, mew=2, label="motor target there")
        what = "the walking request" if part is walking else "the episode"
        axis.set(xlabel=f"{joint_name} angle [rad]", ylabel="its generalised momentum p [kg m$^2$/s]",
                 title=f"{tag} $H_d$ over (q, p) of {joint_name.replace('_', ' ').lower()}; dots: {what} "
                       "(colour: time)")
        axis.legend(fontsize=8, loc="best")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
