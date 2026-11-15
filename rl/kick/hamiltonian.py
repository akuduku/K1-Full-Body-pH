r"""The Hamiltonian of a kick as a figure: where the energy is and where it goes.

Sprangers, Lopes, Babuska ("Reinforcement learning for port-Hamiltonian systems", Fig. 3b) draw the desired
closed-loop Hamiltonian H_d over the pendulum's phase plane (q, p), with the trajectory as black dots.  The K1 has 28
velocities, 22 coil fluxes, contact bristles and a ball: there is no single phase plane.  What this figure shows
instead, from one rollout of the policy on the plant:

  (a) the tanks of H over the request: kinetic, gravity, contact + end stops, coils, ball (each as its change since
      the request started), their sum H, and the closed-loop Hamiltonian
          H_d = H + sum_j 1/2 kp_j (theta_j - target_j)^2
      where the second term is the potential the motor law adds: the driver's PD is a virtual spring to the target
      (energy shaping) and a virtual damper (damping injection).  The policy moves the targets, so it reshapes H_d
      every 20 ms;
  (b) the ledger: the electrical energy supplied since the request started, what the coils, gearboxes and contacts
      turned into heat, and what is left in H: supplied = heat + change of H (the residual is the integrator's error);
  (c, d) Fig. 3b for one joint each: H_d over (theta_j, p_j) of the kicking leg's knee and hip pitch, all other
      coordinates as they are at the touch, with the rollout as black dots (white: the start, red: the touch).  A
      slice through a high-dimensional H_d: the dots before and after the touch live on neighbouring slices.

    rec = Recorder(env)            # a 1-robot KickEnv
    ... every policy step: rec.capture()
    figure(rec, "kick_hamiltonian.png")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ph_k1.model import load_k1
from ph_k1.plant import K1PHPlant

TANKS = ("kinetic", "gravity", "contact", "stops", "coil", "ball", "total")
LEDGER = ("supplied", "copper", "gearbox", "stops", "contact")
KNEE = {0: "Left_Knee_Pitch", 1: "Right_Knee_Pitch"}
HIP = {0: "Left_Hip_Pitch", 1: "Right_Hip_Pitch"}


class Recorder:
    """One robot's energy per policy step (robot `b` of a KickEnv)."""

    def __init__(self, env, b: int = 0):
        self.env, self.b = env, b
        self.rows: list[dict] = []
        self.states: list[dict] = []

    def capture(self) -> None:
        env, b = self.env, self.b
        pl = env.plant
        rows = torch.tensor([b], device=env.device)
        tanks = pl.energy(rows=rows)
        led = pl.ledger
        kp = env.kp[b]
        shaped = 0.5 * (kp * (pl.q[b, 7:] - env.held_target[b]) ** 2).sum()
        row = {f"H_{k}": float(tanks[k][0]) for k in TANKS}
        row.update({f"L_{k}": float(getattr(led, k)[b]) for k in LEDGER})
        row.update(shaped=float(shaped), active=bool(env.active[b]), kicked=bool(env.kicked[b]),
                   touched=bool(env.touched[b]), t_request=float(env.time_in_request()[b]),
                   step=int(env.episode_length_buf[b]), foot=int(env.foot_request[b]))
        self.rows.append(row)
        cpu = lambda x: x[b].detach().to("cpu", torch.float64).clone()  # noqa: E731
        self.states.append({"q": cpu(pl.q), "p": cpu(pl.p), "xi": cpu(pl.xi), "lam": cpu(pl.lam), "ball": cpu(pl.ball),
                            "target": cpu(env.held_target), "kp": cpu(env.kp)})

    # ------------------------------------------------------------ the request's window
    def window(self):
        """Indices of the first request in the rollout (episode steps increasing, request active)."""
        idx = [i for i, r in enumerate(self.rows) if r["active"]]
        if not idx:
            return []
        out = [idx[0]]
        for i in idx[1:]:
            if i != out[-1] + 1 or self.rows[i]["step"] < self.rows[out[-1]]["step"]:
                break
            out.append(i)
        return out

    def touch(self, win):
        hits = [i for i in win if self.rows[i]["touched"]]
        if hits:
            return hits[0]
        return max(win, key=lambda i: self.rows[i]["H_kinetic"]) if win else None


def slice_hd(state: dict, joint: int, q_grid: np.ndarray, p_grid: np.ndarray, ball_params, chunk: int = 512) -> np.ndarray:
    """H_d over (theta_joint, p_joint) with every other coordinate of `state` kept: (len(p_grid), len(q_grid)) [J]."""
    model = load_k1()  # float64, CPU: the reference plant computes H
    Q, Pm = np.meshgrid(q_grid, p_grid)
    n = Q.size
    out = np.empty(n)
    for a in range(0, n, chunk):
        m = min(chunk, n - a)
        pl = K1PHPlant(model=model, batch=m, ball=ball_params)
        pl.q = state["q"].repeat(m, 1)
        pl.p = state["p"].repeat(m, 1)
        pl.xi = state["xi"].repeat(m, 1, 1)
        pl.lam = state["lam"].repeat(m, 1)
        pl.ball = state["ball"].repeat(m, 1)
        pl.q[:, 7 + joint] = torch.as_tensor(Q.reshape(-1)[a:a + m])
        pl.p[:, 6 + joint] = torch.as_tensor(Pm.reshape(-1)[a:a + m])
        shaped = 0.5 * (state["kp"] * (pl.q[:, 7:] - state["target"]) ** 2).sum(-1)
        out[a:a + m] = (pl.energy()["total"] + shaped).numpy()
    return out.reshape(Q.shape)


def figure(rec: Recorder, path: str | Path, dt: float = 0.02, grid: int = 41, title: str | None = None) -> Path | None:
    """Draw the four panels for the first request of the rollout; None if the rollout has no request."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    win = rec.window()
    if len(win) < 5:
        return None
    rows = [rec.rows[i] for i in win]
    t = np.array([r["t_request"] for r in rows])
    k_touch = rec.touch(win)
    t_touch = rec.rows[k_touch]["t_request"] if k_touch is not None and rec.rows[k_touch]["touched"] else None
    A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k.startswith(("H_", "L_")) or k == "shaped"}
    d = lambda x: x - x[0]  # noqa: E731

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    # (a) the tanks
    a = ax[0, 0]
    colors = {"kinetic": "#d95f02", "gravity": "#7570b3", "contact": "#666666", "coil": "#1b9e77", "ball": "#e6ab02"}
    a.plot(t, d(A["H_kinetic"]), color=colors["kinetic"], label="kinetic")
    a.plot(t, d(A["H_gravity"]), color=colors["gravity"], label="gravity")
    a.plot(t, d(A["H_contact"] + A["H_stops"]), color=colors["contact"], label="contact + end stops")
    a.plot(t, d(A["H_coil"]), color=colors["coil"], label="coils")
    a.plot(t, d(A["H_ball"]), color=colors["ball"], label="ball")
    a.plot(t, d(A["H_total"]), color="black", lw=2.2, label="H (sum)")
    a.plot(t, d(A["H_total"] + A["shaped"]), color="black", lw=1.6, ls="--", label=r"$H_d$ = H + $\sum \frac{1}{2}k_p(\theta-\theta^*)^2$")
    a.set(xlabel="time since the request [s]", ylabel="change since the request [J]",
          title="(a) The Hamiltonian's tanks during the kick")
    a.legend(fontsize=8, ncol=2)
    # (b) the ledger
    b = ax[0, 1]
    heat = d(A["L_copper"]) + d(A["L_gearbox"]) + d(A["L_stops"]) + d(A["L_contact"])
    b.plot(t, d(A["L_supplied"]), color="#1b9e77", lw=2.2, label="electrical energy supplied")
    b.plot(t, d(A["L_copper"]), color="#d95f02", label="heat in the coils")
    b.plot(t, d(A["L_gearbox"]), color="#7570b3", label="heat in the gearboxes")
    b.plot(t, d(A["L_contact"]) + d(A["L_stops"]), color="#666666", label="lost in contacts")
    b.plot(t, d(A["H_total"]), color="black", lw=1.6, label="stored: change of H")
    b.plot(t, d(A["L_supplied"]) - heat - d(A["H_total"]), color="red", lw=0.9, ls=":", label="residual (integrator)")
    b.set(xlabel="time since the request [s]", ylabel="energy since the request [J]",
          title="(b) Ledger: supplied = heat + change of H")
    b.legend(fontsize=8)
    for axis in (a, b):
        axis.grid(alpha=0.3)
        if t_touch is not None:
            axis.axvline(t_touch, color="red", lw=0.8)
            axis.text(t_touch, axis.get_ylim()[1], " touch", color="red", va="top", fontsize=8)

    # (c, d) Fig. 3b for the kicking knee and hip
    foot = rows[0]["foot"]
    state = rec.states[k_touch if k_touch is not None else win[len(win) // 2]]
    names = load_k1().joint_names
    for axis, joint_name, tag in ((ax[1, 0], KNEE[foot], "(c)"), (ax[1, 1], HIP[foot], "(d)")):
        j = names.index(joint_name)
        qs = np.array([float(rec.states[i]["q"][7 + j]) for i in win])
        ps = np.array([float(rec.states[i]["p"][6 + j]) for i in win])
        mq, mp = 0.25 * max(np.ptp(qs), 0.4), 0.25 * max(np.ptp(ps), 1.0)
        q_grid = np.linspace(qs.min() - mq, qs.max() + mq, grid)
        p_grid = np.linspace(ps.min() - mp, ps.max() + mp, grid)
        Hd = slice_hd(state, j, q_grid, p_grid, rec.env.ball_params)
        Hd = Hd - Hd.min()
        top = float(np.percentile(Hd, 85))  # walls (a foot in the floor or in the ball) saturate
        levels = np.linspace(0.0, max(top, 1e-6), 15)
        cs = axis.contourf(q_grid, p_grid, np.minimum(Hd, levels[-1]), levels=levels, cmap="Greys_r", extend="max")
        axis.contour(q_grid, p_grid, np.minimum(Hd, levels[-1]), levels=levels[::2], colors="k", linewidths=0.4)
        fig.colorbar(cs, ax=axis, label=r"$H_d - \min H_d$ [J] (cut at the 85th percentile)")
        axis.plot(qs, ps, ".", color="black", ms=4)
        axis.plot(qs[0], ps[0], "o", mfc="white", mec="black", ms=7, label="request starts")
        if k_touch is not None:
            kt = win.index(k_touch)
            axis.plot(qs[kt], ps[kt], "o", color="red", ms=7, label="touch (the slice's state)")
        axis.plot(float(state["target"][j]), 0.0, "x", color="#1b9e77", ms=9, mew=2, label="motor target at the touch")
        axis.set(xlabel=f"{joint_name} angle [rad]", ylabel="its generalised momentum p [kg m$^2$/s]",
                 title=f"{tag} $H_d$ over (q, p) of {joint_name.replace('_', ' ').lower()}, others as at the touch")
        axis.legend(fontsize=8, loc="best")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
