"""Picture and logbook.  MuJoCo only DRAWS here: mj_kinematics, never mj_step.

Our q has exactly MuJoCo's qpos layout (r, quaternion w x y z, 22 hinges),
so drawing is a copy plus forward kinematics for the meshes.
"""

from __future__ import annotations

import csv
from pathlib import Path

import mujoco
import numpy as np

from .plant import K1PHPlant, StepInfo

TANKS = ("kinetic", "gravity", "contact", "stops", "coil")


class Viewer:
    def __init__(self, plant: K1PHPlant, b: int = 0):
        import mujoco.viewer  # noqa: F401  (needs a display)

        self.plant, self.b = plant, b
        self.data = mujoco.MjData(plant.model.mj)
        self.handle = mujoco.viewer.launch_passive(plant.model.mj, self.data)

    def is_running(self) -> bool:
        return self.handle.is_running()

    def sync(self) -> None:
        self.data.qpos[:] = self.plant.q[self.b].detach().cpu().numpy()
        mujoco.mj_kinematics(self.plant.model.mj, self.data)
        self.handle.sync()

    def close(self) -> None:
        self.handle.close()


class Recorder:
    """Energy tanks + ledger + a few kinematic readouts, robot b."""

    def __init__(self, plant: K1PHPlant, b: int = 0):
        self.plant, self.b = plant, b
        self.rows: list[dict] = []
        e = plant.energy()
        self.H0 = float(e["total"][b])
        self.V0 = float(e["gravity"][b])

    def log(self, info: StepInfo | None = None) -> None:
        pl, b = self.plant, self.b
        e = pl.energy()
        led = pl.ledger
        R = pl.kinematics().R[b, 0]
        row = {"t": pl.time}
        for k in TANKS:
            row[k] = float(e[k][b])
        row["gravity"] -= self.V0  # relative to the start
        row["H"] = float(e["total"][b])
        row["supplied"] = float(led.supplied[b])
        row["dissipated"] = float(led.dissipated[b])
        row["copper"] = float(led.copper[b])
        row["residual"] = row["H"] - self.H0 - (row["supplied"] - row["dissipated"] + float(led.mech_input[b]))
        row["trunk_z"] = float(pl.q[b, 2])
        row["tilt_deg"] = float(np.degrees(np.arccos(np.clip(float(R[2, 2]), -1, 1))))
        if info is not None:
            row["contact_points"] = int(info.contact_active[b].sum())
            row["Fz_total"] = float(info.contact_force[b, :, 2].sum())
            row["max_abs_torque"] = float(info.torque[b].abs().max())
        self.rows.append(row)

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)

    def plot(self, path: Path, title: str = "") -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.array([r["t"] for r in self.rows])
        fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for k in TANKS:
            ax[0].plot(t, [r[k] for r in self.rows], label=k)
        ax[0].plot(t, [r["H"] - self.H0 for r in self.rows], "k--", label="H - H(0)")
        ax[0].set_ylabel("energy tanks [J]")
        ax[0].legend(ncol=3, fontsize=8)
        ax[0].set_title(title or "port-Hamiltonian K1: energy tanks")
        ax[1].plot(t, [r["supplied"] for r in self.rows], label="supplied  int v.i dt")
        ax[1].plot(t, [r["dissipated"] for r in self.rows], label="dissipated")
        ax[1].plot(t, [r["residual"] for r in self.rows], label="residual  dH - (in - out)")
        ax[1].set_ylabel("ledger [J]")
        ax[1].legend(fontsize=8)
        ax[2].plot(t, [r["trunk_z"] for r in self.rows], label="trunk z [m]")
        ax2 = ax[2].twinx()
        ax2.plot(t, [r["tilt_deg"] for r in self.rows], "r", label="tilt [deg]")
        ax[2].set_xlabel("t [s]")
        ax[2].set_ylabel("trunk z [m]")
        ax2.set_ylabel("tilt [deg]", color="r")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
