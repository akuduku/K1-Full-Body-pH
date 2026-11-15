r"""How do the pH plant and MuJoCo differ on the same task?  Same policy, same command script, same start, both engines.

    python rl/walk/compare.py                                     # walk_v9, full run (a few minutes on the laptop)
    python rl/walk/compare.py --policy soccer-rolling-walk        # another 75-input walk
    python rl/walk/compare.py --quick                             # fewer seeds, a coarser grid

Writes outputs/compare/<policy>/: report.md (the numbers), results.json, tipover.png, tracking.png, effort.png,
divergence.png.

Scenarios:
    tip-over     no policy: the motors hold HOME and the robot tips over; open loop, so nothing but physics differs
    stand        the policy holds the robot at zero request
    task         speed steps, a turn, a sidestep, a stop, repeated with small random start offsets
    sweep        speed requests up to the firmware's limit: measured speed, power, the top speed before a fall
    pushes       sideways velocity pushes while walking, at four moments of the gait
    divergence   the two engines side by side over time
    sensitivity  the task at other transport delays
    reverse      a policy trained on the pH plant (stand-ph-v1) on both engines

Reading the result: this shows how the two physics differ, not which one is right.  Only the robot can say that
(Model Based RL/mbrl/replay_mujoco.py replays the real get-up logs on both).  The walks were trained in MuJoCo, so
they favour it; the reverse scenario is the counterpart.  Power is |torque x joint speed|: mechanical, and the same
formula on both.  Volts and amperes are left out, because the plant's motor constants are placeholders.

No gait measure uses the contact force.  It is sampled at the end of each 20 ms step, and MuJoCo's reads zero for about a
tenth of the time a foot stands on the floor, which a step counter takes for extra steps.  The rhythm comes from the
knees, the clearance from the feet's height, and a foot is in stance when its lowest corner is within STANCE_Z of the
floor.  Sole slip is the horizontal speed of a foot's lowest corners in stance: the foot body itself moves as the foot
rolls even when nothing slides.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ph_k1.model import load_k1  # noqa: E402
from rl.common.export import load_policy  # noqa: E402
from rl.stand import contract as SC  # noqa: E402
from rl.walk import contract as C  # noqa: E402
from rl.walk.backends import MuJoCoBackend, PHBackend, Walker, load_walk_policy, resolve_policy  # noqa: E402

ENGINES = (PHBackend, MuJoCoBackend)
NAMES = tuple(b.name for b in ENGINES)
BLUE, ORANGE = "#2a78d6", "#eb6834"  # slots 1 and 2 of the validated categorical palette, always pH then MuJoCo
COLOURS = dict(zip(NAMES, (BLUE, ORANGE)))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SETTLE_S = 2.0  # seconds of every task segment left out: the ramp and the change of gait
START_NOISE = 0.01  # rad, uniform on every joint, seeds 1 and up (seed 0 starts exactly at HOME)
MATERIAL = 0.05  # relative difference below which the report does not mark a row
STANCE_Z = 0.003  # m, a foot whose lowest corner is this close to the floor stands on it
LOW_CORNERS = 0.003  # m, corners this close to the lowest one carry the foot
MIN_SWING_RAD = 0.02  # joint motion below this (std) is no gait
LEFT_KNEE, RIGHT_KNEE = 13, 19
JOINT_GROUPS = {"arms": range(2, 10), "hips": (10, 11, 12, 16, 17, 18), "knees": (13, 19), "ankles": (14, 15, 20, 21)}

# label, start, end, request (vx, vy, wz)
SEGMENTS = (("stand", 0.0, 2.0, (0.0, 0.0, 0.0)),
            ("walk 0.5 m/s", 2.0, 8.0, (0.5, 0.0, 0.0)),
            ("walk 1.0 m/s", 8.0, 14.0, (1.0, 0.0, 0.0)),
            ("walk 1.5 m/s", 14.0, 20.0, (1.5, 0.0, 0.0)),
            ("turn 1 rad/s", 20.0, 26.0, (0.5, 0.0, 1.0)),
            ("sidestep 0.3 m/s", 26.0, 32.0, (0.0, 0.3, 0.0)),
            ("stop", 32.0, 36.0, (0.0, 0.0, 0.0)))
TASK_S = SEGMENTS[-1][2]


def request_at(t: float) -> tuple[float, float, float]:
    for _, t0, t1, request in SEGMENTS:
        if t0 <= t < t1:
            return request
    return SEGMENTS[-1][3]


@dataclass
class Log:
    """One run at 50 Hz, up to the fall."""

    t: np.ndarray
    qpos: np.ndarray
    qd: np.ndarray
    torque: np.ndarray
    foot_force: np.ndarray
    foot_pos: np.ndarray
    foot_rot: np.ndarray
    fell_at: float | None

    @property
    def theta(self) -> np.ndarray:
        return self.qpos[:, 7:]

    @property
    def pos(self) -> np.ndarray:
        return self.qpos[:, :3]


def run(walker: Walker, request, seconds: float, pose=None, push_at: float | None = None, push=None) -> Log:
    """request: a function of time. push: one velocity change (3) at push_at, on the first step after it."""
    walker.reset(pose)
    rows = []
    fell_at, pushed = None, False
    for k in range(round(seconds / C.POLICY_DT)):
        t = k * C.POLICY_DT
        dv = None
        if push_at is not None and not pushed and t >= push_at:
            dv, pushed = np.asarray(push, dtype=np.float64), True
        r = walker.tick(request(t), dv)
        rows.append((t, r.qpos, r.qd, r.torque, r.foot_force, r.foot_pos, r.foot_rot))
        if walker.fell:
            fell_at = t
            break
    return Log(*(np.array([row[i] for row in rows]) for i in range(7)), fell_at)


def start_pose(backend, seed: int) -> np.ndarray:
    pose = backend.home.copy()
    if seed > 0:
        pose += np.random.default_rng(seed).uniform(-START_NOISE, START_NOISE, pose.shape)
    return pose


def heading_twist(log: Log) -> np.ndarray:
    """(N-1, 3) vx, vy in the heading frame and yaw rate, from the trunk's motion."""
    yaw = np.array([C.yaw_of(q) for q in log.qpos[:, 3:7]])
    d = np.diff(log.pos[:, :2], axis=0) / C.POLICY_DT
    c, s = np.cos(yaw[1:]), np.sin(yaw[1:])
    turn = np.remainder(np.diff(yaw) + np.pi, 2 * np.pi) - np.pi
    return np.stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], turn / C.POLICY_DT), 1)


def cycle_starts(x: np.ndarray) -> np.ndarray:
    """Fractional sample indices where a periodic signal rises through its mean, one per cycle: it has to go below
    -1/4 of its std first, so small ripples on the way up do not count."""
    x = x - x.mean()
    low, starts = False, []
    threshold = 0.25 * x.std()
    for k in range(1, len(x)):
        if x[k] < -threshold:
            low = True
        elif low and x[k - 1] <= 0.0 < x[k]:
            starts.append(k - 1 + x[k - 1] / (x[k - 1] - x[k]))
            low = False
    return np.array(starts)


def steps_per_second(left_knee: np.ndarray, right_knee: np.ndarray) -> float | None:
    """Steps of both legs per second, from the period of each knee; None when the legs do not swing."""
    total = 0.0
    for knee in (left_knee, right_knee):
        starts = cycle_starts(knee)
        if knee.std() < MIN_SWING_RAD or len(starts) < 3:
            return None
        total += 1.0 / (float(np.diff(starts).mean()) * C.POLICY_DT)
    return total


def foot_clearance(height: np.ndarray) -> float | None:
    """Mean lift of one foot per cycle of its height [m]."""
    starts = cycle_starts(height).astype(int)
    peaks = [height[a:b + 1].max() - height[a:b + 1].min() for a, b in zip(starts, starts[1:])]
    return float(np.mean(peaks)) if peaks else None


def sole_corners(model) -> np.ndarray:
    """(2,4,3) the corners of both soles in their foot frames, the points the plant's contact acts on."""
    m = model
    feet = [m.body_names.index(n) for n in ("left_foot_link", "right_foot_link")]
    return np.stack([m.contact_local[m.contact_is_sole & (m.contact_body == b)].numpy() for b in feet])


def sole_slip(foot_pos, foot_rot, corners) -> np.ndarray:
    """Horizontal speed [m/s] of the lowest corners of each standing foot, one value per foot and sample."""
    out = []
    for side in (0, 1):
        world = foot_pos[:, side, None] + np.einsum("tij,cj->tci", foot_rot[:, side], corners[side])
        speed = np.linalg.norm(np.diff(world[:, :, :2], axis=0), axis=-1) / C.POLICY_DT
        z = world[1:, :, 2]
        lowest = z.min(1, keepdims=True)
        low = z < lowest + LOW_CORNERS
        mean = (speed * low).sum(1) / low.sum(1)
        out.append(mean[lowest[:, 0] < STANCE_Z])
    return np.concatenate(out)


def walking_metrics(log: Log, t0: float, t1: float, mass: float, corners: np.ndarray) -> dict | None:
    """Everything measured over one stretch of a run; None when the run did not reach its end."""
    keep = (log.t >= t0) & (log.t < t1)
    if log.t[-1] < t1 - 2 * C.POLICY_DT or keep.sum() < 10:
        return None
    twist = heading_twist(log)[keep[1:]]
    v = twist.mean(0)
    speed = float(np.hypot(v[0], v[1]))
    feet = log.foot_pos[keep]
    cadence = steps_per_second(log.theta[keep, LEFT_KNEE], log.theta[keep, RIGHT_KNEE])
    lifts = [foot_clearance(feet[:, side, 2]) for side in (0, 1)] if cadence else []
    lifts = [x for x in lifts if x is not None]
    slip = sole_slip(feet, log.foot_rot[keep], corners)
    tilt = np.array([C.roll_pitch(q) for q in log.qpos[keep, 3:7]])
    power = np.abs(log.torque[keep] * log.qd[keep]).sum(1).mean()
    torque_rms = {name: float(np.sqrt((log.torque[keep][:, list(j)] ** 2).mean())) for name, j in JOINT_GROUPS.items()}
    moving = speed > 0.05 and cadence is not None
    return {"vx": float(v[0]), "vy": float(v[1]), "wz": float(v[2]), "speed": speed,
            "steps_per_s": cadence, "step_length_m": speed / cadence if moving else None,
            "foot_clearance_mm": 1e3 * float(np.mean(lifts)) if lifts else None,
            "sole_slip_mean_mm_s": 1e3 * float(slip.mean()) if len(slip) else None,
            "sole_slip_median_mm_s": 1e3 * float(np.median(slip)) if len(slip) else None,
            "trunk_height_m": float(log.pos[keep, 2].mean()),
            "pitch_deg": math.degrees(tilt[:, 1].mean()), "pitch_std_deg": math.degrees(tilt[:, 1].std()),
            "roll_std_deg": math.degrees(tilt[:, 0].std()),
            "power_W": float(power), "cost_of_transport": float(power / (mass * 9.81 * speed)) if speed > 0.05 else None,
            "torque_rms_Nm": torque_rms}


def mean_std(values) -> tuple[float, float] | None:
    x = [v for v in values if v is not None]
    return (float(np.mean(x)), float(np.std(x))) if x else None


# ---------------------------------------------------------------------------- scenarios

def tip_over(walkers: dict) -> dict:
    """No policy: the motors hold HOME, the robot tips over."""
    out = {}
    for name, walker in walkers.items():
        b = walker.backend
        b.reset()
        t, pitch = [], []
        for k in range(150):
            b.step(b.home)
            r = b.read()
            t.append((k + 1) * C.POLICY_DT)
            pitch.append(math.degrees(C.roll_pitch(r.quat)[1]))
        pitch = np.array(pitch)
        over = np.where(np.abs(pitch) > 30.0)[0]
        out[name] = {"t": t, "pitch_deg": pitch.tolist(), "seconds_to_30deg": float(t[over[0]]) if len(over) else None}
    return out


def stand(walkers: dict) -> dict:
    out = {}
    for name, walker in walkers.items():
        log = run(walker, lambda t: (0.0, 0.0, 0.0), 6.0)
        keep = log.t >= 3.0
        out[name] = {"fell": log.fell_at is not None, "trunk_height_m": float(log.pos[keep, 2].mean()),
                     "pitch_deg": math.degrees(np.mean([C.roll_pitch(q)[1] for q in log.qpos[keep, 3:7]])),
                     "joint_offset_deg": math.degrees(float(np.abs(log.theta[keep] - C.HOME).mean())),
                     "foot_load_N": float(log.foot_force[keep].sum(1).mean()),
                     "power_W": float(np.abs(log.torque[keep] * log.qd[keep]).sum(1).mean())}
    return out


def task(walkers: dict, seeds: int, mass: float, corners: np.ndarray) -> tuple[dict, dict]:
    """Per engine and segment: every metric as (mean, std) over the seeds. Also the logs of seed 0."""
    results, first = {}, {}
    for name, walker in walkers.items():
        per_seed, falls = [], 0
        for seed in range(seeds):
            log = run(walker, request_at, TASK_S, pose=start_pose(walker.backend, seed))
            if seed == 0:
                first[name] = log
            falls += log.fell_at is not None
            per_seed.append({label: walking_metrics(log, t0 + (SETTLE_S if label != "stand" else 0.5), t1, mass, corners)
                             for label, t0, t1, _ in SEGMENTS})
        table = {}
        for label, *_ in SEGMENTS:
            rows = [m[label] for m in per_seed if m[label] is not None]
            table[label] = {key: mean_std(r[key] for r in rows) for key in (rows[0] if rows else {})
                            if key != "torque_rms_Nm"}
            if rows:
                table[label]["torque_rms_Nm"] = {g: mean_std(r["torque_rms_Nm"][g] for r in rows) for g in JOINT_GROUPS}
        results[name] = {"seeds": seeds, "falls": falls, "segments": table}
    return results, first


def sweep(walkers: dict, requests, seeds: int, mass: float, corners: np.ndarray) -> dict:
    """A constant forward request for 10 s each."""
    out = {}
    for name, walker in walkers.items():
        rows = []
        for v in requests:
            metrics, falls = [], 0
            for seed in range(seeds):
                log = run(walker, lambda t: (v, 0.0, 0.0) if t >= 1.0 else (0.0, 0.0, 0.0), 10.0,
                          pose=start_pose(walker.backend, seed))
                falls += log.fell_at is not None
                metrics.append(walking_metrics(log, 6.0, 10.0, mass, corners))
            ok = [m for m in metrics if m is not None]
            rows.append({"request": float(v), "falls": falls, "runs": seeds,
                         "speed": mean_std(m["speed"] for m in ok), "power_W": mean_std(m["power_W"] for m in ok),
                         "sole_slip_mean_mm_s": mean_std(m["sole_slip_mean_mm_s"] for m in ok),
                         "cost_of_transport": mean_std(m["cost_of_transport"] for m in ok)})
        top = None
        for row in rows:
            if row["falls"]:
                break
            top = row["request"]
        out[name] = {"rows": rows, "top_request_without_fall": top}
    return out


def pushes(walkers: dict, sizes, request=(0.5, 0.0, 0.0)) -> dict:
    """Walking at `request`: one sideways velocity change, either way, at four moments of the gait."""
    moments = (5.0, 5.1, 5.2, 5.3)
    out = {}
    for name, walker in walkers.items():
        rows = []
        for size in sizes:
            survived, recovery = 0, []
            for moment in moments:
                for side in (1.0, -1.0):
                    log = run(walker, lambda t: request if t >= 1.0 else (0.0, 0.0, 0.0), 8.0, push_at=moment,
                              push=(0.0, side * size, 0.0))
                    if log.fell_at is None:
                        survived += 1
                        roll = np.degrees([C.roll_pitch(q)[0] for q in log.qpos[log.t >= moment, 3:7]])
                        late = np.where(np.abs(roll) > 5.0)[0]
                        recovery.append(float((late[-1] + 1) * C.POLICY_DT) if len(late) else 0.0)
            rows.append({"push_m_s": float(size), "survived": survived, "runs": 2 * len(moments),
                         "recovery_s": mean_std(recovery)})
            if survived == 0:
                break
        safe = None
        for row in rows:
            if row["survived"] < row["runs"]:
                break
            safe = row["push_m_s"]
        out[name] = {"rows": rows, "largest_push_always_survived": safe}
    return out


def aligned_rmse(a: np.ndarray, b: np.ndarray, max_lag: int = 30) -> float:
    """Joint RMSE [deg] between two stretches after shifting one in time: two gaits of the same shape, out of step by a
    little, are far apart at the same moment but close once aligned.  The shift itself means little: a gait repeats, so
    it is only known up to whole strides."""
    best = (math.inf, 0)
    for lag in range(-max_lag, max_lag + 1):
        x, y = a[max_lag:-max_lag], b[max_lag + lag:len(b) - max_lag + lag]
        best = min(best, (float(np.sqrt(((x - y) ** 2).mean())), lag))
    return math.degrees(best[0])


def divergence(first: dict) -> dict:
    """The two engines at the same time, from the same start, under the same commands."""
    a, b = (first[n] for n in NAMES)
    n = min(len(a.t), len(b.t))
    t = a.t[:n]
    rmse = np.degrees(np.sqrt(((a.theta[:n] - b.theta[:n]) ** 2).mean(1)))
    gap = np.linalg.norm(a.pos[:n, :2] - b.pos[:n, :2], axis=1)
    segments = {}
    for label, t0, t1, _ in SEGMENTS:
        keep = (t >= t0 + SETTLE_S) & (t < t1)
        if keep.sum() > 80:
            shape = aligned_rmse(a.theta[:n][keep], b.theta[:n][keep])
            segments[label] = {"same_moment_deg": float(rmse[keep].mean()), "aligned_deg": shape}
    return {"t": t.tolist(), "joint_rmse_deg": rmse.tolist(), "trunk_gap_m": gap.tolist(), "segments": segments}


def sensitivity(model, policy, delays, seeds: int, mass: float, corners: np.ndarray) -> dict:
    out = {}
    for delay in delays:
        walkers = {b.name: Walker(b(model, delay_ms=delay), policy) for b in ENGINES}
        results, _ = task(walkers, seeds, mass, corners)
        out[f"{delay:g}"] = {name: {"falls": r["falls"],
                                    "speed": r["segments"]["walk 1.0 m/s"].get("speed"),
                                    "power_W": r["segments"]["walk 1.0 m/s"].get("power_W"),
                                    "steps_per_s": r["segments"]["walk 1.0 m/s"].get("steps_per_s"),
                                    "sole_slip_mean_mm_s": r["segments"]["walk 1.0 m/s"].get("sole_slip_mean_mm_s")}
                             for name, r in results.items()}
    return out


class StandWalker(Walker):
    """Walker with the stand agent's contract (rl/stand/contract.py): no twist."""

    def __init__(self, backend, policy):
        self.con = SC.Contract(torch.float64)
        super().__init__(backend, policy)

    def tick(self, want=None, push=None):
        r = self.reading
        if not self.fell:
            T = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))[None]  # noqa: E731
            obs = self.con.observe(T(r.theta), T(r.qd), T(r.gyro), T(r.quat), T(self.last_action))
            self.last_action = np.asarray(self.policy(obs.numpy()), dtype=np.float64)[0]
            self.target = self.con.decode(T(self.last_action))[0].numpy()
        self.backend.step(self.target, push)
        if self.backend.healthy():
            self.reading = self.backend.read()
            self.fell = self.fell or C.fallen(float(self.reading.pos[2]), self.reading.quat)
        else:
            self.fell = True
        return self.reading


def reverse(model, path: Path, sizes) -> dict:
    """stand-ph-v1, trained on the pH plant, on both engines."""
    policy = load_policy(path)
    walkers = {b.name: StandWalker(b(model, gain_preset=SC.GAINS, slew=SC.SLEW, home=np.array(SC.DEFAULT_QPOS)), policy)
               for b in ENGINES}
    out = {"policy": path.name}
    for name, walker in walkers.items():
        log = run(walker, lambda t: None, 15.0)
        keep = log.t >= 5.0
        row = {"fell_at": log.fell_at}
        if keep.any():
            tilt = np.degrees([C.roll_pitch(q) for q in log.qpos[keep, 3:7]])
            row.update(pitch_deg=float(tilt[:, 1].mean()), tilt_std_deg=float(tilt.std(0).max()),
                       power_W=float(np.abs(log.torque[keep] * log.qd[keep]).sum(1).mean()))
        safe = None
        for size in sizes:
            logs = [run(walker, lambda t: None, 6.0, push_at=3.0, push=(0.0, side * size, 0.0)) for side in (1.0, -1.0)]
            if any(x.fell_at is not None for x in logs):
                break
            safe = float(size)
        row["largest_push_always_survived"] = safe
        out[name] = row
    return out


# ---------------------------------------------------------------------------- plots

def style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def end_labels(ax, curves) -> None:
    """Name each line at its right end, moved apart if the two ends would collide."""
    ends = sorted((y[-1], name) for name, x, y in curves)
    low, high = ax.get_ylim()
    gap = 0.06 * (high - low)
    placed = [ends[0][0], max(ends[1][0], ends[0][0] + gap)] if len(ends) == 2 else [e[0] for e in ends]
    x_end = max(x[-1] for _, x, _ in curves)
    for (_, name), y in zip(ends, placed):
        ax.annotate(name, (x_end, y), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=10)


def plots(out: Path, r: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    curves = [(n, r["tip_over"][n]["t"], r["tip_over"][n]["pitch_deg"]) for n in NAMES]
    for name, x, y in curves:
        ax.plot(x, y, color=COLOURS[name], lw=2, label=name)
    style(ax, "time [s]", "trunk pitch [deg]")
    ax.set_title("No policy: the motors hold the pose and the robot tips over", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    end_labels(ax, curves)
    fig.tight_layout()
    fig.savefig(out / "tipover.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    curves = []
    for name in NAMES:
        rows = [row for row in r["sweep"][name]["rows"] if row["speed"]]
        x = [row["request"] for row in rows]
        y = [row["speed"][0] for row in rows]
        ax.plot(x, y, color=COLOURS[name], lw=2, marker="o", ms=6, mec="white", mew=1.5, label=name)
        curves.append((name, x, y))
    top = r["sweep"][NAMES[0]]["rows"][-1]["request"]
    ax.plot([0, top], [0, top], color=MUTED, lw=1, ls=(0, (4, 4)), label="request")
    style(ax, "requested speed [m/s]", "measured speed [m/s]")
    ax.set_title("Speed tracking", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    end_labels(ax, curves)
    fig.tight_layout()
    fig.savefig(out / "tracking.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    curves = []
    for name in NAMES:
        rows = [row for row in r["sweep"][name]["rows"] if row["power_W"]]
        x = [row["request"] for row in rows]
        y = [row["power_W"][0] for row in rows]
        ax.plot(x, y, color=COLOURS[name], lw=2, marker="o", ms=6, mec="white", mew=1.5, label=name)
        curves.append((name, x, y))
    style(ax, "requested speed [m/s]", "mechanical power [W]")
    ax.set_title("Joint power at each speed", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    end_labels(ax, curves)
    fig.tight_layout()
    fig.savefig(out / "effort.png", dpi=150)
    plt.close(fig)

    d = r["divergence"]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
    axes[0].plot(d["t"], d["joint_rmse_deg"], color=INK, lw=1.2)
    style(axes[0], "", "joint RMSE [deg]")
    axes[0].set_title("pH against MuJoCo at the same moment of the same run", loc="left", color=INK, pad=34)
    axes[1].plot(d["t"], d["trunk_gap_m"], color=INK, lw=2)
    style(axes[1], "time [s]", "trunk distance [m]")
    for ax in axes:
        for _, t0, _, _ in SEGMENTS[1:]:
            ax.axvline(t0, color=GRID, lw=1)
    for label, t0, t1, _ in SEGMENTS:
        axes[0].text((t0 + t1) / 2, 1.02, label.replace(" ", "\n", 1), transform=axes[0].get_xaxis_transform(),
                     va="bottom", ha="center", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "divergence.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- report

def cell(value, digits: int = 2) -> str:
    if value is None:
        return "–"
    if isinstance(value, (tuple, list)):
        return f"{value[0]:.{digits}f} ± {value[1]:.{digits}f}"
    return f"{value:.{digits}f}"


def difference(a, b, digits: int = 2) -> str:
    """pH minus MuJoCo, marked * when at least MATERIAL apart and more than twice the spread between seeds."""
    if not a or not b:
        return "–"
    delta = a[0] - b[0]
    if abs(b[0]) < 0.1:  # a percentage of nearly nothing says nothing
        return f"{delta:+.{digits}f}"
    relative = delta / abs(b[0])
    marked = abs(relative) >= MATERIAL and abs(delta) > 2 * math.hypot(a[1], b[1])
    return f"{delta:+.{digits}f} ({100 * relative:+.0f} %)" + (" *" if marked else "")


def report(r: dict) -> str:
    a, b = NAMES
    L = [f"# pH plant against MuJoCo: {r['policy']}", "",
         f"Same policy, same commands, same firmware path, {r['delay_ms']:g} ms delay, {r['seeds']} seeds. "
         "This shows how the two physics differ, not which is right.", "",
         "`*` in a difference column: at least 5 % apart and more than twice the spread between seeds. The spread is "
         "not an error bar: a walk settles into the same gait from any small start offset, so it is close to zero.", "",
         "## Tip-over (no policy, open loop)", "", "![tip-over](tipover.png)", "",
         "| | seconds to 30° pitch |", "|---|---|"]
    for name in NAMES:
        L.append(f"| {name} | {cell(r['tip_over'][name]['seconds_to_30deg'])} |")
    L += ["", "## Standing (policy, zero request)", "", f"| | {a} | {b} |", "|---|---|---|"]
    for key, label in (("trunk_height_m", "Trunk height [m]"), ("pitch_deg", "Trunk pitch [deg]"),
                       ("joint_offset_deg", "Mean joint offset from HOME [deg]"), ("foot_load_N", "Foot load [N]"),
                       ("power_W", "Mechanical power [W]")):
        L.append(f"| {label} | {cell(r['stand'][a][key], 3)} | {cell(r['stand'][b][key], 3)} |")
    t = r["task"]
    L += ["", "## Task", "", f"Falls: {a} {t[a]['falls']} of {t[a]['seeds']}, {b} {t[b]['falls']} of {t[b]['seeds']}.", ""]
    rows = (("vx", "Forward speed [m/s]"), ("vy", "Sideways speed [m/s]"), ("wz", "Turn rate [rad/s]"),
            ("steps_per_s", "Steps per second"), ("step_length_m", "Step length [m]"),
            ("foot_clearance_mm", "Foot clearance [mm]"), ("sole_slip_mean_mm_s", "Sole slip, mean [mm/s]"),
            ("sole_slip_median_mm_s", "Sole slip, median [mm/s]"),
            ("trunk_height_m", "Trunk height [m]"), ("pitch_deg", "Trunk pitch [deg]"),
            ("pitch_std_deg", "Pitch sway [deg]"), ("roll_std_deg", "Roll sway [deg]"),
            ("power_W", "Mechanical power [W]"), ("cost_of_transport", "Cost of transport"))
    for label, *_, request in SEGMENTS[1:-1]:
        sa, sb = t[a]["segments"][label], t[b]["segments"][label]
        L += [f"### {label}", "", f"| | {a} | {b} | {a} − {b} |", "|---|---|---|---|"]
        for key, name in rows:
            requested = {"vx": request[0], "vy": request[1], "wz": request[2]}.get(key, 1.0)
            if key in sa and key in sb and requested != 0.0:
                L.append(f"| {name} | {cell(sa[key])} | {cell(sb[key])} | {difference(sa[key], sb[key])} |")
        for group in JOINT_GROUPS:
            if "torque_rms_Nm" in sa and "torque_rms_Nm" in sb:
                x, y = sa["torque_rms_Nm"][group], sb["torque_rms_Nm"][group]
                L.append(f"| Torque RMS, {group} [N m] | {cell(x)} | {cell(y)} | {difference(x, y)} |")
        L.append("")
    s = r["sweep"]
    L += ["## Speed sweep", "", "![tracking](tracking.png)", "", "![effort](effort.png)", "",
          f"Highest request without a fall: {a} {cell(s[a]['top_request_without_fall'])} m/s, "
          f"{b} {cell(s[b]['top_request_without_fall'])} m/s.", "",
          f"| Request [m/s] | {a} speed | {b} speed | {a} power [W] | {b} power [W] | {a} falls | {b} falls |",
          "|---|---|---|---|---|---|---|"]
    for ra, rb in zip(s[a]["rows"], s[b]["rows"]):
        L.append(f"| {ra['request']:.2f} | {cell(ra['speed'])} | {cell(rb['speed'])} | {cell(ra['power_W'], 0)} | "
                 f"{cell(rb['power_W'], 0)} | {ra['falls']}/{ra['runs']} | {rb['falls']}/{rb['runs']} |")
    p = r["pushes"]
    L += ["", "## Sideways pushes while walking at 0.5 m/s", "",
          f"Largest push always survived: {a} {cell(p[a]['largest_push_always_survived'])} m/s, "
          f"{b} {cell(p[b]['largest_push_always_survived'])} m/s.", "",
          f"| Push [m/s] | {a} survived | {b} survived | {a} recovery [s] | {b} recovery [s] |", "|---|---|---|---|---|"]
    by_size = {row["push_m_s"]: row for row in p[b]["rows"]}
    for ra in p[a]["rows"]:
        rb = by_size.get(ra["push_m_s"])
        L.append(f"| {ra['push_m_s']:.2f} | {ra['survived']}/{ra['runs']} | "
                 f"{f'{rb['survived']}/{rb['runs']}' if rb else '–'} | {cell(ra['recovery_s'])} | "
                 f"{cell(rb['recovery_s']) if rb else '–'} |")
    d = r["divergence"]
    L += ["", "## Divergence between the engines", "", "![divergence](divergence.png)", "",
          "At the same moment, a gait that is only a little out of step already shows a large joint error. The aligned "
          "column shifts one run in time first, so it measures the shape of the motion.",
          "", "| Segment | Same moment [deg] | Aligned [deg] |", "|---|---|---|"]
    L += [f"| {label} | {v['same_moment_deg']:.1f} | {v['aligned_deg']:.1f} |" for label, v in d["segments"].items()]
    L += ["", f"Final trunk distance: {d['trunk_gap_m'][-1]:.2f} m.", ""]
    if r.get("sensitivity"):
        L += ["## Sensitivity to the delay (at the 1.0 m/s request)", "",
              f"| Delay [ms] | {a} speed | {b} speed | {a} power [W] | {b} power [W] | {a} slip [mm/s] | {b} slip [mm/s] | Falls |",
              "|---|---|---|---|---|---|---|---|"]
        for delay, row in r["sensitivity"].items():
            x, y = row[a], row[b]
            L.append(f"| {delay} | {cell(x['speed'])} | {cell(y['speed'])} | {cell(x['power_W'], 0)} | "
                     f"{cell(y['power_W'], 0)} | {cell(x['sole_slip_mean_mm_s'], 0)} | {cell(y['sole_slip_mean_mm_s'], 0)} | "
                     f"{x['falls']} / {y['falls']} |")
        L.append("")
    if r.get("reverse"):
        v = r["reverse"]
        L += [f"## Reverse direction: {v['policy']}, trained on the pH plant", "", f"| | {a} | {b} |", "|---|---|---|"]
        fell = [("no fall" if v[n]["fell_at"] is None else f"{v[n]['fell_at']:.1f} s") for n in NAMES]
        L.append(f"| Fell | {fell[0]} | {fell[1]} |")
        for key, label in (("pitch_deg", "Trunk pitch [deg]"),
                           ("tilt_std_deg", "Tilt sway [deg]"), ("power_W", "Mechanical power [W]"),
                           ("largest_push_always_survived", "Largest push always survived [m/s]")):
            L.append(f"| {label} | {cell(v[a].get(key))} | {cell(v[b].get(key))} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="walk_v9", help="a name in k1_sprint/checkpoints, or a .pt / .bin file")
    ap.add_argument("--delay-ms", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="2 seeds, a coarser grid, no sensitivity")
    ap.add_argument("--stand-policy", default=str(REPO / "outputs/kaggle/stand-ph-v1/runs/stand-ph-v1/deploy_2999.bin"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "compare"))
    args = ap.parse_args()

    seeds = 2 if args.quick else args.seeds
    speeds = np.round(np.arange(0.5, 2.51, 0.5 if args.quick else 0.25), 2)
    sizes = np.round(np.arange(0.4, 2.01, 0.4 if args.quick else 0.2), 2)
    model = load_k1()
    mass = float(model.mj.body_subtreemass[1])
    corners = sole_corners(model)
    path = resolve_policy(args.policy)
    policy = load_walk_policy(path)
    walkers = {b.name: Walker(b(model, delay_ms=args.delay_ms), policy) for b in ENGINES}
    say = lambda text: print(f"[compare] {text}", flush=True)  # noqa: E731

    r = {"policy": path.name, "delay_ms": args.delay_ms, "seeds": seeds}
    say("tip-over and standing")
    r["tip_over"] = tip_over(walkers)
    r["stand"] = stand(walkers)
    say("task")
    r["task"], first = task(walkers, seeds, mass, corners)
    say("speed sweep")
    r["sweep"] = sweep(walkers, speeds, min(seeds, 3), mass, corners)
    say("pushes")
    r["pushes"] = pushes(walkers, sizes)
    r["divergence"] = divergence(first)
    if not args.quick:
        say("sensitivity to the delay")
        r["sensitivity"] = sensitivity(model, policy, (10.0, 30.0, 40.0), min(seeds, 3), mass, corners)
    stand_path = Path(args.stand_policy)
    if stand_path.exists():
        say("reverse direction")
        r["reverse"] = reverse(model, stand_path, sizes)

    out = Path(args.out) / path.stem.replace(".policy", "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(r, indent=1))
    plots(out, r)
    (out / "report.md").write_text(report(r))
    say(f"wrote {out}")


if __name__ == "__main__":
    main()
