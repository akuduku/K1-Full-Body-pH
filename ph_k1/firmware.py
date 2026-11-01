r"""The command path of the real K1, from a policy's joint targets to the motor torque.

This is what runs on the robot (K1_Deploy firmware, binary `k1_motion_test`, firmware/src/main.cpp and
firmware/src/safety.cpp), copied term by term so that a policy trained on the pH plant meets the same chain:

    policy target, 22 joint angles [rad], once per policy step (50 Hz)
      -> apply_safety, once per policy step (safety.cpp), each step switchable here:
           1 position clamp to the firmware range          (k1_config.h joint_pos_lower/upper = MJCF range)
           2 slew  |t - t_last| <= max_target_velocity * policy_dt          (per agent on/off: agent->slew())
           3 target clamp so the driver's PD torque  kp (t - q) - kd qdot  stays within +-effort
           4 overspeed guard: qdot > max_joint_velocity -> t <= q   (and mirrored)
      -> held for the whole policy step (main.cpp sends the same LowCmd q/kp/kd every 5 ms)
      -> transport delay (DDS, motor bus, driver): `lag` plant steps, randomised (not identified on the robot;
         the 25 Oct getup log is 10-40 ms slower than the delay-free plant, RL_Core trains 10-70 ms)
      -> motor driver: PD  kp (t - q) - kd qdot, limited by what the motor can deliver at this speed
      -> coil current -> torque: the pH plant.  Volts are a plant-internal quantity, never a policy output.

The robot logs joint angles, the gyro and the accelerometer at 500 Hz; `imu_step` gives the same signals.

K1Firmware is the Python REFERENCE of the chain (it drives K1PHPlant.step(volts)); K1PHPlantNative.policy_step
runs the identical chain in C++/CUDA and is tested against it.  ph_k1/drive.py (K1Drive) is the classical
servo of the physics checks (per-millisecond slew, torque clip): not this chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .drive import GAINS, _group
from .model import NB, NJ, K1Model

# ---------------------------------------------------------------- K1_Deploy firmware/include/k1_config.h
# Joint order of the firmware = MJCF order (check_tables verifies the ranges and efforts against the MJCF).
JOINT_POS_LOWER = (-1.0, -0.349, -3.316, -1.74, -2.27, -2.44, -3.316, -1.57, -2.27, 0.0,
                   -3.0, -0.4, -1.0, 0.0, -0.87, -0.345, -3.0, -1.57, -1.0, 0.0, -0.87, -0.345)
JOINT_POS_UPPER = (1.0, 0.855, 1.22, 1.57, 2.27, 0.0, 1.22, 1.74, 2.27, 2.44,
                   2.21, 1.57, 1.0, 2.23, 0.345, 0.345, 2.21, 0.4, 1.0, 2.23, 0.345, 0.345)
EFFORT_LIMIT = (6.0, 6.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0,
                30.0, 35.0, 20.0, 40.0, 20.0, 20.0, 30.0, 35.0, 20.0, 40.0, 20.0, 20.0)
MAX_JOINT_VELOCITY = (7.85, 7.85, 33.51, 33.51, 33.51, 33.51, 33.51, 33.51, 33.51, 33.51,
                      14.66, 12.57, 17.59, 12.57, 17.59, 17.59, 14.66, 12.57, 17.59, 12.57, 17.59, 17.59)
MAX_TARGET_VELOCITY = (4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                       5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 4.0, 4.0)

# PD gains (kp, kd) the motor driver gets in LowCmd.  The plant must use the gains of the agent that will run
# the policy on the robot.
_FW_STIFF = (3.947842, 3.947842, 3.947842, 3.947842, 3.947842, 3.947842, 3.947842, 3.947842, 3.947842, 3.947842,
             30.200989, 21.447962, 17.846014, 60.401978, 35.692028, 35.692028,
             30.200989, 21.447962, 17.846014, 60.401978, 35.692028, 35.692028)
_FW_DAMP = (0.251327,) * 10 + (3.604978, 2.560162, 2.130211, 4.806637, 4.260422, 4.260422) * 2
GAIN_PRESETS = {
    # Booster-K1-Walk-Genesis walk/config/k1_env.yaml joint_gains (genesis_walk.bin), = drive.GAINS
    "walk": None,
    # k1_config.h stiffness / damping (the mjlab-derived default of k1_motion_test)
    "firmware": (_FW_STIFF, _FW_DAMP),
    # k1_config.h stiffness_stiff_arms / damping_stiff_arms (--stiff-arms)
    "stiff_arms": (_FW_STIFF[:2] + (60.0, 60.0, 40.0, 25.0) * 2 + _FW_STIFF[10:],
                   _FW_DAMP[:2] + (2.0, 2.0, 1.5, 1.0) * 2 + _FW_DAMP[10:]),
    # k1_config.h hold_kp / hold_kd (humanoid_handstand hold policy)
    "hold": ((4.0, 4.0, 60.0, 60.0, 40.0, 25.0, 60.0, 60.0, 40.0, 25.0) + (80.0, 80.0, 80.0, 80.0, 50.0, 50.0) * 2,
             (0.25, 0.25, 2.0, 2.0, 1.5, 1.0, 2.0, 2.0, 1.5, 1.0) + (4.0, 4.0, 4.0, 4.0, 2.0, 2.0) * 2),
    # k1_config.h goalie_kp / goalie_kd
    "goalie": ((10.0, 10.0) + (20.0,) * 8 + (100.0, 100.0, 100.0, 100.0, 50.0, 50.0) * 2,
               (0.5,) * 10 + (2.0, 2.0, 2.0, 2.0, 1.0, 1.0) * 2),
    # k1_config.h Prepare::stiffness / damping (Booster's prepare pose)
    "prepare": ((40.0, 40.0) + (40.0, 50.0, 20.0, 20.0) * 2 + (350.0, 350.0, 180.0, 350.0, 250.0, 250.0) * 2,
                (1.5, 1.5) + (0.5, 1.5, 0.2, 0.2) * 2 + (7.5, 7.5, 3.0, 5.5, 5.0, 5.0) * 2),
}


@dataclass
class FirmwareConfig:
    """Switches of the chain (defaults: the robot's k1_motion_test with an agent that slews)."""

    gains: str = "walk"              # GAIN_PRESETS key
    gain_scale: float = 1.0
    pos_clamp: bool = True           # step 1   (k1_motion_test --no-pos-clamp turns it off)
    slew: bool = True                # step 2   (--no-slew, or an agent trained without it)
    target_clamp: bool = True        # step 3   (always on in the firmware)
    overspeed_guard: bool = True     # step 4   (always on in the firmware)
    policy_dt: float = 0.02          # k1_config.h kDt
    delay_ms: tuple[float, float] = (10.0, 40.0)  # transport delay, drawn per robot at reset
    delay_hold_prob: float | None = None  # None: one lag per episode; p: keep it with prob p every policy step
    max_delay_ms: float = 100.0      # length of the delay line


def check_tables(model: K1Model) -> None:
    """The firmware tables must describe the same robot as the MJCF the plant is built from."""
    for name, table, ref in (("joint_pos_lower", JOINT_POS_LOWER, model.jnt_low),
                             ("joint_pos_upper", JOINT_POS_UPPER, model.jnt_high),
                             ("effort_limit", EFFORT_LIMIT, model.forcerange)):
        diff = (torch.as_tensor(table, dtype=torch.float64) - ref.detach().cpu().double()).abs().max().item()
        if diff > 1e-3:
            raise ValueError(f"K1_Deploy {name} differs from the MJCF by {diff:.3g}: not the same robot model")


def gains(model: K1Model, preset: str = "walk", gain_scale: float = 1.0):
    """(kp, kd) per hinge, (22,) on the model's device."""
    if preset not in GAIN_PRESETS:
        raise KeyError(f"gain preset {preset!r}; known: {sorted(GAIN_PRESETS)}")
    pair = GAIN_PRESETS[preset]
    if pair is None:
        pair = tuple(zip(*(GAINS[_group(n)] for n in model.joint_names)))
    T = lambda a: gain_scale * torch.as_tensor(a, dtype=model.dtype, device=model.device)  # noqa: E731
    return T(pair[0]), T(pair[1])


def firmware_table(model: K1Model, table) -> torch.Tensor:
    return torch.as_tensor(table, dtype=model.dtype, device=model.device)


def safety_filter(model: K1Model, cfg: FirmwareConfig, target, last, q, qd, kp, kd):
    """K1_Deploy apply_safety for a batch: target, last, q, qd, kp, kd (B,22) -> the target the robot sends."""
    t = target
    if cfg.pos_clamp:
        t = torch.minimum(torch.maximum(t, firmware_table(model, JOINT_POS_LOWER)), firmware_table(model, JOINT_POS_UPPER))
    if cfg.slew:
        d = firmware_table(model, MAX_TARGET_VELOCITY) * cfg.policy_dt
        t = torch.minimum(torch.maximum(t, last - d), last + d)
    if cfg.target_clamp:
        eff = firmware_table(model, EFFORT_LIMIT)
        damping_term = kd * qd
        safe_kp = torch.where(kp > 0, kp, torch.ones_like(kp))  # the firmware divides by kp (never 0 on the robot)
        lo = q + (-eff + damping_term) / safe_kp
        hi = q + (eff + damping_term) / safe_kp
        t = torch.where(kp > 0, torch.minimum(torch.maximum(t, lo), hi), t)
    if cfg.overspeed_guard:
        vmax = firmware_table(model, MAX_JOINT_VELOCITY)
        t = torch.where(qd > vmax, torch.minimum(t, q), torch.where(qd < -vmax, torch.maximum(t, q), t))
    return t


def motor_volts(plant, target, kp, kd, qd=None):
    """The motor driver: PD on the (delayed) target, limited by the motor's torque at this speed, and a one-step
    current loop matched to the plant's implicit electrical row (the volts that make the coil carry it)."""
    mot = plant.motors
    q = plant.theta
    if qd is None:
        qd = plant.velocity()[:, NB:]
    i = plant.current
    tau = kp * (target - q) - kd * qd
    avail = ((mot.v_max - mot.k * qd * tau.sign()) * mot.k / mot.R).clamp_min(0)
    cap = torch.minimum(avail, mot.tau_max)
    tau = torch.maximum(torch.minimum(tau, cap), -cap)
    i_star = tau / mot.k
    s = mot.gyrator_scale(i)
    v = mot.L * (i_star - i) / plant.h + mot.R * i_star + s * mot.k * qd
    return v.clamp(-mot.v_max, mot.v_max)


def imu_step(plant, kin_n, nu_n):
    """Gyro and accelerometer after a plant step, trunk frame, as the robot logs them.

    gyro = omega_B(n+1).  acc = specific force of the trunk origin  R^T (a - g):
    a = R_n ((v_B(n+1) - v_B(n)) / h + omega_B(n+1) x v_B(n+1))   (the frame turns during the step).
    Standing still: (0, 0, +9.81); lying on the back: (+9.81, 0, 0); free fall: 0.
    """
    f = plant.free
    nu1 = torch.zeros_like(nu_n)
    nu1[:, f] = torch.linalg.solve(kin_n.M[:, f][:, :, f], plant.p[:, f])  # p_{n+1} = M(q_n) nu_{n+1}
    w1, v1, v0 = nu1[:, :3], nu1[:, 3:6], nu_n[:, 3:6]
    acc = (v1 - v0) / plant.h + torch.cross(w1, v1, dim=-1)
    if plant.gravity:
        from .dynamics import GRAVITY

        g = torch.as_tensor(GRAVITY, dtype=plant.dtype, device=plant.device)
        acc = acc - (kin_n.R[:, 0].transpose(1, 2) @ g)
    return w1, acc


def draw_lags(cfg: FirmwareConfig, n: int, h: float, device, lo_ms=None, hi_ms=None, generator=None) -> torch.Tensor:
    """Integer lags in plant steps, uniform in [lo, hi] ms, (n,) int32."""
    lo_ms = cfg.delay_ms[0] if lo_ms is None else lo_ms
    hi_ms = cfg.delay_ms[1] if hi_ms is None else hi_ms
    lo, hi = round(lo_ms * 1e-3 / h), round(hi_ms * 1e-3 / h)
    cap = delay_line_length(cfg, h) - 1
    if not 0 <= lo <= hi <= cap:
        raise ValueError(f"delay {lo_ms}..{hi_ms} ms must lie in 0..{cfg.max_delay_ms} ms")
    return torch.randint(lo, hi + 1, (n,), dtype=torch.int32, device=device, generator=generator)


def delay_line_length(cfg: FirmwareConfig, h: float) -> int:
    return round(cfg.max_delay_ms * 1e-3 / h) + 1


class K1Firmware:
    """The deploy chain on the Python plant (reference for K1PHPlantNative.policy_step)."""

    def __init__(self, plant, config: FirmwareConfig | None = None):
        self.plant = plant
        self.cfg = config or FirmwareConfig()
        check_tables(plant.model)
        self.reset()

    def reset(self, env_ids=None) -> None:
        pl, B = self.plant, self.plant.B
        D = delay_line_length(self.cfg, pl.h)
        if env_ids is None or not hasattr(self, "ring"):
            self.last_target = pl.theta.clone()
            self.ring = pl.theta[:, None, :].expand(B, D, NJ).clone()
            self.ring_idx = torch.zeros(B, dtype=torch.long, device=pl.device)
            self.lag = draw_lags(self.cfg, B, pl.h, pl.device).long()
            return
        self.last_target[env_ids] = pl.theta[env_ids]
        self.ring[env_ids] = pl.theta[env_ids][:, None, :]
        self.ring_idx[env_ids] = 0
        self.lag[env_ids] = draw_lags(self.cfg, len(env_ids), pl.h, pl.device).long()

    def policy_step(self, target, substeps: int | None = None, kp=None, kd=None):
        pl, cfg = self.plant, self.cfg
        if substeps is None:
            substeps = max(1, round(cfg.policy_dt / pl.h))
        if kp is None or kd is None:
            gkp, gkd = gains(pl.model, cfg.gains, cfg.gain_scale)
            kp = gkp if kp is None else kp
            kd = gkd if kd is None else kd
        B = pl.B
        kp = torch.as_tensor(kp, dtype=pl.dtype, device=pl.device).expand(B, NJ)
        kd = torch.as_tensor(kd, dtype=pl.dtype, device=pl.device).expand(B, NJ)
        target = torch.as_tensor(target, dtype=pl.dtype, device=pl.device).expand(B, NJ)
        if cfg.delay_hold_prob is not None:  # re-draw the lag of some robots (as RL_Core DelayBuffer)
            redraw = torch.rand(B, device=pl.device) >= cfg.delay_hold_prob
            if bool(redraw.any()):
                self.lag[redraw] = draw_lags(cfg, int(redraw.sum()), pl.h, pl.device).long()
        qd = pl.velocity()[:, NB:]
        t = safety_filter(pl.model, cfg, target, self.last_target, pl.theta, qd, kp, kd)
        self.last_target = t
        D = self.ring.shape[1]
        rows = torch.arange(B, device=pl.device)
        info = None
        for _ in range(substeps):
            self.ring[rows, self.ring_idx] = t
            sent = self.ring[rows, (self.ring_idx - self.lag) % D]
            kin = pl.kinematics()
            nu = pl.velocity(kin)
            volts = motor_volts(pl, sent, kp, kd, qd=nu[:, NB:])
            info = pl.step(volts)
            info.extra["gyro"], info.extra["acc"] = imu_step(pl, kin, nu)
            info.extra["target"] = sent
            self.ring_idx = (self.ring_idx + 1) % D
        return info
