r"""K1PHPlantNative: the same plant as K1PHPlant, with the step and the robot's command path in C++/CUDA.

    from ph_k1.native import K1PHPlantNative
    plant = K1PHPlantNative(batch=4096, model=load_k1(dtype=torch.float32, device="cuda"))
    plant.reset()
    info = plant.policy_step(targets)     # 22 joint targets (rad) per robot: one 50 Hz policy step
    info.extra["gyro"], info.extra["acc"]  # what the robot's IMU measures (trunk frame)
    info = plant.step(volts)              # or the plant alone: one plant step h with coil volts

policy_step is the real robot's command path (ph_k1/firmware.py, K1_Deploy safety.cpp): the firmware filter
(position clamp, slew, target clamp, overspeed guard) once per policy step, the transport delay (a delay line
per robot), and the motor driver's PD every plant step.  Its switches, gains and delay range are
`plant.firmware` (a FirmwareConfig); the delay of each robot is drawn at reset (`set_delay` re-draws it).

Everything else (reset, kinematics, velocity, energy, ph_matrices, theta, current) is inherited from
K1PHPlant and stays Python.  Differences to K1PHPlant:
  * q, p, xi, lam, the ledger and the command-path state are updated IN PLACE -- clone what you keep
  * StepInfo.extra: "nu" (nu_{n+1}), "volts", "gyro", "acc", and in policy steps "target" (what the driver held)
  * the parameters (contact, limits, motors, firmware, h, switches) are re-sent to C++ whenever they change

Build once (CPU on the laptop, CUDA where nvcc exists, e.g. Kaggle/Colab):
    cd "Physik C++" && python setup.py build_ext --inplace
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch

from . import firmware as fw
from .dynamics import GRAVITY
from .model import NJ, NV
from .plant import K1PHPlant, Ledger, StepInfo

NATIVE_DIR = Path(__file__).resolve().parents[1] / "Physik C++"
_LEDGER_FIELDS = [f.name for f in dataclasses.fields(Ledger)]  # supplied, mech_input, copper, gearbox, stops, contact


def load_native():
    """The compiled module ph_k1_native (installed, or built in place in "Physik C++")."""
    try:
        import ph_k1_native
    except ImportError:
        if str(NATIVE_DIR) not in sys.path:
            sys.path.insert(0, str(NATIVE_DIR))
        try:
            import ph_k1_native
        except ImportError as exc:
            raise ImportError(
                f'ph_k1_native is not built: cd "{NATIVE_DIR}" && python setup.py build_ext --inplace'
            ) from exc
    return ph_k1_native


class K1PHPlantNative(K1PHPlant):
    def __init__(self, *args, firmware: fw.FirmwareConfig | None = None, **kwargs):
        self._native = load_native()
        self.firmware = firmware or fw.FirmwareConfig()
        self._led = None
        self._md = None
        self._md_key = None
        self._gains = {}
        super().__init__(*args, **kwargs)  # calls reset()
        fw.check_tables(self.model)
        if self.device.type == "cuda" and not self._native.WITH_CUDA:
            raise RuntimeError("ph_k1_native was built without CUDA; rebuild where nvcc exists")

    # ------------------------------------------------------------ ledger
    # one (B, 6) tensor that the C++ step updates; Ledger fields are views of its columns
    @property
    def ledger(self) -> Ledger:
        return Ledger(*self._led.unbind(-1))

    @ledger.setter
    def ledger(self, led: Ledger) -> None:
        cols = [torch.as_tensor(getattr(led, n), dtype=self.dtype, device=self.device).expand(self.B) for n in _LEDGER_FIELDS]
        self._led = torch.stack(cols, -1).contiguous()

    # ------------------------------------------------------------- state
    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self.reset_command_path(kwargs.get("env_ids"))

    def reset_command_path(self, env_ids=None) -> None:
        """Firmware state and delay line start at the current pose; the delays are drawn again."""
        B, dev = self.B, self.device
        D = fw.delay_line_length(self.firmware, self.h)
        theta = self.theta
        if env_ids is None or getattr(self, "ring", None) is None or self.ring.shape[1] != D:
            self.last_target = theta.clone().contiguous()
            self.ring = theta[:, None, :].expand(B, D, NJ).clone().contiguous()
            self.ring_idx = torch.zeros(B, dtype=torch.int32, device=dev)
            self.lag = fw.draw_lags(self.firmware, B, self.h, dev)
            return
        self.last_target[env_ids] = theta[env_ids]
        self.ring[env_ids] = theta[env_ids][:, None, :]
        self.ring_idx[env_ids] = 0
        self.lag[env_ids] = fw.draw_lags(self.firmware, len(env_ids), self.h, dev)

    def set_delay(self, lo_ms: float, hi_ms: float | None = None, env_ids=None) -> None:
        """Draw the transport delay of the robots (all, or env_ids) uniformly in [lo_ms, hi_ms]."""
        hi_ms = lo_ms if hi_ms is None else hi_ms
        n = self.B if env_ids is None else len(env_ids)
        lags = fw.draw_lags(self.firmware, n, self.h, self.device, lo_ms, hi_ms)
        if env_ids is None:
            self.lag = lags
        else:
            self.lag[env_ids] = lags

    def _prepare(self) -> None:
        """State tensors contiguous, with the plant's dtype/device (assignments like pl.p = ... are allowed)."""
        for name in ("q", "p", "xi", "lam", "last_target", "ring"):
            t = getattr(self, name)
            if not t.is_contiguous() or t.dtype != self.dtype or t.device != self.device:
                setattr(self, name, t.to(device=self.device, dtype=self.dtype).contiguous())
        for name in ("ring_idx", "lag"):
            t = getattr(self, name)
            if t.dtype != torch.int32 or t.device != self.device or not t.is_contiguous():
                setattr(self, name, t.to(device=self.device, dtype=torch.int32).contiguous())

    def _model_block(self):
        mot, cfg = self.motors, self.firmware
        key = (
            id(self.model), self.h, self.fixed_base, self.motors_connected, self.gravity, self.dtype, self.device,
            dataclasses.astuple(self.contact), dataclasses.astuple(self.limits), mot.v_max,
            tuple((id(t), t._version) for t in (mot.k, mot.R, mot.L, mot.tau_max, mot.damping)),
            cfg.pos_clamp, cfg.slew, cfg.target_clamp, cfg.overspeed_guard, cfg.policy_dt,
        )
        if key != self._md_key:
            m = self.model
            T = lambda x: torch.as_tensor(x, dtype=torch.float64).detach().cpu()  # noqa: E731
            tensors = {
                "g": T(GRAVITY), "parent": T(m.parent), "body_joint": T(m.body_joint), "jnt_body": T(m.jnt_body),
                "jnt_parent_body": T(m.jnt_parent_body), "body_pos": T(m.body_pos), "body_rot": T(m.body_rot),
                "jnt_axis": T(m.jnt_axis), "jnt_pos": T(m.jnt_pos), "mass": T(m.mass), "ipos": T(m.ipos),
                "I_body": T(m.I_body), "armature": T(m.armature), "jnt_low": T(m.jnt_low), "jnt_high": T(m.jnt_high),
                "stop_low": T(m.stop_low), "stop_high": T(m.stop_high),
                "contact_body": T(m.contact_body), "contact_local": T(m.contact_local),
                "motor_k": T(mot.k), "motor_R": T(mot.R), "motor_L": T(mot.L), "motor_tau_max": T(mot.tau_max),
                "motor_damping": T(mot.damping), "fw_effort": T(fw.EFFORT_LIMIT),
                "fw_max_joint_velocity": T(fw.MAX_JOINT_VELOCITY), "fw_max_target_velocity": T(fw.MAX_TARGET_VELOCITY),
            }
            c, lim = self.contact, self.limits
            scalars = {"h": self.h, "lim_k": lim.k, "lim_c": lim.c, "k_n": c.k_n, "c_n": c.c_n, "k_t": c.k_t,
                       "c_t": c.c_t, "mu": c.mu, "v_max": float(mot.v_max), "policy_dt": cfg.policy_dt}
            ints = {"nbody": m.nbody, "ncontact": m.ncontact, "fixed_base": int(self.fixed_base),
                    "motors_connected": int(self.motors_connected), "gravity": int(self.gravity),
                    "contact_enabled": int(c.enabled), "limits_enabled": int(lim.enabled),
                    "fw_pos_clamp": int(cfg.pos_clamp), "fw_slew": int(cfg.slew),
                    "fw_target_clamp": int(cfg.target_clamp), "fw_overspeed": int(cfg.overspeed_guard)}
            block = self._native.pack_model(tensors, scalars, ints, self.dtype == torch.float64)
            self._md = block.to(self.device)
            self._md_key = key
        return self._md

    def _outputs(self, policy: bool) -> dict:
        B, nc, dev, dt = self.B, self.model.ncontact, self.device, self.dtype
        e = lambda *s: torch.empty(*s, dtype=dt, device=dev)  # noqa: E731
        u = lambda *s: torch.empty(*s, dtype=torch.uint8, device=dev)  # noqa: E731
        out = {"out_nu": e(B, NV), "out_i": e(B, NJ), "out_tau": e(B, NJ), "out_v": e(B, NJ),
               "out_fc": e(B, nc, 3), "out_active": u(B, nc), "out_slip": u(B, nc),
               "out_gyro": e(B, 3), "out_acc": e(B, 3)}
        if policy:
            out["out_target"] = e(B, NJ)
        return out

    @staticmethod
    def _info(o: dict) -> StepInfo:
        extra = {"nu": o["out_nu"], "volts": o["out_v"], "gyro": o["out_gyro"], "acc": o["out_acc"]}
        if "out_target" in o:
            extra["target"] = o["out_target"]
        return StepInfo(current=o["out_i"], torque=o["out_tau"], contact_force=o["out_fc"],
                        contact_active=o["out_active"].bool(), slipping=o["out_slip"].bool(), extra=extra)

    def _rows(self, x):
        return None if x is None else torch.as_tensor(x, dtype=self.dtype, device=self.device).expand(self.B, NJ).contiguous()

    # -------------------------------------------------------------- step
    def step(self, volts: torch.Tensor | None = None, joint_torque: torch.Tensor | None = None) -> StepInfo:
        """The plant alone: advance h seconds with these coil volts (same as K1PHPlant.step)."""
        self._prepare()
        out = self._outputs(policy=False)
        self._native.run(self._model_block(), self.q, self.p, self.xi, self.lam, self._led,
                         volts=self._rows(volts), joint_torque=self._rows(joint_torque), nsub=1, **out)
        self.time += self.h
        return self._info(out)

    def gains(self, preset: str | None = None, gain_scale: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(kp, kd) per hinge for every robot, (B, 22): a firmware.GAIN_PRESETS entry (default: plant.firmware)."""
        preset = self.firmware.gains if preset is None else preset
        gain_scale = self.firmware.gain_scale if gain_scale is None else gain_scale
        key = (preset, gain_scale, self.dtype, self.device, self.B)
        if key not in self._gains:
            kp, kd = fw.gains(self.model, preset, gain_scale)
            self._gains[key] = (kp.to(self.dtype).expand(self.B, NJ).contiguous(),
                                kd.to(self.dtype).expand(self.B, NJ).contiguous())
        return self._gains[key]

    def policy_step(self, targets: torch.Tensor, substeps: int | None = None, kp: torch.Tensor | None = None,
                    kd: torch.Tensor | None = None, gain_scale: float | None = None,
                    push_dv: torch.Tensor | None = None) -> StepInfo:
        """One policy step on the robot's command path: `targets` (B,22) joint angles go through the firmware
        filter, the delay line and the motor driver for `substeps` plant steps (default policy_dt / h).
        kp, kd (B,22) override the gains (domain randomisation); push_dv (B,3) changes the trunk's world velocity
        at the start of the step (a shove; rows of zeros: no push).  Returns the StepInfo of the last plant step."""
        cfg = self.firmware
        if substeps is None:
            substeps = max(1, round(cfg.policy_dt / self.h))
        if kp is None or kd is None:
            gkp, gkd = self.gains(gain_scale=gain_scale)
            kp = gkp if kp is None else kp
            kd = gkd if kd is None else kd
        if cfg.delay_hold_prob is not None:  # re-draw the lag of some robots (as RL_Core DelayBuffer)
            redraw = torch.rand(self.B, device=self.device) >= cfg.delay_hold_prob
            if bool(redraw.any()):
                self.lag[redraw] = fw.draw_lags(cfg, int(redraw.sum()), self.h, self.device)
        self._prepare()
        out = self._outputs(policy=True)
        self._native.run(self._model_block(), self.q, self.p, self.xi, self.lam, self._led,
                         target=self._rows(targets), last_target=self.last_target, ring=self.ring,
                         ring_idx=self.ring_idx, lag=self.lag, kp=self._rows(kp), kd=self._rows(kd),
                         push=None if push_dv is None else torch.as_tensor(push_dv, dtype=self.dtype, device=self.device)
                         .expand(self.B, 3).contiguous(),
                         nsub=int(substeps), **out)
        self.time += substeps * self.h
        return self._info(out)
