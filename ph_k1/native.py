r"""K1PHPlantNative: the same plant as K1PHPlant, with the step in C++/CUDA (folder "Physik C++").

    from ph_k1.native import K1PHPlantNative
    plant = K1PHPlantNative(batch=4096, model=load_k1(dtype=torch.float32, device="cuda"))
    plant.reset()
    info = plant.step(volts)                      # one plant step h, volts (B,22)
    info = plant.policy_step(targets)             # servo (drive.py) + 20 plant steps in ONE launch

Everything else (reset, kinematics, velocity, energy, ph_matrices, theta, current) is inherited from
K1PHPlant and stays Python: the native code only replaces the integrator.  Differences to K1PHPlant:
  * q, p, xi, lam and the ledger are updated IN PLACE (no new tensors per step) -- clone what you keep
  * StepInfo.extra has "nu" (nu_{n+1}) and "volts" (the volts applied)
  * the parameters (contact, limits, motors, h, switches) are re-sent to C++ whenever they change

Build once (CPU on the laptop, CUDA where nvcc exists, e.g. Kaggle/Colab):
    cd "Physik C++" && python setup.py build_ext --inplace
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch

from .drive import GAINS, SLEW, _group
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
    def __init__(self, *args, **kwargs):
        self._native = load_native()
        self._led = None
        self._md = None
        self._md_key = None
        self._gains = {}
        super().__init__(*args, **kwargs)
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
        self.q_star = self.theta.clone().contiguous()  # the servo's slew-limited target (drive.py reset)

    def _prepare(self) -> None:
        """State tensors contiguous, with the plant's dtype/device (assignments like pl.p = ... are allowed)."""
        for name in ("q", "p", "xi", "lam", "q_star"):
            t = getattr(self, name)
            if not t.is_contiguous() or t.dtype != self.dtype or t.device != self.device:
                setattr(self, name, t.to(device=self.device, dtype=self.dtype).contiguous())

    def _model_block(self):
        mot = self.motors
        key = (
            id(self.model), self.h, self.fixed_base, self.motors_connected, self.gravity, self.dtype, self.device,
            dataclasses.astuple(self.contact), dataclasses.astuple(self.limits), mot.v_max,
            tuple((id(t), t._version) for t in (mot.k, mot.R, mot.L, mot.tau_max, mot.damping)),
        )
        if key != self._md_key:
            m = self.model
            T = lambda x: torch.as_tensor(x, dtype=torch.float64).detach().cpu()  # noqa: E731
            tensors = {
                "g": T(GRAVITY), "parent": T(m.parent), "body_joint": T(m.body_joint), "jnt_body": T(m.jnt_body),
                "jnt_parent_body": T(m.jnt_parent_body), "body_pos": T(m.body_pos), "body_rot": T(m.body_rot),
                "jnt_axis": T(m.jnt_axis), "jnt_pos": T(m.jnt_pos), "mass": T(m.mass), "ipos": T(m.ipos),
                "I_body": T(m.I_body), "armature": T(m.armature), "jnt_low": T(m.jnt_low), "jnt_high": T(m.jnt_high),
                "contact_body": T(m.contact_body), "contact_local": T(m.contact_local),
                "motor_k": T(mot.k), "motor_R": T(mot.R), "motor_L": T(mot.L), "motor_tau_max": T(mot.tau_max),
                "motor_damping": T(mot.damping),
            }
            c, lim = self.contact, self.limits
            scalars = {"h": self.h, "lim_k": lim.k, "lim_c": lim.c, "k_n": c.k_n, "c_n": c.c_n, "k_t": c.k_t,
                       "c_t": c.c_t, "mu": c.mu, "v_max": float(mot.v_max), "slew": SLEW}
            ints = {"nbody": m.nbody, "ncontact": m.ncontact, "fixed_base": int(self.fixed_base),
                    "motors_connected": int(self.motors_connected), "gravity": int(self.gravity),
                    "contact_enabled": int(c.enabled), "limits_enabled": int(lim.enabled)}
            block = self._native.pack_model(tensors, scalars, ints, self.dtype == torch.float64)
            self._md = block.to(self.device)
            self._md_key = key
        return self._md

    def _outputs(self) -> dict:
        B, nc, dev, dt = self.B, self.model.ncontact, self.device, self.dtype
        e = lambda *s: torch.empty(*s, dtype=dt, device=dev)  # noqa: E731
        u = lambda *s: torch.empty(*s, dtype=torch.uint8, device=dev)  # noqa: E731
        return {"out_nu": e(B, NV), "out_i": e(B, NJ), "out_tau": e(B, NJ), "out_v": e(B, NJ),
                "out_fc": e(B, nc, 3), "out_active": u(B, nc), "out_slip": u(B, nc)}

    @staticmethod
    def _info(o: dict) -> StepInfo:
        return StepInfo(current=o["out_i"], torque=o["out_tau"], contact_force=o["out_fc"],
                        contact_active=o["out_active"].bool(), slipping=o["out_slip"].bool(),
                        extra={"nu": o["out_nu"], "volts": o["out_v"]})

    def _rows(self, x):
        return None if x is None else torch.as_tensor(x, dtype=self.dtype, device=self.device).expand(self.B, NJ).contiguous()

    # -------------------------------------------------------------- step
    def step(self, volts: torch.Tensor | None = None, joint_torque: torch.Tensor | None = None) -> StepInfo:
        """Advance h seconds (same as K1PHPlant.step)."""
        self._prepare()
        out = self._outputs()
        self._native.run(self._model_block(), self.q, self.p, self.xi, self.lam, self._led,
                         volts=self._rows(volts), joint_torque=self._rows(joint_torque), nsub=1, **out)
        self.time += self.h
        return self._info(out)

    def gains(self, gain_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """drive.py's servo gains (kp, kd) per hinge, (B, 22)."""
        key = (gain_scale, self.dtype, self.device, self.B)
        if key not in self._gains:
            kp, kd = zip(*(GAINS[_group(n)] for n in self.model.joint_names))
            T = lambda a: (gain_scale * torch.as_tensor(a, dtype=self.dtype, device=self.device)).expand(self.B, NJ).contiguous()  # noqa: E731
            self._gains[key] = (T(kp), T(kd))
        return self._gains[key]

    def policy_step(self, targets: torch.Tensor, substeps: int | None = None, kp: torch.Tensor | None = None,
                    kd: torch.Tensor | None = None, gain_scale: float = 1.0) -> StepInfo:
        """The servo (drive.py K1Drive: target angles -> volts) and `substeps` plant steps in one call
        (default: a 50 Hz policy step, 0.02 s / h).  Returns the StepInfo of the last plant step."""
        if substeps is None:
            substeps = max(1, round(0.02 / self.h))
        if kp is None or kd is None:
            gkp, gkd = self.gains(gain_scale)
            kp = gkp if kp is None else kp
            kd = gkd if kd is None else kd
        self._prepare()
        out = self._outputs()
        self._native.run(self._model_block(), self.q, self.p, self.xi, self.lam, self._led,
                         q_star=self.q_star, target=self._rows(targets), kp=self._rows(kp), kd=self._rows(kd),
                         nsub=int(substeps), **out)
        self.time += substeps * self.h
        return self._info(out)
