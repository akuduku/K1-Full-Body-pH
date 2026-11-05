r"""Per-robot physical parameters: one row of NRP numbers per robot on top of the shared constants.

Used for domain randomisation and for system identification (every robot of a batch can be a different
candidate parameter set).  The layout is the C++ one (ph_k1_cuda/include/ph_k1/config.hpp RP_*):

    0..4            factors on the contact constants k_n, c_n, k_t, c_t, mu (ContactParams)
    RP_DAMP + k     passive damping ADDED to joint k [N m s/rad] (MotorBank.damping; most joints have 0)
    RP_TAU  + k     factor on the motor torque ceiling of joint k (MotorBank.tau_max)

No rows (plant.robot_params is None) means exactly the shared values; a row of `neutral_rows` gives the same
numbers.  The driver gains kp/kd are already per robot (policy_step kp=, kd=) and the delay too (set_delay).

    plant.robot_params = neutral_rows(plant.B, plant.dtype, plant.device)
    plant.robot_params[:, RP_MU] = 0.5          # robot-wise friction factor
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import NJ

RP_KN, RP_CN, RP_KT, RP_CT, RP_MU = 0, 1, 2, 3, 4
RP_DAMP = 5
RP_TAU = RP_DAMP + NJ
NRP = RP_TAU + NJ  # 49

NAMES = (["k_n", "c_n", "k_t", "c_t", "mu"] + [f"damp_add[{k}]" for k in range(NJ)]
         + [f"tau_scale[{k}]" for k in range(NJ)])


def neutral_rows(B: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """(B, NRP) rows that change nothing: factors 1, added damping 0."""
    r = torch.ones(B, NRP, dtype=dtype, device=device)
    r[:, RP_DAMP:RP_DAMP + NJ] = 0.0
    return r


@dataclass
class Effective:
    """The parameters one plant step uses, shaped to broadcast against (B, ...) tensors.

    Without rows: Python floats / the (22,) MotorBank tensors.  With rows: contact constants (B, 1),
    damping and tau_max (B, 22)."""

    k_n: torch.Tensor | float
    c_n: torch.Tensor | float
    k_t: torch.Tensor | float
    c_t: torch.Tensor | float
    mu: torch.Tensor | float
    damping: torch.Tensor
    tau_max: torch.Tensor


def effective(contact, motors, rows: torch.Tensor | None) -> Effective:
    if rows is None:
        return Effective(contact.k_n, contact.c_n, contact.k_t, contact.c_t, contact.mu, motors.damping, motors.tau_max)
    if rows.dim() != 2 or rows.shape[1] != NRP:
        raise ValueError(f"robot_params must be (B, {NRP}), got {tuple(rows.shape)}")
    col = lambda i: rows[:, i:i + 1]  # noqa: E731
    return Effective(
        k_n=contact.k_n * col(RP_KN), c_n=contact.c_n * col(RP_CN), k_t=contact.k_t * col(RP_KT),
        c_t=contact.c_t * col(RP_CT), mu=contact.mu * col(RP_MU),
        damping=motors.damping + rows[:, RP_DAMP:RP_DAMP + NJ],
        tau_max=motors.tau_max * rows[:, RP_TAU:RP_TAU + NJ],
    )
