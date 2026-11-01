r"""The policy's PD controller as a port-Hamiltonian system, made passive by an energy tank.

The motor driver's PD  tau = kp (t - q) - kd qdot  is a virtual spring with the energy
        S(kp, t, q) = 1/2 sum_j kp_j (t_j - q_j)^2
and a virtual damper kd.  With FIXED target and stiffness, robot + controller is passive (energy shaping +
damping injection): S + H_robot can only decrease, so nothing can blow up.  A policy that moves the target or
changes the stiffness at a policy step changes S at the same q: it INJECTS
        I = S(kp_new, t_new, q) - S(kp_old, t_old, q)
into the closed loop (that is how it gets the robot up: ~68 J of potential energy on the K1).  The tank
(Ferraguti, Secchi, Fantuzzi 2013; Secchi, Stramigioli, Fantuzzi 2006) makes this exact and bounded:

  * the tank T (>= 0, at most `capacity` J) pays for every injection, I > 0 takes from it, I < 0 (relaxing a
    spring) gives back;
  * it is refilled with what the controller's dampers took out of the robot during the step,
    refill * sum_j kd_j qdot_j^2 dt  (refill <= 1);
  * if T cannot pay, the command change is scaled back: (kp, t) = old + alpha (new - old), with the largest
    alpha in [0, 1] whose injection the tank can pay (bisection).

So over any time interval the controller injects at most T(start) + refill * (its damping dissipation): the
closed loop stays passive by construction, whatever the policy outputs.  Damping changes inject nothing (kd > 0
always).  Everything here is computable on the robot from q, qdot and the commands, at the policy step, with the
target the firmware filter lets through (the clamp is applied inside, so the accounting uses the target the
driver will hold).  Approximations, stated: the accounting happens at command time (the transport delay shifts
when the driver switches), the motor's torque limit can cut the PD torque, and the damper refill is the
trapezoid of qdot^2 over the 20 ms step.
"""

from __future__ import annotations

from typing import Callable

import torch


def spring_energy(kp: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """S = 1/2 sum kp (t - q)^2 per robot [J]."""
    return 0.5 * (kp * (target - q).square()).sum(-1)


class EnergyTank:
    def __init__(self, num_envs: int, num_joints: int, dtype, device, capacity: float = 150.0, refill: float = 1.0,
                 enabled: bool = True, iterations: int = 12):
        z = lambda *s: torch.zeros(*s, dtype=dtype, device=device)  # noqa: E731
        self.capacity = float(capacity)
        self.refill = float(refill)
        self.enabled = enabled
        self.iterations = iterations
        self.level = z(num_envs) + self.capacity
        self.prev_target, self.prev_kp, self.prev_kd = z(num_envs, num_joints), z(num_envs, num_joints), z(num_envs, num_joints)

    def reset(self, ids: torch.Tensor, theta: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor) -> None:
        """Full tank; the driver holds the current pose (no spring energy) with these gains."""
        self.level[ids] = self.capacity
        self.prev_target[ids] = theta[ids]
        self.prev_kp[ids] = kp[ids]
        self.prev_kd[ids] = kd[ids]

    def fraction(self) -> torch.Tensor:
        return self.level / self.capacity

    def step(self, q, qd, qd_prev, dt: float, target, kp, kd, filt: Callable):
        """The command the driver gets this policy step.

        q, qd: joint state now; qd_prev: at the previous policy step; target, kp, kd: what the policy asks for
        (B,22); filt(target, kp, kd) -> the target the firmware filter lets through at this state.
        Returns (target, kp, kd, info) with info: injected, refilled [J], alpha, limited, level, violation."""
        # what the controller's dampers took out of the robot during the last step
        refilled = self.refill * (self.prev_kd * 0.5 * (qd_prev.square() + qd.square())).sum(-1) * dt
        s_old = spring_energy(self.prev_kp, self.prev_target, q)

        def command(alpha):  # alpha (B,1): old + alpha (new - old), through the firmware filter
            k = self.prev_kp + alpha * (kp - self.prev_kp)
            d = self.prev_kd + alpha * (kd - self.prev_kd)
            t = filt(self.prev_target + alpha * (target - self.prev_target), k, d)
            return t, k, d

        one = torch.ones_like(self.level)[:, None]
        t1, k1, d1 = command(one)
        inj1 = spring_energy(k1, t1, q) - s_old
        budget = (self.level + refilled).clamp(max=self.capacity)
        if not self.enabled:  # ablation: no tank, the injection is only measured
            self._store(t1, k1, d1)
            self.level = (budget - inj1).clamp(0.0, self.capacity)
            info = {"injected": inj1, "refilled": refilled, "alpha": one[:, 0],
                    "limited": torch.zeros_like(inj1, dtype=torch.bool), "level": self.level,
                    "violation": torch.zeros_like(inj1)}
            return t1, k1, d1, info
        limited = inj1 > budget
        alpha = one.clone()
        if bool(limited.any()):
            lo, hi = torch.zeros_like(one), one.clone()
            for _ in range(self.iterations):
                mid = 0.5 * (lo + hi)
                t, k, _ = command(mid)
                ok = (spring_energy(k, t, q) - s_old <= budget)[:, None]
                lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
            alpha = torch.where(limited[:, None], lo, one)
        t, k, d = command(alpha)
        inj = spring_energy(k, t, q) - s_old
        level = budget - inj
        violation = (-level).clamp_min(0.0)  # only when even alpha = 0 cannot be paid (the filter moved the old target)
        self.level = level.clamp(0.0, self.capacity)
        self._store(t, k, d)
        info = {"injected": inj, "refilled": refilled, "alpha": alpha[:, 0], "limited": limited, "level": self.level,
                "violation": violation}
        return t, k, d, info

    def _store(self, t, k, d) -> None:
        self.prev_target, self.prev_kp, self.prev_kd = t.clone(), k.clone(), d.clone()

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "refill": self.refill, "enabled": self.enabled}
