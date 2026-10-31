r"""Foot-ground contact as port-Hamiltonian elements (no MuJoCo solver).

Every foot has 4 sole points (corners of the MJCF collision box).  Each point
c with height z_c and penetration delta_c = max(0, -z_c) carries

  normal tank      V_n = 1/2 k_n delta^2           (energy storage, in H)
  normal damper    f_d = -c_n v_z, only while delta > 0     (in D, >= 0)
  tangential tank  V_t = 1/2 k_t |xi|^2,  xi_dot = v_t (stick)
                   xi is a 2-D "bristle": the elastic part of the sole
                   stretch.  Coupled to p through a SKEW block
                   (xi_dot = J_t nu,  p_dot += -J_t^T k_t xi).
  tangential damper  -c_t v_t                      (in D)
  Coulomb cone     |f_t| <= mu f_n.  If the stick force would leave the
                   cone the point slips: the bristle is shortened back onto
                   the cone (elasto-plastic).  Shortening a spring only
                   ever removes energy, so the port stays passive.
  unilateral       f_n >= 0: the ground pushes, never pulls.

Why bristles instead of tanh-Coulomb: a smooth Coulomb law has zero force at
zero slip speed, so a standing robot creeps.  The bristle holds it.

Numbers (placeholders, tune against the robot): 19.7 kg on 8 points.
k_n = 3e4 N/m per point -> about 0.8 mm static sink.  c_n ~ 0.7 of critical
for a 2.5 kg share.  mu = 0.7 is the MJCF ground friction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContactParams:
    k_n: float = 3.0e4  # N/m
    c_n: float = 4.0e2  # N s/m
    k_t: float = 3.0e4  # N/m
    c_t: float = 2.0e2  # N s/m
    mu: float = 0.7
    enabled: bool = True


@dataclass
class LimitParams:
    """Joint end stops as one-sided pH springs + dampers (replaces clip)."""

    k: float = 500.0  # N m/rad
    c: float = 5.0  # N m s/rad
    enabled: bool = True
