r"""THE WHOLE K1 AS ONE PORT-HAMILTONIAN SYSTEM.  No mj_step.

    state   x = (q, p, xi, lambda)
              q      (29)  trunk position, trunk quaternion, 22 hinge angles
              p      (28)  p = M(q) nu   generalised momentum
              xi   (8,2)   tangential bristles of the 8 sole points
              lambda (22)  coil fluxes, lambda = L i

    H(x) = 1/2 p^T M(q)^-1 p + V_grav(q) + V_contact(q, xi) + V_limits(theta)
           + sum_j lambda_j^2 / (2 L_j)

    x_dot = (J(x) - D(x)) dH/dx + G v,      y = G^T dH/dx = i

    J = [ 0        S_q        0      0     ]      D = [0  0   0  0]
        [ -S_q^T   J_gyro(p)  -J_t^T B K_s ]          [0  D_p 0  0]
        [ 0        J_t        0      0     ]          [0  0   0  0]
        [ 0        -K_s B^T   0      0     ]          [0  0   0  R]
    G = [0; 0; 0; I_22]

  S_q        q_dot = S_q(q) nu  (trunk: r_dot = R v_B, Q_dot = E(Q) omega_B)
  J_gyro(p)  skew Lie-algebra block of the free trunk
  J_t        sole-point tangential velocity Jacobian (contact bristles)
  K_s = diag(s k)   motor gyrators,  B selects the hinge rows
  D_p = gearbox damping + contact dampers + end-stop dampers  (>= 0)

The only input is the voltage v (22).  MuJoCo is not called anywhere in this
file.  `ph_matrices()` builds J, D, G and dH/dx explicitly (autograd for
dH/dx) so validation/checks.py can prove that the integrator integrates exactly this.

INTEGRATOR (one step h, semi-implicit, fixes over the arm plant):
  * forces that depend on the configuration (gravity, dT/dq, gyroscopic,
    spring parts of contact/end stops) at x_n;
  * dampers, the stiffness of contact/end-stop springs (linearised), the
    gyrator coupling p <-> lambda and the coil resistance IMPLICIT, solved
    together as one linear system in (nu_{n+1}, i_{n+1});
  * q_{n+1} from nu_{n+1}; trunk orientation by the exponential map.
  The arm plant used forward Euler on the gyrator, which injects energy;
  implicit Euler on these terms can only remove it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch

from .contact import ContactParams, LimitParams
from .dynamics import (
    Kinematics,
    base_gyroscopic,
    forward_kinematics,
    generalized_force,
    gravity_potential,
    gravity_wrenches,
    gyro_matrix,
    kinetic_gradient,
    point_jacobian,
)
from .model import HOME_POSE, NB, NJ, NQ, NV, K1Model, load_k1
from .motors import MotorBank, build_motors
from .robot_params import Effective, effective
from . import ball as bl
from . import self_contact as sc
from .solve import solve as ph_solve
from .spatial import quat_exp, quat_mul, quat_rate_matrix

TIMESTEP = 1e-3


@dataclass
class Ledger:
    """Cumulative energy book-keeping per robot [J]."""

    supplied: torch.Tensor  # integral of v.i (electrical port)
    mech_input: torch.Tensor  # integral of tau_ext.theta_dot + f_ext.v_trunk (test torque port, trunk force)
    copper: torch.Tensor  # i^2 R
    gearbox: torch.Tensor  # d theta_dot^2
    stops: torch.Tensor  # end-stop dampers
    contact: torch.Tensor  # contact dampers + slip

    @staticmethod
    def zeros(B, dtype, device):
        z = lambda: torch.zeros(B, dtype=dtype, device=device)  # noqa: E731
        return Ledger(z(), z(), z(), z(), z(), z())

    @property
    def dissipated(self):
        return self.copper + self.gearbox + self.stops + self.contact


@dataclass
class StepInfo:
    current: torch.Tensor
    torque: torch.Tensor  # motor torque on every hinge (B,22)
    contact_force: torch.Tensor  # (B,nc,3) world force on each sole point
    contact_active: torch.Tensor  # (B,nc) bool
    slipping: torch.Tensor  # (B,nc) bool
    extra: dict = field(default_factory=dict)


class K1PHPlant:
    def __init__(
        self,
        model: K1Model | None = None,
        batch: int = 1,
        timestep: float = TIMESTEP,
        fixed_base: bool = False,
        motors_connected: bool = True,
        contact: ContactParams | None = None,
        limits: LimitParams | None = None,
        motors: MotorBank | None = None,
        gravity: bool = True,
        self_contact: sc.SelfContactParams | None = None,
        ball: bl.BallParams | None = None,
    ):
        self.model = model or load_k1()
        m = self.model
        self.B = batch
        self.h = float(timestep)
        self.fixed_base = fixed_base
        self.motors_connected = motors_connected
        self.contact = contact or ContactParams()
        self.limits = limits or LimitParams()
        self.self_contact = self_contact or sc.SelfContactParams()  # off by default (ph_k1/self_contact.py)
        self.ball_params = ball or bl.BallParams()  # off by default (ph_k1/ball.py)
        self.motors = motors or build_motors(m)
        self.gravity = gravity
        self.dtype, self.device = m.dtype, m.device
        # per-robot parameter rows (B, robot_params.NRP) on top of contact/motors, or None (ph_k1/robot_params.py)
        self.robot_params: torch.Tensor | None = None
        # free dofs and where the hinges sit inside them
        self.free = torch.arange(NB if fixed_base else 0, NV, device=self.device)
        self.jrows = torch.arange(NJ, device=self.device) + (0 if fixed_base else NB)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(
        self,
        pose: dict[str, float] | torch.Tensor | None = None,
        base_pos=(0.0, 0.0, None),
        base_quat=(1.0, 0.0, 0.0, 0.0),
        clearance: float = 0.0,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Zero momentum and current.  base z=None: put the soles on the ground.

        env_ids: reset only these robots (the others are untouched, the clock keeps running).  pose may then be
        one (22,) vector or one row per robot (n, 22); base_quat one quaternion or (n, 4).
        """
        m, B = self.model, self.B
        dev, dt = self.device, self.dtype
        rows = None if env_ids is None else torch.as_tensor(env_ids, device=dev).long().reshape(-1)
        n = B if rows is None else int(rows.numel())
        if pose is None:
            pose = HOME_POSE
        theta = pose.to(device=dev, dtype=dt) if isinstance(pose, torch.Tensor) else m.pose_vector(pose)
        q = torch.zeros(n, NQ, dtype=dt, device=dev)
        q[:, 3:7] = torch.as_tensor(base_quat, dtype=dt, device=dev)
        q[:, 7:] = theta
        x, y, z = base_pos
        q[:, 0], q[:, 1] = x, y
        q[:, 2] = 0.0 if z is None else z
        if z is None and not self.fixed_base and n > 0:
            kin = forward_kinematics(m, q)
            pc = self._sole_points(kin)
            q[:, 2] = -pc[..., 2].min(-1).values + clearance
        if rows is None:
            self.q = q
            self.p = torch.zeros(B, NV, dtype=dt, device=dev)
            self.xi = torch.zeros(B, m.ncontact, 2, dtype=dt, device=dev)
            self.lam = torch.zeros(B, NJ, dtype=dt, device=dev)
            self.time = 0.0
            self.ledger = Ledger.zeros(B, dt, dev)
            self.ball = bl.rest_state(self.ball_params, B, dt, dev)  # parked; set_ball puts it into play
            return
        self.q[rows] = q
        self.p[rows] = 0.0
        self.xi[rows] = 0.0
        self.lam[rows] = 0.0
        led = self.ledger  # fields are tensors (views of one buffer in K1PHPlantNative): zero their rows in place
        for f in dataclasses.fields(Ledger):
            getattr(led, f.name)[rows] = 0.0
        self.ball[rows] = bl.rest_state(self.ball_params, n, dt, dev)

    def set_ball(self, xy, velocity=None, env_ids: torch.Tensor | None = None) -> None:
        """Put the ball on the ground at xy ((2,) or (n,2), world), at rest or with a linear velocity ((3,) or
        (n,3)); a ball that starts with a horizontal velocity starts rolling without slip."""
        dev, dt = self.device, self.dtype
        rows = torch.arange(self.B, device=dev) if env_ids is None else torch.as_tensor(env_ids, device=dev).long().reshape(-1)
        n = int(rows.numel())
        b = bl.rest_state(self.ball_params, n, dt, dev)
        b[:, :2] = torch.as_tensor(xy, dtype=dt, device=dev).expand(n, 2)
        if velocity is not None:
            u = torch.as_tensor(velocity, dtype=dt, device=dev).expand(n, 3)
            b[:, 3:6] = u
            r = self.ball_params.radius
            b[:, 6], b[:, 7] = -u[:, 1] / r, u[:, 0] / r
        self.ball[rows] = b

    def _sole_points(self, kin: Kinematics) -> torch.Tensor:
        m = self.model
        R = kin.R[:, m.contact_body]
        return kin.pos[:, m.contact_body] + (R @ m.contact_local[None, :, :, None]).squeeze(-1)

    def kinematics(self) -> Kinematics:
        return forward_kinematics(self.model, self.q)

    def velocity(self, kin: Kinematics | None = None) -> torch.Tensor:
        """nu = M^-1 p on the free dofs (zeros on a welded trunk)."""
        kin = kin or self.kinematics()
        f = self.free
        nu = torch.zeros_like(self.p)
        nu[:, f] = ph_solve(kin.M[:, f][:, :, f], self.p[:, f])
        return nu

    @property
    def theta(self):
        return self.q[:, 7:]

    def params(self) -> Effective:
        """The contact/motor parameters a step uses: the shared ones, changed per robot by robot_params."""
        return effective(self.contact, self.motors, self.robot_params)

    @property
    def current(self):
        return self.lam / self.motors.L

    # --------------------------------------------------------------- energy
    def energy(self, kin: Kinematics | None = None, rows: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """The tanks of H.  `total` is H(x).  rows: only these robots (kin, if given, must then be theirs)."""
        c, lim, m = self.contact, self.limits, self.model
        e = self.params()
        k_n, k_t = e.k_n, e.k_t
        q, p, xi, lam, ball = self.q, self.p, self.xi, self.lam, self.ball
        if rows is not None:
            q, p, xi, lam, ball = q[rows], p[rows], xi[rows], lam[rows], ball[rows]
            if isinstance(k_n, torch.Tensor):
                k_n, k_t = k_n[rows], k_t[rows]
        kin = kin or forward_kinematics(m, q)
        f = self.free
        nu = torch.zeros_like(p)
        nu[:, f] = ph_solve(kin.M[:, f][:, :, f], p[:, f])
        theta = q[:, 7:]
        out = {"kinetic": 0.5 * (p * nu).sum(-1)}
        zeros_b = torch.zeros(q.shape[0], dtype=self.dtype, device=self.device)
        out["gravity"] = gravity_potential(m, kin) if self.gravity else zeros_b
        if c.enabled and not self.fixed_base:
            delta = (-self._sole_points(kin)[..., 2]).clamp_min(0)
            out["contact"] = (0.5 * _per_robot(k_n) * (delta**2).sum(-1)
                              + 0.5 * _per_robot(k_t) * (xi**2).sum((-1, -2)))
        else:
            out["contact"] = zeros_b
        if self.self_contact.enabled:
            out["contact"] = out["contact"] + sc.potential(self.self_contact, sc.build_geometry(m), kin)
        if lim.enabled:
            over = (theta - m.stop_high).clamp_min(0) + (m.stop_low - theta).clamp_min(0)
            out["stops"] = 0.5 * lim.k * (over**2).sum(-1)
        else:
            out["stops"] = zeros_b
        out["coil"] = (lam**2 / (2 * self.motors.L)).sum(-1)
        if self.ball_params.enabled:
            out["ball"] = bl.energy(self.ball_params, bl.build_boxes(m), kin, ball, self.gravity)
        else:
            out["ball"] = zeros_b
        out["total"] = sum(out[k] for k in ("kinetic", "gravity", "contact", "stops", "coil", "ball"))
        return out

    # ------------------------------------------------------------------ step
    def step(self, volts: torch.Tensor | None = None, joint_torque: torch.Tensor | None = None,
             trunk_force: torch.Tensor | None = None, gen_force: torch.Tensor | None = None) -> StepInfo:
        """Advance h seconds.  volts (B,22) is the ONLY physical input of the robot.

        joint_torque is a test port for validation with motors_connected=False
        (torque straight onto the hinges, like MuJoCo's qfrc_applied).
        trunk_force (B,3) is an external world-frame force on the trunk origin (a training assist, a hand);
        its work enters ledger.mech_input like the joint_torque port's.
        gen_force (B,28) is an external generalised force on (omega_B, v_B, theta_dot), e.g. J^T f of a push on
        any body point; its work also enters ledger.mech_input.
        """
        m, h, B = self.model, self.h, self.B
        mot, c, lim = self.motors, self.contact, self.limits
        e = self.params()  # contact constants (B,1) and damping/tau_max (B,22) when robot_params is set
        k_n, c_n, k_t, c_t, mu = e.k_n, e.c_n, e.k_t, e.c_t, e.mu
        k_t3 = _per_robot(k_t, 3)  # against (B, nc, 2)
        zeros22 = torch.zeros(B, NJ, dtype=self.dtype, device=self.device)
        v = zeros22 if volts is None else torch.as_tensor(volts, dtype=self.dtype, device=self.device).expand(B, NJ)

        kin = self.kinematics()
        M = kin.M
        nu = self.velocity(kin)
        i_n = self.current
        theta = self.theta

        # ---- explicit, configuration-dependent part of p_dot ---------------
        F = torch.zeros(B, NV, dtype=self.dtype, device=self.device)
        if not self.fixed_base:
            F[:, :NB] += base_gyroscopic(self.p, nu)
        F[:, NB:] += kinetic_gradient(m, kin, nu)
        if self.gravity:
            F += generalized_force(kin, gravity_wrenches(m, kin))
        if joint_torque is not None:
            F[:, NB:] += joint_torque
        f_body = None
        if trunk_force is not None and not self.fixed_base:  # generalised force (0, R^T f) on (omega_B, v_B)
            f_w = torch.as_tensor(trunk_force, dtype=self.dtype, device=self.device).expand(B, 3)
            f_body = (kin.R[:, 0].transpose(1, 2) @ f_w[..., None]).squeeze(-1)
            F[:, 3:6] += f_body
        fgen = None
        if gen_force is not None:
            fgen = torch.as_tensor(gen_force, dtype=self.dtype, device=self.device).expand(B, NV)
            F += fgen
        # limbs against limbs (ph_k1/self_contact.py): implicit as the ground, below
        use_self = self.self_contact.enabled
        if use_self:
            sp = self.self_contact
            s_state = sc.pair_state(sc.build_geometry(m), kin)
            Js = sc.jacobian(sc.build_geometry(m), kin, s_state)                 # (B,P,28)
            cs_eff = sp.c + h * sp.k

        # the ball against feet and shanks and against the ground (ph_k1/ball.py): implicit, below
        use_ball = self.ball_params.enabled
        if use_ball:
            bp = self.ball_params
            bx, bu, bom = self.ball[:, 0:3], self.ball[:, 3:6], self.ball[:, 6:9]
            b_state = bl.pair_state(bl.build_boxes(m), kin, bx, bp.radius)
            Jb = bl.jacobian(bl.build_boxes(m), kin, b_state)                    # (B,nb,28)
            cb_eff, cg_eff = bp.c + h * bp.k, bp.c_ground + h * bp.k_ground
            dg = bl.ground_overlap(bp, bx)
            g_vec = bl.gravity_vector(self.dtype, self.device) if self.gravity else torch.zeros(3, dtype=self.dtype, device=self.device)
            eye3 = torch.eye(3, dtype=self.dtype, device=self.device)
            ball_u1 = {}

        # implicit diagonal on the hinge rows: gearbox + end-stop
        diag = e.damping.expand(B, NJ).clone()
        if lim.enabled:
            over = (theta - m.stop_high).clamp_min(0)
            under = (m.stop_low - theta).clamp_min(0)
            stop_on = (over > 0) | (under > 0)
            F[:, NB:] += lim.k * (under - over)
            diag = diag + stop_on * (lim.c + h * lim.k)
        else:
            stop_on = torch.zeros(B, NJ, dtype=torch.bool, device=self.device)

        A0 = M.clone()
        A0[:, NB:, NB:] += h * torch.diag_embed(diag)

        # ---- contact candidates --------------------------------------------
        nc = m.ncontact
        use_contact = c.enabled and not self.fixed_base
        if use_contact:
            pc = self._sole_points(kin)
            delta = -pc[..., 2]
            cand = delta > 0
            Jc = point_jacobian(kin, m.contact_body, pc)  # (B,nc,3,28)
            Jz, Jt = Jc[:, :, 2], Jc[:, :, :2]
            cn_eff, ct_eff = c_n + h * k_n, c_t + h * k_t
            ct_eff3 = _per_robot(ct_eff, 3)
        else:
            cand = torch.zeros(B, nc, dtype=torch.bool, device=self.device)

        Ks = mot.gyrator_scale(i_n, e.tau_max) * mot.k if self.motors_connected else zeros22

        def solve(normal_on, stick_on, f_slip, self_on=None, ball_on=None, ground_on=None):
            A, Ftot = A0, F
            if use_ball:  # the ball's 3 linear velocities eliminated from the system (Schur complement)
                wb = ball_on.to(self.dtype)
                wg = ground_on.to(self.dtype)
                A = A + h * cb_eff * torch.einsum("bp,bpd,bpe->bde", wb, Jb, Jb)
                Ftot = Ftot - torch.einsum("bp,bpd->bd", bp.k * b_state.delta * wb, Jb)
                Abb = bp.mass * eye3 + h * cb_eff * torch.einsum("bp,bpi,bpj->bij", wb, b_state.n, b_state.n)
                Abb[:, 2, 2] += h * cg_eff * wg
                bb = bp.mass * bu + h * (bp.mass * g_vec + torch.einsum("bp,bpi->bi", bp.k * b_state.delta * wb, b_state.n))
                bb[:, 2] += h * bp.k_ground * dg * wg
                Cb = h * cb_eff * torch.einsum("bp,bpd,bpi->bdi", wb, Jb, b_state.n)     # (B,28,3)
                Abb_inv = torch.linalg.inv(Abb)
                A = A - Cb @ Abb_inv @ Cb.transpose(1, 2)
                Ftot = Ftot + (Cb @ (Abb_inv @ bb[..., None])).squeeze(-1) / h
            if self_on is not None:
                A = A + h * cs_eff * torch.einsum("bp,bpd,bpe->bde", self_on.to(self.dtype), Js, Js)
                Ftot = Ftot + torch.einsum("bp,bpd->bd", sp.k * s_state.delta * self_on, Js)
            if use_contact:
                wn = (normal_on * cn_eff).to(self.dtype)
                wt = (stick_on * ct_eff).to(self.dtype)
                A = A + h * (
                    torch.einsum("bc,bcd,bce->bde", wn, Jz, Jz)
                    + torch.einsum("bc,bcid,bcie->bde", wt, Jt, Jt)
                )
                f0 = torch.zeros(B, nc, 3, dtype=self.dtype, device=self.device)
                f0[..., 2] = k_n * delta * normal_on
                f0[..., :2] = -k_t3 * self.xi * stick_on[..., None] + f_slip
                Ftot = Ftot + torch.einsum("bcid,bci->bd", Jc, f0)
            f = self.free
            Af = A[:, f][:, :, f]
            b1 = (self.p + h * Ftot)[:, f]
            E = torch.zeros(B, len(f), NJ, dtype=self.dtype, device=self.device)
            jr = self.jrows  # hinge positions inside the free dofs
            E[:, jr, torch.arange(NJ, device=self.device)] = 1.0
            # A^-1 [b1 | B] in one solve (A is SPD).  Not cholesky/cholesky_solve: some CUDA builds of torch lack
            # their kernels for older GPUs (Colab's torch 2.11+cu130 on a T4: "no kernel image"); linalg.solve works.
            X = ph_solve(Af, torch.cat((b1[..., None], E), -1))
            nu0, Y = X[..., 0], X[..., 1:]  # Y = A^-1 B
            Mj = Y[:, jr]  # B^T A^-1 B
            Ksys = torch.diag_embed(mot.L + h * mot.R) + h * h * Ks[:, :, None] * Mj * Ks[:, None, :]
            rhs = mot.L * i_n + h * v - h * Ks * nu0[:, jr]
            i1 = ph_solve(Ksys, rhs)
            nu_f = nu0 + h * (Y @ (Ks * i1)[..., None]).squeeze(-1)
            nu1 = torch.zeros(B, NV, dtype=self.dtype, device=self.device)
            nu1[:, f] = nu_f
            if use_ball:
                ball_u1["u"] = (Abb_inv @ (bb + (Cb.transpose(1, 2) @ nu1[..., None]).squeeze(-1))[..., None]).squeeze(-1)
            return nu1, i1, A

        no_slip = torch.zeros(B, nc, 2, dtype=self.dtype, device=self.device)
        normal_on, stick_on, f_slip = cand, cand, no_slip
        self_on = s_state.active if use_self else None
        ball_on = b_state.active if use_ball else None
        ground_on = dg > 0 if use_ball else None
        nu1, i1, A = solve(normal_on, stick_on, f_slip, self_on, ball_on, ground_on)
        changed = False
        if use_contact and bool(cand.any()):
            vc1 = torch.einsum("bcid,bd->bci", Jc, nu1)
            fz = k_n * delta - cn_eff * vc1[..., 2]
            ft = -k_t3 * self.xi - ct_eff3 * vc1[..., :2]
            normal2 = cand & (fz > 0)
            fz_c = fz.clamp_min(0) * normal2
            ftn = torch.linalg.norm(ft, dim=-1)
            stick2 = normal2 & (ftn <= mu * fz_c)
            slip2 = normal2 & ~stick2
            f_slip2 = (mu * fz_c / ftn.clamp_min(1e-12))[..., None] * ft * slip2[..., None]
            if not (torch.equal(normal2, normal_on) and torch.equal(stick2, stick_on)):
                normal_on, stick_on, f_slip = normal2, stick2, f_slip2
                changed = True
        if use_self and bool(self_on.any()):  # unilateral: the elements that would pull let go
            self2 = s_state.active & (sp.k * s_state.delta - cs_eff * torch.einsum("bpd,bd->bp", Js, nu1) > 0)
            if not torch.equal(self2, self_on):
                self_on = self2
                changed = True
        if use_ball:  # unilateral: neither a box nor the ground pulls the ball
            u1 = ball_u1["u"]
            gb = torch.einsum("bpi,bi->bp", b_state.n, u1) - torch.einsum("bpd,bd->bp", Jb, nu1)
            ball2 = b_state.active & (bp.k * b_state.delta - cb_eff * gb > 0)
            ground2 = (dg > 0) & (bp.k_ground * dg - cg_eff * u1[:, 2] > 0)
            if not (torch.equal(ball2, ball_on) and torch.equal(ground2, ground_on)):
                ball_on, ground_on = ball2, ground2
                changed = True
        if changed:
            nu1, i1, A = solve(normal_on, stick_on, f_slip, self_on, ball_on, ground_on)

        # ---- post-step: forces, bristles, ledger ---------------------------
        thd1 = nu1[:, NB:]
        f_contact = torch.zeros(B, nc, 3, dtype=self.dtype, device=self.device)
        diss_contact = torch.zeros(B, dtype=self.dtype, device=self.device)
        xi_old = self.xi
        if use_contact:
            vc1 = torch.einsum("bcid,bd->bci", Jc, nu1)
            f_contact[..., 2] = (k_n * delta - cn_eff * vc1[..., 2]) * normal_on
            f_contact[..., :2] = (-k_t3 * xi_old - ct_eff3 * vc1[..., :2]) * stick_on[..., None] + f_slip
            xi_new = torch.where(stick_on[..., None], xi_old + h * vc1[..., :2], torch.zeros_like(xi_old))
            xi_new = torch.where(slip_mask(normal_on, stick_on)[..., None], -f_slip / k_t3, xi_new)
            # losses = work the body does on the contact elements minus what
            # they stored (dampers, slip, bristle released on lift-off)
            work_in = -h * (f_contact * vc1).sum((-1, -2))
            d_new = (delta - h * vc1[..., 2]).clamp_min(0)
            dV_n = 0.5 * _per_robot(k_n) * (d_new**2 - delta.clamp_min(0) ** 2).sum(-1)
            dV_t = 0.5 * _per_robot(k_t) * ((xi_new**2).sum((-1, -2)) - (xi_old**2).sum((-1, -2)))
            diss_contact = work_in - dV_n - dV_t
            self.xi = xi_new

        if use_self:
            vn = torch.einsum("bpd,bd->bp", Js, nu1)
            diss_contact = diss_contact + sc.dissipated(sp, s_state, (sp.k * s_state.delta - cs_eff * vn) * self_on, vn, h)

        if use_ball:
            u1 = ball_u1["u"]
            gb = torch.einsum("bpi,bi->bp", b_state.n, u1) - torch.einsum("bpd,bd->bp", Jb, nu1)   # separation speed
            fb = (bp.k * b_state.delta - cb_eff * gb) * ball_on
            d_new = (b_state.delta - h * gb).clamp_min(0) * b_state.active
            fg = (bp.k_ground * dg - cg_eff * u1[:, 2]) * ground_on
            dg_new = (dg - h * u1[:, 2]).clamp_min(0) * (dg > 0)
            # losses = work of the robot on the ball's elements minus what the ball's tanks gained in this step
            # (dampers, and the implicit step's own damping of the light ball)
            work_in = h * (fb * torch.einsum("bpd,bd->bp", Jb, nu1)).sum(-1)
            d_tanks = (0.5 * bp.mass * ((u1 * u1).sum(-1) - (bu * bu).sum(-1)) - h * bp.mass * (u1 @ g_vec)
                       + 0.5 * bp.k * (d_new**2 - b_state.delta**2).sum(-1) + 0.5 * bp.k_ground * (dg_new**2 - dg**2))
            diss_contact = diss_contact + work_in - d_tanks
            Jc_ball = point_jacobian(kin, bl.build_boxes(m).body, b_state.point)          # (B,nb,3,28)
            vc_ball = torch.einsum("bpid,bd->bpi", Jc_ball, nu1)
            u1f, om1, J_ball, lost_pair = bl.pair_friction(bp, u1, bom, b_state.n, fb, vc_ball, h)
            ball_dp = -torch.einsum("bpid,bpi->bd", Jc_ball, J_ball)                      # the reaction on the robot
            diss_contact = diss_contact + lost_pair
            u2, om2, lost = bl.friction(bp, u1f, om1, fg, h)
            diss_contact = diss_contact + lost
            self.ball = torch.cat((bx + h * u1, u2, om2), -1)
            self.ball_force = fb  # (B,nb) normal force of every box on the ball in this step

        tau_motor = Ks * i1
        led = self.ledger
        led.supplied += h * (v * i1).sum(-1)
        led.copper += h * (mot.R * i1 * i1).sum(-1)
        led.gearbox += h * (e.damping * thd1 * thd1).sum(-1)
        if lim.enabled:
            led.stops += h * (stop_on * (lim.c + h * lim.k) * thd1 * thd1).sum(-1)
        led.contact += diss_contact
        if joint_torque is not None:
            led.mech_input += h * (joint_torque * thd1).sum(-1)
        if f_body is not None:
            led.mech_input += h * (f_body * nu1[:, 3:6]).sum(-1)
        if fgen is not None:
            led.mech_input += h * (fgen * nu1).sum(-1)

        # ---- configuration update ------------------------------------------
        q = self.q.clone()
        q[:, 7:] = theta + h * thd1
        if not self.fixed_base:
            Rb = kin.R[:, 0]
            q[:, :3] = self.q[:, :3] + h * (Rb @ nu1[:, 3:6, None]).squeeze(-1)
            quat = quat_mul(self.q[:, 3:7], quat_exp(h * nu1[:, :3]))
            q[:, 3:7] = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
        self.p = (M @ nu1[..., None]).squeeze(-1)
        if use_ball:
            self.p = self.p + ball_dp
        self.q = q
        self.lam = mot.L * i1
        self.time += h
        return StepInfo(
            current=i1,
            torque=tau_motor,
            contact_force=f_contact,
            contact_active=normal_on,
            slipping=slip_mask(normal_on, stick_on),
        )

    # ------------------------------------------------------- the pH matrices
    def ph_matrices(self, b: int = 0, volts: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Explicit J(x), D(x), G, dH/dx and x_dot for robot b (continuous time).

        Valid in the smooth regime (contact points sticking, no cone/clip
        events).  dH/dx comes from autograd of H(x), not from the dynamics,
        so comparing (J - D) dH/dx + G v with `rhs_direct` is a real test.
        Uses the shared parameters only (not robot_params).
        """
        if self.robot_params is not None:
            raise ValueError("ph_matrices uses the shared contact/motor parameters; set robot_params = None first")
        if self.self_contact.enabled:
            raise ValueError("ph_matrices does not cover self-contact; use a plant with self_contact off")
        m, mot, c, lim = self.model, self.motors, self.contact, self.limits
        dt, dev = self.dtype, self.device
        q0, p0, xi0, lam0 = self.q[b : b + 1], self.p[b : b + 1], self.xi[b : b + 1], self.lam[b : b + 1]
        nc = m.ncontact
        use_contact = c.enabled and not self.fixed_base
        fr = self.free
        nf = len(fr)

        def H_of(q, p, xi, lam):
            kin = forward_kinematics(m, q)
            Mf = kin.M[:, fr][:, :, fr]
            pf = p[:, fr]
            T = 0.5 * (pf * ph_solve(Mf, pf)).sum()
            V = gravity_potential(m, kin).sum() if self.gravity else 0.0
            if use_contact:
                delta = torch.relu(-self._sole_points(kin)[..., 2])
                V = V + 0.5 * c.k_n * (delta**2).sum() + 0.5 * c.k_t * (xi**2).sum()
            if lim.enabled:
                th = q[:, 7:]
                V = V + 0.5 * lim.k * ((torch.relu(th - m.stop_high) ** 2) + torch.relu(m.stop_low - th) ** 2).sum()
            return T + V + (lam**2 / (2 * mot.L)).sum()

        xs = [t.clone().requires_grad_(True) for t in (q0, p0, xi0, lam0)]
        Hval = H_of(*xs)
        grads = torch.autograd.grad(Hval, xs, allow_unused=True)
        gq, gp, gxi, glam = [torch.zeros_like(x) if g is None else g for g, x in zip(grads, xs)]
        gq, gp, gxi, glam = gq[0], gp[0, fr], gxi[0].reshape(-1), glam[0]

        kin = forward_kinematics(m, q0)
        nq = NQ if not self.fixed_base else NJ
        # S_q : q_dot = S_q nu  (rows: q coordinates that move, cols: free dofs)
        Sq = torch.zeros(nq, nf, dtype=dt, device=dev)
        if self.fixed_base:
            Sq[:, :] = torch.eye(NJ, dtype=dt)
            gq_used = gq[7:]
        else:
            Sq[0:3, 3:6] = kin.R[0, 0]
            Sq[3:7, 0:3] = quat_rate_matrix(q0[0, 3:7])
            Sq[7:, 6:] = torch.eye(NJ, dtype=dt)
            gq_used = gq
        nx_xi = 2 * nc if use_contact else 0
        n = nq + nf + nx_xi + NJ
        iq, ip, ix, il = 0, nq, nq + nf, nq + nf + nx_xi
        J = torch.zeros(n, n, dtype=dt, device=dev)
        D = torch.zeros(n, n, dtype=dt, device=dev)
        G = torch.zeros(n, NJ, dtype=dt, device=dev)
        J[iq:ip, ip:ix] = Sq
        J[ip:ix, iq:ip] = -Sq.T
        if not self.fixed_base:
            J[ip : ip + 6, ip : ip + 6] = gyro_matrix(p0[:, :6])[0]
        i_n = lam0[0] / mot.L
        Ks = (mot.gyrator_scale(i_n) * mot.k) if self.motors_connected else torch.zeros(NJ, dtype=dt)
        jr = self.jrows  # hinge positions inside the free dofs
        Bm = torch.zeros(nf, NJ, dtype=dt, device=dev)
        Bm[jr, torch.arange(NJ)] = 1.0
        J[ip:ix, il:] = Bm * Ks
        J[il:, ip:ix] = -(Bm * Ks).T
        Dp = torch.zeros(nf, nf, dtype=dt, device=dev)
        Dp[jr, jr] = mot.damping
        if lim.enabled:
            th = q0[0, 7:]
            on = ((th > m.stop_high) | (th < m.stop_low)).to(dt)
            Dp[jr, jr] += on * lim.c
        if use_contact:
            pc = self._sole_points(kin)
            active = (-pc[0, :, 2] > 0).to(dt)
            Jc = point_jacobian(kin, m.contact_body, pc)[0][:, :, fr]  # (nc,3,nf)
            Jt = (Jc[:, :2] * active[:, None, None]).reshape(2 * nc, nf)
            J[ix:il, ip:ix] = Jt
            J[ip:ix, ix:il] = -Jt.T
            Dp += c.c_n * torch.einsum("c,cd,ce->de", active, Jc[:, 2], Jc[:, 2])
            Dp += c.c_t * Jt.T @ Jt
        D[ip:ix, ip:ix] = Dp
        D[il:, il:] = torch.diag(mot.R)
        G[il:, :] = torch.eye(NJ, dtype=dt)
        grad = torch.cat((gq_used, gp, gxi if use_contact else gxi[:0], glam))
        v = torch.zeros(NJ, dtype=dt) if volts is None else torch.as_tensor(volts, dtype=dt).reshape(NJ)
        rhs = (J - D) @ grad + G @ v

        # the same x_dot written out term by term (what step() integrates)
        nu = torch.zeros(NV, dtype=dt, device=dev)
        nu[fr] = ph_solve(kin.M[0][fr][:, fr], p0[0, fr])
        Fp = torch.zeros(NV, dtype=dt, device=dev)
        if not self.fixed_base:
            Fp[:NB] += base_gyroscopic(p0, nu[None])[0]
        Fp[NB:] += kinetic_gradient(m, kin, nu[None])[0]
        if self.gravity:
            Fp += generalized_force(kin, gravity_wrenches(m, kin))[0]
        th = q0[0, 7:]
        if lim.enabled:
            on = ((th > m.stop_high) | (th < m.stop_low)).to(dt)
            Fp[NB:] += lim.k * ((m.stop_low - th).clamp_min(0) - (th - m.stop_high).clamp_min(0)) - on * lim.c * nu[NB:]
        Fp[NB:] += Ks * i_n - mot.damping * nu[NB:]
        xi_dot = torch.zeros(0, dtype=dt)
        if use_contact:
            Jcf = point_jacobian(kin, m.contact_body, pc)[0]
            vc = Jcf @ nu
            delta = (-pc[0, :, 2]).clamp_min(0)
            f = torch.zeros(nc, 3, dtype=dt)
            f[:, 2] = (c.k_n * delta - c.c_n * vc[:, 2]) * active
            f[:, :2] = (-c.k_t * xi0[0] - c.c_t * vc[:, :2]) * active[:, None]
            Fp += torch.einsum("cid,ci->d", Jcf, f)
            xi_dot = (vc[:, :2] * active[:, None]).reshape(-1)
        if self.fixed_base:
            qdot = nu[NB:]
        else:
            qdot = torch.cat(((kin.R[0, 0] @ nu[3:6]), quat_rate_matrix(q0[0, 3:7]) @ nu[:3], nu[NB:]))
        lam_dot = v - mot.R * i_n - Ks * nu[NB:]
        rhs_direct = torch.cat((qdot, Fp[fr], xi_dot, lam_dot))
        return {"x_grad": grad, "J": J, "D": D, "G": G, "rhs": rhs, "rhs_direct": rhs_direct,
                "y": G.T @ grad, "H": Hval.detach()}


def _per_robot(v, ndim: int = 1):
    """A per-robot (B,1) parameter as (B,) (ndim 1) or (B,1,1) (ndim 3); a shared float stays a float."""
    if not isinstance(v, torch.Tensor):
        return v
    return v.reshape(-1, *([1] * (ndim - 1)))


def slip_mask(normal_on: torch.Tensor, stick_on: torch.Tensor) -> torch.Tensor:
    return normal_on & ~stick_on
