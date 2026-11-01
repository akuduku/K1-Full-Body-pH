#!/usr/bin/env python3
"""Does the full-body pH model do the right physics?  No RL here.

Every check prints PASS/FAIL with the number it measured.

    cd "RL Policy for pH/Full Body"
    ../../Presentation/.venv/bin/python validation/checks.py            # all
    ../../Presentation/.venv/bin/python validation/checks.py mujoco arm # some
    python validation/checks.py --impl native                           # the same checks with the C++ plant

 1  mass      M(q) == MuJoCo mj_fullM, frames == MuJoCo xpos  (random q)
 2  gradient  analytic dT/dtheta == autograd                  (random q, nu)
 3  ph        J = -J^T, D >= 0, (J-D) dH/dx + G v == what step() integrates,
              y = G^T dH/dx = i       (floating base on the ground + welded)
 4  mujoco    accelerations == MuJoCo mj_forward qacc (torque port, no contact)
 5  flight    zero gravity, no losses: momentum and energy errors -> 0
              linearly with h  (first-order integrator)
 6  drop      drop on the floor, 0 V on all coils: energy ledger
              H(t) - H(0) = supplied - dissipated, residual -> 0 with h
 7  motors    torque-speed of every motor type vs the datasheet envelope;
              shorted-coil brake k^2/R
 9  hold      classical servo (volts only) keeps the robot standing 3 s
 8  arm       welded trunk, right arm: same trajectory as the 4-DOF arm plant
              (Presentation/K1/physik/Right Hand Model/plant.py), with the
              arm's MuJoCo contact oracle and clip stops switched off
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]  # the repo root (this script lives one folder down)
sys.path.insert(0, str(HERE))

from ph_k1.dynamics import forward_kinematics, kinetic_gradient  # noqa: E402
from ph_k1.model import NB, NJ, load_k1  # noqa: E402
from ph_k1.motors import DATASHEET, build_motors  # noqa: E402
from ph_k1.plant import K1PHPlant  # noqa: E402

IMPL = "python"
if "--impl" in sys.argv:  # --impl native: every check steps the C++ plant (ph_k1_cuda)
    _i = sys.argv.index("--impl")
    IMPL = sys.argv[_i + 1]
    del sys.argv[_i : _i + 2]
    if IMPL == "native":
        from ph_k1.native import K1PHPlantNative as K1PHPlant  # noqa: E402,F811
    elif IMPL != "python":
        raise SystemExit(f"--impl {IMPL}: use python or native")

ARM_DIR = HERE.parents[1] / "Presentation" / "K1" / "physik" / "Right Hand Model"
RESULTS: list[tuple[str, bool | None, str]] = []
MODEL = load_k1()
torch.manual_seed(0)


def report(name: str, ok: bool, msg: str) -> None:
    RESULTS.append((name, ok, msg))
    tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    print(f"[{tag}] {name:9s} {msg}")


def random_q(B: int) -> torch.Tensor:
    m = MODEL
    q = torch.zeros(B, 29, dtype=torch.float64)
    q[:, :3] = torch.randn(B, 3, dtype=torch.float64)
    qu = torch.randn(B, 4, dtype=torch.float64)
    q[:, 3:7] = qu / qu.norm(dim=-1, keepdim=True)
    q[:, 7:] = m.jnt_low + (m.jnt_high - m.jnt_low) * torch.rand(B, NJ, dtype=torch.float64)
    return q


def nu_to_qvel(R: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """ours (omega_B, v_B, theta_dot) -> MuJoCo (r_dot world, omega_B, theta_dot)."""
    return np.concatenate((R @ nu[3:6], nu[:3], nu[6:]))


# ---------------------------------------------------------------------------
def check_mass():
    mj, d = MODEL.mj, mujoco.MjData(MODEL.mj)
    q = random_q(5)
    kin = forward_kinematics(MODEL, q)
    err_M = err_x = 0.0
    for b in range(q.shape[0]):
        d.qpos[:] = q[b].numpy()
        mujoco.mj_forward(mj, d)
        Mm = np.zeros((28, 28))
        mujoco.mj_fullM(mj, d, Mm)
        R = kin.R[b, 0].numpy()
        A = np.zeros((28, 28))  # qvel = A nu
        A[0:3, 3:6], A[3:6, 0:3], A[6:, 6:] = R, np.eye(3), np.eye(NJ)
        err_M = max(err_M, np.abs(A.T @ Mm @ A - kin.M[b].numpy()).max())
        err_x = max(err_x, np.abs(d.xpos[1:] - kin.pos[b].numpy()).max(), np.abs(d.xipos[1:] - kin.com[b].numpy()).max())
    report("mass", err_M < 1e-10 and err_x < 1e-12, f"max|M - M_mujoco| = {err_M:.1e}, frames {err_x:.1e}")


def check_gradient():
    q = random_q(4)
    nu = torch.randn(4, 28, dtype=torch.float64)
    th = q[:, 7:].clone().requires_grad_(True)
    kin = forward_kinematics(MODEL, torch.cat((q[:, :7], th), 1))
    T = 0.5 * torch.einsum("bd,bde,be->b", nu, kin.M, nu).sum()
    (g,) = torch.autograd.grad(T, th)
    an = kinetic_gradient(MODEL, forward_kinematics(MODEL, q), nu)
    err = (g - an).abs().max().item() / g.abs().max().item()
    report("gradient", err < 1e-12, f"relative error of analytic dT/dtheta = {err:.1e}")


def check_ph():
    msgs, ok = [], True
    for fixed in (False, True):
        pl = K1PHPlant(model=MODEL, fixed_base=fixed)
        pl.reset(clearance=-0.002)  # soles 2 mm into the floor -> contacts active
        pl.p = 0.3 * torch.randn_like(pl.p)
        pl.lam = 1e-3 * torch.randn_like(pl.lam)
        if not fixed:
            pl.xi = 1e-4 * torch.randn_like(pl.xi)
        out = pl.ph_matrices(volts=torch.randn(NJ, dtype=torch.float64))
        J, D = out["J"], out["D"]
        skew = (J + J.T).abs().max().item()
        dmin = torch.linalg.eigvalsh(D).min().item()
        rel = ((out["rhs"] - out["rhs_direct"]).abs().max() / out["rhs"].abs().max()).item()
        yerr = (out["y"] - pl.current[0]).abs().max().item()
        ok &= skew == 0 and dmin > -1e-10 and rel < 1e-12 and yerr < 1e-12
        msgs.append(f"{'welded' if fixed else 'floating'}: |J+J^T|={skew:.0e} min eig D={dmin:.0e} rhs rel err={rel:.0e}")
    report("ph", ok, "; ".join(msgs))


def check_mujoco():
    """Instantaneous accelerations under a joint torque, against MuJoCo."""
    mj = MODEL.mj
    mj.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    d = mujoco.MjData(mj)
    pl = K1PHPlant(model=MODEL, motors_connected=False)
    pl.contact.enabled = False
    pl.limits.enabled = False
    worst = 0.0
    for trial in range(3):
        q = random_q(1)
        pl.q = q
        kin = forward_kinematics(MODEL, q)
        nu = torch.randn(1, 28, dtype=torch.float64)
        pl.p = (kin.M @ nu[..., None]).squeeze(-1)
        tau = 2.0 * torch.randn(1, NJ, dtype=torch.float64)
        # our p_dot from the pH right-hand side
        out = pl.ph_matrices()
        pdot = out["rhs_direct"][29:57].clone()
        pdot[NB:] += tau[0]
        # M nu_dot = p_dot - Mdot nu   (Mdot nu by central difference along the flow)
        eps = 1e-6

        def moved(s):
            qq = q.clone()
            qq[:, 7:] += s * nu[:, 6:]
            return forward_kinematics(MODEL, qq).M[0]

        Mdot_nu = (moved(eps) - moved(-eps)) @ nu[0] / (2 * eps)
        nudot = torch.linalg.solve(kin.M[0], pdot - Mdot_nu).numpy()
        R = kin.R[0, 0].numpy()
        n = nu[0].numpy()
        qacc_ours = np.concatenate((R @ (np.cross(n[:3], n[3:6]) + nudot[3:6]), nudot[:3], nudot[6:]))
        d.qpos[:] = q[0].numpy()
        d.qvel[:] = nu_to_qvel(R, n)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[6:] = tau[0].numpy() - pl.motors.damping.numpy() * n[6:]
        mujoco.mj_forward(mj, d)
        worst = max(worst, np.abs(qacc_ours - d.qacc).max() / np.abs(d.qacc).max())
    mj.opt.disableflags = 0
    report("mujoco", worst < 1e-6, f"relative error of accelerations vs MuJoCo = {worst:.1e}")


def check_flight():
    errs = []
    for h in (1e-3, 5e-4):
        pl = K1PHPlant(model=MODEL, timestep=h, motors_connected=False, gravity=False)
        pl.contact.enabled = False
        pl.limits.enabled = False
        pl.motors.damping.zero_()
        pl.reset(base_pos=(0, 0, 1.0))
        kin = pl.kinematics()
        nu = 0.5 * torch.randn(1, 28, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        pl.p = (kin.M @ nu[..., None]).squeeze(-1)

        def world_momentum():
            k = pl.kinematics()
            V = torch.einsum("bnid,bd->bni", k.J, pl.velocity(k))
            return torch.einsum("bnij,bnj->bi", k.I, V).sum(0)

        h0, E0 = world_momentum(), pl.energy()["total"].item()
        for _ in range(int(0.5 / h)):
            pl.step()
        dh = (world_momentum() - h0).abs().max().item()
        errs.append((h, dh, abs(pl.energy()["total"].item() - E0) / E0))
    # first-order integrator: both errors must halve with h
    ok = errs[1][1] < 0.7 * errs[0][1] and errs[1][2] < 0.7 * errs[0][2] and errs[1][2] < 1e-4
    report("flight", ok, " ; ".join(f"h={h:g}: |dMomentum|={dh:.0e}, |dE|/E={de:.1e}" for h, dh, de in errs))


def check_drop():
    rows = []
    for h in (1e-3, 5e-4):
        pl = K1PHPlant(model=MODEL, timestep=h)
        pl.reset(clearance=0.05)
        E0 = pl.energy()["total"].item()
        for _ in range(int(1.0 / h)):
            info = pl.step()
        E = pl.energy()
        led = pl.ledger
        res = E["total"].item() - E0 - (led.supplied.item() - led.dissipated.item())
        rows.append((h, res, led.dissipated.item(), pl.q[0, 2].item(), info.contact_force[0, :, 2].sum().item()))
    ok = abs(rows[1][1]) < 0.7 * abs(rows[0][1]) and abs(rows[1][1]) < 0.05 * rows[1][2]
    mg = MODEL.total_mass * 9.81
    msg = " ; ".join(f"h={h:g}: residual={r:+.3f} J of {dis:.2f} J dissipated" for h, r, dis, _, _ in rows)
    report("drop", ok, msg + f" ; after 1 s trunk z={rows[1][3]:.3f} m, sum Fz={rows[1][4]:.1f} N (mg={mg:.1f})")


def check_motors():
    mot = build_motors(MODEL)
    worst, msgs = 0.0, []
    for t, (effort, w_max, w_knee) in DATASHEET.items():
        j = mot.types.index(t)
        k, R, cap = mot.k[j].item(), mot.R[j].item(), mot.tau_max[j].item()
        for w in np.linspace(0, w_max, 9):
            i = (mot.v_max - k * w) / R  # steady current at full voltage
            tau = mot.torque(torch.tensor([i], dtype=torch.float64) * torch.ones(NJ, dtype=torch.float64))[j].item()
            line = effort * (w_max - w) / (w_max - w_knee) if w_knee < w_max else effort * (1 - w / w_max)
            want = min(cap, effort, line) if w_knee < w_max else min(cap, line)
            worst = max(worst, abs(tau - want) / effort)
        msgs.append(f"{t}: k={k:.3f} R={R:.4f} brake k^2/R={k * k / R:.2f}")
    report("motors", worst < 1e-9, f"max envelope error {worst:.1e} (ceiling = MJCF forcerange) | " + ", ".join(msgs))


def check_arm():
    if not (ARM_DIR / "plant.py").exists():
        report("arm", None, f"arm plant not found at {ARM_DIR} (only in the thesis folder)")
        return
    sys.path.insert(0, str(ARM_DIR))
    import plant as arm_ph  # noqa: E402

    am = arm_ph.load_model()
    ad = mujoco.MjData(am)
    arm = arm_ph.RightArmPlant(am, ad)
    # Compare the pH EQUATIONS: switch off what the two plants model
    # differently -- the arm's MuJoCo self-contact oracle and its clip stops
    # (ours: no self-contact yet, spring stops).
    arm.oracle.torque = lambda q, qd, tau: np.zeros(arm_ph.N)
    arm.chain.limit_low = np.full(arm_ph.N, -50.0)
    arm.chain.limit_high = np.full(arm_ph.N, 50.0)
    h = arm.timestep
    start = np.array([-0.5, 0.9, 0.6, 1.2])
    arm.state[arm_ph.SLICE_Q] = start

    pl = K1PHPlant(model=MODEL, fixed_base=True, timestep=h)
    pl.limits.enabled = False
    names = arm_ph.RIGHT_ARM
    idx = [MODEL.joint_index(n) for n in names]
    pl.reset(pose=dict(zip(names, start)))
    pl.motors.damping[:] = arm_ph.JOINT_DAMPING
    worst_q = worst_c = 0.0
    for s in range(int(1.0 / h)):
        t = s * h
        va = np.array([0.6 * np.sin(2 * np.pi * t), 0.35 * np.sin(3 * t) + 0.25,
                       0.5 * np.sin(4 * t), -0.3 * np.sin(5 * t)])
        arm.step(va)
        v = torch.zeros(1, NJ, dtype=torch.float64)
        v[0, idx] = torch.as_tensor(va)
        pl.step(v)
        worst_q = max(worst_q, np.abs(pl.theta[0, idx].numpy() - arm.q).max())
        worst_c = max(worst_c, np.abs(pl.current[0, idx].numpy() - arm.current).max())
    report("arm", worst_q < np.deg2rad(1.0),
           f"1 s voltage drive at h={h * 1e3:g} ms: max |q - q_arm| = {np.rad2deg(worst_q):.3f} deg, "
           f"max |i - i_arm| = {worst_c:.3f} A (residue = integrator difference; at h=0.5 ms it is 0.16 deg)")


def check_hold():
    from ph_k1.drive import K1Drive

    pl = K1PHPlant(model=MODEL)
    pl.reset()
    drive = K1Drive(pl, gain_scale=3.0)
    drive.reset()
    target = pl.theta.clone()
    E0 = pl.energy()["total"].item()
    for _ in range(3000):
        info = pl.step(drive.volts(target))
    R = pl.kinematics().R[0, 0]
    tilt = float(torch.rad2deg(torch.acos(R[2, 2].clamp(-1, 1))))
    led = pl.ledger
    res = pl.energy()["total"].item() - E0 - (led.supplied.item() - led.dissipated.item())
    soles_only = bool((info.contact_active[0] <= MODEL.contact_is_sole).all())
    Fz = info.contact_force[0, :, 2].sum().item()
    mg = MODEL.total_mass * 9.81
    ok = tilt < 5 and soles_only and abs(Fz - mg) < 0.02 * mg and abs(res) < 0.1
    report("hold", ok, f"servo (gain x3) holds the stand pose 3 s on volts only: tilt={tilt:.2f} deg, "
           f"sum Fz={Fz:.1f} N vs mg={mg:.1f} N, only soles touch={soles_only}, energy residual={res:+.3f} J")


CHECKS = {
    "mass": check_mass,
    "gradient": check_gradient,
    "ph": check_ph,
    "mujoco": check_mujoco,
    "flight": check_flight,
    "drop": check_drop,
    "motors": check_motors,
    "arm": check_arm,
    "hold": check_hold,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(CHECKS)
    for name in wanted:
        CHECKS[name]()
    bad = [n for n, ok, _ in RESULTS if ok is False]
    ran = [n for n, ok, _ in RESULTS if ok is not None]
    print(f"\n{len(ran) - len(bad)}/{len(ran)} passed" + (f"; failed: {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)
