"""The C++ plant (K1PHPlantNative) against the Python reference (K1PHPlant), step by step.

    cd "Physik C++" && python setup.py build_ext --inplace
    python -m pytest "Physik C++/tests" -q                     # from the repo root
    PH_TEST_DEVICE=cuda python -m pytest "Physik C++/tests" -q # on a GPU machine (Kaggle, Colab)

float64 cases must agree to round-off (one step ~1e-13, long trajectories with contact switching
~1e-8).  The float32 case checks the physics instead: the servo holds the stand pose and the energy
ledger stays balanced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ph_k1.drive import K1Drive  # noqa: E402
from ph_k1.model import NJ, load_k1  # noqa: E402
from ph_k1.native import K1PHPlantNative  # noqa: E402
from ph_k1.plant import K1PHPlant  # noqa: E402

DEVICE = os.environ.get("PH_TEST_DEVICE", "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    pytest.skip("PH_TEST_DEVICE=cuda but no GPU", allow_module_level=True)


@pytest.fixture(scope="module")
def model64():
    return load_k1(dtype=torch.float64, device=DEVICE)


def pair(model, **kw):
    return K1PHPlant(model=model, **kw), K1PHPlantNative(model=model, **kw)


def same_state(py, nat, tol, what=""):
    for name in ("q", "p", "xi", "lam"):
        err = (getattr(py, name) - getattr(nat, name)).abs().max().item()
        assert err < tol, f"{what} {name}: max |python - native| = {err:.3e} >= {tol:.0e}"
    for f in ("supplied", "mech_input", "copper", "gearbox", "stops", "contact"):
        err = (getattr(py.ledger, f) - getattr(nat.ledger, f)).abs().max().item()
        assert err < tol, f"{what} ledger.{f}: {err:.3e}"


def copy_state(src, dst):
    for name in ("q", "p", "xi", "lam"):
        setattr(dst, name, getattr(src, name).clone())


def test_one_step_with_contacts(model64):
    """Random momenta/currents/bristles with the soles 2 mm in the floor: stick, slip and lift-off."""
    torch.manual_seed(1)
    B = 8
    py, nat = pair(model64, batch=B)
    py.reset(clearance=-0.002)
    py.p = 0.3 * torch.randn_like(py.p)
    py.lam = 1e-3 * torch.randn_like(py.lam)
    py.xi = 1e-4 * torch.randn_like(py.xi)
    copy_state(py, nat)
    nat.q_star = py.theta.clone()
    v = torch.randn(B, NJ, dtype=torch.float64, device=DEVICE)
    a, b = py.step(v), nat.step(v)
    same_state(py, nat, 1e-10, "one step")
    assert torch.equal(a.contact_active, b.contact_active)
    assert torch.equal(a.slipping, b.slipping)
    assert (a.current - b.current).abs().max() < 1e-10
    assert (a.contact_force - b.contact_force).abs().max() < 1e-9
    assert a.slipping.any() and a.contact_active.any(), "the case should exercise slip and stick"


def test_drop_trajectory(model64):
    """Dropped from 5 cm with 0 V (coils shorted): impact, bouncing contacts, energy ledger."""
    py, nat = pair(model64, batch=2)
    for pl in (py, nat):
        pl.reset(clearance=0.05)
    for s in range(300):
        py.step()
        nat.step()
    same_state(py, nat, 1e-8, "drop 300 steps")


def test_fixed_base_torque_port_and_end_stops(model64):
    """Welded trunk, motors disconnected, random test torques, hinges started past their end stops."""
    torch.manual_seed(2)
    B = 4
    py, nat = pair(model64, batch=B, fixed_base=True, motors_connected=False)
    py.reset()
    m = model64
    span = m.jnt_high - m.jnt_low
    py.q[:, 7:] = m.jnt_low - 0.05 * span + 1.1 * span * torch.rand(B, NJ, dtype=torch.float64, device=DEVICE)
    copy_state(py, nat)
    for s in range(100):
        tau = 2.0 * torch.randn(B, NJ, dtype=torch.float64, device=DEVICE)
        py.step(joint_torque=tau)
        nat.step(joint_torque=tau)
    same_state(py, nat, 1e-8, "fixed base")
    assert py.ledger.stops.abs().max() > 0, "end stops should have been hit"


def test_flight_without_gravity(model64):
    """No gravity, contact, limits or motors: the free-floating gyroscopic dynamics alone."""
    torch.manual_seed(3)
    py, nat = pair(model64, batch=3, motors_connected=False, gravity=False)
    for pl in (py, nat):
        pl.contact.enabled = False
        pl.limits.enabled = False
        pl.reset(base_pos=(0.0, 0.0, 1.0))
    py.p = 0.5 * torch.randn_like(py.p)
    copy_state(py, nat)
    for s in range(200):
        py.step()
        nat.step()
    same_state(py, nat, 1e-9, "flight")


def test_saturated_motors(model64):
    """Full voltage on every coil: torques hit the force range (gyrator scale s < 1)."""
    torch.manual_seed(4)
    py, nat = pair(model64, batch=2)
    for pl in (py, nat):
        pl.reset()
    v = 7.0 * torch.sign(torch.randn(2, NJ, dtype=torch.float64, device=DEVICE))
    hit = False
    for s in range(60):
        a = py.step(v)
        nat.step(v)
        hit = hit or bool(((model64.forcerange - a.torque.abs()) < 1e-9).any())
    same_state(py, nat, 1e-7, "saturated")
    assert hit, "some motor should have reached its torque ceiling"


def test_servo_policy_step_matches_drive(model64):
    """policy_step (servo + 20 plant steps in C++) == K1Drive.volts + K1PHPlant.step in Python."""
    torch.manual_seed(5)
    B = 3
    py, nat = pair(model64, batch=B)
    for pl in (py, nat):
        pl.reset()
    drive = K1Drive(py, gain_scale=3.0)
    drive.reset()
    target = py.theta + 0.2 * torch.randn(B, NJ, dtype=torch.float64, device=DEVICE)
    for it in range(3):
        for s in range(20):
            py.step(drive.volts(target))
        nat.policy_step(target, substeps=20, gain_scale=3.0)
    same_state(py, nat, 1e-8, "servo")
    assert (drive.q_star - nat.q_star).abs().max() < 1e-12


def test_parameter_change_is_picked_up(model64):
    """Changing contact/motor parameters after construction reaches the C++ side."""
    py, nat = pair(model64, batch=2)
    for pl in (py, nat):  # each plant has its own ContactParams and MotorBank
        pl.reset(clearance=-0.002)
        pl.contact.mu = 0.2
        pl.motors.damping.mul_(3.0)
    v = torch.full((2, NJ), 0.5, dtype=torch.float64, device=DEVICE)
    for s in range(50):
        py.step(v)
        nat.step(v)
    same_state(py, nat, 1e-8, "parameters")


def test_float32_servo_hold():
    """float32 (the training type): the gain-3 servo holds the stand pose, the ledger balances."""
    model = load_k1(dtype=torch.float32, device=DEVICE)
    nat = K1PHPlantNative(model=model, batch=4)
    nat.reset()
    target = nat.theta.clone()
    H0 = nat.energy()["total"].clone()
    for it in range(100):  # 2 s
        nat.policy_step(target, gain_scale=3.0)
    q = nat.q
    tilt = torch.rad2deg(torch.arccos((1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).clamp(-1, 1)))
    led = nat.ledger
    residual = (nat.energy()["total"] - H0) - (led.supplied - led.dissipated)
    assert torch.isfinite(q).all()
    assert tilt.max() < 1.0, f"tilt {tilt.max():.2f} deg"
    assert (q[:, 2] - 0.515).abs().max() < 0.01
    assert residual.abs().max() < 0.05, f"ledger residual {residual.abs().max():.3f} J"
