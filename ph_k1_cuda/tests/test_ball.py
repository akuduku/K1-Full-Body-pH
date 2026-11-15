"""The ball (ph_k1/ball.py): the C++ step against the Python reference, and the ball's physics and energy ledger.

    python -m pytest ph_k1_cuda/tests/test_ball.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ph_k1.ball import BallParams, build_boxes  # noqa: E402
from ph_k1.model import load_k1  # noqa: E402
from ph_k1.native import K1PHPlantNative  # noqa: E402
from ph_k1.plant import K1PHPlant  # noqa: E402

DEVICE = os.environ.get("PH_TEST_DEVICE", "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    pytest.skip("PH_TEST_DEVICE=cuda but no GPU", allow_module_level=True)
BALL = BallParams(enabled=True)


@pytest.fixture(scope="module")
def model():
    return load_k1(dtype=torch.float64, device=DEVICE)


def balance(pl):
    """H + dissipated - supplied - mechanical input: constant if every joule is booked."""
    led = pl.ledger
    return pl.energy()["total"] + led.dissipated - led.supplied - led.mech_input


def test_boxes_are_feet_and_shanks(model):
    boxes = build_boxes(model)
    assert [model.body_names[int(b)] for b in boxes.body] == ["left_foot_link", "right_foot_link", "Left_Shank",
                                                              "Right_Shank"]


def test_native_equals_reference_with_ball(model):
    """Balls shot at the feet and shanks, one dropped, one rolling past: 300 steps, both steps agree."""
    B = 6
    T = lambda x: torch.tensor(x, dtype=torch.float64, device=DEVICE)  # noqa: E731
    xy = T([[0.35, 0.085], [0.35, -0.085], [0.30, 0.0], [0.2, 0.3], [1.0, 0.0], [0.25, 0.10]])
    vel = T([[-4.0, 0, 0], [-4.0, 0, 0], [-3.0, 0.2, 0], [-1.0, -3.0, 0], [2.0, 1.0, 0], [-6.0, 0.5, 0.0]])
    ref = K1PHPlant(model=model, batch=B, ball=BALL)
    nat = K1PHPlantNative(model=model, batch=B, ball=BALL)
    for pl in (ref, nat):
        pl.set_ball(xy, vel)
        pl.ball[4, 2] = 0.4
    torch.manual_seed(0)
    volts = 2 * torch.randn(B, 22, dtype=torch.float64, device=DEVICE)
    touched = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    for _ in range(300):
        ref.step(volts)
        nat.step(volts)
        touched |= (ref.ball_force > 0).any(-1)
    assert touched[:4].all() and not touched[4]          # the test does hit the robot
    assert (ref.q - nat.q).abs().max() < 1e-9
    assert (ref.ball - nat.ball).abs().max() < 1e-8
    assert (ref.ledger.contact - nat.ledger.contact).abs().max() < 1e-8


def test_plant_without_ball_is_unchanged(model):
    """A parked ball changes nothing for the robot; with the ball off its state is not touched."""
    torch.manual_seed(1)
    volts = 2 * torch.randn(2, 22, dtype=torch.float64, device=DEVICE)
    off = K1PHPlantNative(model=model, batch=2)
    on = K1PHPlantNative(model=model, batch=2, ball=BALL)
    ball0 = off.ball.clone()
    for _ in range(100):
        off.step(volts)
        on.step(volts)
    assert torch.equal(off.ball, ball0)
    assert (off.q - on.q).abs().max() < 1e-10
    assert (on.ball[:, :3] - ball0[:, :3]).abs().max() < 1e-6   # it stays where it is parked


def test_drop_bounces_and_comes_to_rest(model):
    pl = K1PHPlantNative(model=model, batch=1, ball=BALL, fixed_base=True)
    pl.set_ball((1.0, 0.0))
    pl.ball[:, 2] = 0.5
    e0 = balance(pl)
    top, falling = 0.0, True
    for _ in range(1500):
        z0 = float(pl.ball[0, 2])
        pl.step()
        if float(pl.ball[0, 5]) > 0:
            falling = False
        if not falling:
            top = max(top, z0)
    assert 0.09 < top < 0.35                                       # it bounces, lower than the drop
    assert abs(float(pl.ball[0, 2]) - BALL.rest_height()) < 1e-4   # and rests on the ground spring
    assert abs(float(balance(pl) - e0)) < 0.02 * float(pl.ledger.contact)


def test_rolling_and_sliding(model):
    """A rolling ball slows down by 0.6 * roll * g (hollow sphere); a sliding one rolls from 3/5 of its speed on
    (after (3 - 1.8) / (mu g) = 0.2 s) and then slows down in the same way."""
    pl = K1PHPlantNative(model=model, batch=2, ball=BALL, fixed_base=True)
    pl.set_ball((1.0, 0.0), (3.0, 0.0, 0.0))
    pl.ball[1, 6:] = 0.0                                           # robot 1: no spin, it slides first
    e0 = balance(pl)
    for _ in range(1000):
        pl.step()
    u, om = pl.ball[:, 3], pl.ball[:, 7]
    assert (u - BALL.radius * om).abs().max() < 1e-6               # both roll without slip
    decel = 0.6 * BALL.roll * 9.81
    assert abs(float(u[0]) - (3.0 - decel)) < 0.01
    assert 1.1 < float(u[1]) < 1.8 - 0.8 * decel   # below the estimate: the rolling resistance also acts while it slides
    assert (balance(pl) - e0).abs().max() < 0.01


def test_impact_on_the_foot_is_booked(model):
    """A ball at 4 m/s into the toes of a free, unpowered robot: it comes back slower, the robot got the difference
    in momentum, no step books a negative loss and the energy balance holds."""
    pl = K1PHPlantNative(model=model, batch=1, ball=BALL)
    pl.set_ball((0.35, 0.085), (-4.0, 0.0, 0.0))
    e0 = balance(pl)
    least = 0.0
    for _ in range(300):
        c0 = pl.ledger.contact.clone()
        pl.step()
        least = min(least, float(pl.ledger.contact - c0))
    assert 0.5 < float(pl.ball[0, 3]) < 4.0
    assert least > -1e-9
    assert abs(float(balance(pl) - e0)) < 0.05


def test_float32_ball(model):
    m32 = load_k1(dtype=torch.float32, device=DEVICE)
    pl = K1PHPlantNative(model=m32, batch=4, ball=BALL)
    pl.set_ball((0.35, 0.085), (-4.0, 0.0, 0.0))
    for _ in range(300):
        pl.step()
    assert torch.isfinite(pl.ball).all() and torch.isfinite(pl.q).all()
    assert (pl.ball[:, 3] > 0.5).all()
