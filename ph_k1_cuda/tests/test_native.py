"""The C++ plant (K1PHPlantNative) against the Python references (K1PHPlant, firmware.K1Firmware), step by step.

    cd ph_k1_cuda && python setup.py build_ext --inplace
    python -m pytest ph_k1_cuda/tests -q                     # from the repo root
    PH_TEST_DEVICE=cuda python -m pytest ph_k1_cuda/tests -q # on a GPU machine (Kaggle, Colab)

float64 cases must agree to round-off (one step ~1e-13, long trajectories with contact switching
~1e-8).  The command-path cases (firmware filter, delay line, motor driver, IMU) are checked against
ph_k1/firmware.py and against what the real K1 logged; the float32 case checks the physics: the servo
holds the stand pose and the energy ledger stays balanced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ph_k1.firmware import MAX_TARGET_VELOCITY, FirmwareConfig, K1Firmware  # noqa: E402
from ph_k1.model import MEASURED_RANGE, NB, NJ, load_k1  # noqa: E402
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
    span = m.stop_high - m.stop_low
    py.q[:, 7:] = m.stop_low - 0.05 * span + 1.1 * span * torch.rand(B, NJ, dtype=torch.float64, device=DEVICE)
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


def _command_pair(model, cfg, B, lags):
    py, nat = pair(model, batch=B)
    py.reset()
    nat.firmware = cfg
    nat.reset()
    ref = K1Firmware(py, cfg)
    ref.lag = torch.as_tensor(lags, device=DEVICE).long()
    nat.lag = torch.as_tensor(lags, device=DEVICE).int()
    return py, nat, ref


def _same_command_state(ref, nat, a, b, tol, what):
    for name, x, y in (("last_target", ref.last_target, nat.last_target), ("held target", a.extra["target"], b.extra["target"]),
                       ("gyro", a.extra["gyro"], b.extra["gyro"]), ("acc", a.extra["acc"], b.extra["acc"]),
                       ("torque", a.torque, b.torque)):
        err = (x - y).abs().max().item()
        assert err < tol, f"{what} {name}: {err:.3e}"


@pytest.mark.parametrize("switches", ["on", "off"])
def test_command_path_matches_reference(model64, switches):
    """policy_step (firmware filter + delay line + motor driver in C++) == firmware.K1Firmware in Python,
    with a different delay per robot and targets far enough away that clamp and slew act."""
    torch.manual_seed(5)
    cfg = FirmwareConfig() if switches == "on" else FirmwareConfig(pos_clamp=False, slew=False, target_clamp=False,
                                                                   overspeed_guard=False)
    py, nat, ref = _command_pair(model64, cfg, 4, [0, 7, 23, 40])
    target = py.theta + 0.6 * torch.randn(4, NJ, dtype=torch.float64, device=DEVICE)
    for it in range(4):
        a, b = ref.policy_step(target), nat.policy_step(target)
        same_state(py, nat, 1e-8, f"command path {switches}, policy step {it}")
        _same_command_state(ref, nat, a, b, 1e-8, f"command path {switches}")
    moved = (ref.last_target - target).abs().max().item()
    assert (moved > 0.1) == (switches == "on"), f"filter moved the target by {moved:.3f} rad"


def test_slew_limits_the_sent_target(model64):
    """With slew on, the sent target moves at most max_target_velocity * policy_dt per policy step."""
    cfg = FirmwareConfig(pos_clamp=False, target_clamp=False, overspeed_guard=False, delay_ms=(0.0, 0.0))
    nat = K1PHPlantNative(model=model64, batch=1, firmware=cfg)
    nat.reset()
    step = torch.as_tensor(MAX_TARGET_VELOCITY, dtype=torch.float64, device=DEVICE) * cfg.policy_dt
    prev = nat.last_target.clone()
    for it in range(3):
        nat.policy_step(prev + 1.0)
        d = (nat.last_target - prev)[0]
        assert torch.allclose(d, step, atol=1e-12), f"policy step {it}: moved {d.tolist()}"
        prev = nat.last_target.clone()


def test_target_clamp_and_overspeed_guard(model64):
    """The target clamp keeps the driver's PD torque within the effort limit; a joint faster than its
    max_joint_velocity is not pushed further.  Both against the reference, from a fast-moving state."""
    torch.manual_seed(6)
    cfg = FirmwareConfig(delay_ms=(0.0, 0.0))
    py, nat, ref = _command_pair(model64, cfg, 2, [0, 0])
    kin = py.kinematics()
    nu = torch.zeros(2, NB + NJ, dtype=torch.float64, device=DEVICE)
    nu[:, NB:] = 25.0 * torch.sign(torch.randn(2, NJ, dtype=torch.float64, device=DEVICE))  # past most joints' max speed
    py.p = (kin.M @ nu[..., None]).squeeze(-1)
    nat.p = py.p.clone()
    q, qd = py.theta.clone(), nu[:, NB:].clone()
    target = q + 2.0 * torch.sign(qd)  # push further in the direction of motion
    a, b = ref.policy_step(target, substeps=1), nat.policy_step(target, substeps=1)
    _same_command_state(ref, nat, a, b, 1e-9, "fast state")
    same_state(py, nat, 1e-9, "fast state")
    sent = nat.last_target
    fast = qd.abs() > torch.as_tensor((7.85,) * 2 + (33.51,) * 8 + (14.66, 12.57, 17.59, 12.57, 17.59, 17.59) * 2,
                                      dtype=torch.float64, device=DEVICE)
    assert fast.any() and (~fast).any()
    assert ((sent - q) * torch.sign(qd))[fast].max() <= 1e-12, "overspeed guard: a too fast joint was pushed further"
    kp, kd = nat.gains()
    tau = kp * (sent - q) - kd * qd
    # the overspeed guard comes after the target clamp (safety.cpp): t = q then asks for -kd qdot, which may exceed
    # the effort limit (the motor driver caps it).  Every other joint is within it.
    over = (tau.abs() - model64.forcerange)[~fast]
    assert over.max() <= 1e-9, "target clamp: PD torque beyond the effort limit"


def test_delay_shifts_the_response_exactly(model64):
    """At rest without gravity, a robot with lag d does exactly what a robot with lag 0 did d plant steps earlier."""
    d = 9
    cfg = FirmwareConfig(pos_clamp=False, slew=False, target_clamp=False, overspeed_guard=False)
    nat = K1PHPlantNative(model=model64, batch=2, fixed_base=True, gravity=False, firmware=cfg)
    nat.reset()
    nat.lag = torch.tensor([0, d], dtype=torch.int32, device=DEVICE)
    target = nat.theta.clone()
    target[:, 10] += 0.3  # left hip pitch
    start = nat.theta[0, 10].clone()
    traj = []
    for k in range(40):
        nat.policy_step(target, substeps=1)
        traj.append(nat.theta[:, 10].clone())
    traj = torch.stack(traj)  # (k, robot)
    assert (traj[:d, 1] - start).abs().max() < 1e-12, "the delayed robot moved before its delay was over"
    err = (traj[d:, 1] - traj[:-d, 0]).abs().max().item()
    assert err < 1e-12, f"delayed robot differs from the shifted undelayed one by {err:.2e}"
    assert (traj[-1, 0] - start).abs() > 0.02, "the joint should have moved"


def test_end_stops_contain_the_measured_range(model64):
    """The real K1 went past the MJCF range in its get-ups; the plant's end stops must not push back there."""
    m = model64
    assert (m.stop_low <= m.jnt_low).all() and (m.stop_high >= m.jnt_high).all()
    pl = K1PHPlant(model=m, batch=2)
    pl.reset()
    for b, side in enumerate((0, 1)):
        for j, name in enumerate(m.joint_names):
            pl.q[b, 7 + j] = MEASURED_RANGE[name][side]
    assert (pl.energy()["stops"] == 0).all(), "a measured joint angle lies in an end-stop spring"


def test_imu_standing_lying_falling(model64):
    """Accelerometer/gyro as the robot logs them: standing (0,0,+g), lying on the back (+g,0,0), free fall 0."""
    g = 9.81
    stand = K1PHPlantNative(model=model64, batch=1, firmware=FirmwareConfig(delay_ms=(0.0, 0.0)))
    stand.reset()
    target = stand.theta.clone()
    for it in range(50):
        info = stand.policy_step(target, gain_scale=3.0)
    acc, gyro = info.extra["acc"][0], info.extra["gyro"][0]
    assert (acc - acc.new_tensor([0.0, 0.0, g])).abs().max() < 0.3, f"standing: acc {acc.tolist()}"
    assert gyro.abs().max() < 0.05

    s = 0.5 ** 0.5  # trunk x axis (forward) pointing up: lying on the back, as at the start of getup_2236.csv
    lie = K1PHPlantNative(model=model64, batch=1, firmware=FirmwareConfig(delay_ms=(0.0, 0.0)))
    lie.reset(base_pos=(0.0, 0.0, 0.25), base_quat=(s, 0.0, -s, 0.0))
    target = lie.theta.clone()
    for it in range(100):
        info = lie.policy_step(target)
    acc = info.extra["acc"][0]
    assert acc[0] > 0.9 * g and acc.norm().sub(g).abs() < 0.5, f"lying on the back: acc {acc.tolist()}"

    fall = K1PHPlantNative(model=model64, batch=1, motors_connected=False)
    fall.contact.enabled = False
    fall.reset(base_pos=(0.0, 0.0, 2.0))
    for it in range(20):
        info = fall.step()
    assert info.extra["acc"][0].abs().max() < 1e-6, f"free fall: acc {info.extra['acc'][0].tolist()}"


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
    nat = K1PHPlantNative(model=model, batch=4)  # default command path: firmware filter, 10-40 ms delay
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


def test_reset_of_some_robots_leaves_the_others(model64):
    """RL resets fallen robots only: the other rows (plant state, ledger, command path) stay bit-identical."""
    torch.manual_seed(7)
    nat = K1PHPlantNative(model=model64, batch=4, firmware=FirmwareConfig(delay_ms=(5.0, 30.0)))
    nat.reset()
    target = nat.theta + 0.2 * torch.randn(4, NJ, dtype=torch.float64, device=DEVICE)
    for it in range(5):
        nat.policy_step(target)
    keep = torch.tensor([0, 2], device=DEVICE)
    before = {n: getattr(nat, n)[keep].clone() for n in ("q", "p", "xi", "lam", "last_target", "ring", "ring_idx", "lag")}
    led_before = nat._led[keep].clone()
    pose = model64.pose_vector() + 0.01
    nat.reset(pose=pose, env_ids=torch.tensor([1, 3], device=DEVICE))
    for n, v in before.items():
        assert torch.equal(getattr(nat, n)[keep], v), f"{n} of an untouched robot changed"
    assert torch.equal(nat._led[keep], led_before)
    reset_rows = torch.tensor([1, 3], device=DEVICE)
    assert (nat.theta[reset_rows] - pose).abs().max() == 0
    assert nat.p[reset_rows].abs().max() == 0 and nat._led[reset_rows].abs().max() == 0
    assert (nat.last_target[reset_rows] - pose).abs().max() == 0 and (nat.ring_idx[reset_rows] == 0).all()
    # the reset robots stand on the floor again: lowest sole point at z = 0
    pc = nat._sole_points(nat.kinematics())
    assert pc[reset_rows, :, 2].min(-1).values.abs().max() < 1e-12


def test_push_matches_reference(model64):
    """A shove (world-frame trunk velocity change at the policy step) in C++ == firmware.K1Firmware."""
    torch.manual_seed(8)
    cfg = FirmwareConfig(delay_ms=(0.0, 0.0))
    py, nat, ref = _command_pair(model64, cfg, 3, [0, 4, 9])
    target = py.theta.clone()
    dv = torch.tensor([[0.3, 0.0, 0.0], [0.0, -0.25, 0.05], [0.0, 0.0, 0.0]], dtype=torch.float64, device=DEVICE)
    a, b = ref.policy_step(target, push_dv=dv), nat.policy_step(target, push_dv=dv)
    same_state(py, nat, 1e-9, "push")
    _same_command_state(ref, nat, a, b, 1e-9, "push")
    for it in range(5):
        a, b = ref.policy_step(target), nat.policy_step(target)
    same_state(py, nat, 1e-8, "after the push")
    # the pushed robots move in the push direction, the third one does not
    v = b.extra["nu"][:, 3:6]
    R = nat.kinematics().R[:, 0]
    v_world = (R @ v[..., None]).squeeze(-1)
    assert v_world[0, 0] > 0.05 and v_world[1, 1] < -0.05 and v_world[2].abs().max() < 0.05
