"""The walk: contract == k1_deploy's sprint agent, the firmware's ramp, a walk on both engines, the slider panel.

    python -m pytest rl/walk/tests -q                       (needs the built ph_k1_native)
    PH_TEST_DISPLAY=1 python -m pytest rl/walk/tests -q     (also opens the slider panel for a moment)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DEPLOY = REPO.parent / "K1_Deploy"

from ph_k1.model import load_k1  # noqa: E402
from rl.walk import contract as C  # noqa: E402
from rl.walk.backends import CHECKPOINTS, MuJoCoBackend, PHBackend, Walker, load_walk_policy  # noqa: E402


def deploy_sprint():
    if not DEPLOY.exists():
        pytest.skip("K1_Deploy is not next to this repository")
    if str(DEPLOY) not in sys.path:
        sys.path.insert(0, str(DEPLOY))
    from k1_deploy.agents import sprint

    return sprint


def test_contract_matches_k1_deploy():
    sprint = deploy_sprint()
    np.testing.assert_allclose(C.HOME, sprint.HOME)
    np.testing.assert_allclose(C.ACTION_SCALE, sprint.ACTION_SCALE, rtol=1e-6)
    np.testing.assert_allclose(C.TWIST_HIGH, sprint.TWIST.hi)
    assert C.OBS_DIM == sprint.OBS_DIM


def test_observation_and_decode_match_k1_deploy():
    sprint = deploy_sprint()
    from k1_deploy.agents.base import RobotState
    from k1_deploy.contract import projected_gravity

    rng = np.random.default_rng(0)
    q, dq = rng.normal(size=22), rng.normal(size=22)
    gyro, rpy = rng.normal(size=3), rng.uniform(-0.4, 0.4, size=3)
    last = rng.normal(size=22)
    twist = np.array((0.7, -0.3, 0.5))
    state = RobotState(q=q.astype(np.float32), dq=dq.astype(np.float32), gyro=gyro.astype(np.float32),
                       rpy=rpy.astype(np.float32))
    agent = sprint.Sprint()
    expected = agent.observe(state, last, {"twist": twist})
    gravity = projected_gravity(rpy.astype(np.float32))
    ours = C.observe(gyro, gravity, q, dq, last, twist)
    np.testing.assert_allclose(ours, expected, atol=1e-5)
    np.testing.assert_allclose(C.decode(last), agent.decode(last.astype(np.float32)), atol=1e-5)


def test_gait_clock_matches_k1_deploy_walk_clock():
    """The gait clock (W5c): training's two clock inputs equal the twin's walk_clock agent (the firmware's
    WalkClockAgent) step by step, through standing, walking, a request under the gate, a turn and a slow sidestep."""
    sprint = deploy_sprint()
    if not hasattr(sprint, "WalkClock"):
        pytest.skip("this K1_Deploy has no walk_clock agent (branch walk-clock)")
    import torch
    from k1_deploy.agents.base import RobotState
    from rl.walk.task import WalkTask

    task = WalkTask(load_k1(dtype=torch.float64))
    agent = sprint.WalkClock()
    agent.reset()
    state = RobotState(q=sprint.HOME.copy(), dq=np.zeros(22, np.float32), gyro=np.zeros(3, np.float32),
                       rpy=np.zeros(3, np.float32))
    mem = {"gait_phase": torch.zeros(1, dtype=torch.float64)}
    plan = [(0.0, 0.0, 0.0)] * 3 + [(0.5, 0.0, 0.0)] * 60 + [(0.1, 0.0, 0.0)] * 2 + [(0.0, 0.0, 0.3)] * 5 \
        + [(0.15, 0.1, 0.0)] * 2 + [(3.5, 0.0, 0.0)] * 3
    seen = []
    for cmd in plan:
        mem["commands"] = torch.tensor([cmd], dtype=torch.float64)
        ours = task.clock(mem)[0].numpy()
        theirs = agent.observe(state, np.zeros(22, np.float32), {"twist": np.array(cmd, dtype=np.float32)})[75:77]
        np.testing.assert_allclose(ours, theirs, atol=1e-5)
        seen.append(ours)
        mem["gait_phase"] = task.next_phase(mem)
    assert np.allclose(seen[3], (0.0, 1.0)) and np.allclose(seen[63], (0.0, 0.0)) and np.allclose(seen[65], (0.0, 1.0))
    assert np.allclose(seen[70], (0.0, 0.0))               # 0.18 m/s sideways and forward: under the gate


def test_projected_gravity_matches_k1_deploy():
    deploy_sprint()
    from k1_deploy.contract import projected_gravity

    roll, pitch, yaw = 0.3, -0.2, 0.9
    cr, sr, cp, sp, cy, sy = (f(a / 2) for a in (roll, pitch, yaw) for f in (np.cos, np.sin))
    quat = np.array((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy))
    np.testing.assert_allclose(C.projected_gravity(quat), projected_gravity(np.array((roll, pitch, yaw))), atol=1e-5)


def test_ramp_is_the_firmwares():
    twist = np.zeros(3)
    want = np.array((2.5, -1.0, 2.0))
    twist = C.ramp_twist(twist, want)
    np.testing.assert_allclose(twist, (1.5 * 0.02, -2.0 * 0.02, 4.0 * 0.02))
    for _ in range(500):
        twist = C.ramp_twist(twist, want)
    np.testing.assert_allclose(twist, want)


def test_a_request_beyond_the_limits_is_clipped():
    twist = np.zeros(3)
    for _ in range(500):
        twist = C.ramp_twist(twist, (9.0, -9.0, 9.0))
    np.testing.assert_allclose(twist, (2.5, -1.0, 2.0))


@pytest.mark.parametrize("backend", (PHBackend, MuJoCoBackend))
def test_walk_v9_walks_on_both_engines(backend):
    if not (CHECKPOINTS / "walk_v9.pt").exists():
        pytest.skip("k1_sprint is not next to this repository")
    walker = Walker(backend(load_k1()), load_walk_policy("walk_v9"))
    start = walker.reading.pos.copy()
    for _ in range(200):  # 4 s
        walker.tick((0.5, 0.0, 0.0))
    assert not walker.fell
    assert walker.reading.pos[0] - start[0] > 0.8


@pytest.mark.parametrize("backend", (PHBackend, MuJoCoBackend))
def test_standing_robot_carries_its_weight(backend):
    """The readout both comparisons rest on: the feet carry the robot's weight when it stands still.  The policy holds
    it up: with the sprint's gains alone the robot tips over within 2 s, on both engines."""
    if not (CHECKPOINTS / "walk_v9.pt").exists():
        pytest.skip("k1_sprint is not next to this repository")
    model = load_k1()
    walker = Walker(backend(model), load_walk_policy("walk_v9"))
    for _ in range(200):
        walker.tick((0.0, 0.0, 0.0))
    assert not walker.fell
    weight = float(model.mj.body_subtreemass[1]) * 9.81
    assert walker.reading.foot_force.sum() == pytest.approx(weight, rel=0.01)


def test_reset_restores_the_start():
    walker = Walker(PHBackend(load_k1()), load_walk_policy("walk_v9"))
    start = walker.reading.qpos.copy()
    for _ in range(50):
        walker.tick((0.5, 0.0, 0.0))
    walker.reset()
    np.testing.assert_allclose(walker.reading.qpos, start)
    np.testing.assert_allclose(walker.twist, 0.0)


def test_steps_come_from_the_motion_not_the_force():
    """Both knees at 2 cycles per second are 4 steps per second, whatever the contact force does."""
    from rl.walk.compare import steps_per_second

    t = np.arange(0, 4.0, C.POLICY_DT)
    left, right = 0.8 + 0.3 * np.sin(2 * np.pi * 2.0 * t), 0.8 - 0.3 * np.sin(2 * np.pi * 2.0 * t)
    assert steps_per_second(left, right) == pytest.approx(4.0, rel=0.01)
    assert steps_per_second(0.8 + 1e-3 * t, 0.8 + 0 * t) is None


def test_ripple_on_a_cycle_is_not_a_second_cycle():
    from rl.walk.compare import cycle_starts

    t = np.arange(0, 4.0, C.POLICY_DT)
    x = np.sin(2 * np.pi * t) + 0.1 * np.sin(2 * np.pi * 15.0 * t)
    assert len(cycle_starts(x)) == pytest.approx(4, abs=1)


def test_a_foot_rolling_on_its_toe_does_not_slip():
    """The foot body sweeps through space while its toe stays planted: that is no slip, and a sliding foot is."""
    from rl.walk.compare import sole_corners, sole_slip

    corners = sole_corners(load_k1())
    angle = np.linspace(0.0, 0.5, 50)
    c, s = np.cos(angle), np.sin(angle)
    R = np.zeros((50, 3, 3))
    R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s, 1.0, -s, c  # nose down, about y
    feet = np.zeros((50, 2, 3))
    for side in (0, 1):
        feet[:, side] = (0.1, 0.1 * side, 0.0) - np.einsum("tij,j->ti", R, corners[side, 2])
    rot = np.stack((R, R), 1)
    assert sole_slip(feet, rot, corners).max() < 1e-3  # m/s; the heel is still in the low band as it starts to lift
    assert np.linalg.norm(np.diff(feet[:, 0], axis=0), axis=1).max() / C.POLICY_DT > 0.05
    sliding = feet + np.linspace(0, 1, 50)[:, None, None] * np.array((0.2, 0.0, 0.0))
    assert sole_slip(sliding, rot, corners).mean() > 0.1


@pytest.mark.skipif(not os.environ.get("PH_TEST_DISPLAY"), reason="opens a window: set PH_TEST_DISPLAY=1")
def test_a_slider_stays_where_it_is_left():
    from rl.walk.drive import TwistPanel

    panel = TwistPanel()
    wanted = (1.25, -0.4, 1.5)
    for scale, value in zip(panel.scales, wanted):
        scale.set(value)
    for _ in range(100):
        panel.pump()
    np.testing.assert_allclose(panel.values(), wanted)
    panel.stop()
    panel.pump()
    np.testing.assert_allclose(panel.values(), 0.0)
    panel._close()
