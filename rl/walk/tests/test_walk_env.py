r"""The walk task (rl/walk/task.py) and its environment on the C++ plant (rl/walk/env.py).

    python -m pytest rl/walk/tests/test_walk_env.py -q
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ph_k1.model import load_k1  # noqa: E402
from rl.walk import contract as C  # noqa: E402
from rl.walk.backends import CHECKPOINTS  # noqa: E402
from rl.walk.task import COUNTS, REWARDS, Signals, WalkTask  # noqa: E402

torch.set_num_threads(1)
B = 3


@pytest.fixture(scope="module")
def task():
    return WalkTask(load_k1())


def rand_fn(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return lambda *shape: torch.rand(*shape, generator=gen, dtype=torch.float64)


def standing(task, n=B, **changes) -> Signals:
    """A robot standing still in the home pose, both feet down, carrying its weight."""
    m = task.model
    q = torch.zeros(n, 29, dtype=torch.float64)
    q[:, 2], q[:, 3], q[:, 7:] = 0.52, 1.0, task.home
    active = torch.zeros(n, m.ncontact, dtype=torch.float64) + task.sole.sum(0)
    force = torch.zeros(n, m.ncontact, 3, dtype=torch.float64)
    force[..., 2] = active * m.total_mass * 9.81 / 8
    s = task.signals(q, torch.zeros(n, 28, dtype=torch.float64), torch.zeros(n, 22, dtype=torch.float64), active, force)
    return replace(s, **changes)


def terms_of(task, s, mem, power=0.0, terminated=None, actions=None):
    n = s.theta.shape[0]
    a = [torch.zeros(n, 22, dtype=torch.float64)] * 3 if actions is None else actions
    fell = torch.zeros(n, dtype=torch.bool) if terminated is None else terminated
    return task.terms(s, mem, *a, torch.full((n,), power, dtype=torch.float64), fell)


# ------------------------------------------------------------------ the signals against MuJoCo
def test_signals_match_mujoco_on_a_moving_robot(task):
    m = task.model
    mj = m.mj
    d = mujoco.MjData(mj)
    gen = torch.Generator().manual_seed(4)
    quat = torch.nn.functional.normalize(torch.randn(1, 4, generator=gen, dtype=torch.float64), dim=-1)
    q = torch.zeros(1, 29, dtype=torch.float64)
    q[:, :3], q[:, 3:7] = torch.tensor([0.3, -0.2, 0.7]), quat
    q[:, 7:] = task.home + 0.3 * torch.randn(1, 22, generator=gen, dtype=torch.float64)
    nu = torch.randn(1, 28, generator=gen, dtype=torch.float64)
    s = task.signals(q, nu, nu[:, 6:], torch.zeros(1, m.ncontact, dtype=torch.float64),
                     torch.zeros(1, m.ncontact, 3, dtype=torch.float64))
    R0 = np.zeros(9)
    mujoco.mju_quat2Mat(R0, q[0, 3:7].numpy())
    d.qpos[:] = q[0].numpy()
    d.qvel[:3] = R0.reshape(3, 3) @ nu[0, 3:6].numpy()      # MuJoCo free joint: linear velocity in the world frame,
    d.qvel[3:] = nu[0, [0, 1, 2] + list(range(6, 28))].numpy()   # angular velocity in the trunk frame
    mujoco.mj_forward(mj, d)
    mujoco.mj_subtreeVel(mj, d)
    trunk = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
    np.testing.assert_allclose(s.com[0].numpy(), d.subtree_com[trunk], atol=1e-9)
    np.testing.assert_allclose(s.angular_momentum[0].numpy(), d.subtree_angmom[trunk], atol=1e-8)
    for i, name in enumerate(("left_foot_link", "right_foot_link")):
        b = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, name)
        np.testing.assert_allclose(s.foot_pos[0, i].numpy(), d.xpos[b], atol=1e-9)
        v = np.zeros(6)
        mujoco.mj_objectVelocity(mj, d, mujoco.mjtObj.mjOBJ_XBODY, b, v, 0)    # at the body origin: (angular, linear), world
        np.testing.assert_allclose(s.foot_vel[0, i].numpy(), v[3:], atol=1e-8)
    np.testing.assert_allclose(s.lin_vel_w[0].numpy(), d.qvel[:3], atol=1e-10)
    assert float(s.yaw[0]) == pytest.approx(C.yaw_of(q[0, 3:7].numpy()))


def test_a_standing_robot_reads_as_standing(task):
    s = standing(task)
    assert bool(s.contact.all()) and not bool(s.ground.any())
    assert float(s.foot_height.max()) < 0.03                                   # the soles are on the floor
    assert float(s.contact_force[..., 2].sum(-1)[0]) == pytest.approx(task.model.total_mass * 9.81)
    assert float(s.angular_momentum.abs().max()) == 0.0
    assert 0.15 < float(s.foot_pos[0, 0, 1] - s.foot_pos[0, 1, 1]) < 0.35       # left foot is the first one
    assert float(s.hand_pos[0, 0, 1]) > 0 > float(s.hand_pos[0, 1, 1])
    assert not bool(task.fallen(s).any())


# ------------------------------------------------------------------ single terms against Eric's formulas
def test_standing_still_on_a_zero_request_is_paid_in_full(task):
    mem = task.set_command(task.new_memory(B, rand_fn()), (0.0, 0.0, 0.0))
    t, _ = terms_of(task, standing(task), mem)
    assert set(t) == set(REWARDS)
    for name in ("track_linear_velocity", "track_angular_velocity", "upright", "yaw_rate_fine", "velocity_fine"):
        assert float(t[name][0]) == pytest.approx(1.0, abs=1e-6), name
    for name in ("body_ang_vel", "action_rate", "air_time", "foot_slip", "toe_clearance", "overspeed", "knee_depth",
                 "heading_hold", "terminated", "foot_touchdown", "reference_lag", "energy", "standing_pose", "leg_acc"):
        assert float(t[name].abs().max()) < 1e-9, name
    assert float(task.reward(t)[0]) > 0.0


def test_tracking_is_erics_formula(task):
    p = task.p
    mem = task.set_command(task.new_memory(B, rand_fn()), (1.0, 0.0, 0.5))
    v = torch.tensor([[0.7, 0.1, 0.05]] * B, dtype=torch.float64)
    w = torch.tensor([[0.1, -0.2, 0.3]] * B, dtype=torch.float64)
    t, _ = terms_of(task, standing(task, base_lin_vel=v, base_ang_vel=w), mem)
    error = 0.3**2 + 0.1**2
    sigma = min(max(p.linear_relative_std * 1.0, p.linear_std_at_rest), p.linear_std)
    want = 0.5 * math.exp(-(error + 0.05**2) / sigma**2) + 0.5 * (1 - math.sqrt(error) / 1.0)
    assert float(t["track_linear_velocity"][0]) == pytest.approx(want, abs=1e-6)
    want = math.exp(-((0.5 - 0.3) ** 2 + 0.1**2 + 0.2**2) / p.angular_std**2)
    assert float(t["track_angular_velocity"][0]) == pytest.approx(want, abs=1e-9)


def test_feet_too_close_cost_the_square_of_the_gap(task):
    mem = task.set_command(task.new_memory(B, rand_fn()), (0.5, 0.0, 0.0))
    s = standing(task)
    foot = s.foot_pos.clone()
    foot[:, 0, 1], foot[:, 1, 1] = 0.10, -0.10            # 0.20 m apart: 3 cm inside the band's lower edge
    t, _ = terms_of(task, replace(s, foot_pos=foot), mem)
    assert float(t["foot_spacing"][0]) == pytest.approx(0.03**2, abs=1e-9)
    mem = task.set_command(mem, (0.5, 0.0, 1.0))           # turning: no spacing cost
    assert float(terms_of(task, replace(s, foot_pos=foot), mem)[0]["foot_spacing"][0]) == 0.0


def test_air_time_counts_feet_in_the_air_and_the_landing_step(task):
    mem = task.set_command(task.new_memory(1, rand_fn()), (0.5, 0.0, 0.0))
    s = standing(task, 1)
    up = replace(s, contact=torch.tensor([[False, True]]), foot_height=torch.tensor([[0.05, 0.0]], dtype=torch.float64))
    paid = []
    for _ in range(5):                                     # 0.02 ... 0.10 s in the air
        t, mem = terms_of(task, up, mem)
        paid.append(float(t["air_time"][0]))
    assert paid == [0.0, 0.0, 1.0, 1.0, 1.0]               # paid while 0.05 s < air time < 0.5 s
    assert float(mem["swing_peak"][0, 0]) == pytest.approx(0.05)
    t, mem = terms_of(task, s, mem)                        # the foot lands
    assert float(t["foot_swing_height"][0]) == pytest.approx((0.05 / 0.06 - 1) ** 2)
    assert float(t["soft_landing"][0]) == pytest.approx(float(torch.linalg.vector_norm(s.contact_force[0, 0])) / 1.5)
    assert float(mem["swing_peak"][0, 0]) == 0.0 and float(mem["air_time"][0, 0]) == 0.0
    t, mem = terms_of(task, s, mem)                        # the step after: not a landing any more
    assert float(t["soft_landing"][0]) == 0.0


def test_heading_hold_charges_drift_but_not_a_requested_turn(task):
    mem = task.set_command(task.new_memory(1, rand_fn()), (0.5, 0.0, 0.0))
    s = standing(task, 1)
    _, mem = terms_of(task, s, mem)                        # the reference heading is taken here
    t, mem = terms_of(task, replace(s, yaw=s.yaw + 0.2), mem)
    assert float(t["heading_hold"][0]) == pytest.approx(0.04)
    mem = task.set_command(mem, (0.5, 0.0, 1.0))
    t, _ = terms_of(task, replace(s, yaw=s.yaw + 0.5), mem)
    assert float(t["heading_hold"][0]) == 0.0


def test_the_memory_is_never_written_in_place(task):
    mem = task.new_memory(B, rand_fn())
    before = {k: v.clone() for k, v in mem.items()}
    s = standing(task, contact=torch.zeros(B, 2, dtype=torch.bool))
    _, new = terms_of(task, s, mem)
    task.update_commands(new, s.yaw, torch.full((B,), 5.0, dtype=torch.float64), rand_fn(1))    # 5 s: every request expires
    task.reset_memory(new, torch.arange(B), rand_fn(2))
    for k, v in before.items():
        assert torch.equal(mem[k], v) or bool(torch.isnan(v).all()), k


def test_which_terms_have_a_derivative(task):
    """One foot down and landing, one in the air, the robot moving: the counting terms give no derivative, the
    terms a gradient learner lives on do."""
    m = task.model
    gen = torch.Generator().manual_seed(1)
    mem = task.set_command(task.new_memory(1, rand_fn()), (-0.6, 0.2, 0.1))
    q = torch.zeros(1, 29, dtype=torch.float64)
    q[:, 2], q[:, 3], q[:, 7:] = 0.56, 1.0, task.home + 0.4 * torch.randn(1, 22, generator=gen, dtype=torch.float64)
    q[:, 3:7] = torch.nn.functional.normalize(q[:, 3:7] + 0.1 * torch.randn(1, 4, generator=gen, dtype=torch.float64), dim=-1)
    q[:, 26] = 1.3                                           # the right knee bent beyond knee_depth
    nu = 12.0 * torch.randn(1, 28, generator=gen, dtype=torch.float64)
    active = torch.zeros(1, m.ncontact, dtype=torch.float64)
    active[:, task.sole_ids[1]] = 1.0                        # right foot down (it lands this step), left in the air
    force = torch.zeros(1, m.ncontact, 3, dtype=torch.float64)
    force[:, task.sole_ids[1], 2] = torch.tensor([40.0, 30.0, 20.0, 10.0], dtype=torch.float64)
    leaves = [x.requires_grad_(True) for x in (q, nu, force)]
    a = [torch.randn(1, 22, generator=gen, dtype=torch.float64).requires_grad_(True) for _ in range(3)]
    power = torch.tensor([50.0], dtype=torch.float64, requires_grad=True)
    s = task.signals(q, nu, torch.zeros(1, 22, dtype=torch.float64), active, force)
    t, _ = task.terms(s, mem, *a, power, torch.zeros(1, dtype=torch.bool))
    alive = set()
    for name, value in t.items():
        assert bool(torch.isfinite(value).all()), name
        if value.requires_grad:
            grads = torch.autograd.grad(value.sum(), leaves + a + [power], allow_unused=True, retain_graph=True)
            assert all(g is None or bool(torch.isfinite(g).all()) for g in grads), name
            if any(g is not None and bool(g.abs().sum() > 0) for g in grads):
                alive.add(name)
    assert not alive & set(COUNTS)
    assert alive >= {"track_linear_velocity", "track_angular_velocity", "upright", "body_ang_vel", "angular_momentum",
                     "action_rate", "foot_clearance", "foot_slip", "soft_landing", "upper_body_posture",
                     "feet_parallel", "arm_swing", "knee_depth", "yaw_rate_fine", "action_jerk", "leg_acc",
                     "trunk_bounce", "velocity_fine", "heel_walking", "energy"}, sorted(set(t) - alive)


# ------------------------------------------------------------------ commands
def test_requests_follow_the_shares_and_the_reference_ramps(task):
    n = 4000
    mem = task.new_memory(n, rand_fn(5))
    cmd = mem["commands"]
    sprint = mem["is_sprint"]
    assert 0.25 < float(sprint.float().mean()) < 0.35
    assert bool((cmd[sprint, 1] == 0).all()) and float(cmd[sprint, 0].min()) >= 0.6 * task.top_speed - 1e-9
    assert float(cmd[:, 0].min()) >= -1.5 and float(cmd[:, 1].abs().max()) <= 1.0 and float(cmd[:, 2].abs().max()) <= 2.0
    assert bool((mem["target"] == 0).all())
    mem = task.update_commands(mem, torch.zeros(n, dtype=torch.float64), torch.full((n,), 0.02, dtype=torch.float64), rand_fn(6))
    step = task.p.accel * task.dt
    assert float(mem["target"][:, :2].abs().max()) <= step + 1e-12                 # 2 m/s^2: 4 cm/s per policy step
    assert bool((mem["commands"][mem["is_standing"]] == 0).all())
    scaled = WalkTask(task.model)
    scaled.command_scale = 0.0
    assert bool((scaled.new_memory(50, rand_fn(7))["commands"] == 0).all())      # the stand end of the curriculum


# ------------------------------------------------------------------ the environment on the C++ plant
def walk_env(robots=4, **changes):
    from rl.walk.env import WalkConfig, WalkEnv, quiet

    return WalkEnv(quiet(WalkConfig(num_envs=robots, device="cpu", dtype="float32", seed=3), **changes))


def test_walk_v9_follows_a_command_in_the_environment():
    """Eric's released walk in our environment: a check of the environment (it never enters the method)."""
    if not (CHECKPOINTS / "walk_v9.pt").exists():
        pytest.skip("k1_sprint is not next to this repository")
    from rl.walk.evaluate import load, play

    env = walk_env()
    with torch.no_grad():
        r = play(env, load("walk_v9", env), (0.5, 0.0, 0.0), seconds=5.0)
    assert r["fell"] == 0.0
    assert r["speed"] == pytest.approx(0.5, abs=0.1) and r["speed_error"] < 0.15
    terms = env.extras["terms"]
    assert set(terms) == set(REWARDS) and all(bool(torch.isfinite(v).all()) for v in terms.values())
    assert float(terms["track_linear_velocity"].mean()) > 0.6


def test_doing_nothing_stands_and_a_fall_ends_the_episode():
    env = walk_env(fixed_command=(0.0, 0.0, 0.0))
    zero = torch.zeros(env.num_envs, 22)
    for _ in range(50):
        obs, reward, dones, extras = env.step(zero)
    assert obs["policy"].shape == (4, C.OBS_DIM) and obs["critic"].shape == (4, C.OBS_DIM + 15)
    assert not bool(dones.any()) and float(reward.min()) > 0.0                     # the home pose stands by itself
    q = env.plant.q.clone()
    q[0, 3:7] = torch.tensor([math.cos(0.5), 0.0, math.sin(0.5), 0.0])              # robot 0 pitched by 1 rad
    env.plant.q = q
    _, reward, dones, extras = env.step(zero)
    assert bool(dones[0]) and bool(extras["terminated"][0]) and not bool(dones[1:].any())
    assert float(reward[0]) < -1.0                                                 # -100 x 0.02 for the fall
    assert int(env.episode_length_buf[0]) == 0                                     # and it started again


def test_a_snapshot_continues_as_the_original():
    env = walk_env(fixed_command=(0.4, 0.0, 0.2))
    gen = torch.Generator().manual_seed(0)
    acts = 0.3 * torch.randn(30, env.num_envs, 22, generator=gen)
    for a in acts[:10]:
        env.step(a)
    snap = env.snapshot()
    first = [env.step(a)[1].clone() for a in acts[10:]]
    q_end = env.plant.q.clone()
    env.restore(snap)
    second = [env.step(a)[1].clone() for a in acts[10:]]
    assert torch.equal(torch.stack(first), torch.stack(second)) and torch.equal(q_end, env.plant.q)
