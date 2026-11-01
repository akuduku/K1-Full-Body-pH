"""The get-up task: contract (= StandupAgent + gains), the energy tank, the lying library, the env, the export.

    python -m pytest rl/getup/tests -q          (needs the built ph_k1_native, rsl-rl-lib and tensordict)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ph_k1 import firmware as fw  # noqa: E402
from ph_k1.model import NJ, load_k1  # noqa: E402
from rl.getup import contract as C  # noqa: E402
from rl.getup.impedance import EnergyTank, spring_energy  # noqa: E402
from rl.getup.lying import PRONE, SUPINE, quat_from_rpy  # noqa: E402

DEVICE = os.environ.get("PH_TEST_DEVICE", "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    pytest.skip("PH_TEST_DEVICE=cuda but no GPU", allow_module_level=True)


def test_contract_is_the_standup_agent_plus_gains():
    """Targets = sprint.cpp decode (kHome + 0.25 kEffort / kKp a); gains x0.35..x2.83 of kKp / kKd, always > 0;
    zero gain actions = the firmware's gains; observation layout 3|3|22|22|66|1."""
    C.check_joint_order(load_k1().joint_names)
    con = C.Contract(torch.float64)
    a = torch.randn(5, C.NUM_ACTIONS, dtype=torch.float64) * 3
    t, kp, kd = con.decode(a)
    for r in range(NJ):
        want = C.HOME[r] + 0.25 * C.EFFORT_DECODE[r] / C.KP_REF[r] * a[:, r]
        assert torch.allclose(t[:, r], want)
    lo, hi = 2.0 ** -C.GAIN_RANGE, 2.0 ** C.GAIN_RANGE
    assert (kp / con.kp_ref >= lo - 1e-12).all() and (kp / con.kp_ref <= hi + 1e-12).all() and (kd > 0).all()
    _, kp0, kd0 = con.decode(torch.zeros(1, C.NUM_ACTIONS, dtype=torch.float64))
    assert torch.allclose(kp0[0], con.kp_ref) and torch.allclose(kd0[0], con.kd_ref)
    kp_fw, kd_fw = fw.gains(load_k1(), C.GAINS)
    assert torch.allclose(kp_fw, con.kp_ref) and torch.allclose(kd_fw, con.kd_ref)
    f64 = lambda shape, v: torch.full(shape, v, dtype=torch.float64)  # noqa: E731
    obs = con.observe(f64((2, 3), 1.0), f64((2, 3), 2.0), con.home.expand(2, NJ), f64((2, NJ), 4.0),
                      f64((2, C.NUM_ACTIONS), 5.0), f64((2,), 0.5))
    assert obs.shape == (2, C.OBS_DIM) == (2, 117)
    assert (obs[:, 0:3] == 1).all() and (obs[:, 3:6] == 2).all() and (obs[:, 6:28] == 0).all()
    assert (obs[:, 28:50] == 4).all() and (obs[:, 50:116] == 5).all() and (obs[:, 116] == 0.5).all()


def _tank_case(B=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, NJ, generator=g, dtype=torch.float64) * 0.3
    kp_ref = torch.as_tensor(C.KP_REF, dtype=torch.float64)
    return q, kp_ref


def test_tank_pays_every_injection_and_never_goes_negative():
    """Random commands: the injected energy is always paid from the tank (level >= 0, <= capacity), the level
    changes by exactly refill - injection, and alpha = 1 whenever the tank can pay."""
    torch.manual_seed(1)
    B = 64
    q, kp_ref = _tank_case(B)
    kd_ref = torch.as_tensor(C.KD_REF, dtype=torch.float64)
    tank = EnergyTank(B, NJ, torch.float64, "cpu", capacity=5.0, refill=1.0)
    tank.reset(torch.arange(B), q, kp_ref.expand(B, NJ), kd_ref.expand(B, NJ))
    ident = lambda t, k, d: t  # noqa: E731
    qd_prev = torch.zeros(B, NJ, dtype=torch.float64)
    for step in range(30):
        qd = torch.randn(B, NJ, dtype=torch.float64)
        q = q + 0.02 * qd
        target = q + torch.randn(B, NJ, dtype=torch.float64) * 0.4
        kp = kp_ref * torch.exp2(torch.randn(B, NJ, dtype=torch.float64))
        kd = kd_ref * torch.exp2(torch.randn(B, NJ, dtype=torch.float64))
        before = tank.level.clone()
        prev_kd = tank.prev_kd.clone()
        t, k, d, info = tank.step(q, qd, qd_prev, 0.02, target, kp, kd, ident)
        refill = (prev_kd * 0.5 * (qd_prev.square() + qd.square())).sum(-1) * 0.02
        assert torch.allclose(info["refilled"], refill)
        assert (info["level"] >= 0).all() and (info["level"] <= 5.0 + 1e-12).all()
        budget = (before + refill).clamp(max=5.0)
        assert (info["injected"] <= budget + 1e-9).all(), "an injection the tank could not pay"
        expect = (budget - info["injected"]).clamp(0.0, 5.0)
        assert torch.allclose(info["level"], expect)
        free = ~info["limited"]
        assert torch.allclose(t[free], target[free]) and torch.allclose(k[free], kp[free]), "alpha = 1 when affordable"
        qd_prev = qd
    assert info["limited"].any(), "a 5 J tank should have limited some commands"


def test_tank_scales_back_to_what_it_can_pay():
    """An empty tank and a command that stretches the springs: alpha < 1, the injection is (almost) the budget."""
    B = 4
    q, kp_ref = _tank_case(B)
    kd = torch.as_tensor(C.KD_REF, dtype=torch.float64).expand(B, NJ)
    tank = EnergyTank(B, NJ, torch.float64, "cpu", capacity=1.0, refill=0.0)
    tank.reset(torch.arange(B), q, kp_ref.expand(B, NJ), kd)
    zeros = torch.zeros(B, NJ, dtype=torch.float64)
    t, k, d, info = tank.step(q, zeros, zeros, 0.02, q + 1.0, kp_ref.expand(B, NJ), kd, lambda t, k, d: t)
    assert info["limited"].all() and (info["alpha"] < 1).all()
    assert (info["injected"] <= 1.0 + 1e-9).all() and (info["injected"] > 0.99).all()
    assert torch.allclose(spring_energy(k, t, q), info["injected"])  # from S = 0
    # relaxing the springs pays back
    t2, _, _, info2 = tank.step(q, zeros, zeros, 0.02, q.clone(), k, kd, lambda t, k, d: t)
    assert (info2["injected"] < 0).all() and (info2["level"] > 0.99).all()


def test_tank_off_is_the_plain_command():
    B = 3
    q, kp_ref = _tank_case(B)
    kd = torch.as_tensor(C.KD_REF, dtype=torch.float64).expand(B, NJ)
    tank = EnergyTank(B, NJ, torch.float64, "cpu", capacity=0.1, enabled=False)
    tank.reset(torch.arange(B), q, kp_ref.expand(B, NJ), kd)
    z = torch.zeros(B, NJ, dtype=torch.float64)
    t, k, d, info = tank.step(q, z, z, 0.02, q + 2.0, kp_ref.expand(B, NJ) * 2, kd, lambda t, k, d: t)
    assert torch.allclose(t, q + 2.0) and torch.allclose(k, kp_ref * 2) and not info["limited"].any()


def test_quaternion_of_the_lying_kinds():
    """Pitch -pi/2: on the back, gravity along -x of the trunk; +pi/2: on the front, +x (the IMU convention)."""
    z = torch.zeros(1, dtype=torch.float64)
    for pitch, gx in ((-math.pi / 2, -1.0), (math.pi / 2, 1.0)):
        quat = quat_from_rpy(z, z + pitch, z)
        g = C.Contract.projected_gravity(quat)[0]
        assert abs(g[0].item() - gx) < 1e-12 and g[1:].abs().max() < 1e-12


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("tensordict")
    from rl.getup.env import GetUpConfig, GetUpEnv

    return GetUpEnv(GetUpConfig(num_envs=12, device=DEVICE, dtype="float32", library_size=24, push_dv=0.0))


def test_library_lies_on_the_floor(env):
    lib = env.library
    assert len(lib.index(SUPINE)) > 0 and len(lib.index(PRONE)) > 0
    assert (lib.q[:, 2] < 0.3).all() and (lib.com_z < 0.2).all()
    g = C.Contract.projected_gravity(lib.q[:, 3:7])
    assert (g[lib.index(SUPINE), 0] < -0.5).float().mean() > 0.8, "back drops should mostly lie on the back"
    assert (g[lib.index(PRONE), 0] > 0.5).float().mean() > 0.8, "front drops should mostly lie on the front"


def test_env_steps_and_resets(env):
    from rl.getup.env import REWARDS

    obs = env.get_observations()
    assert obs["policy"].shape == (12, C.OBS_DIM) and obs["critic"].shape == (12, C.OBS_DIM + 12)
    env._reset_idx(torch.arange(12, device=env.device))
    lying = env.start_type != 0
    assert lying.any() and (~lying).any()
    assert (env.plant.q[lying, 2] < 0.3).all() and (env.plant.q[~lying, 2] > 0.45).all()
    for k in range(25):
        obs, rew, dones, extras = env.step(torch.zeros(12, C.NUM_ACTIONS, device=env.device))
        assert torch.isfinite(rew).all() and torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
    assert set(env.episode_sums) == set(REWARDS)
    log = extras["log"]
    assert float(log["Energy/battery_W"]) > 0 and float(log["Energy/tank_J"]) <= env.cfg.tank_capacity
    # the standing starts stand after half a second of HOME at the firmware's gains
    assert env.standing[~lying].float().mean() > 0.5
    # the assist pushes the lying robots up, not the standing ones
    f = env._assist()
    assert f is not None and (f[lying & ~env.standing, 2] > 0).all() and (f[env.standing, 2] == 0).all()


def test_curriculum_state_round_trip(env):
    env.success_ema = {k: 0.7 for k in env.success_ema}
    env.head_start_iter, env.head_scale, env.common_step_counter = 12.0, 0.4, 999
    s = env.curriculum_state()
    env.success_ema = {k: 0.0 for k in env.success_ema}
    env.head_scale, env.common_step_counter = 0.0, 0
    env.load_curriculum_state(s)
    assert all(v == 0.7 for v in env.success_ema.values()) and env.head_scale == 0.4 and env.common_step_counter == 999


def test_export_round_trip(tmp_path):
    """deploy.bin (117 -> 66) read back and run with the firmware's forward pass == the rsl-rl actor."""
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict

    from rl.common.export import bin_policy, read_bin
    from rl.getup.export import export_bin
    from rsl_rl.models import MLPModel

    torch.manual_seed(2)
    obs = TensorDict({"policy": torch.randn(32, C.OBS_DIM)}, batch_size=[32])
    actor = MLPModel(obs, {"actor": ["policy"]}, "actor", C.NUM_ACTIONS, hidden_dims=list(C.HIDDEN_DIMS),
                     activation="elu", obs_normalization=True,
                     distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.8})
    actor.update_normalization(TensorDict({"policy": 2 * torch.randn(128, C.OBS_DIM) + 1}, batch_size=[128]))
    ck = tmp_path / "model_3.pt"
    torch.save({"actor_state_dict": actor.state_dict()}, ck)
    mats = read_bin(export_bin(ck, tmp_path / "deploy_3.bin"))
    assert mats["actor.0.weight"].shape == (512, C.OBS_DIM) and mats["actor.6.weight"].shape == (C.NUM_ACTIONS, 128)
    actor.eval()
    with torch.no_grad():
        want = actor(obs).double().numpy()
    assert np.abs(bin_policy(mats)(obs["policy"].double().numpy()) - want).max() < 1e-5
