"""The stand task: contract == firmware obs.h, exporter == firmware forward pass, the environment's mechanics.

    python -m pytest stand/tests -q          (needs the built ph_k1_native, rsl-rl-lib and tensordict)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ph_k1.model import NJ, load_k1  # noqa: E402
from stand import contract as C  # noqa: E402

DEVICE = os.environ.get("PH_TEST_DEVICE", "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    pytest.skip("PH_TEST_DEVICE=cuda but no GPU", allow_module_level=True)


def test_contract_matches_firmware_obs_h():
    """fill_proprio / decode_stand / projected_gravity written out by hand, index by index, as in the C++."""
    torch.manual_seed(0)
    C.check_joint_order(load_k1().joint_names)
    con = C.Contract(torch.float64)
    q = torch.randn(1, NJ, dtype=torch.float64)
    dq = torch.randn(1, NJ, dtype=torch.float64)
    gyro = torch.randn(1, 3, dtype=torch.float64)
    quat = torch.randn(1, 4, dtype=torch.float64)
    quat = quat / quat.norm()
    last = torch.randn(1, NJ, dtype=torch.float64)
    obs = con.observe(q, dq, gyro, quat, last)[0]
    # math.h projected_gravity: rotate (0,0,-1) by the inverse trunk rotation
    w, x, y, z = quat[0].tolist()
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    g = R.T @ np.array([0.0, 0.0, -1.0])
    want = np.zeros(72)
    want[0:3] = gyro[0].numpy() * 0.25
    want[3:6] = g
    for k in range(22):
        r = C.REAL2SIM[k]
        want[6 + k] = (q[0, r].item() - C.DEFAULT_QPOS[r]) * 1.0
        want[28 + k] = dq[0, r].item() * 0.05
        want[50 + k] = last[0, k].item()
    assert np.abs(obs.numpy() - want).max() < 1e-12
    raw = torch.randn(1, NJ, dtype=torch.float64)
    target = con.decode(raw)[0]
    for r in range(22):
        assert abs(target[r].item() - (C.DEFAULT_QPOS[r] + 1.0 * C.ACTION_SCALE[r] * raw[0, C.SIM2REAL[r]].item())) < 1e-12


def test_roll_pitch_and_gravity_of_simple_poses():
    con = C.Contract(torch.float64)
    a = 0.4
    roll = torch.tensor([[np.cos(a / 2), np.sin(a / 2), 0.0, 0.0]], dtype=torch.float64)
    pitch = torch.tensor([[np.cos(a / 2), 0.0, np.sin(a / 2), 0.0]], dtype=torch.float64)
    assert torch.allclose(con.roll_pitch(roll), torch.tensor([[a, 0.0]], dtype=torch.float64))
    assert torch.allclose(con.roll_pitch(pitch), torch.tensor([[0.0, a]], dtype=torch.float64))
    up = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(con.projected_gravity(up), torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64))


def test_export_round_trip(tmp_path):
    """deploy.bin read back and run with the firmware's forward pass == the rsl-rl actor (normalizer included)."""
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict

    from rsl_rl.models import MLPModel
    from stand.export import bin_policy, export_bin, read_bin

    torch.manual_seed(1)
    obs = TensorDict({"policy": torch.randn(64, C.OBS_DIM)}, batch_size=[64])
    actor = MLPModel(obs, {"actor": ["policy"]}, "actor", C.NUM_ACTIONS, hidden_dims=list(C.HIDDEN_DIMS),
                     activation="elu", obs_normalization=True,
                     distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5})
    for _ in range(3):  # give the normalizer non-trivial statistics
        actor.update_normalization(TensorDict({"policy": 3 * torch.randn(256, C.OBS_DIM) + 1}, batch_size=[256]))
    with torch.no_grad():
        for prm in actor.parameters():
            prm.add_(0.1 * torch.randn_like(prm))
    ck = tmp_path / "model_7.pt"
    torch.save({"actor_state_dict": actor.state_dict()}, ck)
    out = export_bin(ck, tmp_path / "deploy_7.bin")
    mats = read_bin(out)
    assert sorted(mats) == sorted([f"actor.{i}.{p}" for i in (0, 2, 4, 6) for p in ("weight", "bias")]
                                  + ["obs_normalizer._mean", "obs_normalizer._std"])
    actor.eval()
    with torch.no_grad():
        want = actor(obs).double().numpy()
    got = bin_policy(mats)(obs["policy"].double().numpy())
    assert np.abs(got - want).max() < 1e-5


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("tensordict")
    from stand.env import StandConfig, StandEnv

    return StandEnv(StandConfig(num_envs=8, device=DEVICE, dtype="float32", push_dv=0.0))


def test_env_shapes_and_reset_pose(env):
    obs = env.get_observations()
    assert obs["policy"].shape == (8, C.OBS_DIM) and obs["critic"].shape == (8, C.OBS_DIM + 3 + 1 + 2)
    theta = env.plant.theta
    assert (theta - env.contract.default_qpos).abs().max() <= env.cfg.joint_noise + 1e-6
    pc = env.plant._sole_points(env.plant.kinematics())
    assert pc[..., 2].min(-1).values.abs().max() < 1e-5, "soles on the floor at reset"
    assert 0.4 < env.stand_height < 0.7


def test_zero_action_falls_and_is_reset(env):
    """The untrained policy (a = 0: targets = the firmware's default pose, firmware gains) tips over within a few
    seconds; the firmware abort terminates the episode and the robot is reset to the start pose."""
    zeros = torch.zeros(8, C.NUM_ACTIONS, device=env.device)
    terminated_at = None
    for k in range(400):  # 8 s
        obs, rew, dones, extras = env.step(zeros)
        assert torch.isfinite(rew).all() and torch.isfinite(obs["policy"]).all()
        term = dones & ~extras["time_outs"]
        if bool(term.any()):
            terminated_at = k
            assert "Episode_Termination/fell" in extras["log"]
            row = int(term.nonzero()[0, 0])
            assert env.episode_length_buf[row] == 0 and env.plant.p[row].abs().max() == 0
            break
    assert terminated_at is not None, "a zero-action robot should fall with the firmware gains"
    assert terminated_at * env.cfg.policy_dt > 0.3, "it should not fall instantly"


def test_rewards_are_finite_and_alive_dominates_while_standing(env):
    env._reset_idx(torch.arange(8, device=env.device))
    _, rew, _, _ = env.step(torch.zeros(8, C.NUM_ACTIONS, device=env.device))
    assert torch.isfinite(rew).all()
    assert (rew > 0).all(), "a clean standing step must have positive reward (else falling on purpose pays)"
