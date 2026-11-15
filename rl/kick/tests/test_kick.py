"""The walk-and-kick task: the contract's layout, the environment's mechanics, and a MuJoCo-trained kick on the plant.

    python -m pytest rl/kick/tests -q          (needs the built ph_k1_native, rsl-rl-lib and tensordict)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rl.kick import contract as C  # noqa: E402

DEVICE = os.environ.get("PH_TEST_DEVICE", "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    pytest.skip("PH_TEST_DEVICE=cuda but no GPU", allow_module_level=True)
pytest.importorskip("tensordict")

from rl.kick import policy as P  # noqa: E402
from rl.kick.env import SHOT, KickConfig, KickEnv  # noqa: E402
from rl.kick.evaluate import play, quiet, summarize  # noqa: E402

KICK_018 = REPO.parent / "k1_kick" / "checkpoints" / "kick_018.pt"   # Eric Plass's k1_kick, trained in MuJoCo


def config(**kw):
    return KickConfig(**{"num_envs": 8, "device": DEVICE, "dtype": "float32", "seed": 3, **kw})


def test_layout_is_k1_kicks():
    """k1_kick env.py actor_layout(frames=1, speed=1, camera=1, walk=1): 165 inputs, the walk's 75 first."""
    blocks, width = C.layout()
    assert width == C.OBS_DIM == 165
    assert blocks["frame"] == (0, 72) and blocks["command"] == (72, 3) and blocks["request"] == (75, 1)
    assert blocks["ball"] == (76, 6) and blocks["target"] == (82, 2) and blocks["speed"] == (84, 1)
    assert blocks["reports"] == (85, 80)
    assert C.layout(phase=True)[1] == 166


def test_observation_blocks():
    """Without a request everything after the flag is zero; with one the blocks hold what the contract says."""
    env = KickEnv(config(stand_prob=0.0, obs_noise={}))
    obs = env.get_observations()["policy"]
    assert obs.shape == (8, 165)
    assert not env.active.any()
    assert (obs[:, 75:] == 0).all()
    assert torch.allclose(obs[:, 72:75], env.commands)
    env = KickEnv(config(stand_prob=1.0, obs_noise={}))       # every robot: its request at once
    obs = env.get_observations()["policy"]
    assert env.active.all() and (obs[:, 75] == 1).all()
    assert (obs[:, 72:75] == 0).all()                         # no walk command during a request
    ball = C.to_heading(env.yaw, env.ball_pos - env.base_pos)[:, :2]
    assert torch.allclose(obs[:, 76:78], ball, atol=1e-4)     # the camera's first report is the ball itself
    assert torch.allclose(obs[:, 82:84], C.to_heading(env.yaw, env.target_w)[:, :2], atol=1e-5)
    assert torch.allclose(obs[:, 84], (env.target_speed - 3.0) / 2.0)
    assert torch.allclose(obs[:, 85:165].reshape(8, 40, 2), ball[:, None].expand(8, 40, 2), atol=1e-4)
    phased = KickEnv(config(stand_prob=1.0, phase_input=True))
    assert phased.get_observations()["policy"].shape == (8, 166)


def test_ball_is_ahead_and_clear_of_the_feet():
    env = KickEnv(config(stand_prob=1.0, num_envs=64))
    local = C.to_heading(env.yaw, env.ball_pos - env.base_pos)
    c = env.cfg
    assert (local[:, 0] >= c.ball_x[0] - 1e-4).all() and (local[:, 0] <= c.ball_x[1] + 1e-4).all()
    assert (local[:, 1].abs() <= c.ball_y[1] + 1e-4).all()
    feet = C.to_heading(env.yaw, env.foot_pos - env.base_pos[:, None])
    gap = torch.linalg.vector_norm(local[:, None, :2] - feet[..., :2], dim=-1).amin(-1)
    assert (gap > env.cfg.ball_radius + 0.09).all()            # the ball does not start inside a foot
    assert torch.allclose(env.ball_pos[:, 2], torch.full_like(env.ball_pos[:, 2], env.ball_params.rest_height()), atol=1e-5)


def test_episodes_walk_then_request_then_end():
    """With zero actions (the stance's targets) the robot stands: the walk part passes, the request starts, the
    episode ends at the request's timeout, and the logs carry the energy and kick keys."""
    env = KickEnv(config(num_envs=4, walk_s=(0.2, 0.3), stop_prob=1.0, stand_prob=0.0, request_s=0.6, push_dv=0.0,
                         still_prob=1.0))
    z = torch.zeros(4, 22, device=env.device)
    seen_request, ended, keys = False, torch.zeros(4, dtype=torch.bool, device=env.device), set()
    for _ in range(60):
        _, rew, dones, extras = env.step(z)
        seen_request |= bool(env.active.any())
        ended |= dones
        keys |= set(extras["log"])
        assert torch.isfinite(rew).all()
    assert seen_request and ended.all()
    for key in ("Energy/H_total_J", "Energy/H_ball_J", "Energy/supplied_W", "Kick/kick_rate", "Kick/fell_kicking",
                "Motors/at_torque_limit", "Episode_Reward/launch_direction"):
        assert key in keys
    assert env.last_shot.shape == (4, len(SHOT))
    assert (env.last_shot[:, SHOT.index("requested")] == 1).all()
    assert (env.last_shot[:, SHOT.index("kicked")] == 0).all()   # nobody kicked: the robots stood


def test_touch_and_launch_are_measured():
    """A ball that is set rolling counts as touched; the launch is its fastest velocity in the window."""
    env = KickEnv(config(num_envs=2, stand_prob=1.0, push_dv=0.0))
    z = torch.zeros(2, 22, device=env.device)
    env.step(z)
    xy = env.ball_pos[:, :2].clone() + 2.0 * env.target_w[:, :2]          # 2 m away, along the direction
    env.plant.set_ball(xy, 3.0 * env.target_w)
    env.step(z)
    assert env.kicked.all() and env.launching.all()
    for _ in range(20):
        env.step(z)
    assert not env.launching.any()
    along = (env.launch_vel * env.target_w).sum(-1)
    assert (along > 2.7).all() and (along < 3.05).all()
    assert (C.direction_error(env.launch_vel, env.target_w) < 0.02).all()


@pytest.mark.skipif(not KICK_018.exists(), reason="k1_kick checkpoints are not next to this repository")
def test_mujoco_trained_kick_works_on_the_ph_plant():
    """k1_kick's kick-018 (trained in MuJoCo, runs on the robot) in this env, unchanged: it walks, kicks nearly
    every ball in about the requested direction and stays up."""
    policy, n = P.load(KICK_018, DEVICE)
    assert n == 165
    env = KickEnv(quiet(config(num_envs=48, seed=11)))
    with torch.no_grad():
        s = summarize(play(env, lambda o: policy(o["policy"])))
    assert s["requests"] >= 44
    assert s["kick_rate"] > 0.9
    assert s["fell"] < 0.1
    assert 2.5 < s["launch_speed"] < 5.5
    assert s["direction_error_deg"] < 20.0


def test_hamiltonian_slice_passes_through_the_state():
    """The figure's H_d slice, evaluated at the robot's own (theta, p), is the plant's H plus the shaped potential."""
    from rl.kick.hamiltonian import Recorder, slice_hd

    env = KickEnv(quiet(config(num_envs=1, stand_prob=1.0, device="cpu", energy_log_envs=0)))
    z = torch.zeros(1, 22)
    rec = Recorder(env)
    for _ in range(5):
        env.step(z + 0.3)
        rec.capture()
    st, row = rec.states[-1], rec.rows[-1]
    j = 19  # right knee
    import numpy as np

    hd = slice_hd(st, j, np.array([float(st["q"][7 + j])]), np.array([float(st["p"][6 + j])]), env.ball_params)
    assert abs(float(hd[0, 0]) - (row["H_total"] + row["shaped"])) < 1e-3 * max(1.0, abs(row["H_total"]))
