r"""Run a stand policy on the pH plant and report how well it stands: one evaluation for every way of training it.

    python rl/stand/evaluate.py --policy outputs/kaggle/stand-ph-v1/runs/stand-ph-v1/deploy_2999.bin -B 256
    python rl/stand/evaluate.py --policy <file> --quiet          # no pushes, no sensor noise, nominal gains

Every robot plays one episode (10 s) from the start pose through the firmware's command path.  Default: the
training's conditions (pushes, sensor noise, gain errors, delays).  StandEval is the same for a training loop:
evaluate(policy, it) -> {Eval/...}.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rl.stand.env import StandConfig, StandEnv  # noqa: E402


def quiet(cfg: StandConfig, **changes) -> StandConfig:
    return replace(cfg, obs_noise={}, push_dv=0.0, gain_dr=0.0, delay_ms=(20.0, 20.0), **changes)


def play(env: StandEnv, policy, seconds: float | None = None) -> dict:
    """One episode per robot.  policy: the env's observations (TensorDict) -> actions."""
    B, dt = env.num_envs, env.cfg.policy_dt
    env._reset_idx(torch.arange(B, device=env.device))
    obs = env.get_observations()
    steps = int(round((seconds or env.cfg.episode_s) / dt))
    alive = torch.ones(B, dtype=torch.bool, device=env.device)
    stood = torch.zeros(B, device=env.device)
    watts = torch.zeros(B, device=env.device)
    tilt = torch.zeros(B, device=env.device)
    for _ in range(steps):
        s0 = env.plant.ledger.supplied.clone()
        obs, _, dones, extras = env.step(policy(obs))
        fell = dones.bool() & ~extras["time_outs"].bool()
        rp = env.contract.roll_pitch(env.plant.q[:, 3:7]).abs().amax(-1)
        live = alive & ~dones.bool()
        stood += alive.float() * dt
        watts += live.float() * ((env.plant.ledger.supplied - s0) / dt).float()
        tilt += live.float() * torch.rad2deg(rp).float()
        alive &= ~fell
        if not bool(alive.any()):
            break
    n = (stood / dt).clamp(min=1.0)
    return {"fell": float((~alive).float().mean()), "stood_s": float(stood.mean()), "watts": float((watts / n).mean()),
            "tilt_deg": float((tilt / n).mean()), "robots": B, "seconds": steps * dt}


class StandEval:
    """evaluate(policy, it) for a training loop: one episode on `robots` robots under the training's conditions and
    one without disturbances."""

    def __init__(self, cfg: StandConfig, robots: int = 64, device: str = "cpu", seconds: float | None = None):
        self.cfgs = {"": replace(cfg, num_envs=robots, device=device, seed=cfg.seed + 7),
                     "quiet_": quiet(cfg, num_envs=robots, device=device, seed=cfg.seed + 8)}
        self.envs, self.seconds = {}, seconds

    def __call__(self, policy, it: int = 0) -> dict:
        out = {}
        with torch.no_grad():
            for tag, cfg in self.cfgs.items():
                if tag not in self.envs:
                    self.envs[tag] = StandEnv(cfg)
                r = play(self.envs[tag], policy, self.seconds)
                out.update({f"Eval/{tag}fell": r["fell"], f"Eval/{tag}stood_s": r["stood_s"],
                            f"Eval/{tag}watts": r["watts"], f"Eval/{tag}tilt_deg": r["tilt_deg"]})
        return out


def main():
    from rl.kick import policy as P  # the same loader for .bin, rsl-rl checkpoints and TorchScript

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="firmware .bin or a checkpoint; 'zero' = always the start pose")
    ap.add_argument("-B", "--num-envs", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quiet", action="store_true", help="no pushes, no sensor noise, nominal gains, 20 ms delay")
    args = ap.parse_args()
    cfg = StandConfig(num_envs=args.num_envs, device=args.device, dtype="float32", seed=args.seed)
    if args.quiet:
        cfg = quiet(cfg)
    env = StandEnv(cfg)
    if args.policy == "zero":
        act = lambda o: torch.zeros(env.num_envs, env.num_actions, device=env.device)  # noqa: E731
    else:
        policy, _ = P.load(args.policy, args.device)
        act = lambda o: policy(o["policy"])  # noqa: E731
    with torch.no_grad():
        r = play(env, act)
    print(f"{Path(args.policy).name} on the pH plant ({r['robots']} robots, {r['seconds']:g} s"
          f"{', quiet' if args.quiet else ', pushes and noise'}): fell {100 * r['fell']:.1f} %, stood "
          f"{r['stood_s']:.2f} s on average, {r['watts']:.1f} W, tilt {r['tilt_deg']:.2f} deg")


if __name__ == "__main__":
    main()
