r"""Run a walk policy on the pH plant and report how well it follows velocity commands: one evaluation for every
way of training it.

    python rl/walk/evaluate.py --policy walk_v9 -B 64                 # a name in k1_sprint/checkpoints, or a file
    python rl/walk/evaluate.py --policy logs/x/deploy_100.bin --quiet # no pushes, no sensor noise, nominal gains
    python rl/walk/evaluate.py --policy zero                          # always the home pose

Every robot plays one episode per command (10 s) from standing, through the firmware's command path.  Per command:
how many fell, the speed error after the first 2 s (the reference still ramps before), the electrical power and the
energy per metre.  WalkEval is the same for a training loop: evaluate(policy, it) -> {Eval/...}.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rl.walk.env import WalkConfig, WalkEnv, quiet  # noqa: E402

# (vx, vy, wz): stand, walk, walk fast, back, sidestep, turn
COMMANDS = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0))


def play(env: WalkEnv, policy, command=None, seconds: float = 10.0, settle: float = 2.0) -> dict:
    """One episode per robot.  policy: the env's observations (TensorDict) -> actions.  command (vx, vy, wz): every
    robot follows it; None: the task's own random commands (what the training sees)."""
    B, dt = env.num_envs, env.cfg.policy_dt
    env.cfg.fixed_command = None if command is None else tuple(float(v) for v in command)
    env._reset_idx(torch.arange(B, device=env.device))
    obs = env.get_observations()
    steps, first = int(round(seconds / dt)), int(round(settle / dt))
    alive = torch.ones(B, dtype=torch.bool, device=env.device)
    z = lambda: torch.zeros(B, device=env.device)  # noqa: E731
    n, error, yaw_error, watts, along, stood = z(), z(), z(), z(), z(), z()
    for i in range(steps):
        target = env.mem["target"].clone()
        obs, _, dones, extras = env.step(policy(obs))
        s = env.sig
        fell = extras["terminated"].bool()
        live = (alive & ~dones.bool()).float()
        stood += alive.float() * dt
        if i >= first:
            track = s.heading_vel if env.cfg.params.heading_frame else s.base_lin_vel
            n += live
            error += live * torch.linalg.vector_norm(target[:, :2] - track[:, :2], dim=1).float()
            yaw_error += live * (target[:, 2] - s.base_ang_vel[:, 2]).abs().float()
            watts += live * extras["supplied_W"].float()
            speed = torch.linalg.vector_norm(target[:, :2], dim=1).clamp(min=1e-6)
            along += live * ((track[:, :2] * target[:, :2]).sum(1) / speed).float()
        alive &= ~fell
        if not bool(alive.any()):
            break
    m = n.clamp(min=1.0)
    mean = lambda x: float((x / m)[n > 0].mean()) if bool((n > 0).any()) else float("nan")  # noqa: E731
    speed, power = mean(along), mean(watts)
    return {"fell": float((~alive).float().mean()), "stood_s": float(stood.mean()), "speed_error": mean(error),
            "yaw_rate_error": mean(yaw_error), "watts": power, "speed": speed,
            "joule_per_m": power / speed if speed > 0.1 else float("nan"), "robots": B, "seconds": steps * dt}


def command_name(command) -> str:
    """(vx, vy, wz) -> a short name for logs: stand, fwd0.5, back0.5, side0.5, turn1.0, or vx_vy_wz."""
    vx, vy, wz = (float(v) for v in command)
    if vx == vy == wz == 0.0:
        return "stand"
    if vy == wz == 0.0:
        return f"{'fwd' if vx > 0 else 'back'}{abs(vx):g}"
    if vx == wz == 0.0:
        return f"side{vy:g}"
    if vx == vy == 0.0:
        return f"turn{wz:g}"
    return f"{vx:g}_{vy:g}_{wz:g}"


class WalkEval:
    """evaluate(policy, it) for a training loop: one episode per fixed command (COMMANDS: stand, walk 0.5 and 1.0 m/s,
    back, sidestep, turn) on `robots` robots under the training's conditions (pushes, noise, gain errors, delays).
    Fixed commands, so evaluations stay comparable while a curriculum changes what the training asks for.

    -> Eval/fell, Eval/speed_error, Eval/yaw_rate_error, Eval/watts, Eval/stood_s (means over the commands) and
    EvalCommand/<name>/<number> per command."""

    def __init__(self, cfg: WalkConfig, robots: int = 64, device: str = "cpu", seconds: float = 10.0,
                 commands=COMMANDS):
        self.cfg = replace(cfg, num_envs=robots, device=device, seed=cfg.seed + 7)
        self.env, self.seconds, self.commands = None, seconds, tuple(commands)

    def __call__(self, policy, it: int = 0) -> dict:
        if self.env is None:
            self.env = WalkEnv(self.cfg)
        keys = ("fell", "stood_s", "speed_error", "yaw_rate_error", "watts")
        out, per = {}, {k: [] for k in keys}
        with torch.no_grad():
            for command in self.commands:
                r = play(self.env, policy, command, self.seconds)
                for k in keys:
                    out[f"EvalCommand/{command_name(command)}/{k}"] = r[k]
                    if r[k] == r[k]:                # a NaN (everybody fell before the first 2 s) is not averaged
                        per[k].append(r[k])
        for k in keys:
            out[f"Eval/{k}"] = sum(per[k]) / len(per[k]) if per[k] else float("nan")
        return out


def load(policy: str, env: WalkEnv):
    """A policy file (or a name in k1_sprint/checkpoints, or 'zero') as observations (TensorDict) -> actions."""
    from rl.kick import policy as P
    from rl.walk.backends import resolve_policy

    if policy == "zero":
        return lambda o: torch.zeros(env.num_envs, env.num_actions, dtype=env.dtype, device=env.device)
    fn, _ = P.load(resolve_policy(policy), env.device)
    return lambda o: fn(o["policy"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="a file (.bin, checkpoint, TorchScript), a name in k1_sprint/checkpoints, or 'zero'")
    ap.add_argument("-B", "--num-envs", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--quiet", action="store_true", help="no pushes, no sensor noise, nominal gains, 20 ms delay")
    ap.add_argument("--gait-clock", action="store_true", help="a walk_clock policy (W5c): 77 inputs, the gait clock")
    ap.add_argument("--command", type=float, nargs=3, action="append", metavar=("VX", "VY", "WZ"),
                    help="a command to test (repeatable); default: stand, walk 0.5 and 1.0, back, sidestep, turn")
    args = ap.parse_args()
    cfg = WalkConfig(num_envs=args.num_envs, device=args.device, dtype="float32", seed=args.seed, gait_clock=args.gait_clock)
    env = WalkEnv(quiet(cfg) if args.quiet else cfg)
    act = load(args.policy, env)
    print(f"{Path(args.policy).name} on the pH plant ({args.num_envs} robots, {args.seconds:g} s per command"
          f"{', quiet' if args.quiet else ', pushes and noise'})")
    print(f"{'command vx vy wz':>18s} {'fell':>7s} {'speed error':>12s} {'yaw error':>10s} {'speed':>7s} {'power':>8s} {'J per m':>8s}")
    with torch.no_grad():
        for command in (args.command or COMMANDS):
            r = play(env, act, command, args.seconds)
            print(f"{command[0]:6.2f}{command[1]:6.2f}{command[2]:6.2f} {100 * r['fell']:6.1f}% {r['speed_error']:8.3f} m/s "
                  f"{r['yaw_rate_error']:6.3f}/s {r['speed']:6.2f} {r['watts']:6.1f} W {r['joule_per_m']:8.1f}", flush=True)


if __name__ == "__main__":
    main()
