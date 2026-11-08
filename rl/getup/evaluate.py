r"""The get-up policy WITHOUT action noise (its mean: what the robot runs), on a CPU copy of the task.

The training's success rates (Metrics/success_<kind>) belong to the stochastic policy, and its action noise is part
of what it learnt to rely on: getup-mb-v1d got up from the front in 95 % of its training episodes, but only 12 %
without the noise.  This runs the deterministic policy for one episode from the lying starts (half of the robots
per kind), without the assist and the pushes, with the training's observation noise, delays and gain errors, and
returns per start kind:

    Eval/success_<kind>        standing at the end (as Metrics/success_<kind>: standing when the episode ends)
    Eval/stood_<kind>          stood up at some time
    Eval/time_to_stand_<kind>  [s] until the first standing, of those that stood
    Eval/getup_J_<kind>        battery energy from the start to the first standing [J], of those that stood
    Eval/tank_limited          share of the robot-steps where the tank scaled the command back

At every call the copy takes the training env's curriculum (gain scale, and with it the tank switch and the masked
gain outputs); its own episodes never move a curriculum.
"""

from __future__ import annotations

from typing import Callable

import torch

from .lying import KIND_NAMES


class NoNoiseEval:
    def __init__(self, make_env: Callable, train_env, seconds: float | None = None):
        self.make_env = make_env      # () -> GetUpEnv, e.g. on the CPU
        self.train_env = train_env    # its curriculum is copied at every call
        self.seconds = seconds        # None: one training episode (cfg.episode_s)
        self.env = None

    def __call__(self, policy, it: int = 0) -> dict:
        with torch.no_grad():  # no_grad, not inference_mode: the env's buffers are written by later resets
            if self.env is None:
                self.env = self.make_env()
                self.env._update_curriculum = lambda ids: None  # its own episodes must not move the curriculum
            env = self.env
            n, kinds = env.num_envs, list(env.kinds)
            ids = torch.arange(n, device=env.device)
            for i, part in enumerate(torch.tensor_split(ids, len(kinds))):
                env.cfg.start = KIND_NAMES[kinds[i]]
                env._reset_idx(part)
            env.cfg.start = None
            env.load_curriculum_state(self.train_env.curriculum_state())
            group = env.start_type.clone()
            steps = int(round((self.seconds or env.cfg.episode_s) / env.cfg.policy_dt))
            env.max_episode_length = steps + 1  # no timeout inside; a robot that explodes is reset and counts as failed
            try:
                dev = next(policy.parameters()).device
            except (AttributeError, StopIteration):
                dev = env.device
            failed = torch.zeros(n, dtype=torch.bool, device=env.device)
            stood = torch.zeros_like(failed)
            limited = 0.0
            obs = env.get_observations()
            for _ in range(steps):
                obs, _, dones, extras = env.step(policy(obs.to(dev)))
                failed |= dones.to(env.device).bool()
                stood |= env.standing & ~failed
                limited += float(extras["log"]["Energy/tank_limited"])
            success = env.standing & ~failed
            out = {"Eval/tank_limited": limited / steps}
            for k in kinds:
                rows = group == k
                name = KIND_NAMES[k]
                out[f"Eval/success_{name}"] = float(success[rows].float().mean())
                out[f"Eval/stood_{name}"] = float(stood[rows].float().mean())
                ok = rows & stood & torch.isfinite(env.time_to_stand)
                if bool(ok.any()):
                    out[f"Eval/time_to_stand_{name}"] = float(env.time_to_stand[ok].mean())
                    e = env.ep_to_stand[ok]
                    e = e[torch.isfinite(e)]
                    if len(e):
                        out[f"Eval/getup_J_{name}"] = float(e.mean())
            return out
