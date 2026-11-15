r"""Run a walk-and-kick policy on the pH plant and report its shots.

    python rl/kick/evaluate.py --policy ../k1_kick/checkpoints/kick_018.pt -B 256      # Eric's MuJoCo-trained policy
    python rl/kick/evaluate.py --policy logs/<run>/model_1000.pt -B 256 --noise        # with the training's noise

Every robot plays one episode (walk, request, 4 s), all of them through the firmware's command path.  Without
--noise: no sensor noise, no pushes, nominal gains, 20 ms delay (the camera keeps its frame rate, latency and noise:
it is part of the task; --clean-camera removes that too).  The report follows k1_kick's README columns.

KickEval is the same as a training-time evaluation (rl/common/harness.py evaluate=): the deterministic policy on a
small CPU or GPU env, numbers into W&B under Eval/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco is imported (ph_k1 imports it)
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rl.kick import policy as P  # noqa: E402
from rl.kick.camera import CAMERA_CLEAN  # noqa: E402
from rl.kick.env import SHOT, KickConfig, KickEnv  # noqa: E402


def quiet(cfg: KickConfig, clean_camera: bool = False, **changes) -> KickConfig:
    """cfg without sensor noise, pushes and gain errors, with a 20 ms delay."""
    out = replace(cfg, obs_noise={}, push_dv=0.0, gain_dr=0.0, delay_ms=(20.0, 20.0), **changes)
    if clean_camera:
        out = replace(out, camera=dict(CAMERA_CLEAN))
    return out


def play(env: KickEnv, policy, max_steps: int | None = None) -> torch.Tensor:
    """One episode per robot: (B, len(SHOT)) rows of env.last_shot, taken when each robot's first episode ends.
    policy: the env's observations (TensorDict) -> actions (B,22)."""
    B = env.num_envs
    env._reset_idx(env.index)
    env._refresh()
    env._start_requests()
    obs = env.get_observations()
    done = torch.zeros(B, dtype=torch.bool, device=env.device)
    shots = torch.zeros(B, len(SHOT), dtype=env.dtype, device=env.device)
    for _ in range(max_steps or 2 * env.max_episode_length):
        obs, _, dones, _ = env.step(policy(obs))
        new = dones & ~done
        shots[new] = env.last_shot[new]
        done |= dones
        if bool(done.all()):
            break
    shots[~done, SHOT.index("requested")] = 0.0
    return shots


def summarize(shots: torch.Tensor, full_power: float = 5.0) -> dict:
    col = {k: shots[:, i] for i, k in enumerate(SHOT)}
    req, kicked, fell = col["requested"] > 0, col["kicked"] > 0, col["fell"] > 0
    ok = req & kicked

    def mean(x, mask):
        return float(x[mask].mean()) if bool(mask.any()) else float("nan")

    def share(mask, of):
        return float(mask[of].float().mean()) if bool(of.any()) else float("nan")

    out = {
        "episodes": int(shots.shape[0]), "requests": int(req.sum()),
        "kick_rate": share(kicked, req),
        "launch_speed": mean(col["along"], ok),
        "direction_error_deg": mean(col["direction_deg"], ok),
        "within_5deg": share(col["direction_deg"] < 5.0, ok),
        "within_10deg": share(col["direction_deg"] < 10.0, ok),
        "loft": mean(col["loft"], ok),
        "speed_error": mean(col["speed_error"], ok & (col["want"] < full_power)),
        "launch_speed_full_power": mean(col["along"], ok & (col["want"] >= full_power)),
        "touch_time": mean(col["touch_time"], ok),
        "fell": share(fell, torch.ones_like(fell)),
        "fell_walking": share(fell & ~req, torch.ones_like(fell)),
        "fell_kicking": share(fell, req),
        "kick_rate_far": share(kicked, req & (col["range"] >= 0.6)),
        "direction_error_deg_far": mean(col["direction_deg"], ok & (col["range"] >= 0.6)),
        "direction_error_deg_near": mean(col["direction_deg"], ok & (col["range"] < 0.6)),
        "kick_rate_moving": share(kicked, req & (col["robot_speed"] >= 0.3)),
        "direction_error_deg_moving": mean(col["direction_deg"], ok & (col["robot_speed"] >= 0.3)),
        "direction_error_deg_stopped": mean(col["direction_deg"], ok & (col["robot_speed"] < 0.3)),
        "kick_energy_J": mean(col["energy_J"], ok),
        "ball_energy_J": mean(col["ball_J"], ok),
    }
    for lo in range(1, 6):
        band = ok & (col["want"] >= lo) & (col["want"] < lo + 1)
        out[f"speed_at_{lo}_{lo + 1}"] = mean(col["speed"], band)
    return out


class KickEval:
    """evaluate(policy, it) for the harness: the deterministic policy, one episode on each of `robots` robots."""

    def __init__(self, cfg: KickConfig, robots: int = 64, device: str = "cpu", env_cls=KickEnv, **changes):
        self.cfg = quiet(cfg, num_envs=robots, device=device, seed=cfg.seed + 7, **changes)
        self.env_cls, self.env = env_cls, None

    def __call__(self, policy, it: int) -> dict:
        with torch.no_grad():
            if self.env is None:
                self.env = self.env_cls(self.cfg)
            env = self.env
            try:
                dev = next(policy.parameters()).device
            except (AttributeError, StopIteration):
                dev = env.device
            s = summarize(play(env, lambda o: policy(o.to(dev)).to(env.device)), env.cfg.full_power)
        keep = ("kick_rate", "launch_speed", "direction_error_deg", "within_10deg", "loft", "speed_error",
                "launch_speed_full_power", "fell_walking", "fell_kicking", "kick_energy_J")
        return {f"Eval/{k}": s[k] for k in keep if s[k] == s[k]}


def report(s: dict) -> str:
    pct = lambda x: "  n/a" if x != x else f"{100 * x:5.1f} %"  # noqa: E731
    num = lambda x, u="": "  n/a" if x != x else f"{x:5.2f}{u}"  # noqa: E731
    lines = [
        f"episodes {s['episodes']}, requests {s['requests']}",
        f"  kicked                {pct(s['kick_rate'])}   (far balls {pct(s['kick_rate_far'])}, mid-walk {pct(s['kick_rate_moving'])})",
        f"  launch speed          {num(s['launch_speed'], ' m/s')} along the direction; full power {num(s['launch_speed_full_power'], ' m/s')}",
        "  ball speed by request " + " / ".join(num(s[f'speed_at_{lo}_{lo + 1}']) for lo in range(1, 6)) + " m/s  (requests 1-2 / 2-3 / 3-4 / 4-5 / 5-6)",
        f"  relative speed error  {num(s['speed_error'])}",
        f"  direction error       {num(s['direction_error_deg'], ' deg')}  (within 5 deg {pct(s['within_5deg'])}, within 10 deg {pct(s['within_10deg'])})",
        f"     near {num(s['direction_error_deg_near'])}, far {num(s['direction_error_deg_far'])}, stopped {num(s['direction_error_deg_stopped'])}, moving {num(s['direction_error_deg_moving'])} deg",
        f"  loft                  {num(s['loft'], ' m/s')};  touch after {num(s['touch_time'], ' s')}",
        f"  falls                 {pct(s['fell'])}   (walking {pct(s['fell_walking'])}, of the requests {pct(s['fell_kicking'])})",
        f"  energy per kick       {num(s['kick_energy_J'], ' J')} electrical, {num(s['ball_energy_J'], ' J')} in the ball",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="TorchScript, rsl-rl checkpoint or firmware .bin")
    ap.add_argument("-B", "--num-envs", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--noise", action="store_true", help="the training's sensor noise, pushes, gain errors and delays")
    ap.add_argument("--clean-camera", action="store_true", help="a camera without frame drops, noise and drift")
    ap.add_argument("--stand", action="store_true", help="every request from the stance (no walk before)")
    ap.add_argument("--output", default=None, help="write the summary and the shots as JSON")
    ap.add_argument("--video", default=None, help="also write a video of one walk and kick to this folder, with the "
                                                  "Hamiltonian figure of that rollout (rl/kick/hamiltonian.py)")
    ap.add_argument("--video-seed", type=int, default=3)
    args = ap.parse_args()

    policy, obs_dim = P.load(args.policy, args.device, getattr(torch, args.dtype))
    cfg = KickConfig(num_envs=args.num_envs, device=args.device, dtype=args.dtype, seed=args.seed,
                     phase_input=obs_dim == 166)
    if args.stand:
        cfg = replace(cfg, stand_prob=1.0)
    if not args.noise:
        cfg = quiet(cfg, clean_camera=args.clean_camera)
    elif args.clean_camera:
        cfg = replace(cfg, camera=dict(CAMERA_CLEAN))
    env = KickEnv(cfg)
    with torch.no_grad():
        shots = play(env, lambda o: policy(o["policy"]))
    s = summarize(shots, cfg.full_power)
    print(f"{Path(args.policy).name} on the pH plant ({args.num_envs} robots, seed {args.seed}"
          f"{', noise' if args.noise else ''}{', from the stance' if args.stand else ''})")
    print(report(s))
    if args.video:
        from rl.kick.train import KickVideo  # here: train.py sets MUJOCO_GL before mujoco is imported by ph_k1

        one = quiet(cfg, num_envs=1, device="cpu", seed=args.video_seed, energy_log_envs=0)
        video = KickVideo(lambda: KickEnv(one), 6.0, cfg.policy_dt, Path(args.video), Path(args.policy).stem)
        cpu_policy, _ = P.load(args.policy, "cpu", torch.float32)
        out = video(lambda o: cpu_policy(o["policy"]), 0)
        print("video:", out)
    if args.output:
        Path(args.output).write_text(json.dumps({"summary": s, "columns": SHOT, "shots": shots.tolist()}, indent=1))


if __name__ == "__main__":
    main()
