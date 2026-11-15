r"""Train walk-and-kick on the pH plant with rsl-rl PPO (Kaggle/Colab GPU; a CPU smoke run works too).

    python rl/kick/train.py --num-envs 4096 --hours 6 --run-name kick-ph-v1 \
        --init-actor ../k1_kick/checkpoints/kick_018.pt                      # continue a MuJoCo-trained kick on pH
    python rl/kick/train.py --device cpu --num-envs 16 --iterations 3 --no-wandb       # smoke test

--init-actor: the actor starts from another kick_walk policy with the same 165 inputs (an rsl-rl checkpoint, e.g.
Eric's k1_kick/checkpoints/kick_018.pt, which already kicks on the pH plant: rl/kick/evaluate.py); the critic starts
new.  --walk-teacher: the deployed walk (TorchScript, 75 -> 22) for the walk_imitation reward.

W&B (unless --no-wandb): project "Akos pH Kick", run = --run-name.  Logged per iteration: Kick/* (kick rate, launch
speed, speed and direction error, loft, falls), Energy/* (the Hamiltonian's tanks, the ledger's power flows, the
electrical energy per kick, the ball's energy, their ratio), Motors/*, Walk/*, Policy/*, Episode_Reward/*; every
30 min Eval/* (the policy without action noise) and a video with the Hamiltonian figure of that rollout
(rl/kick/hamiltonian.py).  The harness is rl/common/harness.py (clean stop, resume, checkpoints with deploy files).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco is imported (ph_k1 imports it)
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rl.common import harness as H  # noqa: E402
from rl.kick import contract as C  # noqa: E402
from rl.kick.env import KickConfig, KickEnv  # noqa: E402
from rl.kick.evaluate import KickEval, quiet  # noqa: E402
from rl.kick.export import export_bin  # noqa: E402
from rl.kick.hamiltonian import Recorder, figure  # noqa: E402


class KickVideo(H.VideoMaker):
    """A video of one walk and kick (the ball and the requested direction drawn in), and from the same rollout the
    Hamiltonian figure (rl/kick/hamiltonian.py), logged as Energy/hamiltonian."""

    def __init__(self, make_env, seconds, policy_dt, out_dir, prefix, walk_s: float = 1.5):
        super().__init__(make_env, seconds, policy_dt, out_dir, prefix, prepare=self._prepare)
        self.walk_steps = int(round(walk_s / policy_dt))
        self.ham = None

    def _prepare(self, env, it):
        ids = torch.arange(1, device=env.device)
        env._reset_idx(ids)
        env.request_step[ids] = self.walk_steps       # a short walk, then the request
        env.stop_step[ids] = max(self.walk_steps - 25, 0)
        env.active[ids] = False
        env._refresh()

    def on_start(self, env, it):
        self.ham = Recorder(env)

    def on_step(self, env, rec):
        arrow = None
        if bool(env.active[0]):  # the requested direction, from where the ball lay
            if not hasattr(self, "_from") or int(env.episode_length_buf[0]) == int(env.request_step[0]) + 1:
                self._from = env.ball_pos[0].detach().cpu().numpy().copy()
            a = self._from.copy()
            a[2] = 0.01
            d = env.target_w[0].detach().cpu().numpy()
            arrow = [*a, *(a + 0.6 * d)]
        rec.capture(arrow=arrow)
        self.ham.capture()

    def on_end(self, env, path, it):
        try:
            png = figure(self.ham, path.with_name(f"hamiltonian_{it:05d}.png"), env.cfg.policy_dt,
                         title=f"Kick on the pH plant, iteration {it}")
        except Exception as exc:  # the figure must not cost the video
            print(f"[video] Hamiltonian figure failed: {type(exc).__name__}: {exc}", flush=True)
            png = None
        return (path, {"Energy/hamiltonian": png}) if png else path


def init_actor(path: str):
    """-> before_learn(runner, resumed): the actor's weights and normalizer from another policy with the same inputs."""

    def before_learn(runner, resumed: bool):
        if resumed:
            return
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck["actor_state_dict"]
        actor = runner.alg.actor
        own = actor.state_dict()
        keep = {k: v for k, v in state.items() if k in own and tuple(v.shape) == tuple(own[k].shape)
                and not k.startswith("distribution")}  # the action noise starts at --init-std
        missing = [k for k in own if k not in keep and not k.startswith("distribution")]
        if missing:
            raise ValueError(f"{path}: its actor does not fit this agent (missing or other shape: {missing[:4]})")
        actor.load_state_dict({**own, **keep})
        print(f"[kick] actor from {Path(path).name}: {len(keep)} tensors", flush=True)

    return before_learn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    H.add_arguments(ap, run_prefix="kick-ph", wandb_project="Akos pH Kick", iterations=100000, save_every=0,
                    video_every=0, save_every_min=10.0, video_every_min=30.0, init_std=0.3, entropy_coef=0.003,
                    eval_every_min=30.0, first_eval_min=3.0)
    ap.add_argument("--init-actor", default=None, help="rsl-rl checkpoint of a kick_walk policy with the same inputs")
    ap.add_argument("--walk-teacher", default=None, help="TorchScript walk policy (75 -> 22) for walk_imitation")
    ap.add_argument("--phase-input", action="store_true", help="166 inputs: the request clock (contract.py)")
    ap.add_argument("--delay-ms", default="0,70")
    ap.add_argument("--push-dv", type=float, default=0.25)
    ap.add_argument("--eval-robots", type=int, default=64)
    ap.set_defaults(video_seconds=6.0, first_video_min=3.0)
    args = ap.parse_args()

    lo, hi = (float(x) for x in args.delay_ms.split(","))
    cfg = KickConfig(num_envs=args.num_envs, device=args.device, dtype=args.dtype, seed=args.seed,
                     delay_ms=(lo, hi), push_dv=args.push_dv, phase_input=args.phase_input,
                     walk_teacher=args.walk_teacher)
    env = KickEnv(cfg)
    run_dir = Path(args.log_root) / args.run_name
    print(f"[kick] {args.num_envs} robots on {args.device}, {env.contract.obs_dim} inputs, "
          f"{env.substeps} plant steps per policy step, logs -> {run_dir}", flush=True)

    video = KickVideo(lambda: KickEnv(quiet(cfg, num_envs=1, device="cpu", seed=cfg.seed + 1, energy_log_envs=0)),
                      args.video_seconds, cfg.policy_dt, run_dir / "videos", "kick")
    evaluate = KickEval(cfg, args.eval_robots) if args.eval_robots > 0 else None
    obs_dim = env.contract.obs_dim
    H.train(env, args, tag="kick", hidden_dims=C.HIDDEN_DIMS, export=lambda ck, out: export_bin(ck, out, obs_dim),
            video=video, evaluate=evaluate, artifact="k1-kick-ph",
            artifact_meta={"agent": "kick_walk", "obs_dim": obs_dim, "num_actions": C.NUM_ACTIONS},
            before_learn=init_actor(args.init_actor) if args.init_actor else None)


if __name__ == "__main__":
    main()
