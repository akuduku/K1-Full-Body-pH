r"""Train the stand policy on the pH plant with rsl-rl PPO (Kaggle/Colab GPU; a CPU smoke run works too).

    python stand/train.py --num-envs 4096 --iterations 3000 --run-name stand-ph-v1          # GPU
    python stand/train.py --device cpu --num-envs 64 --iterations 20 --no-wandb             # smoke test

Long runs: --max-minutes stops cleanly before a session limit (final checkpoint + deploy.bin + artifact, the W&B
run finishes normally); --resume-wandb RUN continues from the newest checkpoint of an earlier W&B run (then
--iterations counts the additional iterations; the iteration numbers continue).

W&B (unless --no-wandb; needs WANDB_API_KEY): entity WANDB_ENTITY (default karl-jahnel-universit-t-leipzig),
project "Akos pH Stand", run = --run-name.  Logged: every iteration's losses/rewards/terminations/energy, a video
of the current policy every --video-every iterations (headless MuJoCo drawing, MUJOCO_GL=egl), every checkpoint
model_<it>.pt with its firmware file deploy_<it>.bin, and at the end the artifact "k1-stand-ph".
Output folder: logs/<run-name>/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco is imported (ph_k1 imports it)
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from stand import contract as C  # noqa: E402
from stand.env import REWARDS, StandConfig, StandEnv  # noqa: E402
from stand.export import export_bin  # noqa: E402


class TimeBudgetReached(Exception):
    """Raised after an iteration's log once --max-minutes is used up: rsl-rl's learn loop ends, the checkpoint is
    saved and the W&B run is finished normally (see the abort hook in main)."""


def fetch_wandb_checkpoint(entity: str, project: str, run_ref: str, dest: Path) -> Path:
    """The newest model_<it>.pt of a W&B run (given by its name or its id), downloaded into dest."""
    import re

    import wandb

    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_ref}")
    except Exception:
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_ref}, order="-created_at")
        if not len(runs):
            raise SystemExit(f"no W&B run {run_ref!r} in {entity}/{project}")
        run = runs[0]
    found = [(int(m.group(1)), f) for f in run.files() if (m := re.fullmatch(r"model_(\d+)\.pt", f.name))]
    if not found:
        raise SystemExit(f"W&B run {run.name} ({run.id}) has no model_<it>.pt yet")
    it, f = max(found, key=lambda x: x[0])
    dest.mkdir(parents=True, exist_ok=True)
    f.download(root=str(dest), replace=True)
    print(f"[stand] resume from {run.name} ({run.id}) model_{it}.pt", flush=True)
    return dest / f.name


def train_cfg(args) -> dict:
    """rsl-rl 5.5 OnPolicyRunner config; actor = the firmware's network (512-256-128 ELU + obs normalizer)."""
    logger = ({"class_name": "WandbLogWriter", "project_name": args.wandb_project} if not args.no_wandb
              else "tensorboard")
    return {
        "num_steps_per_env": args.steps_per_env,
        "save_interval": args.save_every,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "logger": logger,
        "check_for_nan": True,
        "multi_gpu": None,
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": args.entropy_coef,
            "learning_rate": args.learning_rate,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": list(C.HIDDEN_DIMS),
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": args.init_std},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": list(C.HIDDEN_DIMS), "activation": "elu",
                   "obs_normalization": True},
    }


class VideoMaker:
    """A rollout of the current policy on a 1-robot copy of the task, written as MP4 into the run folder (the
    rsl-rl logger uploads every new MP4 in the run folder to W&B at that iteration)."""

    def __init__(self, cfg: StandConfig, seconds: float, out_dir: Path):
        self.cfg = StandConfig(**{**cfg.__dict__, "num_envs": 1, "seed": cfg.seed + 1})
        self.steps = int(round(seconds / cfg.policy_dt))
        self.out_dir = out_dir
        self.env = None
        self.broken = False

    def __call__(self, policy, it: int) -> Path | None:
        if self.broken:
            return None
        try:
            from ph_k1.video import VideoRecorder

            # no_grad, not inference_mode: inference tensors in the env's buffers could not be written by a later
            # reset outside inference mode, and the plant reads its parameters' version counters
            if self.env is None:
                self.env = StandEnv(self.cfg)
            with torch.no_grad():
                env = self.env
                env._reset_idx(torch.arange(1, device=env.device))
                rec = VideoRecorder(env.plant, fps=1.0 / self.cfg.policy_dt)
                obs = env.get_observations()
                for _ in range(self.steps):
                    obs, _, _, _ = env.step(policy(obs))
                    rec.capture()
            path = rec.save(self.out_dir / f"stand_{it:05d}.mp4")
            rec.close()
            return path
        except Exception as exc:  # no EGL/OSMesa or no MP4 writer: train on, without videos
            print(f"[video] switched off: {type(exc).__name__}: {exc}", flush=True)
            self.broken = True
            return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=time.strftime("stand-ph-%Y%m%d-%H%M"))
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps-per-env", type=int, default=24)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--video-every", type=int, default=500)
    ap.add_argument("--video-seconds", type=float, default=8.0)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--entropy-coef", type=float, default=0.005)
    ap.add_argument("--init-std", type=float, default=0.5)
    ap.add_argument("--no-slew", action="store_true", help="train for an agent without the firmware slew limiter")
    ap.add_argument("--delay-ms", default="10,40", help="transport delay range, ms")
    ap.add_argument("--push-dv", type=float, default=0.3)
    ap.add_argument("--resume", default=None, help="model_<it>.pt to continue from")
    ap.add_argument("--resume-wandb", default=None,
                    help="continue from the newest checkpoint of this W&B run (name or id) in --wandb-project")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop cleanly after this much training time (0 = no limit), e.g. before a session cap")
    ap.add_argument("--log-root", default=str(REPO / "logs"))
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="Akos pH Stand")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "karl-jahnel-universit-t-leipzig"))
    args = ap.parse_args()

    from rsl_rl.runners import OnPolicyRunner

    if not args.no_wandb:
        os.environ.setdefault("WANDB_USERNAME", args.wandb_entity)  # rsl-rl's W&B writer reads the entity from here
    lo, hi = (float(x) for x in args.delay_ms.split(","))
    cfg = StandConfig(num_envs=args.num_envs, device=args.device, dtype=args.dtype, seed=args.seed,
                      slew=not args.no_slew, delay_ms=(lo, hi), push_dv=args.push_dv, rewards=dict(REWARDS))
    env = StandEnv(cfg)
    log_dir = Path(args.log_root) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stand] {args.num_envs} robots on {args.device}, stand height {env.stand_height:.3f} m, "
          f"{env.substeps} plant steps per policy step, logs -> {log_dir}", flush=True)

    tcfg = train_cfg(args)
    runner = OnPolicyRunner(env, tcfg, str(log_dir), device=str(env.device))
    if (REPO / ".git").exists():
        runner.add_git_repo_to_log(str(REPO))
    if args.resume_wandb:
        args.resume = str(fetch_wandb_checkpoint(args.wandb_entity, args.wandb_project, args.resume_wandb,
                                                 log_dir / "resume"))
    if args.resume:
        runner.load(args.resume)
        print(f"[stand] continuing at iteration {runner.current_learning_iteration}", flush=True)

    t0 = time.time()

    # every checkpoint also as the firmware's file, uploaded next to it
    save = runner.save

    def save_with_deploy(path, infos=None):
        save(path, infos)
        out = Path(path).with_name(Path(path).stem.replace("model", "deploy") + ".bin")
        try:
            export_bin(path, out)
            if runner.logger.writer is not None and hasattr(runner.logger.writer, "save_file"):
                runner.logger.writer.save_file(str(out))
        except Exception as exc:
            print(f"[export] {out.name} failed: {type(exc).__name__}: {exc}", flush=True)

    runner.save = save_with_deploy

    # a video every --video-every iterations, written just before the logger looks for new MP4s
    video = VideoMaker(cfg, args.video_seconds, log_dir / "videos")
    log = runner.logger.log
    run_id = {}

    def log_with_video(*a, **kw):
        it = kw.get("it", a[0] if a else 0)
        last = it == kw.get("total_it", 0) - 1
        if args.video_every and it > 0 and (it % args.video_every == 0 or last):
            policy = runner.get_inference_policy(device=str(env.device))
            path = video(policy, it)
            runner.alg.train_mode()
            if path:
                print(f"[video] {path.name}", flush=True)
        if not run_id and not args.no_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.config.update({"stand": env.config_dict()}, allow_val_change=True)
                    run_id["id"] = wandb.run.id
            except Exception:
                pass
        log(*a, **kw)
        if args.max_minutes > 0 and (time.time() - t0) / 60 > args.max_minutes:
            budget["hit"] = it
            raise TimeBudgetReached(it)

    runner.logger.log = log_with_video

    # time budget: rsl-rl ends learn() on any exception by aborting the logger; for ours, save the checkpoint
    # (and its deploy.bin, uploaded) and finish the W&B run normally instead
    budget = {}
    abort = runner.logger.abort_logging_writer

    def abort_or_finish():
        if "hit" in budget:
            runner.save(str(log_dir / f"model_{runner.current_learning_iteration}.pt"))
            runner.logger.stop_logging_writer()
        else:
            abort()

    runner.logger.abort_logging_writer = abort_or_finish

    try:
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        print(f"[stand] {args.iterations} iterations in {(time.time() - t0) / 60:.1f} min", flush=True)
    except TimeBudgetReached:
        print(f"[stand] time budget of {args.max_minutes:.0f} min used up: stopped cleanly after iteration "
              f"{budget['hit']}; continue with --resume-wandb {args.run_name}", flush=True)

    # the final policy as a W&B artifact on the same run
    finals = sorted(log_dir.glob("deploy_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    if finals and run_id and not args.no_wandb:
        import wandb

        wandb.init(entity=args.wandb_entity, project=args.wandb_project, id=run_id["id"], resume="must")
        art = wandb.Artifact("k1-stand-ph", type="model",
                             metadata={"agent": "stand", "obs_dim": C.OBS_DIM, "iterations": args.iterations,
                                       "file": finals[-1].name, "run": args.run_name})
        art.add_file(str(finals[-1]), name="deploy.bin")
        ck = finals[-1].with_name(finals[-1].name.replace("deploy", "model").replace(".bin", ".pt"))
        if ck.exists():
            art.add_file(str(ck))
        wandb.log_artifact(art, aliases=["latest", args.run_name])
        wandb.finish()
        print(f"[stand] artifact k1-stand-ph <- {finals[-1].name}", flush=True)


if __name__ == "__main__":
    main()
