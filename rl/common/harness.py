r"""The PPO training harness shared by the skills in rl/ (rsl-rl 5.5 OnPolicyRunner on a pH-plant VecEnv).

What it adds around rsl-rl's learn loop:
  * W&B (WandbLogWriter): every iteration's losses and env logs, the skill's config, every checkpoint
    model_<it>.pt with the firmware file deploy_<it>.bin next to it, a video, and at the end an artifact
  * checkpoints and videos every N iterations (--save-every / --video-every) or every N minutes of training
    (--save-every-min / --video-every-min: on a GPU session that can end at any time, a disconnect loses at most
    that much)
  * a clean stop: after --max-minutes / --hours of training, or on SIGINT (Ctrl-C, colab-train's session limit),
    the running iteration finishes, the checkpoint + deploy file are saved and uploaded and the W&B run is
    finished normally (a second SIGINT aborts at once)
  * resume: --resume <model_it.pt>, or --resume-wandb <run> (the newest checkpoint of that W&B run); the env's
    curriculum state (success rates, ramps) is saved inside every checkpoint and restored with it

An env may define curriculum_state() -> dict and load_curriculum_state(dict), and config_dict() -> dict.
"""

from __future__ import annotations

import math
import os
import re
import signal
import time
from pathlib import Path
from typing import Callable

import torch

REPO = Path(__file__).resolve().parents[2]


class StopTraining(Exception):
    """Raised after an iteration's log once the time budget is used up or SIGINT came: rsl-rl's learn loop ends,
    the checkpoint is saved and the W&B run is finished normally (see the abort hook in train)."""


TimeBudgetReached = StopTraining  # the name rl/stand used before


def add_arguments(ap, *, run_prefix: str, wandb_project: str, iterations: int = 3000, save_every: int = 250,
                  video_every: int = 500, save_every_min: float = 0.0, video_every_min: float = 0.0,
                  entropy_coef: float = 0.005, init_std: float = 0.5, learning_rate: float = 1e-3,
                  eval_every_min: float = 0.0, first_eval_min: float = 0.0) -> None:
    ap.add_argument("--run-name", default=time.strftime(f"{run_prefix}-%Y%m%d-%H%M"))
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=iterations,
                    help="upper bound; with --hours/--max-minutes the time budget usually ends the run first")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps-per-env", type=int, default=24)
    ap.add_argument("--save-every", type=int, default=save_every, help="checkpoint every N iterations (0: off)")
    ap.add_argument("--video-every", type=int, default=video_every, help="video every N iterations (0: off)")
    ap.add_argument("--save-every-min", type=float, default=save_every_min,
                    help="checkpoint every N minutes of training (0: off)")
    ap.add_argument("--video-every-min", type=float, default=video_every_min,
                    help="video every N minutes of training (0: off)")
    ap.add_argument("--first-video-min", type=float, default=0.0,
                    help="with --video-every-min: the first video after this many minutes (0: after one interval)")
    ap.add_argument("--video-seconds", type=float, default=8.0)
    ap.add_argument("--eval-every-min", type=float, default=eval_every_min,
                    help="evaluate the policy without action noise every N minutes of training, if the skill has an "
                         "evaluation (0: off)")
    ap.add_argument("--first-eval-min", type=float, default=first_eval_min,
                    help="with --eval-every-min: the first evaluation after this many minutes (0: after one interval)")
    ap.add_argument("--learning-rate", type=float, default=learning_rate)
    ap.add_argument("--entropy-coef", type=float, default=entropy_coef)
    ap.add_argument("--init-std", type=float, default=init_std)
    ap.add_argument("--resume", default=None, help="model_<it>.pt to continue from")
    ap.add_argument("--resume-wandb", default=None,
                    help="continue from the newest checkpoint of this W&B run (name or id) in --wandb-project")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop cleanly after this much training time (0 = no limit), e.g. before a session cap")
    ap.add_argument("--hours", type=float, default=0.0, help="the same in hours (the larger of the two counts)")
    ap.add_argument("--log-root", default=str(REPO / "logs"))
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default=wandb_project)
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "karl-jahnel-universit-t-leipzig"))


def ppo_config(args, hidden_dims, *, num_mini_batches: int = 4, learning_epochs: int = 5, gamma: float = 0.99,
               lam: float = 0.95, desired_kl: float = 0.01) -> dict:
    """rsl-rl 5.5 OnPolicyRunner config; actor = the firmware's network (512-256-128 ELU + obs normalizer)."""
    logger = ({"class_name": "WandbLogWriter", "project_name": args.wandb_project} if not args.no_wandb
              else "tensorboard")
    every = args.save_every if args.save_every > 0 else 10**9  # time-based saves only: rsl-rl's own at it 0 and the end
    return {
        "num_steps_per_env": args.steps_per_env,
        "save_interval": every,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "logger": logger,
        "check_for_nan": True,
        "multi_gpu": None,
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": learning_epochs,
            "num_mini_batches": num_mini_batches,
            "clip_param": 0.2,
            "gamma": gamma,
            "lam": lam,
            "value_loss_coef": 1.0,
            "entropy_coef": args.entropy_coef,
            "learning_rate": args.learning_rate,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": desired_kl,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": list(hidden_dims),
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": args.init_std},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": list(hidden_dims), "activation": "elu",
                   "obs_normalization": True},
    }


def fetch_wandb_checkpoint(entity: str, project: str, run_ref: str, dest: Path, tag: str = "train") -> Path:
    """The newest model_<it>.pt of a W&B run (given by its name or its id), downloaded into dest."""
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
    print(f"[{tag}] resume from {run.name} ({run.id}) model_{it}.pt", flush=True)
    return dest / f.name


class VideoMaker:
    """A rollout of the current policy on a 1-robot copy of the task, written as MP4 into the run folder (the
    rsl-rl logger uploads every new MP4 in the run folder to W&B at that iteration).

    make_env() builds the 1-robot env (once, lazily); prepare(env, it) puts it into the start state of the video
    (default: reset robot 0).  The env may live on another device than the policy (e.g. the CPU while training on
    the GPU): the observations are moved to the policy's device, the env takes the actions to its own."""

    def __init__(self, make_env: Callable, seconds: float, policy_dt: float, out_dir: Path, prefix: str,
                 prepare: Callable | None = None):
        self.make_env = make_env
        self.steps = int(round(seconds / policy_dt))
        self.fps = 1.0 / policy_dt
        self.out_dir = out_dir
        self.prefix = prefix
        self.prepare = prepare or (lambda env, it: env._reset_idx(torch.arange(1, device=env.device)))
        self.env = None
        self.broken = False
        self.policy_device = None  # set by train(): where the policy runs

    def __call__(self, policy, it: int) -> Path | None:
        if self.broken:
            return None
        try:
            from ph_k1.video import VideoRecorder

            # no_grad, not inference_mode: inference tensors in the env's buffers could not be written by a later
            # reset outside inference mode, and the plant reads its parameters' version counters
            with torch.no_grad():
                if self.env is None:
                    self.env = self.make_env()
                env = self.env
                self.prepare(env, it)
                rec = VideoRecorder(env.plant, fps=self.fps)
                to_policy = (lambda o: o) if self.policy_device is None else (lambda o: o.to(self.policy_device))
                obs = env.get_observations()
                for _ in range(self.steps):
                    obs, _, _, _ = env.step(policy(to_policy(obs)))
                    rec.capture()
            path = rec.save(self.out_dir / f"{self.prefix}_{it:05d}.mp4")
            rec.close()
            return path
        except Exception as exc:  # no EGL/OSMesa or no MP4 writer: train on, without videos
            print(f"[video] switched off: {type(exc).__name__}: {exc}", flush=True)
            self.broken = True
            return None


def hold_locked_std(policy, mask: torch.Tensor | None, cap: float) -> None:
    """Clamp the action std of the outputs in mask ((num_actions,) bool) to at most cap, in place.

    For outputs that act on nothing (an env's locked_actions()): PPO's entropy bonus raises their std and nothing in
    the return pulls it back (getup-mb-v1d: 0.8 -> 3.4 for the 44 locked gain outputs)."""
    dist = getattr(policy, "distribution", None)
    if dist is None or mask is None or not bool(mask.any()):
        return
    with torch.no_grad():
        if hasattr(dist, "std_param"):
            p, top = dist.std_param, cap
        elif hasattr(dist, "log_std_param"):
            p, top = dist.log_std_param, math.log(cap)
        else:
            return
        m = mask.to(p.device)
        p[m] = p[m].clamp(max=top)


def action_std(policy) -> torch.Tensor | None:
    """(num_actions,) std of a state-independent Gaussian policy, from its parameter (output_std is batch-shaped)."""
    dist = getattr(policy, "distribution", None)
    if hasattr(dist, "std_param"):
        return dist.std_param.detach()
    if hasattr(dist, "log_std_param"):
        return dist.log_std_param.detach().exp()
    return None


def train(env, args, *, tag: str, hidden_dims, export: Callable | None = None, video: VideoMaker | None = None,
          evaluate: Callable | None = None, artifact: str | None = None, artifact_meta: dict | None = None,
          ppo: dict | None = None) -> Path:
    """Run PPO on env with the flags of add_arguments; returns the run folder.

    export(checkpoint_path, out_path) writes the firmware file; artifact: W&B artifact name for the final
    deploy.bin + checkpoint.  evaluate(policy, it) -> {name: number}: run every --eval-every-min minutes with the
    deterministic policy (no action noise), logged at that iteration.  An env with locked_actions() -> (num_actions,)
    bool gets the std of those outputs held at --init-std."""
    from rsl_rl.runners import OnPolicyRunner

    if not args.no_wandb:
        os.environ.setdefault("WANDB_USERNAME", args.wandb_entity)  # rsl-rl's W&B writer reads the entity from here
    log_dir = Path(args.log_root) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, ppo or ppo_config(args, hidden_dims), str(log_dir), device=str(env.device))
    if (REPO / ".git").exists():
        runner.add_git_repo_to_log(str(REPO))
    if args.resume_wandb:
        args.resume = str(fetch_wandb_checkpoint(args.wandb_entity, args.wandb_project, args.resume_wandb,
                                                 log_dir / "resume", tag))
    if args.resume:
        infos = runner.load(args.resume)
        if isinstance(infos, dict) and infos.get("curriculum") is not None and hasattr(env, "load_curriculum_state"):
            env.load_curriculum_state(infos["curriculum"])
            print(f"[{tag}] curriculum restored: {infos['curriculum']}", flush=True)
        print(f"[{tag}] continuing at iteration {runner.current_learning_iteration}", flush=True)

    def hold_std():
        if hasattr(env, "locked_actions"):
            hold_locked_std(runner.alg.get_policy(), env.locked_actions(), args.init_std)

    hold_std()  # a resumed run: before its first rollout

    budget_min = max(args.max_minutes, 60.0 * args.hours)
    t0 = time.time()
    state = {"last_save": t0, "last_video": t0, "last_eval": t0, "reason": None, "sigint": False, "run_id": None}
    if args.first_video_min > 0:  # an early video proves the video path works
        state["last_video"] = t0 - max(args.video_every_min - args.first_video_min, 0.0) * 60
    eval_every = getattr(args, "eval_every_min", 0.0) if evaluate is not None else 0.0
    if eval_every > 0 and getattr(args, "first_eval_min", 0.0) > 0:  # an early one: the start value, the path works
        state["last_eval"] = t0 - max(eval_every - args.first_eval_min, 0.0) * 60

    # every checkpoint also as the firmware's file, uploaded next to it; the env's curriculum goes inside
    save = runner.save

    def save_with_deploy(path, infos=None):
        infos = dict(infos or {})
        if hasattr(env, "curriculum_state"):
            infos["curriculum"] = env.curriculum_state()
        save(path, infos)
        if export is None:
            return
        out = Path(path).with_name(Path(path).stem.replace("model", "deploy") + ".bin")
        try:
            export(path, out)
            if runner.logger.writer is not None and hasattr(runner.logger.writer, "save_file"):
                runner.logger.writer.save_file(str(out))
        except Exception as exc:
            print(f"[export] {out.name} failed: {type(exc).__name__}: {exc}", flush=True)

    runner.save = save_with_deploy

    def make_video(it):
        if video is None:
            return
        policy = runner.get_inference_policy(device=str(env.device))
        video.policy_device = env.device
        path = video(policy, it)
        runner.alg.train_mode()
        if path:
            print(f"[video] {path.name}", flush=True)

    def run_eval(it):
        policy = runner.get_inference_policy(device=str(env.device))
        t = time.time()
        try:
            results = evaluate(policy, it)
        except Exception as exc:  # an evaluation must never end a training
            print(f"[eval] failed: {type(exc).__name__}: {exc}", flush=True)
            results = {}
        finally:
            runner.alg.train_mode()
        results = {**results, "Eval/seconds": time.time() - t}
        writer = runner.logger.writer
        if writer is not None:
            for k, v in results.items():
                writer.add_scalar(k, float(v), it)
        print(f"[eval] it {it}: " + ", ".join(f"{k.split('/')[-1]} {float(v):.3g}" for k, v in results.items()),
              flush=True)

    # videos and time-based checkpoints just before / after the logger (it uploads the new MP4s of this iteration)
    log = runner.logger.log

    def log_hook(*a, **kw):
        it = kw.get("it", a[0] if a else 0)
        now = time.time()
        last = it == kw.get("total_it", 0) - 1
        over = budget_min > 0 and (now - t0) / 60 > budget_min
        if over and state["reason"] is None:
            state["reason"] = f"time budget of {budget_min:.1f} min used up"
        if state["sigint"] and state["reason"] is None:
            state["reason"] = "SIGINT"
        stopping = state["reason"] is not None
        by_it = args.video_every and it > 0 and (it % args.video_every == 0 or last)
        by_min = args.video_every_min > 0 and it > 0 and (now - state["last_video"]) / 60 >= args.video_every_min
        if by_it or by_min or (stopping and it > 0 and (args.video_every or args.video_every_min)):
            make_video(it)
            state["last_video"] = time.time()
        if eval_every > 0 and it > 0 and ((now - state["last_eval"]) / 60 >= eval_every or stopping):
            run_eval(it)
            state["last_eval"] = time.time()
        hold_std()  # after this iteration's update, before the next rollout
        if hasattr(env, "locked_actions") and runner.logger.writer is not None:
            free = ~env.locked_actions()
            std = action_std(runner.alg.get_policy())
            if std is not None and bool(free.any()) and std.numel() == free.numel():  # mean_std counts locked ones
                runner.logger.writer.add_scalar("Policy/std_free_outputs", float(std[free.to(std.device)].mean()), it)
        if state["run_id"] is None and not args.no_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    if hasattr(env, "config_dict"):
                        wandb.config.update({tag: env.config_dict()}, allow_val_change=True)
                    state["run_id"] = wandb.run.id
            except Exception:
                pass
        log(*a, **kw)
        if (args.save_every_min > 0 and not stopping and runner.logger.writer is not None
                and (time.time() - state["last_save"]) / 60 >= args.save_every_min):
            runner.save(str(log_dir / f"model_{it}.pt"))
            state["last_save"] = time.time()
        if stopping:
            raise StopTraining(it)

    runner.logger.log = log_hook

    # a clean stop: rsl-rl ends learn() on any exception by aborting the logger; for ours, save the checkpoint
    # (and its deploy file, uploaded) and finish the W&B run normally instead
    abort = runner.logger.abort_logging_writer

    def abort_or_finish():
        if state["reason"] is not None:
            runner.save(str(log_dir / f"model_{runner.current_learning_iteration}.pt"))
            runner.logger.stop_logging_writer()
        else:
            abort()

    runner.logger.abort_logging_writer = abort_or_finish

    def on_sigint(signum, frame):
        if state["sigint"]:
            signal.default_int_handler(signum, frame)  # the second one: KeyboardInterrupt, abort at once
        state["sigint"] = True
        print(f"[{tag}] SIGINT: stopping cleanly after this iteration (again to abort)", flush=True)

    old_handler = signal.signal(signal.SIGINT, on_sigint)
    try:
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        print(f"[{tag}] {args.iterations} iterations in {(time.time() - t0) / 60:.1f} min", flush=True)
    except StopTraining:
        print(f"[{tag}] {state["reason"]}: stopped cleanly after iteration {runner.current_learning_iteration} "
              f"({(time.time() - t0) / 60:.1f} min); continue with --resume-wandb {args.run_name}", flush=True)
    finally:
        signal.signal(signal.SIGINT, old_handler)

    # the final policy as a W&B artifact on the same run
    finals = sorted(log_dir.glob("deploy_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    if artifact and finals and state["run_id"] and not args.no_wandb:
        import wandb

        wandb.init(entity=args.wandb_entity, project=args.wandb_project, id=state["run_id"], resume="must")
        art = wandb.Artifact(artifact, type="model", metadata={**(artifact_meta or {}), "file": finals[-1].name,
                                                                "run": args.run_name,
                                                                "iteration": runner.current_learning_iteration})
        art.add_file(str(finals[-1]), name="deploy.bin")
        ck = finals[-1].with_name(finals[-1].name.replace("deploy", "model").replace(".bin", ".pt"))
        if ck.exists():
            art.add_file(str(ck))
        wandb.log_artifact(art, aliases=["latest", args.run_name])
        wandb.finish()
        print(f"[{tag}] artifact {artifact} <- {finals[-1].name}", flush=True)
    return log_dir
