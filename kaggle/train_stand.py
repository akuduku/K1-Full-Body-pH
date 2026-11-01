"""Kaggle kernel: train the stand policy on the pH plant (stand/train.py) on a T4, with W&B.

The code comes from git: this repo (akuduku/K1-Full-Body-pH, private) is cloned at BRANCH with GITHUB_TOKEN; the
W&B key is WANDB_API_KEY.  Both come from the private dataset <user>/k1-secrets (attached by kaggle/run.sh; made
by upload_secrets.sh of the k1-walk repo), or from Kaggle Secrets / the environment.

Steps: GPU + network check -> clone -> install (the preinstalled CUDA torch is kept) -> build the C++/CUDA plant
-> its tests ON THE GPU (C++ vs Python, the command path, the stand task; training starts only if they pass)
-> stand/train.py (W&B project WANDB_PROJECT, run RUN_NAME, a video every VIDEO_EVERY iterations, deploy_<it>.bin
with every checkpoint) -> runs/<RUN_NAME>/ in the kernel output.  kaggle/run.sh overrides the settings below:
    TASK=stand kaggle/run.sh ITERATIONS=3000 RUN_NAME=stand-ph-v1
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

BRANCH = "native-cuda"                    # git branch of akuduku/K1-Full-Body-pH to train
RUN_NAME = "stand-ph-v1"
NUM_ENVS = 4096
ITERATIONS = 3000
VIDEO_EVERY = 500
SAVE_EVERY = 250
WANDB_ENTITY = "karl-jahnel-universit-t-leipzig"
WANDB_PROJECT = "Akos pH Stand"
TRAIN_ARGS = ""                           # extra stand/train.py arguments, e.g. "--no-slew --delay-ms 10,30"
RUN_TESTS = True                          # GPU tests of the plant before training (stop if they fail)

REPO = "https://github.com/akuduku/K1-Full-Body-pH.git"
WORK = Path("/kaggle/working")
SRC = Path("/tmp/phstand")
LOGS = WORK / "logs"


def log(msg):
    print(f"[kaggle] {msg}", flush=True)


def run(cmd, logfile, env=None, cwd=None, check=True):
    log("$ " + " ".join(map(str, cmd)))
    with open(LOGS / logfile, "a") as out:
        try:
            proc = subprocess.Popen(list(map(str, cmd)), env={**os.environ, **(env or {})}, cwd=cwd,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except OSError as exc:
            out.write(f"{cmd[0]}: {exc}\n")
            print(f"{cmd[0]}: {exc}", flush=True)
            code = 127
        else:
            for line in proc.stdout:
                sys.stdout.write(line)
                out.write(line)
            code = proc.wait()
    if check and code:
        raise SystemExit(f"{cmd[0]} failed with exit code {code} (logs/{logfile})")
    return code


def secrets_file():
    import glob
    for path in sorted(glob.glob("/kaggle/input/**/k1_secrets.json", recursive=True)):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            pass
    return {}


def secret(name):
    if os.environ.get(name):
        return os.environ[name]
    if secrets_file().get(name):
        return secrets_file()[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def reachable(host):
    try:
        socket.create_connection((host, 443), timeout=5).close()
        return True
    except OSError:
        return False


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    settings = {k: globals()[k] for k in ("BRANCH", "RUN_NAME", "NUM_ENVS", "ITERATIONS", "VIDEO_EVERY", "SAVE_EVERY",
                                          "WANDB_ENTITY", "WANDB_PROJECT", "TRAIN_ARGS", "RUN_TESTS")}
    log(json.dumps(settings))
    log(f"network: github {'ok' if reachable('github.com') else 'NO'}, pypi {'ok' if reachable('pypi.org') else 'NO'}")
    github, wandb_key = secret("GITHUB_TOKEN"), secret("WANDB_API_KEY")
    missing = [n for n, v in (("GITHUB_TOKEN", github), ("WANDB_API_KEY", wandb_key)) if not v]
    if missing:
        raise SystemExit("secrets missing: " + ", ".join(missing) + " (private dataset <user>/k1-secrets: run "
                         "upload_secrets.sh of the k1-walk repo once; kaggle/run.sh attaches it)")
    if run(["nvidia-smi"], "install.log", check=False) and not os.environ.get("PH_ALLOW_CPU"):
        raise SystemExit("no GPU in this session: Kaggle gives GPU and internet only to phone-verified accounts; "
                         "the kernel's Settings need Accelerator GPU T4 x2 and Internet on")

    # the code, from git
    if SRC.exists():
        shutil.rmtree(SRC)
    helper = '!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'
    run(["git", "-c", "credential.helper=", "-c", f"credential.helper={helper}", "clone", "--depth", "1",
         "--branch", BRANCH, REPO, SRC], "install.log", env={"GITHUB_TOKEN": github})
    run(["git", "-C", SRC, "log", "-1", "--format=commit %h %s (%cd)"], "install.log")

    # dependencies; the preinstalled CUDA torch stays (a constraint, so tensordict follows torch, not vice versa)
    py = sys.executable
    import torch
    log(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    constraints = WORK / "constraints.txt"
    constraints.write_text(f"torch=={torch.__version__}\n")
    run([py, "-m", "pip", "install", "-q", "-c", constraints, "rsl-rl-lib==5.5.2", "tensordict", "mujoco", "wandb",
         "imageio", "imageio-ffmpeg", "tensorboard", "pytest"], "install.log")
    run(["bash", "-c", "apt-get install -y -qq libegl1 libgl1 libglvnd0 > /dev/null 2>&1 || true"], "install.log",
        check=False)

    # the C++/CUDA plant
    run([py, "setup.py", "build_ext", "--inplace"], "build.log", cwd=SRC / "Physik C++",
        env={"PH_K1_CUDA": os.environ.get("PH_K1_CUDA", "1")})  # CUDA required (a local CPU test sets 0)
    if RUN_TESTS:
        code = run([py, "-m", "pytest", SRC / "Physik C++" / "tests", SRC / "stand" / "tests", "-q",
                    "-p", "no:cacheprovider"], "tests.log", cwd=SRC,
                   env={"PH_TEST_DEVICE": os.environ.get("PH_TEST_DEVICE", "cuda")}, check=False)
        if code:
            raise SystemExit("GPU tests failed (logs/tests.log): not training on a plant that disagrees with its "
                             "reference")

    # training
    env = {"WANDB_API_KEY": wandb_key, "WANDB_ENTITY": WANDB_ENTITY, "WANDB_USERNAME": WANDB_ENTITY,
           "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1"}
    cmd = [py, SRC / "stand" / "train.py", "--run-name", RUN_NAME, "--num-envs", NUM_ENVS, "--iterations", ITERATIONS,
           "--video-every", VIDEO_EVERY, "--save-every", SAVE_EVERY, "--wandb-project", WANDB_PROJECT,
           "--wandb-entity", WANDB_ENTITY, "--log-root", SRC / "logs", *TRAIN_ARGS.split()]
    start = time.time()
    code = run(cmd, "train.log", env=env, cwd=SRC, check=False)
    log(f"training exit code {code} after {(time.time() - start) / 60:.1f} min")
    run_dir = SRC / "logs" / RUN_NAME
    if run_dir.exists():
        shutil.copytree(run_dir, WORK / "runs" / RUN_NAME, dirs_exist_ok=True)
    if code:
        raise SystemExit(f"training failed (exit code {code}), see logs/train.log")
    log("done")


if __name__ == "__main__":
    main()
