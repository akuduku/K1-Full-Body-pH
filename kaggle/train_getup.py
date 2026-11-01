"""Kaggle kernel: train the get-up policy on the pH plant (rl/getup/train.py) on a T4, with W&B, by time.

The same training as the Colab job colab/k1-getup.yaml, as a Kaggle kernel (12 h sessions, the browser can be
closed).  Code from git (this repo, branch BRANCH, cloned with GITHUB_TOKEN); keys from the private dataset
<user>/k1-secrets (attached by kaggle/run.sh).

Steps: GPU + network check -> clone -> colab/setup_gpu.sh (packages next to the preinstalled CUDA torch, the CUDA
build of ph_k1_cuda, its tests ON THE GPU: no training if they fail) -> rl/getup/train.py for HOURS hours of
training (checkpoint every 10 min, video every 30 min, clean stop: final checkpoint + deploy.bin + W&B artifact)
-> runs/<RUN_NAME>/ in the kernel output, written there live.

    TASK=getup kaggle/run.sh                                    # fresh run getup-ph-kaggle-v1
    TASK=getup kaggle/run.sh RUN_NAME=getup-ph-kaggle-v1b RESUME=getup-ph-kaggle-v1   # continue in a new session
"""

import json
import os
import time

KERNEL_START = time.time()  # the session clock starts with the kernel
import shutil
import socket
import subprocess
import sys
from pathlib import Path

BRANCH = "native-cuda"                    # git branch of akuduku/K1-Full-Body-pH to train
RUN_NAME = "getup-ph-kaggle-v1"           # W&B run (the Colab job uses getup-ph-v1)
NUM_ENVS = 4096
HOURS = 7.0                               # training time of this session (Kaggle's cap: 12 h incl. setup)
WANDB_ENTITY = "karl-jahnel-universit-t-leipzig"
WANDB_PROJECT = "Akos pH GetUp"
TRAIN_ARGS = ""                           # extra rl/getup/train.py arguments, e.g. "--no-tank"
RUN_TESTS = True                          # GPU tests before training (stop if they fail)
RESUME = ""                               # W&B run (name or id) in WANDB_PROJECT to continue from; "" = fresh
SESSION_HOURS = 12.0                      # Kaggle's session cap: HOURS is cut to fit into it
MARGIN_MINUTES = 20

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
    settings = {k: globals()[k] for k in ("BRANCH", "RUN_NAME", "NUM_ENVS", "HOURS", "WANDB_ENTITY", "WANDB_PROJECT",
                                          "TRAIN_ARGS", "RUN_TESTS", "RESUME", "SESSION_HOURS", "MARGIN_MINUTES")}
    log(json.dumps(settings))
    log(f"network: github {'ok' if reachable('github.com') else 'NO'}, pypi {'ok' if reachable('pypi.org') else 'NO'}")
    github, wandb_key = secret("GITHUB_TOKEN"), secret("WANDB_API_KEY")
    missing = [n for n, v in (("GITHUB_TOKEN", github), ("WANDB_API_KEY", wandb_key)) if not v]
    if missing:
        raise SystemExit("secrets missing: " + ", ".join(missing) + " (private dataset <user>/k1-secrets: run "
                         "upload_secrets.sh of the k1-walk repo once; kaggle/run.sh attaches it)")
    if run(["nvidia-smi"], "install.log", check=False) and not os.environ.get("PH_ALLOW_CPU"):
        raise SystemExit("no GPU in this session: the kernel needs Accelerator GPU T4 and Internet on")

    # the code, from git
    if SRC.exists():
        shutil.rmtree(SRC)
    helper = '!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'
    run(["git", "-c", "credential.helper=", "-c", f"credential.helper={helper}", "clone", "--depth", "1",
         "--branch", BRANCH, REPO, SRC], "install.log", env={"GITHUB_TOKEN": github})
    run(["git", "-C", SRC, "log", "-1", "--format=commit %h %s (%cd)"], "install.log")

    # packages, the CUDA build and the GPU tests: the same script as on Colab
    py = sys.executable
    run(["bash", "-c", "apt-get install -y -qq libegl1 libgl1 libglvnd0 > /dev/null 2>&1 || true"], "install.log",
        check=False)
    setup_env = {"PYTHON": py, "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5+PTX")}
    if not RUN_TESTS:
        setup_env["PH_TEST_DEVICE"] = "skip"
    code = run(["bash", SRC / "colab" / "setup_gpu.sh"], "setup.log", env=setup_env, cwd=SRC, check=False)
    if code and RUN_TESTS:
        raise SystemExit("setup or GPU tests failed (logs/setup.log): not training on a plant that disagrees with "
                         "its reference")

    # training, by time
    env = {"WANDB_API_KEY": wandb_key, "WANDB_ENTITY": WANDB_ENTITY, "WANDB_USERNAME": WANDB_ENTITY,
           "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1"}
    left = SESSION_HOURS * 60 - (time.time() - KERNEL_START) / 60 - MARGIN_MINUTES
    minutes = max(1.0, min(HOURS * 60, left))
    log(f"training time {minutes:.0f} min (HOURS {HOURS:g}, session {SESSION_HOURS:g} h, {MARGIN_MINUTES} min margin)")
    # the run folder IS in the kernel output: checkpoints, deploy.bin and videos are kept even if Kaggle kills it
    cmd = [py, SRC / "rl" / "getup" / "train.py", "--run-name", RUN_NAME, "--num-envs", NUM_ENVS,
           "--max-minutes", f"{minutes:.1f}", "--save-every-min", "10", "--video-every-min", "30",
           "--wandb-project", WANDB_PROJECT, "--wandb-entity", WANDB_ENTITY, "--log-root", WORK / "runs",
           *(["--resume-wandb", RESUME] if RESUME else []), *TRAIN_ARGS.split()]
    start = time.time()
    code = run(cmd, "train.log", env=env, cwd=SRC, check=False)
    log(f"training exit code {code} after {(time.time() - start) / 60:.1f} min")
    if code:
        raise SystemExit(f"training failed (exit code {code}), see logs/train.log")
    log("done")


if __name__ == "__main__":
    main()
