"""Speed of the pH plant on a T4 GPU, next to MJX on the same GPU (Kaggle kernel or Colab notebook).

The code comes from git: this repo (akuduku/K1-Full-Body-pH, private) is cloned at BRANCH with the
GITHUB_TOKEN secret (Kaggle: Add-ons > Secrets; Colab: the key icon; or an environment variable).
Push your changes before you start a run: the kernel sees only what is on GitHub.

Everything worth keeping lands in WORK (on Kaggle /kaggle/working, the kernel output):
  results/ph_cuda.json, results/ph_cpu.json, results/mjx.json   raw numbers
  logs/*.log                                                     full output of every step
  summary.txt / summary.json                                     the comparison table
run.sh overrides the settings below in the copy it pushes. Elsewhere (the Colab notebook,
a local test) environment variables do: BENCH_WORK, BENCH_SRC for the paths, BENCH_REPO for
the repo URL (a local path works too) and BENCH_<SETTING> (e.g. BENCH_BATCHES=4) for any setting.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BRANCH = "kaggle-bench"          # git branch of akuduku/K1-Full-Body-pH to benchmark
BATCHES = "256,1024,4096,8192"   # pH plant on the GPU
DTYPES = "float32,float64"
ITERS = 3                        # timed policy steps per configuration
HOLD_S = 2.0                     # physics quality check (servo hold, gain x3) on the GPU
PROFILE_BATCH = 4096             # torch profiler table for one policy step (0 = skip)
CPU_REF = True                   # same benchmark on Kaggle's CPU, small batches
MJX = True                       # MJX reference on the same GPU (separate venv)
MJX_BATCHES = "1024,4096,8192"
MJX_TIMESTEPS = "0.005,0.001"    # 5 ms = what RL_Core trains with, 1 ms = the pH plant's step

SETTINGS = ("BRANCH", "BATCHES", "DTYPES", "ITERS", "HOLD_S", "PROFILE_BATCH", "CPU_REF", "MJX", "MJX_BATCHES",
            "MJX_TIMESTEPS")
for _key in SETTINGS:  # BENCH_<SETTING> overrides, cast to the default's type
    _val = os.environ.get(f"BENCH_{_key}")
    if _val is not None:
        _old = globals()[_key]
        globals()[_key] = (_val.lower() in ("1", "true", "yes") if isinstance(_old, bool)
                           else type(_old)(_val))

REPO = os.environ.get("BENCH_REPO", "https://github.com/akuduku/K1-Full-Body-pH.git")
ON_KAGGLE = Path("/kaggle/working").exists()
WORK = Path(os.environ.get("BENCH_WORK", "/kaggle/working"))
SRC = Path(os.environ.get("BENCH_SRC", "/tmp/phbench"))
LOGS = WORK / "logs"
RESULTS = WORK / "results"
MJX_VENV = Path("/tmp/mjxenv")
timings = {}


def sh(cmd, log, cwd=None, env=None, check=True):
    """Run ``cmd``, streaming its output to the kernel log and ``logs/<log>.log``."""
    print(f"$ {' '.join(map(str, cmd))}", flush=True)
    t0 = time.time()
    with open(LOGS / f"{log}.log", "a") as out:
        try:
            proc = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env={**os.environ, **(env or {})},
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except OSError as exc:  # program missing: a failed step, not a crashed kernel
            msg = f"{cmd[0]}: {exc}\n"
            sys.stdout.write(msg)
            out.write(msg)
            code = 127
        else:
            for line in proc.stdout:
                sys.stdout.write(line)
                out.write(line)
            code = proc.wait()
    timings[log] = round(timings.get(log, 0) + time.time() - t0, 1)
    (WORK / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    if check and code:
        raise SystemExit(f"{cmd[0]} failed with exit code {code} (logs/{log}.log)")
    return code


def secret(name):
    """A secret from the environment, Kaggle Secrets or Colab Secrets (None when none has it)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except ImportError:
        pass
    except Exception as exc:  # Kaggle's reason, never the value
        print(f"secret {name} not available from Kaggle: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def network_report():
    """Can this session reach GitHub, PyPI and the Kaggle secret service? Logged at the start of a run."""
    import socket
    out = []
    for host in ("github.com", "pypi.org"):
        try:
            socket.create_connection((host, 443), timeout=5).close()
            out.append(f"{host}: ok")
        except OSError as exc:
            out.append(f"{host}: NO ({type(exc).__name__})")
    out.append("secret service token: " + ("present" if os.environ.get("KAGGLE_USER_SECRETS_TOKEN") else "MISSING"))
    return "network: " + ", ".join(out)


def clone_source():
    """git clone --depth 1 of REPO at BRANCH into SRC -> True on success. The token goes through a
    credential helper that reads it from the environment, so it is never on a command line or in a log."""
    if (SRC / "ph_k1" / "plant.py").exists():
        print(f"using the existing checkout {SRC}", flush=True)
        return True
    env = {}
    if REPO.startswith("https://github.com/"):
        token = secret("GITHUB_TOKEN")
        if not token:
            raise SystemExit("GITHUB_TOKEN missing: the repo is private. Kaggle: Edit > Add-ons > Secrets > add "
                             "GITHUB_TOKEN and tick it for this notebook. Colab: key icon > GITHUB_TOKEN.")
        env["GITHUB_TOKEN"] = token
    helper = "!f() { echo username=x-access-token; echo \"password=$GITHUB_TOKEN\"; }; f"
    code = sh(["git", "-c", "credential.helper=", "-c", f"credential.helper={helper}", "clone", "--depth", "1",
               "--branch", BRANCH, REPO, SRC], "install", env=env, check=False)
    if code == 0:
        sh(["git", "-C", SRC, "log", "-1", "--format=commit %h %s (%cd)"], "install", check=False)
    return code == 0


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def summarize():
    """One table: best pH number per batch vs MJX at 5 ms and 1 ms."""
    ph, cpu, mjx = load(RESULTS / "ph_cuda.json"), load(RESULTS / "ph_cpu.json"), load(RESULTS / "mjx.json")
    lines = []
    if ph:
        lines.append(f"GPU: {ph.get('gpu')}   torch {ph.get('torch')}")
    if mjx:
        lines.append(f"MJX: jax {mjx.get('jax')}  mujoco {mjx.get('mujoco')}  model {mjx.get('xml')}")
    lines.append("")
    lines.append("policy steps/s (50 Hz policy, all robots together; higher is better)")
    lines.append(f"{'batch':>6} {'pH f32':>10} {'pH f64':>10} {'MJX 5ms':>10} {'MJX 1ms':>10} "
                 f"{'MJX5/pH':>8} {'MJX1/pH':>8} {'pH: 1e8 steps':>14}")

    def sps(rows, **match):
        for r in rows or []:
            if all(r.get(k) == v for k, v in match.items()) and "policy_steps_per_s" in r:
                return r["policy_steps_per_s"]
        return None

    batches = sorted({r["batch"] for src in (ph, mjx) if src for r in src["speed"] if "batch" in r})
    best = None
    for B in batches:
        p32 = sps(ph and ph["speed"], batch=B, dtype="float32")
        p64 = sps(ph and ph["speed"], batch=B, dtype="float64")
        m5 = sps(mjx and mjx["speed"], batch=B, timestep=0.005)
        m1 = sps(mjx and mjx["speed"], batch=B, timestep=0.001)
        p = max([x for x in (p32, p64) if x] or [0]) or None
        if p and (best is None or p > best[1]):
            best = (B, p)
        f = lambda x: f"{x:>10.0f}" if x else f"{'-':>10}"  # noqa: E731
        r = lambda a, b: f"{a / b:>7.0f}x" if a and b else f"{'-':>8}"  # noqa: E731
        eta = f"{1e8 / p / 3600:>12.1f} h" if p else f"{'-':>14}"
        lines.append(f"{B:>6} {f(p32)} {f(p64)} {f(m5)} {f(m1)} {r(m5, p)} {r(m1, p)} {eta}")
    if best:
        lines.append("")
        lines.append(f"best pH: {best[1]:.0f} policy steps/s at batch {best[0]} "
                     f"-> 1e8 steps in {1e8 / best[1] / 3600:.1f} h")
    if cpu:
        c = max((r.get("policy_steps_per_s", 0) for r in cpu["speed"]), default=0)
        lines.append(f"pH on this machine's CPU ({cpu.get('threads')} threads): best {c:.0f} policy steps/s")
    if ph and ph.get("quality"):
        for q in ph["quality"]:
            lines.append(f"quality on GPU ({q['dtype']}, gain x{q['gain']:g}, {q['seconds']:g} s hold): "
                         f"tilt {q['tilt_deg']:.2f} deg, ledger residual {q['residual_J']:.3f} J, "
                         f"finite={q['finite']}")
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    (WORK / "summary.txt").write_text(text + "\n")
    (WORK / "summary.json").write_text(json.dumps({"ph_cuda": ph, "ph_cpu": cpu, "mjx": mjx,
                                                   "timings": timings}, indent=2) + "\n")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    settings = {k: globals()[k] for k in SETTINGS}
    print(json.dumps(settings), flush=True)
    gpu = sh(["nvidia-smi"], "install", check=False) == 0
    print(network_report(), flush=True)
    cloned = clone_source()
    py = sys.executable
    installed = cloned and sh([py, "-m", "pip", "install", "-q", "mujoco"], "install", check=False) == 0
    if not (gpu and cloned and installed):
        # Kaggle silently gives a CPU-only, offline session when the account may not use GPU/internet
        log = (LOGS / "install.log").read_text()
        why = [] if gpu else ["no GPU in this session (nvidia-smi missing)"]
        if any(k in log for k in ("Could not resolve host", "pypi.org", "Temporary failure in name resolution")):
            why.append("no internet (GitHub/PyPI not reachable)")
        elif not cloned:
            why.append(f"git clone of {REPO} branch {BRANCH} failed (pushed? token valid?), see logs/install.log")
        elif not installed:
            why.append("pip install mujoco failed, see logs/install.log")
        fix = ("Kaggle grants GPU and internet only to phone-verified accounts: kaggle.com/settings -> Phone "
               "verification, then check the kernel's Settings (Accelerator: GPU T4 x2, Internet: on).\n"
               if ON_KAGGLE else
               "Colab: Runtime > Change runtime type > T4 GPU, then Runtime > Run all again.\n")
        text = "BENCHMARK NOT RUN: " + "; ".join(why) + ".\n" + fix
        print(text, flush=True)
        (WORK / "summary.txt").write_text(text)
        raise SystemExit(1)
    sh([py, "-c", "import torch, mujoco; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'mujoco', mujoco.__version__)"],
       "install")

    bench = SRC / "benchmarks" / "benchmark_ph.py"
    ph_path = SRC
    common = [py, bench, "--ph-path", ph_path, "--iters", ITERS]
    # 1. the number that matters: pH plant on the GPU, big batches, both float types
    sh(common + ["--device", "cuda", "--batches", BATCHES, "--dtypes", DTYPES, "--hold-s", HOLD_S,
                 "--json", RESULTS / "ph_cuda.json"], "ph_cuda", check=False)
    # 2. where the GPU time goes
    if PROFILE_BATCH:
        sh(common + ["--device", "cuda", "--batches", PROFILE_BATCH, "--dtypes", "float32", "--iters", 1,
                     "--hold-s", 0, "--profile"], "ph_profile", check=False)
    # 3. CPU reference on Kaggle's machine
    if CPU_REF:
        sh(common + ["--device", "cpu", "--batches", "64,256", "--dtypes", "float32", "--hold-s", 0,
                     "--json", RESULTS / "ph_cpu.json"], "ph_cpu", check=False)
    # 4. MJX on the same GPU, in its own venv (JAX must not touch the system torch install)
    if MJX:
        uv = [py, "-m", "uv"]
        if sh([py, "-m", "pip", "install", "-q", "uv"], "mjx_install", check=False) == 0 and \
                sh(uv + ["venv", MJX_VENV, "--python", "3.12"], "mjx_install", check=False) == 0 and \
                sh(uv + ["pip", "install", "--python", MJX_VENV / "bin" / "python", "mujoco", "mujoco-mjx",
                         "jax[cuda12]"], "mjx_install", check=False) == 0:
            sh([MJX_VENV / "bin" / "python", SRC / "benchmarks" / "benchmark_mjx.py",
                "--xml", SRC / "assets" / "K1" / "K1_22dof_mjx_prims.xml",
                "--batches", MJX_BATCHES, "--timesteps", MJX_TIMESTEPS, "--json", RESULTS / "mjx.json"],
               "mjx", check=False, env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
        else:
            print("MJX install failed; see logs/mjx_install.log", flush=True)
    summarize()


if __name__ == "__main__":
    main()
