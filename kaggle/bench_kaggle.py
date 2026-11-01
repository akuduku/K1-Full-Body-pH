"""The pH plant on a T4 GPU (Kaggle kernel or Colab notebook): the C++/CUDA plant is built, tested against
the Python one and benchmarked next to the PyTorch plant and MJX; a video of the servo hold + push goes to W&B.

The code comes from git: this repo (akuduku/K1-Full-Body-pH, private) is cloned at BRANCH with the
GITHUB_TOKEN (Kaggle: the private dataset <user>/k1-secrets, attached by run.sh; Colab: the key icon;
or an environment variable).
Push your changes before you start a run: the kernel sees only what is on GitHub.

Everything worth keeping lands in WORK (on Kaggle /kaggle/working, the kernel output):
  results/ph_*.json, results/mjx.json        raw numbers (ph_native_* = the C++/CUDA plant)
  logs/*.log                                 full output of every step (native_build, native_tests, ...)
  videos/hold_push.mp4                       servo hold + push on the native plant (also on W&B)
  summary.txt / summary.json                 the comparison table
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
NATIVE = True                    # build + test + benchmark the C++/CUDA plant (Physik C++)
NATIVE_BATCHES = "256,1024,4096,8192,16384"
VIDEO = True                     # servo hold + push video of the native plant, to W&B if WANDB_API_KEY is set
WANDB_ENTITY = "karl-jahnel-universit-t-leipzig"
WANDB_PROJECT = "Akos pH"

SETTINGS = ("BRANCH", "BATCHES", "DTYPES", "ITERS", "HOLD_S", "PROFILE_BATCH", "CPU_REF", "MJX", "MJX_BATCHES",
            "MJX_TIMESTEPS", "NATIVE", "NATIVE_BATCHES", "VIDEO", "WANDB_ENTITY", "WANDB_PROJECT")
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


def secrets_file():
    """The keys file of the attached private dataset <user>/k1-secrets (see kaggle/upload_secrets.sh)."""
    import glob
    import json
    for path in sorted(glob.glob("/kaggle/input/**/k1_secrets.json", recursive=True)):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            pass
    return {}


def secret(name):
    """A secret from the environment, the private dataset k1-secrets, Kaggle Secrets or Colab Secrets."""
    if os.environ.get(name):
        return os.environ[name]
    if secrets_file().get(name):
        return secrets_file()[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except ImportError:
        pass
    except Exception as exc:  # Kaggle's reason, never the value
        cause = exc.__cause__ or exc.__context__
        body = ""
        if cause is not None and hasattr(cause, "read"):  # Kaggle's error text (no value: the request failed)
            try:
                body = cause.read().decode(errors="replace")[:400]
            except Exception:
                pass
        print(f"secret {name} not available from Kaggle: {type(exc).__name__}: {str(exc)[:300]}"
              + (f" <- {type(cause).__name__}: {str(cause)[:300]}" if cause else "")
              + (f" | response: {body}" if body else ""), flush=True)
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
    from urllib.parse import urlparse
    for var in ("KAGGLE_URL_BASE", "KAGGLE_DATA_PROXY_URL"):  # service addresses, not secrets
        url = os.environ.get(var)
        if not url:
            out.append(f"{var}: unset")
            continue
        u = urlparse(url)
        try:
            socket.create_connection((u.hostname, u.port or (443 if u.scheme == "https" else 80)), timeout=5).close()
            out.append(f"{var}={u.scheme}://{u.hostname}: ok")
        except OSError as exc:
            out.append(f"{var}={u.scheme}://{u.hostname}: NO ({type(exc).__name__})")
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
            raise SystemExit("GITHUB_TOKEN missing: the repo is private. Kaggle: run upload_secrets.sh of the "
                             "k1-walk repo once (private dataset <user>/k1-secrets). Colab: key icon > GITHUB_TOKEN.")
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
    nat, nat_cpu = load(RESULTS / "ph_native_cuda.json"), load(RESULTS / "ph_native_cpu.json")
    lines = []
    if ph:
        lines.append(f"GPU: {ph.get('gpu')}   torch {ph.get('torch')}")
    if mjx:
        lines.append(f"MJX: jax {mjx.get('jax')}  mujoco {mjx.get('mujoco')}  model {mjx.get('xml')}")
    lines.append("")
    lines.append("policy steps/s (50 Hz policy, all robots together; higher is better)")
    lines.append(f"{'batch':>6} {'torch f32':>10} {'C++ f32':>10} {'C++ f64':>10} {'MJX 5ms':>10} {'MJX 1ms':>10} "
                 f"{'C++/torch':>9} {'MJX5/C++':>9} {'MJX1/C++':>9} {'C++: 1e8 steps':>15}")

    def sps(rows, **match):
        for r in rows or []:
            if all(r.get(k) == v for k, v in match.items()) and "policy_steps_per_s" in r:
                return r["policy_steps_per_s"]
        return None

    batches = sorted({r["batch"] for src in (ph, mjx, nat) if src for r in src["speed"] if "batch" in r})
    best = None
    for B in batches:
        t32 = sps(ph and ph["speed"], batch=B, dtype="float32")
        n32 = sps(nat and nat["speed"], batch=B, dtype="float32")
        n64 = sps(nat and nat["speed"], batch=B, dtype="float64")
        m5 = sps(mjx and mjx["speed"], batch=B, timestep=0.005)
        m1 = sps(mjx and mjx["speed"], batch=B, timestep=0.001)
        if n32 and (best is None or n32 > best[1]):
            best = (B, n32)
        f = lambda x: f"{x:>10.0f}" if x else f"{'-':>10}"  # noqa: E731
        r = lambda a, b: f"{a / b:>8.1f}x" if a and b else f"{'-':>9}"  # noqa: E731
        eta = f"{1e8 / n32 / 3600:>13.1f} h" if n32 else f"{'-':>15}"
        lines.append(f"{B:>6} {f(t32)} {f(n32)} {f(n64)} {f(m5)} {f(m1)} {r(n32, t32)} {r(m5, n32)} {r(m1, n32)} {eta}")
    if best:
        lines.append("")
        lines.append(f"best C++ plant: {best[1]:.0f} policy steps/s at batch {best[0]} (float32) "
                     f"-> 1e8 steps in {1e8 / best[1] / 3600:.1f} h")
    if nat_cpu:
        c = max((r.get("policy_steps_per_s", 0) for r in nat_cpu["speed"]), default=0)
        lines.append(f"C++ plant on this machine's CPU ({nat_cpu.get('threads')} threads): best {c:.0f} policy steps/s")
    for name in ("native_build", "native_tests"):
        log = LOGS / f"{name}.log"
        if log.exists():
            tail = [l for l in log.read_text().splitlines() if l.strip()][-1:]
            lines.append(f"{name}: {tail[0][:160] if tail else '(empty)'}")
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
    (WORK / "summary.json").write_text(json.dumps({"ph_cuda": ph, "ph_cpu": cpu, "mjx": mjx, "native_cuda": nat,
                                                   "native_cpu": nat_cpu, "timings": timings}, indent=2) + "\n")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    settings = {k: globals()[k] for k in SETTINGS}
    print(json.dumps(settings), flush=True)
    gpu = sh(["nvidia-smi"], "install", check=False) == 0
    print(network_report(), flush=True)
    cloned = clone_source()
    py = sys.executable
    installed = cloned and sh([py, "-m", "pip", "install", "-q", "mujoco", "imageio", "imageio-ffmpeg", "pytest",
                               "wandb"], "install", check=False) == 0
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
    # 0. the C++/CUDA plant: build (CUDA required), test against the Python plant on the GPU, benchmark
    built = False
    if NATIVE:
        built = sh([py, "setup.py", "build_ext", "--inplace"], "native_build", cwd=SRC / "Physik C++",
                   env={"PH_K1_CUDA": os.environ.get("PH_K1_CUDA", "1")}, check=False) == 0
        if built:
            sh([py, "-m", "pytest", SRC / "Physik C++" / "tests", "-q", "-p", "no:cacheprovider"], "native_tests",
               cwd=SRC, env={"PH_TEST_DEVICE": "cuda"}, check=False)
            sh(common + ["--impl", "native", "--device", "cuda", "--batches", NATIVE_BATCHES, "--dtypes", DTYPES,
                         "--hold-s", HOLD_S, "--json", RESULTS / "ph_native_cuda.json"], "ph_native_cuda", check=False)
            sh(common + ["--impl", "native", "--device", "cpu", "--batches", "64,256,1024", "--dtypes", "float32",
                         "--hold-s", 0, "--json", RESULTS / "ph_native_cpu.json"], "ph_native_cpu", check=False)
        else:
            print("native build failed; see logs/native_build.log", flush=True)
    # 1. the PyTorch plant on the GPU, big batches, both float types
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
    # 5. video of the native plant (servo hold + push), to W&B when a key is available
    if VIDEO and built:
        cmd = [py, SRC / "benchmarks" / "servo_video.py", "--device", "cuda", "--out", WORK / "videos" / "hold_push.mp4",
               "--summary", WORK / "summary.json"]
        env = {}
        key = secret("WANDB_API_KEY")
        if key:
            cmd += ["--wandb-project", WANDB_PROJECT, "--wandb-entity", WANDB_ENTITY]
            env["WANDB_API_KEY"] = key
        else:
            print("no WANDB_API_KEY: the video stays in videos/ only", flush=True)
        ok = sh(cmd, "video", env={**env, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}, check=False) == 0
        if not ok and not (WORK / "videos" / "hold_push.mp4").exists():  # no EGL: software rendering
            sh(["bash", "-c", "apt-get install -y -qq libosmesa6 > /dev/null 2>&1 || true"], "video", check=False)
            sh(cmd, "video", env={**env, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}, check=False)


if __name__ == "__main__":
    main()
