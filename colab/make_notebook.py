"""Write the Colab notebook that benchmarks the pH plant on a T4 GPU (colab/ph_bench_colab.ipynb).

The notebook clones this repo from GitHub (private: Colab Secret GITHUB_TOKEN, the key icon in the
left bar) at a branch and runs the same driver as the Kaggle kernel, ``kaggle/bench_kaggle.py``.
Push your changes first: the notebook sees only what is on GitHub.

    python colab/make_notebook.py            # -> colab/ph_bench_colab.ipynb

Then: colab.research.google.com -> File > Upload notebook (or open it from GitHub) -> Runtime > Run all.
Results go to Google Drive (MyDrive/ph_bench/<time>/) and are offered as a zip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


# Plain Python (no ! magics), so a test can run the cells outside Colab with COLAB_ROOT and BENCH_REPO set.
CELL_SETUP = r'''
#@title 1. GPU check and where the results go
import datetime, os, subprocess, sys
from pathlib import Path

SAVE_TO_DRIVE = True  #@param {type:"boolean"}
ROOT = Path(os.environ.get("COLAB_ROOT", "/content"))

try:
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "nvidia-smi: no output")
except OSError:
    print("!!! NO GPU: Runtime > Change runtime type > T4 GPU, then Runtime > Run all again.")

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
OUT = ROOT / "ph_bench_out" / stamp
if SAVE_TO_DRIVE:
    try:
        from google.colab import drive
        drive.mount(str(ROOT / "drive"))
        OUT = ROOT / "drive" / "MyDrive" / "ph_bench" / stamp
    except Exception as exc:  # not on Colab, or the Drive consent was declined
        print(f"Google Drive not used ({type(exc).__name__}); results stay in {OUT}")
OUT.mkdir(parents=True, exist_ok=True)
print("results ->", OUT)
'''

CELL_CLONE = r'''
#@title 2. Clone the code from GitHub (needs the Colab Secret GITHUB_TOKEN)
BRANCH = "kaggle-bench"  #@param {type:"string"}
REPO = os.environ.get("BENCH_REPO", "https://github.com/akuduku/K1-Full-Body-pH.git")
SRC = ROOT / "phbench"
env = dict(os.environ)
if REPO.startswith("https://github.com/"):
    try:
        from google.colab import userdata
        env["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
    except Exception as exc:
        raise SystemExit("Add the Colab Secret GITHUB_TOKEN (key icon, left bar) with notebook access, "
                         f"then run this cell again ({type(exc).__name__})")
helper = '!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'
if SRC.exists():
    subprocess.run(["rm", "-rf", str(SRC)], check=True)
subprocess.run(["git", "-c", "credential.helper=", "-c", f"credential.helper={helper}", "clone", "--depth", "1",
                "--branch", BRANCH, REPO, str(SRC)], env=env, check=True)
print(subprocess.run(["git", "-C", str(SRC), "log", "-1", "--format=%h %s (%cd)"],
                     capture_output=True, text=True).stdout)
'''

CELL_RUN = r'''
#@title 3. Run the benchmark (about 30-60 min; keep this tab open)
#@markdown Leave the defaults for the full comparison. Smaller lists finish sooner.
BATCHES = "256,1024,4096,8192"  #@param {type:"string"}
DTYPES = "float32,float64"  #@param {type:"string"}
MJX = True  #@param {type:"boolean"}
MJX_BATCHES = "1024,4096,8192"  #@param {type:"string"}
run_env = {**os.environ, "BENCH_WORK": str(OUT), "BENCH_SRC": str(SRC), "BENCH_BRANCH": BRANCH,
           "BENCH_BATCHES": BATCHES, "BENCH_DTYPES": DTYPES, "BENCH_MJX": str(MJX),
           "BENCH_MJX_BATCHES": MJX_BATCHES}
run_env.update({k: v for k, v in os.environ.items() if k.startswith("BENCH_") and k not in
                ("BENCH_WORK", "BENCH_SRC", "BENCH_BRANCH")})  # test overrides win
proc = subprocess.Popen([sys.executable, str(SRC / "kaggle" / "bench_kaggle.py")], env=run_env, cwd=str(ROOT),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="")
print("\nexit code", proc.wait())
'''

CELL_RESULTS = r'''
#@title 4. Results (also saved in the folder printed in step 1)
import shutil
summary = OUT / "summary.txt"
print(summary.read_text() if summary.exists() else "no summary.txt - see the output of step 3")
archive = shutil.make_archive(str(ROOT / f"ph_bench_{stamp}"), "zip", OUT)
print("zip:", archive)
try:
    from google.colab import files
    files.download(archive)
except Exception:
    pass
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "ph_bench_colab.ipynb")
    args = ap.parse_args()
    cells = [
        markdown("""
# pH plant vs MJX on a Colab T4

Measures how many policy steps per second the port-Hamiltonian K1 plant (`ph_k1`) reaches on this GPU, next to
MJX on the same GPU and robot. The code is cloned from GitHub (`akuduku/K1-Full-Body-pH`, private).

1. Key icon in the left bar: add the secret **`GITHUB_TOKEN`** (a GitHub token with read access to the repo) and
   switch on *Notebook access*.
2. **Runtime > Change runtime type > T4 GPU** (should already be selected).
3. **Runtime > Run all**. Allow Google Drive access when asked: the results are saved there
   (`MyDrive/ph_bench/<time>/`), so you can read them later even after this tab is closed.
4. Keep the tab open until step 4 prints the table (about 30-60 min).
"""),
        code(CELL_SETUP),
        code(CELL_CLONE),
        code(CELL_RUN),
        code(CELL_RESULTS),
    ]
    nb = {"nbformat": 4, "nbformat_minor": 0, "cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": args.out.name},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}}
    args.out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
