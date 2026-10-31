# pH plant speed on a Kaggle T4

Measures how many policy steps per second the pH plant (`ph_k1`) reaches on a Kaggle T4 GPU, and runs MJX on
the same GPU and robot as the reference (`benchmarks/benchmark_ph.py`, `benchmarks/benchmark_mjx.py`).

The kernel `<user>/ph-k1-bench` is private and **clones this repo from GitHub** at a branch. Commit and push
before a run: the kernel sees only what is on GitHub.

One-time setup:
- the `kaggle` CLI, logged in, on a phone-verified account (otherwise Kaggle gives no GPU and no internet)
- a GitHub fine-grained token with read-only *Contents* on this repo, added as Kaggle Secret `GITHUB_TOKEN`
  and ticked for the kernel (kernel page → Edit → Add-ons → Secrets)

```bash
git push origin <branch>
kaggle/run.sh                 # push the kernel; it clones the current branch (~30-60 min on a T4)
kaggle/fetch.sh t4            # wait, download to outputs/kaggle/t4/, print summary.txt
```

`run.sh KEY=VALUE ...` overrides the settings at the top of `bench_kaggle.py` (`DRY_RUN=1` shows the staged
kernel without pushing):

| Setting | Default | |
|---|---|---|
| `BRANCH` | current branch | branch the kernel clones |
| `BATCHES` | `256,1024,4096,8192` | robots in parallel for the pH plant on the GPU |
| `DTYPES` | `float32,float64` | float32 gives the same physics (hold tilt 0.9°, ledger 0.015 J) |
| `ITERS` | 3 | timed policy steps per configuration |
| `HOLD_S` | 2.0 | physics check on the GPU: servo hold (gain x3), tilt and energy-ledger residual |
| `PROFILE_BATCH` | 4096 | torch profiler table of one policy step (0 = skip) |
| `CPU_REF` | 1 | the same benchmark on Kaggle's CPU |
| `MJX` | 1 | MJX on the same GPU, own venv; model `assets/K1/K1_22dof_mjx_prims.xml` (the plain K1 XML has cylinder-mesh contacts MJX cannot do) |
| `MJX_BATCHES` | `1024,4096,8192` | |
| `MJX_TIMESTEPS` | `0.005,0.001` | 5 ms = what RL_Core trains with; 1 ms = the pH plant's step |

Output: `summary.txt` (the comparison table), `results/*.json`, `logs/*.log` (`ph_profile.log` = where the
GPU time goes). Locally the benchmarks run on CPU: `python benchmarks/benchmark_ph.py`.
