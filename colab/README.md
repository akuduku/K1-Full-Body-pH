# Colab

Two things run on Colab from this folder. Both clone this repo from GitHub at a branch, so **push first**.

## The get-up training (`k1-getup.yaml`, with colab-train)

[colab-train](https://github.com/Barathaner/colab-gpu-trainer) turns `k1-getup.yaml` into a notebook. The
training itself is `rl/getup/train.py` (see `rl/getup/README.md`).

```bash
colab-train validate colab/k1-getup.yaml     # the job, the resolved train command
colab-train doctor colab/k1-getup.yaml       # local setup, secret names, repo reachable
colab-train launch --browser colab/k1-getup.yaml
```

`launch --browser` uploads the notebook in colab-train's own Chrome, sets the T4 and presses **Run all**; the
one-time sign-in is `colab-train browser setup colab/k1-getup.yaml`. Without `--browser`, it prints the steps.

Colab Secrets (key icon, **Notebook access** on):
- `GITHUB_TOKEN`: read access to `akuduku/K1-Full-Body-pH`;
- `WANDB_API_KEY`.

**One session**, up to about 7 h, with the tab kept open:
1. `setup_gpu.sh`: packages next to Colab's CUDA torch, the CUDA build of `ph_k1_cuda`, its tests on the GPU
   (no training on a plant that fails them).
2. 6.75 h of training, a checkpoint every 10 min, a video every 30 min.
3. A clean stop: final checkpoint, `deploy.bin`, the W&B run finished.

**The next session:** **Run all** again. The newest checkpoint of the same W&B run (`getup-ph-v1`) is
restored, and the training continues the run and its curriculum. A disconnect in between loses at most 10 min.

**Free-tier limits** (not published, they change):
- a T4 when one is free;
- sessions end after a few hours at the latest (12 h hard maximum), and idle tabs time out;
- one session per account;
- a daily GPU quota: on "usage limit", wait some hours.

That is why the training is planned in time (6.75 h sessions), not in iterations. If Colab ends a session earlier, at
most 10 min are lost.

## The speed benchmark (`ph_bench_colab.ipynb`)

The same benchmark as `kaggle/` (pH plant on the GPU, physics check, profile, CPU reference, MJX on the same
GPU), as a Colab notebook.

1. Open https://colab.research.google.com, **File > Upload notebook**, pick `colab/ph_bench_colab.ipynb`.
2. Key icon in the left bar: add the secret `GITHUB_TOKEN` (read access to this repo), switch on notebook access.
3. Check **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**. Allow Google Drive access:
   results go to `MyDrive/ph_bench/<time>/`.
4. Keep the tab open until step 4 prints the table (about 30–60 min).

`python colab/make_notebook.py` rewrites the notebook after changing its cells.
