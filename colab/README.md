# pH plant speed on a Colab T4

The same benchmark as `kaggle/` (pH plant on the GPU, physics check, profile, CPU reference, MJX on the same
GPU) as a Colab notebook. It clones this repo from GitHub at a branch, so **push first**.

1. Open https://colab.research.google.com, **File > Upload notebook**, pick `colab/ph_bench_colab.ipynb`.
2. Key icon in the left bar: add the secret `GITHUB_TOKEN` (read access to this repo), switch on notebook access.
3. Check **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**. Allow Google Drive access:
   results go to `MyDrive/ph_bench/<time>/`.
4. Keep the tab open until step 4 prints the table (about 30–60 min).

`python colab/make_notebook.py` rewrites the notebook after changing its cells.
