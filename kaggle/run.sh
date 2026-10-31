#!/bin/bash
# Push the benchmark kernel <user>/ph-k1-bench. The kernel clones this repo from GitHub at BRANCH,
# so commit and push first: the run sees only what is on GitHub.
#   kaggle/run.sh                              (BRANCH = the current branch)
#   kaggle/run.sh BATCHES=1024,4096 MJX=0
#   DRY_RUN=1 kaggle/run.sh ...                (show the staged kernel, push nothing)
set -euo pipefail
. "$(dirname "$0")/common.sh"

branch=$(git -C "$REPO" branch --show-current)
case " $* " in *" BRANCH="*) ;; *) set -- "BRANCH=$branch" "$@" ;; esac
want=$(printf '%s\n' "$@" | sed -n 's/^BRANCH=//p' | tail -1)
git -C "$REPO" fetch -q origin "$want" 2>/dev/null || { echo "branch $want is not on GitHub: git push -u origin $want" >&2; exit 1; }
if [ "$want" = "$branch" ] && [ "$(git -C "$REPO" rev-parse HEAD)" != "$(git -C "$REPO" rev-parse "origin/$want")" ]; then
    echo "HEAD differs from origin/$want: push first (git push origin $want), the kernel clones GitHub" >&2; exit 1
fi
echo "kernel clones $want @ $(git -C "$REPO" rev-parse --short "origin/$want")"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp "$KIT_DIR/bench_kaggle.py" "$KIT_DIR/kernel-metadata.json" "$STAGE/"
python3 - "$STAGE" "$KAGGLE_USER" "$KERNEL_SLUG" "$@" <<'PY'
import json, os, re, sys
from pathlib import Path

stage, user, slug, *pairs = sys.argv[1:]
stage = Path(stage)
script = (stage / "bench_kaggle.py").read_text()
meta = json.loads((stage / "kernel-metadata.json").read_text())
meta["id"] = f"{user}/{slug}"
meta["title"] = slug
# the keys only (upload_secrets.sh of the k1-walk repo); the code comes from git
secrets = os.environ.get("SECRETS_DATASET", f"{user}/k1-secrets")
meta["dataset_sources"] = [secrets] if secrets else []
for pair in pairs:
    key, value = pair.split("=", 1)
    line = re.search(rf"^{key} = (.*?)(\s+#.*)?$", script, re.M)
    if line is None:
        sys.exit(f"unknown setting {key}")
    old = line.group(1)
    if old.startswith('"'):
        new = json.dumps(value)
    elif old in ("True", "False"):
        new = str(value.lower() in ("1", "true", "yes"))
    elif "." in old:
        new = str(float(value))
    else:
        new = str(int(value))
    script = script[:line.start(1)] + new + script[line.end(1):]
    print(f"{key} = {new}")
(stage / "bench_kaggle.py").write_text(script)
(stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
PY
timeout=${TIMEOUT:-5400}   # 90 min: install + benchmarks + MJX compile
if [ -n "${DRY_RUN:-}" ]; then diff "$KIT_DIR/bench_kaggle.py" "$STAGE/bench_kaggle.py" || true; cat "$STAGE/kernel-metadata.json"; exit; fi
"$KAGGLE" kernels push -p "$STAGE" --accelerator NvidiaTeslaT4 -t "$timeout"
echo "pushed https://www.kaggle.com/code/$KERNEL (timeout ${timeout}s); fetch with kaggle/fetch.sh"
