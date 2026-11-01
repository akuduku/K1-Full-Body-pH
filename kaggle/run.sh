#!/bin/bash
# Push a kernel that clones this repo from GitHub at BRANCH (commit and push first: the run sees only GitHub).
#   kaggle/run.sh                                         benchmark kernel <user>/ph-k1-bench (bench_kaggle.py)
#   kaggle/run.sh BATCHES=1024,4096 MJX=0
#   TASK=stand kaggle/run.sh                              stand training <user>/ph-k1-stand (train_stand.py)
#   TASK=stand kaggle/run.sh ITERATIONS=3000 RUN_NAME=stand-ph-v1
#   DRY_RUN=1 kaggle/run.sh ...                           (show the staged kernel, push nothing)
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
cp "$KIT_DIR/$SCRIPT" "$KIT_DIR/kernel-metadata.json" "$STAGE/"
python3 - "$STAGE" "$KAGGLE_USER" "$KERNEL_SLUG" "$SCRIPT" "$@" <<'PY'
import json, os, re, sys
from pathlib import Path

stage, user, slug, code_file, *pairs = sys.argv[1:]
stage = Path(stage)
script = (stage / code_file).read_text()
meta = json.loads((stage / "kernel-metadata.json").read_text())
meta["id"] = f"{user}/{slug}"
meta["code_file"] = code_file
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
(stage / code_file).write_text(script)
(stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
PY
timeout=${TIMEOUT:-$DEFAULT_TIMEOUT}   # bench 90 min, stand 12 h (Kaggle's session cap)
if [ -n "${DRY_RUN:-}" ]; then diff "$KIT_DIR/$SCRIPT" "$STAGE/$SCRIPT" || true; cat "$STAGE/kernel-metadata.json"; exit; fi
"$KAGGLE" kernels push -p "$STAGE" --accelerator NvidiaTeslaT4 -t "$timeout"
echo "pushed https://www.kaggle.com/code/$KERNEL (timeout ${timeout}s); fetch with TASK=$TASK kaggle/fetch.sh"
