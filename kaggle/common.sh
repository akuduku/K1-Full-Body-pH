# Sourced by the Kaggle helpers: paths, the Kaggle CLI, the user name and the kernel id.
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$KIT_DIR/.." && pwd)
KAGGLE=${KAGGLE:-kaggle}   # e.g. KAGGLE=/path/to/venv/bin/kaggle
command -v "$KAGGLE" > /dev/null || { echo "kaggle CLI not found (pip install kaggle, or set KAGGLE=...)" >&2; exit 1; }
KAGGLE_USER=${KAGGLE_USER:-$("$KAGGLE" config view | sed -n 's/^- username: //p')}
[ -n "$KAGGLE_USER" ] || { echo "no Kaggle user name (set KAGGLE_USER)" >&2; exit 1; }
# TASK=bench (default): speed benchmark + plant validation; TASK=stand / getup: train that policy
TASK=${TASK:-bench}
case "$TASK" in
    bench) SCRIPT=bench_kaggle.py; DEFAULT_SLUG=ph-k1-bench; DEFAULT_TIMEOUT=5400 ;;
    stand) SCRIPT=train_stand.py; DEFAULT_SLUG=ph-k1-stand; DEFAULT_TIMEOUT=43200 ;;   # Kaggle's 12 h session cap
    getup) SCRIPT=train_getup.py; DEFAULT_SLUG=ph-k1-getup; DEFAULT_TIMEOUT=43200 ;;
    *) echo "TASK=$TASK: use bench, stand or getup" >&2; exit 1 ;;
esac
KERNEL_SLUG=${KERNEL_SLUG:-$DEFAULT_SLUG}
KERNEL="$KAGGLE_USER/$KERNEL_SLUG"
