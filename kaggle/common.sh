# Sourced by the Kaggle helpers: paths, the Kaggle CLI, the user name and the kernel id.
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$KIT_DIR/.." && pwd)
KAGGLE=${KAGGLE:-kaggle}   # e.g. KAGGLE=/path/to/venv/bin/kaggle
command -v "$KAGGLE" > /dev/null || { echo "kaggle CLI not found (pip install kaggle, or set KAGGLE=...)" >&2; exit 1; }
KAGGLE_USER=${KAGGLE_USER:-$("$KAGGLE" config view | sed -n 's/^- username: //p')}
[ -n "$KAGGLE_USER" ] || { echo "no Kaggle user name (set KAGGLE_USER)" >&2; exit 1; }
KERNEL_SLUG=${KERNEL_SLUG:-ph-k1-bench}
KERNEL="$KAGGLE_USER/$KERNEL_SLUG"
