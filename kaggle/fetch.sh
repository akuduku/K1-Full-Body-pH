#!/bin/bash
# Wait for <user>/ph-k1-bench to finish, download its output to outputs/kaggle/<name>/ and print the summary.
#   kaggle/fetch.sh [name]      (name defaults to a timestamp)
set -euo pipefail
. "$(dirname "$0")/common.sh"
OUT="$REPO/outputs/kaggle/${1:-$(date +%Y%m%d-%H%M)}"
while true; do
    status=$("$KAGGLE" kernels status "$KERNEL" 2>&1 | tail -1)
    echo "$(date +%H:%M:%S) $status"
    case "$status" in
        *[Rr]unning*|*RUNNING*|*[Qq]ueued*|*QUEUED*) sleep "${POLL:-60}" ;;
        *) break ;;
    esac
done
mkdir -p "$OUT"
"$KAGGLE" kernels output "$KERNEL" -p "$OUT" -o -q
echo "downloaded to $OUT"
[ -f "$OUT/summary.txt" ] && cat "$OUT/summary.txt"
case "$status" in *[Cc]omplete*|*COMPLETE*) ;; *) echo "kernel ended with: $status" >&2; exit 1 ;; esac
