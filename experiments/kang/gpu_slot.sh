#!/usr/bin/env bash
#
# Run a command pinned to one exclusively-held GPU.
#
#   gpu_slot.sh python -u experiments/kang/02_architecture_ablation.py ...
#
# Snakemake's `resources: gpu=1` caps how many GPU jobs run at once, but it hands
# out a count, not a device id, so without this every job would land on device 0
# and the other two cards would idle. Each slot is an flock on a file under
# $GPU_LOCKDIR; the lock is held for exactly as long as the command runs, because
# flock itself execs the command and the kernel drops the lock when that process
# exits. No cleanup path to get wrong, and a killed job frees its GPU.
#
# -E 99 is what makes the loop safe: without it flock's "could not acquire"
# collides with a command that genuinely exited 1, and a failing job would be
# silently retried on the next card.
set -uo pipefail

N_GPUS="${N_GPUS:-3}"
LOCKDIR="${GPU_LOCKDIR:-.snakemake/gpu_slots}"
mkdir -p "$LOCKDIR"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

for i in $(seq 0 $((N_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES="$i" flock -n -E 99 "$LOCKDIR/gpu$i.lock" "$@"
  rc=$?
  if [ "$rc" -ne 99 ]; then
    exit "$rc"
  fi
done

echo "[gpu_slot] every GPU in 0..$((N_GPUS - 1)) is busy; run snakemake with" \
     "--resources gpu=$N_GPUS so it never oversubscribes" >&2
exit 1
