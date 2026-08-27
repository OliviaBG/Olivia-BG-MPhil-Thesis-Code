#!/usr/bin/env bash
# Runs ON THE POD. Starts the screen detached and returns immediately.
#
#   bash run_screen.sh # --stage screen with defaults
#   bash run_screen.sh --stage screen --seeds 5 --md 800
#   bash run_screen.sh --paralogues KPNA2
#
# Everything is addressed by absolute path on purpose. An SSH *login* shell reads
# ~/.bash_profile rather than ~/.bashrc, so a conda env activated in.bashrc is not
# present and `python` silently resolves to the system interpreter. This script never
# depends on any rc file having been sourced.
set -euo pipefail

WORK="${WORK:-/workspace/ack1}"
FORGE="${FORGE:-/opt/miniforge3}"
ENVN="${ENVN:-ack1}"
THREADS="${THREADS:-13}"
PY="${PY:-$FORGE/envs/$ENVN/bin/python}"

cd "$WORK"

if [ ! -x "$PY" ]; then
  echo "Interpreter not found: $PY"
  echo "The conda env did not build. Re-run the bootstrap:"
  echo "   WORK=$WORK THREADS=$THREADS bash $WORK/pod_bootstrap.sh"
  exit 1
fi

if [ -f screen.pid ] && kill -0 "$(cat screen.pid)" 2>/dev/null; then
  echo "already running, pid $(cat screen.pid)"
  tail -n 5 screen.log 2>/dev/null || true
  exit 0
fi

export OPENMM_CPU_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS"
export ACK1_THREADS="$THREADS"

ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--stage screen)

# setsid detaches from the SSH channel so ssh returns instead of holding the connection
# open; < /dev/null stops it blocking on stdin; -u keeps the log readable while running.
setsid nohup "$PY" -u ack1_importin_gpu.py "${ARGS[@]}" \
    > screen.log 2>&1 < /dev/null &
echo $! > screen.pid

sleep 5
echo "started, pid $(cat screen.pid)   args: ${ARGS[*]}   threads: $THREADS"
echo "--- first log lines ---"
tail -n 12 screen.log 2>/dev/null || echo "(nothing yet)"
