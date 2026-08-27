#!/usr/bin/env bash
# Control the pod run from WSL, without ever needing tmux.
#
#   bash pod.sh start launch the screen under nohup and come straight back
#   bash pod.sh check interpreter, packages, CUDA, thread cap
#   bash pod.sh status is it alive, how far through, GPU utilisation
#   bash pod.sh log tail the last 40 lines
#   bash pod.sh follow stream the log live (Ctrl-C stops watching, not the job)
#   bash pod.sh analyse run the analysis on the pod and print it
#   bash pod.sh fetch copy results back into this folder, once
#   bash pod.sh sync keep copying them back every few minutes (run with nohup)
#   bash pod.sh stop kill the run (it can be restarted; finished work is kept)
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
POD_HOST="${POD_HOST:-<pod-host>}"
POD_PORT="${POD_PORT:-13087}"
POD_USER="${POD_USER:-root}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/ack1}"
ENVN="${ENVN:-ack1}"
FORGE="${FORGE:-/opt/miniforge3}"
ARGS="${ARGS:---stage screen}"
# Absolute interpreter path. Do NOT rely on `conda activate`: an SSH login shell reads
# ~/.bash_profile, not ~/.bashrc, so an env activated there is simply not present.
PY="${PY:-$FORGE/envs/$ENVN/bin/python}"
THREADS="${THREADS:-13}"
ENVVARS="OPENMM_CPU_THREADS=$THREADS OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS ACK1_THREADS=$THREADS"

SSH=(ssh -i "$KEY" -p "$POD_PORT" -o StrictHostKeyChecking=accept-new
     -o ServerAliveInterval=30 "${POD_USER}@${POD_HOST}")
SCP=(scp -i "$KEY" -P "$POD_PORT" -o StrictHostKeyChecking=accept-new)

# every remote command runs in the work dir with the thread caps set; the interpreter
# is addressed by absolute path so no shell rc file has to have been sourced
remote() { "${SSH[@]}" "cd $REMOTE_DIR && export $ENVVARS && $*"; }

case "${1:-status}" in

start)
  # setsid detaches the process from the ssh channel entirely, so ssh returns
  # immediately instead of hanging on to an open stdout. -u keeps python unbuffered
  # so the log is readable while it runs.
  # ship the launcher first so a stale copy on the pod cannot bite, then invoke it.
  # Launching via a script rather than a long ssh one-liner avoids the quoting trap
  # where line continuations inside quotes detach `cd` from the command it applies to.
  "${SCP[@]}" "$LOCAL_DIR/run_screen.sh" "${POD_USER}@${POD_HOST}:${REMOTE_DIR}/" >/dev/null
  "${SSH[@]}" "WORK=$REMOTE_DIR FORGE=$FORGE ENVN=$ENVN THREADS=$THREADS \
               bash $REMOTE_DIR/run_screen.sh $ARGS"
  echo
  echo "Running detached. It survives you closing this terminal and the ssh session."
  echo "Check on it with:  bash pod.sh status"
  ;;

check)
  remote "ls $PY >/dev/null 2>&1 && echo 'interpreter: ok' || echo 'interpreter: MISSING'; \
          $PY -c 'import openmm,pdbfixer,Bio,scipy; print(\"openmm\", openmm.version.version)' 2>&1 | tail -2; \
          $PY -c 'import openmm as m; print(\"platforms:\", [m.Platform.getPlatform(i).getName() for i in range(m.Platform.getNumPlatforms())])' 2>&1 | tail -1; \
          echo -n 'thread cap seen by python: '; $PY -c 'import os;print(os.environ.get(\"OPENMM_CPU_THREADS\"))'"
  ;;

status)
  remote "if [ -f screen.pid ] && kill -0 \$(cat screen.pid) 2>/dev/null; then \
            echo \"RUNNING  pid \$(cat screen.pid)\"; \
          else echo 'NOT RUNNING'; fi; \
          echo -n 'rows in screen_results.tsv: '; \
          if [ -f screen_results.tsv ]; then echo \$(( \$(wc -l < screen_results.tsv) - 1 )); \
          else echo 0; fi; \
          if [ -s screen_results.tsv ]; then \
            total=\$(grep -oE '^[0-9]+ runs to do' screen.log 2>/dev/null | head -1 | awk '{print \$1}'); \
            awk -F'\t' -v tot=\"\$total\" 'NR>1{s+=\$17;n++} END{ \
              if(n>0){ m=s/n; printf \"mean %.0f s/run\", m; \
                if(tot>0){ rem=tot-n; printf \"   remaining %d runs   ETA %.1f h\", rem, rem*m/3600 } \
                printf \"\\n\" } }' screen_results.tsv; \
          fi; \
          echo '--- last 3 log lines ---'; tail -n 3 screen.log 2>/dev/null || echo '(no log yet)'; \
          echo '--- gpu ---'; \
          nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader 2>/dev/null || echo '(no nvidia-smi)'"
  ;;

log)     remote "tail -n 40 screen.log" ;;
follow)  echo "streaming; Ctrl-C stops watching, the job keeps running"
         remote "tail -f screen.log" ;;
analyse) remote "$PY ack1_importin_gpu.py --stage analyse | tee analysis.txt" ;;

fetch)
  mkdir -p results
  for f in screen_results.tsv screen.log analysis.txt pocket_map.json; do
    "${SCP[@]}" "${POD_USER}@${POD_HOST}:${REMOTE_DIR}/$f" results/ 2>/dev/null \
      && echo "  got $f" || echo "  (no $f yet)"
  done
  ;;

sync)
  # Pull results back on a loop so a pod failure can never cost more than one interval.
  # A dated snapshot is kept only when the row count actually changes, so the backup
  # folder tracks real progress instead of filling with identical copies.
  INTERVAL="${INTERVAL:-300}"
  mkdir -p results results/snapshots
  echo "syncing every ${INTERVAL}s into $LOCAL_DIR/results  (Ctrl-C to stop)"
  last=-1
  while true; do
    if "${SCP[@]}" "${POD_USER}@${POD_HOST}:${REMOTE_DIR}/screen_results.tsv" \
         results/screen_results.tsv 2>/dev/null; then
      n=$(( $(wc -l < results/screen_results.tsv) - 1 ))
      if [ "$n" -ne "$last" ]; then
        stamp=$(date +%Y%m%d-%H%M%S)
        cp results/screen_results.tsv "results/snapshots/screen_results_${stamp}_${n}rows.tsv"
        echo "$(date +%H:%M:%S)  $n rows  -> snapshot saved"
        last=$n
      else
        echo "$(date +%H:%M:%S)  $n rows  (unchanged)"
      fi
      "${SCP[@]}" "${POD_USER}@${POD_HOST}:${REMOTE_DIR}/screen.log" \
        results/screen.log 2>/dev/null || true
    else
      echo "$(date +%H:%M:%S)  pod unreachable - will retry"
    fi
    sleep "$INTERVAL"
  done
  ;;

stop)
  remote "if [ -f screen.pid ]; then kill \$(cat screen.pid) 2>/dev/null && \
            echo 'stopped'; rm -f screen.pid; else echo 'no pid file'; fi"
  echo "Finished runs are already in screen_results.tsv; 'start' resumes from there."
  ;;

*) sed -n '2,12p' "$0" ;;
esac
