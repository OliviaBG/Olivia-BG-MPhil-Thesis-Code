#!/usr/bin/env bash
# Drive several pods at once, one shard of the panel each.
#
#   bash pods.sh deploy copy files + bootstrap every pod (idempotent)
#   bash pods.sh push copy the pipeline script to every pod (fast, no bootstrap)
#   bash pods.sh check interpreter / CUDA / thread cap on each
#   bash pods.sh start launch shard i of N on pod i
#   bash pods.sh status progress and ETA for each, plus the combined total
#   bash pods.sh sync pull every shard back on a loop (run with nohup)
#   bash pods.sh fetch pull every shard back once
#   bash pods.sh merge combine the shards and run the analysis locally
#   bash pods.sh stop stop all of them
#
# Pods are listed one per line in pods.txt as: host port [user]
#
#   <pod-host> 13087
#   <pod-host> 40011
#...
#
# Sharding is interleaved, not blocked: pod 1 takes jobs 1, 5, 9... So if one pod
# dies you still have an even cross-section of the panel rather than one corner of it,
# and its shard can simply be re-run on any surviving pod.
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PODS_FILE="${PODS_FILE:-$LOCAL_DIR/pods.txt}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
# RunPod network volumes can be attached to several pods at once, so /workspace may be
# SHARED. Every pod therefore gets its own subdirectory; a shared working directory
# means shared scratch files, shared logs and mutual corruption.
REMOTE_BASE="${REMOTE_BASE:-/workspace/ack1}"
rdir() { echo "$REMOTE_BASE/pod$(( $1 + 1 ))"; }
FORGE="${FORGE:-/opt/miniforge3}"
ENVN="${ENVN:-ack1}"
THREADS="${THREADS:-13}"
ARGS="${ARGS:---stage screen}"
PY="$FORGE/envs/$ENVN/bin/python"

[ -f "$PODS_FILE" ] || { echo "No $PODS_FILE. See the header of this script."; exit 1; }

mapfile -t PODLINES < <(grep -vE '^\s*(#|$)' "$PODS_FILE")
N=${#PODLINES[@]}
[ "$N" -gt 0 ] || { echo "$PODS_FILE is empty"; exit 1; }

host_of() { echo "${PODLINES[$1]}" | awk '{print $1}'; }
port_of() { echo "${PODLINES[$1]}" | awk '{print $2}'; }
user_of() { echo "${PODLINES[$1]}" | awk '{print ($3==""?"root":$3)}'; }

# RunPod's basic proxy (ssh.runpod.io) refuses non-interactive commands without a PTY,
# so force one for proxy hosts. Note the proxy still cannot do scp/sftp at all -- for
# file transfer a pod must expose direct TCP.
is_proxy() { [[ "$(host_of $1)" == *runpod.io ]]; }
ssh_to()  { local t=(); is_proxy $1 && t=(-tt)
            ssh "${t[@]}" -i "$KEY" -p "$(port_of $1)" \
                -o StrictHostKeyChecking=accept-new \
                -o ServerAliveInterval=30 "$(user_of $1)@$(host_of $1)" "${@:2}"; }
scp_to()  { scp -i "$KEY" -P "$(port_of $1)" -o StrictHostKeyChecking=accept-new \
                "${@:2}" "$(user_of $1)@$(host_of $1):$(rdir $1)/"; }
scp_from() { scp -i "$KEY" -P "$(port_of $1)" -o StrictHostKeyChecking=accept-new \
                "$(user_of $1)@$(host_of $1):$(rdir $1)/$2" "$3" 2>/dev/null; }

FILES=(ack1_importin_gpu.py 1EJL.pdb sam_dimer_fixed.pdb kpna_seqs.fasta
       pod_bootstrap.sh run_screen.sh fix_cuda.sh)
# NOTE: every pod gets its own copy of the inputs inside its own directory

banner() { printf '\n\033[1m--- pod %d/%d  %s:%s ---\033[0m\n' \
           "$(( $1 + 1 ))" "$N" "$(host_of $1)" "$(port_of $1)"; }

case "${1:-status}" in

deploy)
  for i in $(seq 0 $((N-1))); do
    banner $i
    if is_proxy $i; then
      echo "  SKIPPED: $(host_of $i) is RunPod's SSH proxy, which does not support"
      echo "  scp/sftp, so files cannot be copied here. Get this pod's direct TCP"
      echo "  endpoint from the RunPod console (Connect -> SSH over exposed TCP) and"
      echo "  put 'IP PORT root' in pods.txt instead."
      continue
    fi
    ssh_to $i "mkdir -p $(rdir $i)"
    for f in "${FILES[@]}"; do
      [ -f "$LOCAL_DIR/$f" ] && scp_to $i "$LOCAL_DIR/$f" >/dev/null && echo "  sent $f"
    done
    # verify the inputs arrived intact before anything is built from them
    for f in 1EJL.pdb sam_dimer_fixed.pdb kpna_seqs.fasta ack1_importin_gpu.py; do
      lsum=$(md5sum "$LOCAL_DIR/$f" | awk '{print $1}')
      rsum=$(ssh_to $i "md5sum $(rdir $i)/$f 2>/dev/null | awk '{print \$1}'" | tr -d '\r')
      if [ "$lsum" != "$rsum" ]; then
        echo "  CHECKSUM MISMATCH on $f (local $lsum, remote $rsum) - resending"
        scp_to $i "$LOCAL_DIR/$f" >/dev/null
      fi
    done
    ssh_to $i "WORK=$(rdir $i) THREADS=$THREADS bash $(rdir $i)/pod_bootstrap.sh" \
      2>&1 | tail -25
    # the PTX/driver mismatch is the one failure that always needs fixing afterwards
    ssh_to $i "bash -s" < "$LOCAL_DIR/fix_cuda.sh" 2>&1 | tail -6 || true
    # last, so the bootstrap's own generated launcher cannot overwrite ours
    scp_to $i "$LOCAL_DIR/run_screen.sh" >/dev/null && echo "  launcher installed"
  done
  ;;

push)
  # just the code, no bootstrap: for pushing a fix to pods that are already set up
  for i in $(seq 0 $((N-1))); do
    banner $i
    for f in ack1_importin_gpu.py run_screen.sh; do
      scp_to $i "$LOCAL_DIR/$f" >/dev/null && echo "  sent $f"
    done
    lsum=$(md5sum "$LOCAL_DIR/ack1_importin_gpu.py" | awk '{print $1}')
    rsum=$(ssh_to $i "md5sum $(rdir $i)/ack1_importin_gpu.py | awk '{print \$1}'" | tr -d '\r')
    [ "$lsum" = "$rsum" ] && echo "  checksum ok" || echo "  CHECKSUM MISMATCH"
  done
  ;;

check)
  for i in $(seq 0 $((N-1))); do
    banner $i
    ssh_to $i "$PY -c 'import openmm as m; print(\"openmm\", m.version.version)'; \
               $PY -c \"
import openmm as mm
from openmm import unit
s=mm.System(); s.addParticle(1.0)
f=mm.NonbondedForce(); f.addParticle(0.0,0.1,0.1); s.addForce(f)
try:
    c=mm.Context(s, mm.VerletIntegrator(0.001*unit.picoseconds),
                 mm.Platform.getPlatformByName('CUDA'))
    c.setPositions([[0,0,0]]); c.getState(getEnergy=True); print('CUDA: WORKS')
except Exception as e: print('CUDA: FAILS ->', e)
\"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader" 2>&1 | tail -5
  done
  ;;

start)
  # Collect everything already finished anywhere and push it to every pod, so no run
  # is repeated -- this matters when one pod has been running unsharded beforehand.
  mkdir -p "$LOCAL_DIR/results"
  echo "== collecting completed runs from all pods"
  for i in $(seq 0 $((N-1))); do
    # a glob, not a fixed name: pods may hold results from an earlier run with a
    # different shard count, and those runs must still count as done
    mkdir -p "$LOCAL_DIR/results/pod$((i+1))"
    scp -i "$KEY" -P "$(port_of $i)" -o StrictHostKeyChecking=accept-new \
        "$(user_of $i)@$(host_of $i):$(rdir $i)/screen_results*.tsv" \
        "$LOCAL_DIR/results/pod$((i+1))/" 2>/dev/null || true
  done
  KNOWN="$LOCAL_DIR/results/screen_results_known.tsv"
  cat "$LOCAL_DIR"/results/pod*/screen_results*.tsv 2>/dev/null \
    | grep -v '^paralogue' | sort -u > "$KNOWN.body" || true
  if [ -s "$KNOWN.body" ]; then
    printf '%s\n' "$(cat "$LOCAL_DIR"/results/pod*/screen_results*.tsv 2>/dev/null \
      | grep -m1 '^paralogue' || echo)" > "$KNOWN"
    cat "$KNOWN.body" >> "$KNOWN"
    echo "  $(wc -l < "$KNOWN.body") completed runs found; distributing to all pods"
  else
    : > "$KNOWN"
    echo "  none found (clean start)"
  fi
  rm -f "$KNOWN.body"

  for i in $(seq 0 $((N-1))); do
    banner $i
    scp_to $i "$LOCAL_DIR/run_screen.sh" >/dev/null
    # Derived files live on a network volume and can be left NUL-padded by a hard
    # kill; they are cheap to rebuild, so always start from clean ones.
    # no orphan may survive into the new run and sit on the GPU
    ssh_to $i "pkill -9 -f ack1_importin_gpu.py 2>/dev/null; sleep 2; true"
    ssh_to $i "mkdir -p $(rdir $i) && cd $(rdir $i) && mkdir -p old && \
               mv -f screen_results.tsv screen_results_shard*.tsv old/ 2>/dev/null; \
               rm -f _site_*.pdb _sasa_tmp.pdb _carve_*.pdb screen.log; true"
    [ -s "$KNOWN" ] && scp_to $i "$KNOWN" >/dev/null
    ssh_to $i "WORK=$(rdir $i) FORGE=$FORGE ENVN=$ENVN THREADS=$THREADS \
               bash $(rdir $i)/run_screen.sh $ARGS --shard $((i+1))/$N" 2>&1 | tail -8
  done
  echo
  echo "All $N shards launched. Combined progress:  bash pods.sh status"
  ;;

status)
  tot_done=0; tot_left=0
  for i in $(seq 0 $((N-1))); do
    banner $i
    out=$(ssh_to $i "cd $(rdir $i) 2>/dev/null || exit 1; \
      f=screen_results_shard$((i+1))of${N}.tsv; \
      if [ -f screen.pid ] && kill -0 \$(cat screen.pid) 2>/dev/null; then \
        echo 'RUNNING'; else echo 'NOT RUNNING'; fi; \
      if [ -s \$f ]; then \
        total=\$(grep -aoE '^[0-9]+ runs to do' screen.log | head -1 | awk '{print \$1}'); \
        awk -F'\t' -v tot=\"\$total\" 'NR>1{s+=\$17;n++} END{ \
          if(n>0){m=s/n; printf \"%d done   mean %.0f s   left %d   ETA %.1f h\n\", \
                  n, m, tot-n, (tot-n)*m/3600} }' \$f; \
      else echo '0 done'; fi; \
      tail -n 1 screen.log 2>/dev/null | tr -d '\\000'; \
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>&1)
    echo "$out"
    d=$(echo "$out" | grep -oE '^[0-9]+ done' | head -1 | awk '{print $1}')
    l=$(echo "$out" | grep -oE 'left [0-9]+' | head -1 | awk '{print $2}')
    tot_done=$(( tot_done + ${d:-0} )); tot_left=$(( tot_left + ${l:-0} ))
  done
  printf '\n\033[1mCOMBINED: %d runs done, %d remaining\033[0m\n' "$tot_done" "$tot_left"
  ;;

fetch)
  mkdir -p "$LOCAL_DIR/results"
  for i in $(seq 0 $((N-1))); do
    f="screen_results_shard$((i+1))of${N}.tsv"
    scp_from $i "$f" "$LOCAL_DIR/results/$f" && echo "  got $f" || echo "  (no $f yet)"
    scp_from $i "screen.log" "$LOCAL_DIR/results/screen_pod$((i+1)).log" || true
  done
  ;;

sync)
  INTERVAL="${INTERVAL:-300}"
  mkdir -p "$LOCAL_DIR/results" "$LOCAL_DIR/results/snapshots"
  echo "syncing $N pods every ${INTERVAL}s into $LOCAL_DIR/results  (Ctrl-C to stop)"
  last=-1
  while true; do
    for i in $(seq 0 $((N-1))); do
      f="screen_results_shard$((i+1))of${N}.tsv"
      scp_from $i "$f" "$LOCAL_DIR/results/$f" || \
        echo "$(date +%H:%M:%S)  pod $((i+1)) unreachable"
    done
    n=$(cat "$LOCAL_DIR"/results/screen_results_shard*.tsv 2>/dev/null \
        | grep -vc '^paralogue' || echo 0)
    if [ "$n" -ne "$last" ]; then
      stamp=$(date +%Y%m%d-%H%M%S)
      tar -czf "$LOCAL_DIR/results/snapshots/shards_${stamp}_${n}rows.tgz" \
          -C "$LOCAL_DIR/results" $(cd "$LOCAL_DIR/results" && ls screen_results_shard*.tsv 2>/dev/null) 2>/dev/null || true
      echo "$(date +%H:%M:%S)  $n rows total -> snapshot"
      last=$n
    else
      echo "$(date +%H:%M:%S)  $n rows total (unchanged)"
    fi
    sleep "$INTERVAL"
  done
  ;;

merge)
  cd "$LOCAL_DIR/results" 2>/dev/null || { echo "run 'bash pods.sh fetch' first"; exit 1; }
  head -1 "$(ls screen_results_shard*.tsv | head -1)" > screen_results.tsv
  for f in screen_results_shard*.tsv; do tail -n +2 "$f"; done \
    | sort -u >> screen_results.tsv
  echo "merged $(( $(wc -l < screen_results.tsv) - 1 )) unique rows into results/screen_results.tsv"
  if command -v python3 >/dev/null && python3 -c 'import numpy' 2>/dev/null; then
    python3 "$LOCAL_DIR/ack1_importin_gpu.py" --stage analyse --results screen_results.tsv \
      | tee analysis.txt
  else
    echo "numpy not available locally; run the analysis on a pod:"
    echo "  bash pods.sh ... or scp results/screen_results.tsv to a pod and run --stage analyse"
  fi
  ;;

stop)
  for i in $(seq 0 $((N-1))); do
    banner $i
    # Kill by name as well as by pid file. A relaunch overwrites screen.pid, so an
    # earlier process becomes invisible to the pid file while still holding the GPU --
    # that is how a pod ends up "NOT RUNNING" at 100% utilisation.
    ssh_to $i "kill \$(cat $(rdir $i)/screen.pid 2>/dev/null) 2>/dev/null; \
               pkill -9 -f ack1_importin_gpu.py 2>/dev/null; \
               rm -f $(rdir $i)/screen.pid; sleep 2; \
               echo -n 'remaining ack1 processes: '; \
               pgrep -cf ack1_importin_gpu.py || echo 0; \
               nvidia-smi --query-gpu=memory.used --format=csv,noheader"
  done
  ;;

*) sed -n '2,20p' "$0" ;;
esac
