#!/usr/bin/env bash
# Set up and launch ONE pod of a five-way split of the beta screen.
# Deliberately not clever: no loops, no set -e, every step says what it did.
#
#   bash beta_pod5.sh <host> <port> <shard>
#
#   bash beta_pod5.sh <pod-host> 24072 1
#   bash beta_pod5.sh <pod-host> 23191 2
#   bash beta_pod5.sh <pod-host> 15693 3
#   bash beta_pod5.sh <pod-host> 16656 4
#   bash beta_pod5.sh <pod-host> 32148 5
#
# RUN rescue_pod1.sh FIRST. This script kills whatever is running on the pod, so any
# result not already pulled to this machine is lost.
#
# WHY THE SPLIT IS CLEAN
#   beta_screen.py removes completed runs from the queue BEFORE applying the shard
#   modulo. So every pod must see the SAME set of completed results, or they compute
#   different queues and the k % n split stops being a partition -- some runs done
#   twice, others never. That is why step 4 ships every local beta_results*.tsv to
#   every pod. Do not launch a pod that skipped step 4.
#
#   --priority is applied after sharding, so it only reorders within a shard. The 12
#   scramble runs spread over 5 pods, ~2-3 each, and all land in the first few minutes.

set -u
HOST="${1:-}"; PORT="${2:-}"; SHARD="${3:-}"
TOTAL="${TOTAL:-5}"
[ -z "$SHARD" ] && { sed -n '2,20p' "$0"; exit 1; }

KEY="${KEY:-$HOME/.ssh/id_ed25519}"
USER_="${USER_:-root}"
DIR="/workspace/ack1/beta${SHARD}"          # per-pod dir: RunPod network volumes can
                                            # be SHARED between pods, and a common
                                            # working directory silently interleaves
                                            # two pods' output into one file.
FORGE=/opt/miniforge3
PY="$FORGE/envs/ack1/bin/python"
THREADS="${THREADS:-13}"
SEEDS="${SEEDS:-6}"; MD="${MD:-1000}"; FRAMES="${FRAMES:-25}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

S=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new
   -o ServerAliveInterval=30 -o ConnectTimeout=25 "${USER_}@${HOST}")
C=(scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25)

step() { printf '\n\033[1m[pod%s] %s\033[0m\n' "$SHARD" "$*"; }

step "target ${USER_}@${HOST}:${PORT}  ->  $DIR   (shard ${SHARD}/${TOTAL})"

step "1. killing anything already running"
"${S[@]}" "pkill -9 -f beta_screen.py; pkill -9 -f ack1_importin_gpu.py; sleep 2; \
  echo -n 'python screens still up: '; \
  ps -eo comm,cmd | awk '\$1 ~ /^python/ && /beta_screen.py/' | wc -l; \
  nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true"

step "2. private working directory"
"${S[@]}" "mkdir -p $DIR && cd $DIR && rm -f _beta_*.pdb _sasa_tmp*.pdb beta.log && \
  echo \"$DIR ready on \$(hostname)\""

step "3. copying code and structures"
for f in beta_screen.py ack1_importin_gpu.py 1M5N.pdb 5J3V.pdb \
         pod_bootstrap.sh fix_cuda.sh; do
  if [ -f "$HERE/$f" ]; then
    "${C[@]}" "$HERE/$f" "${USER_}@${HOST}:${DIR}/" >/dev/null 2>&1 \
      && echo "   sent $f" || echo "   FAILED to send $f"
  else
    echo "   MISSING locally: $f"
  fi
done

step "4. copying ALL completed results (identical on every pod, or the split breaks)"
n=0
for f in "$HERE"/results/beta_results*.tsv; do
  [ -s "$f" ] || continue
  "${C[@]}" "$f" "${USER_}@${HOST}:${DIR}/" >/dev/null 2>&1 \
    && { r=$(grep -vc '^receptor' "$f"); n=$((n + r)); \
         echo "   sent $(basename "$f")  ($r rows)"; } \
    || echo "   FAILED to send $(basename "$f")"
done
echo "   ------------------------------------------------"
echo "   $n rows shipped to pod$SHARD"
echo "   this number MUST be the same on all five pods"

step "5. environment (bootstraps only if the interpreter is absent)"
"${S[@]}" "if [ -x $PY ]; then echo 'interpreter: ok'; else \
  echo 'interpreter MISSING - bootstrapping, takes a few minutes'; \
  WORK=$DIR THREADS=$THREADS bash $DIR/pod_bootstrap.sh 2>&1 | tail -15; \
  bash $DIR/fix_cuda.sh 2>&1 | tail -5; fi"

step "6. verifying CUDA"
"${S[@]}" "$PY -c \"
import openmm as mm
from openmm import unit
s=mm.System(); s.addParticle(1.0)
f=mm.NonbondedForce(); f.addParticle(0.0,0.1,0.1); s.addForce(f)
try:
    c=mm.Context(s, mm.VerletIntegrator(0.001*unit.picoseconds),
                 mm.Platform.getPlatformByName('CUDA'))
    c.setPositions([[0,0,0]]); c.getState(getEnergy=True); print('CUDA: WORKS')
except Exception as e:
    print('CUDA: FAILS ->', e)
\""

step "7. launching shard ${SHARD}/${TOTAL}"
"${S[@]}" "cd $DIR && \
  export OPENMM_CPU_THREADS=$THREADS OMP_NUM_THREADS=$THREADS ACK1_THREADS=$THREADS && \
  setsid nohup $PY -u beta_screen.py --shard ${SHARD}/${TOTAL} --seeds $SEEDS \
      --md $MD --frames $FRAMES \
      --priority NEG_scramble ACK1_71-73 ACK1_64-67 \
      > beta.log 2>&1 < /dev/null & \
  sleep 12; cd $DIR && echo '--- log ---' && head -n 14 beta.log"

step "done -- check with: bash check_beta5.sh"
