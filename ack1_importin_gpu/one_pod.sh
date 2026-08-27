#!/usr/bin/env bash
# Set up and launch ONE pod. Deliberately not clever: no loops, no set -e, every step
# reports what it did. Run it once per pod.
#
#   bash one_pod.sh <host> <port> <shard> <total> [known.tsv]
#
#   bash one_pod.sh <pod-host> 13087 1 6 results/screen_results_known.tsv
#   bash one_pod.sh <pod-host> 33946 2 6 results/screen_results_known.tsv
#   bash one_pod.sh <pod-host> 22965 3 6 results/screen_results_known.tsv
#   bash one_pod.sh <pod-host> 10251 4 6 results/screen_results_known.tsv
#   bash one_pod.sh <pod-host> 12493 5 6 results/screen_results_known.tsv
#   bash one_pod.sh <pod-host> 16073 6 6 results/screen_results_known.tsv
#
# If a pod fails you see exactly where, and the other five are untouched.

HOST="$1"; PORT="$2"; SHARD="$3"; TOTAL="$4"; KNOWN="${5:-}"
[ -z "$TOTAL" ] && { sed -n '2,18p' "$0"; exit 1; }

KEY="${KEY:-$HOME/.ssh/id_ed25519}"
USER_="${USER_:-root}"
DIR="/workspace/ack1/pod${SHARD}"
FORGE=/opt/miniforge3
PY="$FORGE/envs/ack1/bin/python"
THREADS="${THREADS:-13}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

S=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new
   -o ServerAliveInterval=30 "${USER_}@${HOST}")
C=(scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new)

step() { printf '\n\033[1m[%s] %s\033[0m\n' "pod$SHARD" "$*"; }

step "target $USER_@$HOST:$PORT  ->  $DIR   (shard $SHARD/$TOTAL)"

step "1. killing anything already running"
"${S[@]}" "pkill -9 -f ack1_importin_gpu.py; sleep 2; \
           echo -n 'ack1 processes now: '; pgrep -cf ack1_importin_gpu.py || echo 0; \
           nvidia-smi --query-gpu=memory.used --format=csv,noheader"

step "2. making a private working directory"
"${S[@]}" "mkdir -p $DIR && cd $DIR && rm -f _site_*.pdb _sasa_tmp*.pdb screen.log \
           screen.pid && echo \"$DIR ready\" && hostname"

step "3. copying files"
for f in ack1_importin_gpu.py 1EJL.pdb sam_dimer_fixed.pdb kpna_seqs.fasta \
         pod_bootstrap.sh run_screen.sh fix_cuda.sh; do
  if [ -f "$HERE/$f" ]; then
    "${C[@]}" "$HERE/$f" "${USER_}@${HOST}:${DIR}/" >/dev/null 2>&1 \
      && echo "   sent $f" || echo "   FAILED to send $f"
  fi
done
if [ -n "$KNOWN" ] && [ -s "$HERE/$KNOWN" ]; then
  "${C[@]}" "$HERE/$KNOWN" "${USER_}@${HOST}:${DIR}/screen_results_known.tsv" \
    >/dev/null 2>&1 && echo "   sent known-results ($(( $(wc -l < "$HERE/$KNOWN") - 1 )) rows)"
fi

step "4. checking the environment"
"${S[@]}" "if [ -x $PY ]; then echo 'interpreter: ok'; else \
             echo 'interpreter MISSING - running bootstrap'; \
             WORK=$DIR THREADS=$THREADS bash $DIR/pod_bootstrap.sh 2>&1 | tail -15; \
             bash $DIR/fix_cuda.sh 2>&1 | tail -5; fi"

step "5. verifying CUDA really works"
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

step "6. launching shard $SHARD/$TOTAL"
"${S[@]}" "cd $DIR && \
  export OPENMM_CPU_THREADS=$THREADS OMP_NUM_THREADS=$THREADS ACK1_THREADS=$THREADS && \
  setsid nohup $PY -u ack1_importin_gpu.py --stage screen --shard $SHARD/$TOTAL \
      > screen.log 2>&1 < /dev/null & \
  sleep 6; cd $DIR && echo '--- first log lines ---' && head -n 8 screen.log && \
  echo '--- pwd of the running process ---' && \
  pgrep -af ack1_importin_gpu.py | head -3"

step "done. check progress with:"
echo "   ssh -i $KEY -p $PORT ${USER_}@${HOST} \"tail -n 5 $DIR/screen.log; \\"
echo "       wc -l < $DIR/screen_results_shard${SHARD}of${TOTAL}.tsv\""
