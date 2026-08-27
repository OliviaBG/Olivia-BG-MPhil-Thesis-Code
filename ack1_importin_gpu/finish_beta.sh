#!/usr/bin/env bash
# Finish the beta-family screen on ONE pod, resuming from whatever is already done.
#
#   bash finish_beta.sh <host> <port>
#   bash finish_beta.sh <pod-host> 13087
#
# HOW THE RESUME WORKS
#   beta_screen.py globs beta_results*.tsv in its working directory and builds a set of
#   (receptor, peptide, seed) triples that are already finished. Anything in that set is
#   dropped from the queue. So this script copies every shard you have already pulled
#   into the new pod's working directory, and the screen simply skips them.
#
#   Nothing is recomputed and nothing is lost. Output goes to beta_results.tsv, a new
#   file that does not collide with the six shard names.
#
# ORDERING
#   One pod has to do the whole remainder serially, so the queue is reordered to put
#   the composition-matched scrambles and the ACK1 registers first. Those are the runs
#   the matched-pair test needs; everything else is context. You get the answer that
#   decides the transportin question long before the queue drains.

set -u
HOST="${1:-}"; PORT="${2:-}"
[ -z "$PORT" ] && { sed -n '2,22p' "$0"; exit 1; }

KEY="${KEY:-$HOME/.ssh/id_ed25519}"
USER_="${USER_:-root}"
DIR="${DIR:-/workspace/ack1/finish}"
FORGE=/opt/miniforge3
PY="$FORGE/envs/ack1/bin/python"
THREADS="${THREADS:-13}"
SEEDS="${SEEDS:-6}"; MD="${MD:-1000}"; FRAMES="${FRAMES:-25}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

S=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new
   -o ServerAliveInterval=30 "${USER_}@${HOST}")
C=(scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new)

step() { printf '\n\033[1m[finish] %s\033[0m\n' "$*"; }

step "target ${USER_}@${HOST}:${PORT}  ->  $DIR"

step "1. killing anything already running"
"${S[@]}" "pkill -9 -f beta_screen.py; pkill -9 -f ack1_importin_gpu.py; sleep 2; \
           echo -n 'screen processes now: '; pgrep -cf '[b]eta_screen.py' || echo 0; \
           nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true"

step "2. making a clean working directory"
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

step "4. copying completed results so they are not repeated"
n=0
for f in "$HERE"/results/beta_results_shard*.tsv; do
  [ -s "$f" ] || continue
  "${C[@]}" "$f" "${USER_}@${HOST}:${DIR}/" >/dev/null 2>&1 \
    && { r=$(grep -vc '^receptor' "$f"); n=$((n + r)); \
         echo "   sent $(basename "$f")  ($r rows)"; } \
    || echo "   FAILED to send $(basename "$f")"
done
echo "   ---------------------------------------------"
echo "   $n completed runs shipped; the screen will skip every one of them"
[ "$n" -eq 0 ] && echo "   WARNING: no results found in $HERE/results -- it will start from scratch"

step "5. checking the environment (bootstraps only if the interpreter is absent)"
"${S[@]}" "if [ -x $PY ]; then echo 'interpreter: ok'; else \
             echo 'interpreter MISSING - running bootstrap, this takes a few minutes'; \
             WORK=$DIR THREADS=$THREADS bash $DIR/pod_bootstrap.sh 2>&1 | tail -15; \
             bash $DIR/fix_cuda.sh 2>&1 | tail -5; fi"

step "6. verifying CUDA really works"
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

step "7. launching the remainder (no --shard: this pod takes everything left)"
"${S[@]}" "cd $DIR && \
  export OPENMM_CPU_THREADS=$THREADS OMP_NUM_THREADS=$THREADS ACK1_THREADS=$THREADS && \
  setsid nohup $PY -u beta_screen.py --seeds $SEEDS --md $MD --frames $FRAMES \
      --priority NEG_scramble ACK1_71-73 ACK1_64-67 \
      > beta.log 2>&1 < /dev/null & \
  sleep 10; cd $DIR && echo '--- first log lines ---' && head -n 14 beta.log && \
  echo '--- process ---' && pgrep -af '[b]eta_screen.py' | head -3"

step "done"
echo "   watch it:   bash check_finish.sh $HOST $PORT"
echo "   pull it:    bash pull_finish.sh $HOST $PORT"
