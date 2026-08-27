#!/usr/bin/env bash
# Top the importin-alpha panel up from 3 seeds to 6, on ONE bare pod.
# Run once per pod, from WSL, in ~/AlphaFold/ack1_importin_gpu.
#
#   bash alpha_topup_pod.sh <host> <port> <shard>
#
#   bash alpha_topup_pod.sh <pod-host>  19167 1
#   bash alpha_topup_pod.sh <pod-host> 26471 2
#   bash alpha_topup_pod.sh <pod-host> 10569 3
#
# WHY
#   pod_results.tsv holds 408 unique runs = 8 importin-alpha surfaces x 17 peptides
#   x 3 seeds. The beta-family panel used 6 seeds. At n=3 per group a proper Welch
#   t-test leaves most matched pairs short of Holm significance, so this run takes
#   importin-alpha to 6 seeds and makes the two panels directly comparable.
#
# THE PROTOCOL MUST MATCH THE ORIGINAL RUNS
#   Seeds 3-5 are only poolable with seeds 0-2 if they are sampled identically.
#   The original alpha panel used 100 ps equilibration, 400 ps production, 20 frames
#   (n_frames=20 in every row of pod_results.tsv). Those are passed explicitly below
#   rather than left to defaults, so this cannot drift. Do NOT copy the beta-family
#   settings (150/1000/25) here.
#
# HOW THE RESUME WORKS
#   ack1_importin_gpu.py globs screen_results*.tsv in its working directory and skips
#   any (paralogue, site, peptide, seed) already present. Step 4 ships the existing
#   408 runs as screen_results_known.tsv, so --seeds 6 produces exactly seeds 3, 4, 5.
#   Output goes to screen_results_topupNof3.tsv, a name that collides with nothing.
#
#   Every pod must receive the SAME completed-run file: the done set is applied before
#   the shard modulo, so a pod that saw a different history would compute a different
#   queue and the split would stop being a partition.

set -u
HOST="${1:-}"; PORT="${2:-}"; SHARD="${3:-}"
TOTAL="${TOTAL:-3}"
[ -z "$SHARD" ] && { sed -n '2,12p' "$0"; exit 1; }

KEY="${KEY:-$HOME/.ssh/id_ed25519}"
USER_="${USER_:-root}"
DIR="/workspace/ack1/top${SHARD}"     # per-pod dir: RunPod network volumes can be
                                      # shared between pods, and a common working
                                      # directory silently interleaves two pods' output
FORGE=/opt/miniforge3
PY="$FORGE/envs/ack1/bin/python"
THREADS="${THREADS:-13}"
SEEDS="${SEEDS:-6}"
EQUIL="${EQUIL:-100}"; MD="${MD:-400}"; FRAMES="${FRAMES:-20}"
OUT="screen_results_topup${SHARD}of${TOTAL}.tsv"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

S=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new
   -o ServerAliveInterval=30 -o ConnectTimeout=30 "${USER_}@${HOST}")
C=(scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)

step() { printf '\n\033[1m[pod%s] %s\033[0m\n' "$SHARD" "$*"; }

step "target ${USER_}@${HOST}:${PORT}  ->  $DIR   (shard ${SHARD}/${TOTAL})"

step "1. killing anything already running"
"${S[@]}" "pkill -9 -f ack1_importin_gpu.py; pkill -9 -f beta_screen.py; sleep 2; \
  echo -n 'python screens still up: '; \
  ps -eo comm,cmd | awk '\$1 ~ /^python/ && /ack1_importin_gpu.py/' | wc -l; \
  nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null || \
    echo 'no nvidia-smi'"

step "2. private working directory"
"${S[@]}" "mkdir -p $DIR && cd $DIR && rm -f _site_*.pdb _sasa_tmp*.pdb screen.log && \
  echo \"$DIR ready on \$(hostname)\""

step "3. copying code and structures"
for f in ack1_importin_gpu.py 1EJL.pdb sam_dimer_fixed.pdb kpna_seqs.fasta \
         pod_bootstrap.sh fix_cuda.sh; do
  if [ -f "$HERE/$f" ]; then
    "${C[@]}" "$HERE/$f" "${USER_}@${HOST}:${DIR}/" >/dev/null 2>&1 \
      && echo "   sent $f" || echo "   FAILED to send $f"
  else
    echo "   MISSING locally: $f"
  fi
done

step "4. shipping the 408 completed runs so they are not repeated"
KNOWN=""
for cand in "$HERE/results/merged_results.tsv" "$HERE/../ack1_nls_af3/pod_results.tsv" \
            "$HERE/pod_results.tsv" "$HERE/results/pod_results.tsv"; do
  [ -s "$cand" ] && { KNOWN="$cand"; break; }
done
if [ -n "$KNOWN" ]; then
  "${C[@]}" "$KNOWN" "${USER_}@${HOST}:${DIR}/screen_results_known.tsv" >/dev/null 2>&1 \
    && echo "   sent $(basename "$KNOWN") as screen_results_known.tsv ($(grep -vc '^paralogue' "$KNOWN") rows)" \
    || echo "   FAILED to send the known-results file"
else
  echo "   !! could not find merged_results.tsv or pod_results.tsv locally."
  echo "   !! WITHOUT IT THIS POD WILL RE-RUN SEEDS 0-2. Stop and find the file."
fi
# any per-shard files from the original run also count towards the done set
for f in "$HERE"/results/screen_results_shard*.tsv; do
  [ -s "$f" ] || continue
  "${C[@]}" "$f" "${USER_}@${HOST}:${DIR}/" >/dev/null 2>&1 \
    && echo "   sent $(basename "$f")"
done

step "5. bootstrap: miniforge, conda env, openmm, pdbfixer, biopython (~10 min on a bare pod)"
"${S[@]}" "if [ -x $PY ]; then echo 'interpreter already present, skipping bootstrap'; \
  else WORK=$DIR THREADS=$THREADS bash $DIR/pod_bootstrap.sh 2>&1 | tail -32; fi"

step "6. CUDA check, and repair if the PTX version is unsupported"
"${S[@]}" "cd $DIR && $PY - <<'EOF'
import openmm as mm
from openmm import unit
s = mm.System(); s.addParticle(1.0)
f = mm.NonbondedForce(); f.addParticle(0.0, 0.1, 0.1); s.addForce(f)
try:
    c = mm.Context(s, mm.VerletIntegrator(0.001*unit.picoseconds),
                   mm.Platform.getPlatformByName('CUDA'))
    c.setPositions([[0,0,0]]); c.getState(getEnergy=True); print('CUDA: WORKS')
except Exception as e:
    print('CUDA: FAILS ->', type(e).__name__, e)
EOF" | tee /tmp/_cuda_$SHARD.txt
if grep -q 'CUDA: FAILS' /tmp/_cuda_$SHARD.txt; then
  step "6b. running fix_cuda.sh (pins the CUDA stack to 12.6, a few minutes)"
  "${S[@]}" "bash $DIR/fix_cuda.sh 2>&1 | tail -12"
fi
rm -f /tmp/_cuda_$SHARD.txt

step "7. launching shard ${SHARD}/${TOTAL}  --  seeds 6, protocol matched to the original panel"
"${S[@]}" "cd $DIR && \
  export OPENMM_CPU_THREADS=$THREADS OMP_NUM_THREADS=$THREADS ACK1_THREADS=$THREADS && \
  setsid nohup $PY -u ack1_importin_gpu.py --stage screen \
      --shard ${SHARD}/${TOTAL} --seeds $SEEDS \
      --equil $EQUIL --md $MD --frames $FRAMES \
      --results $OUT \
      --priority ACK1_71-73 ACK1_64-67 NEG_scramble \
      > screen.log 2>&1 < /dev/null & \
  sleep 12; cd $DIR && echo '--- log ---' && head -n 20 screen.log"

step "done"
echo "   watch:  bash check_alpha.sh"
echo "   pull:   bash pull_alpha.sh"
