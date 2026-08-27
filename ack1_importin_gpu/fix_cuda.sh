#!/usr/bin/env bash
# Fix CUDA_ERROR_UNSUPPORTED_PTX_VERSION on the ack1 pod.
#
#   ssh -i ~/.ssh/id_ed25519 -p 13087 root@<pod-host> 'bash -s' < fix_cuda.sh
#
# Cause: conda pulled cuda-nvrtc 13.3, but the installed driver only supports up to
# CUDA 13.0. OpenMM compiles its kernels at runtime with nvrtc, so it emits PTX the
# driver cannot load. nvidia-smi keeps working because it uses libnvidia-ml, a
# different library that never touches the compile path.
#
# pin the CUDA stack below the driver's ceiling. CUDA 12.x PTX is accepted by
# any 5xx driver, so 12.6 is the safe choice rather than shaving to exactly 13.0.
set -uo pipefail

FORGE=/opt/miniforge3
PY=$FORGE/envs/ack1/bin/python
CONDA=$FORGE/bin/conda

line() { printf '\n===== %s =====\n' "$*"; }

test_cuda() {
  $PY - <<'PY' 2>&1 | tail -3
import openmm as mm
from openmm import unit
s = mm.System(); s.addParticle(1.0)
f = mm.NonbondedForce(); f.addParticle(0.0, 0.1, 0.1); s.addForce(f)
try:
    p = mm.Platform.getPlatformByName('CUDA')
    c = mm.Context(s, mm.VerletIntegrator(0.001*unit.picoseconds), p)
    c.setPositions([[0, 0, 0]])
    c.getState(getEnergy=True)
    print('CUDA_OK')
except Exception as e:
    print('CUDA_FAIL', type(e).__name__, e)
PY
}

line "driver ceiling"
nvidia-smi 2>/dev/null | grep -o 'CUDA Version: *[0-9.]*' | head -1
echo -n "installed nvrtc: "
$CONDA list -n ack1 2>/dev/null | grep -E '^cuda-nvrtc' | awk '{print $2}'

line "before"
test_cuda

line "pinning the CUDA stack to 12.6 (this takes a few minutes)"
$CONDA install -y -n ack1 -c conda-forge 'cuda-version=12.6' 2>&1 | tail -15

line "after"
RESULT="$(test_cuda)"
echo "$RESULT"

if echo "$RESULT" | grep -q CUDA_OK; then
  line "fixed"
  $CONDA list -n ack1 2>/dev/null | grep -E '^(openmm|cuda-nvrtc|cuda-version)'
  echo
  echo "Start the run:   bash /workspace/ack1/run_screen.sh --stage screen"
  exit 0
fi

line "still failing - trying a system nvcc instead of nvrtc"
# OpenMM will use an external nvcc if OPENMM_CUDA_COMPILER points at one; a 12.x nvcc
# emits PTX the driver accepts.
NVCC="$(ls -d /usr/local/cuda-12*/bin/nvcc 2>/dev/null | head -1)"
if [ -n "$NVCC" ]; then
  echo "found $NVCC"
  export OPENMM_CUDA_COMPILER="$NVCC"
  RESULT2="$(test_cuda)"
  echo "$RESULT2"
  if echo "$RESULT2" | grep -q CUDA_OK; then
    # make it stick for every future run
    for rc in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
      grep -q OPENMM_CUDA_COMPILER "$rc" 2>/dev/null || \
        echo "export OPENMM_CUDA_COMPILER=$NVCC" >> "$rc"
    done
    grep -q OPENMM_CUDA_COMPILER /workspace/ack1/run_screen.sh 2>/dev/null || \
      sed -i "s|^cd \$WORK|export OPENMM_CUDA_COMPILER=$NVCC\ncd \$WORK|" \
          /workspace/ack1/run_screen.sh 2>/dev/null || true
    echo
    echo "Fixed via nvcc. Start the run:  bash /workspace/ack1/run_screen.sh --stage screen"
    exit 0
  fi
else
  echo "no system nvcc found under /usr/local/cuda-12*"
fi

line "not fixed"
cat <<'EOF'
CUDA still will not compile kernels on this pod.

Two options:

  1. Run on CPU. It works and the science is identical, just slow:
         bash /workspace/ack1/run_screen.sh --stage screen --platform CPU \
              --paralogues KPNA2 --seeds 2 --md 200 --frames 10
     Cut to one paralogue and shorter sampling, or it will take days.

  2. Start a fresh pod with a driver that matches the toolkit, and redeploy.
     The bootstrap is idempotent and takes about ten minutes.
EOF
exit 1
