#!/usr/bin/env bash
# Runs ON THE POD. Installs Miniforge, creates the `ack1` environment, installs
# everything the pipeline needs, and caps OpenMM to 13 CPU threads permanently.
# Safe to re-run: every step is skipped if it is already done.
set -euo pipefail

WORK="${WORK:-/workspace/ack1}"
THREADS="${THREADS:-13}"
FORGE_DIR="${FORGE_DIR:-/opt/miniforge3}"
ENV_NAME=ack1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "pod: $(hostname)   work dir: $WORK   thread cap: $THREADS"
mkdir -p "$WORK"

# ---------------------------------------------------------------- base packages
say "base packages"
if ! command -v curl >/dev/null 2>&1 || ! command -v bzip2 >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends curl ca-certificates bzip2 git tmux
fi
command -v tmux >/dev/null 2>&1 || apt-get install -y -qq tmux || true

say "GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
else
  echo "WARNING: nvidia-smi not found. The pipeline will fall back to the CPU platform."
fi
echo "CPU cores available: $(nproc)"

# ---------------------------------------------------------------------- miniforge
if [ ! -x "$FORGE_DIR/bin/conda" ]; then
  say "installing Miniforge to $FORGE_DIR"
  ARCH="$(uname -m)"
  URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh"
  curl -fsSL "$URL" -o /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$FORGE_DIR"
  rm -f /tmp/miniforge.sh
else
  say "Miniforge already present at $FORGE_DIR"
fi

export PATH="$FORGE_DIR/bin:$PATH"
set +u                      # conda's own scripts trip over `set -u`
# shellcheck disable=SC1091
source "$FORGE_DIR/etc/profile.d/conda.sh"

# ------------------------------------------------------------------ environment
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  say "creating conda env '${ENV_NAME}'"
  conda create -y -n "$ENV_NAME" -c conda-forge \
      python=3.11 openmm pdbfixer biopython numpy scipy pandas matplotlib
else
  say "env '${ENV_NAME}' already exists, ensuring packages are present"
  conda install -y -n "$ENV_NAME" -c conda-forge \
      python=3.11 openmm pdbfixer biopython numpy scipy pandas matplotlib
fi

conda activate "$ENV_NAME"
set -u

# ------------------------------------------------------------------ thread caps
say "capping threads at $THREADS"
ACT="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACT"
cat > "$ACT/ack1_threads.sh" <<EOF
export OPENMM_CPU_THREADS=$THREADS
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export VECLIB_MAXIMUM_THREADS=$THREADS
export ACK1_THREADS=$THREADS
EOF
# shellcheck disable=SC1090
source "$ACT/ack1_threads.sh"

# Make the env and the caps the default for future shells. An SSH *login* shell reads
#.bash_profile /.profile and NOT.bashrc, so all three are written; otherwise `python`
# silently resolves to the system interpreter and openmm appears to be missing.
for rc in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
  if ! grep -q 'ack1_bootstrap_marker' "$rc" 2>/dev/null; then
    cat >> "$rc" <<EOF

# ack1_bootstrap_marker
source $FORGE_DIR/etc/profile.d/conda.sh 2>/dev/null && conda activate $ENV_NAME
export OPENMM_CPU_THREADS=$THREADS OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS ACK1_THREADS=$THREADS
cd $WORK 2>/dev/null || true
EOF
  fi
done

# A launcher that does not depend on any rc file having been sourced.
# NOTE: WORK is resolved at RUN time, not here. Baking the path in at generation time
# meant the launcher always cd'd to the bootstrap-time directory and silently ignored
# the per-pod directory it was later invoked with.
if [ ! -f "$WORK/run_screen.sh" ]; then
cat > "$WORK/run_screen.sh" <<EOF
#!/usr/bin/env bash
# Start the screen detached. Interpreter by absolute path on purpose.
WORK="\${WORK:-$WORK}"
THREADS="\${THREADS:-$THREADS}"
cd "\$WORK" || exit 1
export OPENMM_CPU_THREADS=\$THREADS OMP_NUM_THREADS=\$THREADS MKL_NUM_THREADS=\$THREADS
export OPENBLAS_NUM_THREADS=\$THREADS NUMEXPR_NUM_THREADS=\$THREADS ACK1_THREADS=\$THREADS
if [ -f screen.pid ] && kill -0 \$(cat screen.pid) 2>/dev/null; then
  echo "already running, pid \$(cat screen.pid)"; exit 0
fi
setsid nohup $FORGE_DIR/envs/$ENV_NAME/bin/python -u ack1_importin_gpu.py \
    "\$@" > screen.log 2>&1 < /dev/null &
echo \$! > screen.pid
sleep 4
echo "started in \$WORK, pid \$(cat screen.pid)"
tail -n 8 screen.log
EOF
fi
chmod +x "$WORK/run_screen.sh"

# --------------------------------------------------------------------- verify
say "OpenMM installation test"
python -m openmm.testInstallation || true

say "versions and platform check"
python - <<'PY'
import os
import openmm, openmm.app, pdbfixer, Bio, numpy, scipy
print('openmm  ', openmm.version.version)
print('pdbfixer', getattr(pdbfixer, '__version__', 'ok'))
print('biopython', Bio.__version__)
print('numpy   ', numpy.__version__)
print('scipy   ', scipy.__version__)
names = [openmm.Platform.getPlatform(i).getName()
         for i in range(openmm.Platform.getNumPlatforms())]
print('platforms:', names)
print('OPENMM_CPU_THREADS =', os.environ.get('OPENMM_CPU_THREADS'))
if 'CUDA' not in names:
    print()
    print('*** CUDA platform NOT available ***')
    print('  The pipeline will still run on the CPU platform, just far slower.')
    print('  Usual fixes, in order:')
    print('    1. nvidia-smi          - if this fails the pod has no GPU driver')
    print('    2. conda install -y -n ack1 -c conda-forge "openmm=*=*cuda*"')
    print('    3. conda install -y -n ack1 -c conda-forge cuda-version=12')
PY

# ------------------------------------------------------------------- input check
say "input files in $WORK"
cd "$WORK"
missing=0
for f in ack1_importin_gpu.py 1EJL.pdb sam_dimer_fixed.pdb kpna_seqs.fasta; do
  if [ -f "$f" ]; then printf '  ok      %s\n' "$f"
  else printf '  MISSING %s\n' "$f"; missing=1; fi
done
[ "$missing" -eq 0 ] || { echo; echo "Copy the missing files across, then re-run."; exit 1; }

say "smoke test (a few minutes)"
python ack1_importin_gpu.py --stage pockets
python ack1_importin_gpu.py --stage tether --tether-samples 800 --unwind 0 2 4

cat <<EOF

================================================================================
Ready.

Start the full screen, detached with nohup so it survives you logging out:

    WORK=$WORK bash $WORK/run_screen.sh --stage screen

Watch it:

    tail -f screen.log            # Ctrl-C stops watching, not the job
    kill -0 \$(cat screen.pid) && echo running

Analyse, at any point, part-way is fine:

    python ack1_importin_gpu.py --stage analyse | tee analysis.txt

Results accumulate in screen_results.tsv and completed runs are skipped on
restart, so interrupting is harmless.

From your WSL side you can drive all of this without logging in at all:

    bash pod.sh start | status | log | follow | analyse | fetch | stop
================================================================================
EOF
