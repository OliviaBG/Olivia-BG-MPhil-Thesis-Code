#!/bin/bash
#
# pod_setup.sh
# ============================================================
# Provisions a BARE RunPod GPU pod to run this project's CRM1 docking MD
# (md_refinement.NESMDRefiner) on the GPU. Run this ON THE POD, once, after
# push_to_pod.sh has copied the code across.
#
#   ssh root@<pod-host> -p 27676 -i ~/.ssh/id_ed25519
#   cd /root/AlphaFold && bash pod_setup.sh
#
# WHAT IT DOES
#   1. Installs Miniforge3 (conda-forge by default -- OpenMM's CUDA builds
#      live there, and the plain pip 'openmm' wheel is CPU-only, which is
#      the usual reason a pod "defaults to CPU").
#   2. Creates the 'md' env with a CUDA-enabled OpenMM matched to the pod's
#      actual driver, plus PDBFixer and mdtraj.
#   3. Installs the project's pip dependencies.
#   4. Pins thread counts to 13 in the env's activate hook, so they apply
#      every time the env is activated (not just this shell).
#   5. HARD-VERIFIES that OpenMM can actually see and use the CUDA platform,
#      and exits non-zero if it can't -- rather than letting you discover it
#      three hours into a silent CPU run.
#
# Safe to re-run: every step is idempotent.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/AlphaFold}"
CONDA_DIR="${CONDA_DIR:-/root/miniforge3}"
ENV_NAME="${ENV_NAME:-md}"
PY_VERSION="${PY_VERSION:-3.11}"
THREADS="${THREADS:-13}"

banner() { echo; echo "============================================================"; echo "$1"; echo "============================================================"; }

# ------------------------------------------------------------------
banner "0. Pod / GPU sanity check"
# ------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. This pod has no NVIDIA driver visible."
    echo "There is no point installing a CUDA OpenMM here -- check you booted a GPU pod."
    exit 1
fi
nvidia-smi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Driver's max supported CUDA runtime, e.g. "12.4" from the nvidia-smi header.
DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)"
if [ -z "$DRIVER_CUDA" ]; then
    echo "WARNING: could not parse a CUDA version from nvidia-smi; defaulting to 12.4"
    DRIVER_CUDA="12.4"
fi
echo
echo "GPU            : $GPU_NAME"
echo "Driver supports: CUDA $DRIVER_CUDA"

# conda-forge's cuda-version metapackage. CUDA 12 has minor-version
# compatibility, so asking for the driver's own minor is the safe request.
CUDA_PIN="$DRIVER_CUDA"

# ------------------------------------------------------------------
banner "1. Miniforge3"
# ------------------------------------------------------------------
if [ -d "$CONDA_DIR" ]; then
    echo "Already installed at $CONDA_DIR -- skipping."
else
    echo "Installing Miniforge3 to $CONDA_DIR ..."
    command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl; }
    ARCH="$(uname -m)"
    curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh"
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
    rm -f /tmp/miniforge.sh
fi

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Make conda available in future non-interactive SSH sessions too.
if ! grep -q "miniforge3/etc/profile.d/conda.sh" /root/.bashrc 2>/dev/null; then
    echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> /root/.bashrc
    echo "Added conda to /root/.bashrc"
fi

# ------------------------------------------------------------------
banner "2. Conda env '$ENV_NAME' with CUDA-enabled OpenMM"
# ------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Env '$ENV_NAME' already exists -- skipping creation."
else
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

conda activate "$ENV_NAME"
echo "Python: $(which python) ($(python --version 2>&1))"

echo
echo "Installing openmm + pdbfixer + mdtraj against CUDA $CUDA_PIN ..."
# cuda-version pins the metapackage that selects the CUDA-enabled openmm build.
# If the exact minor isn't available, fall back to the major series.
if ! conda install -y -c conda-forge openmm pdbfixer mdtraj "cuda-version=$CUDA_PIN"; then
    echo "Pinned cuda-version=$CUDA_PIN failed; retrying against the major series ..."
    CUDA_MAJOR="${CUDA_PIN%%.*}"
    conda install -y -c conda-forge openmm pdbfixer mdtraj "cuda-version=${CUDA_MAJOR}.*"
fi

# ------------------------------------------------------------------
banner "3. Project pip dependencies"
# ------------------------------------------------------------------
# numpy/scipy come from conda above (they're openmm/mdtraj deps) -- installing
# them again via pip risks a mismatched ABI, so they're deliberately not listed.
python -m pip install --upgrade pip
# werkzeug MUST be pinned to the 2.3 series. Flask 2.3.0's testing.py reads
# werkzeug.__version__, which Werkzeug 3.x removed -- letting pip resolve
# werkzeug freely installs 3.x and every flask_app.test_client() call dies with
# "module 'werkzeug' has no attribute '__version__'". That is the exact call
# run_ack1_rank1_50ns.py uses to resolve the AlphaFold model_id, so the run
# fails after setup looks like it succeeded.
python -m pip install \
    flask==2.3.0 \
    "werkzeug==2.3.8" \
    flask-cors==4.0.0 \
    requests \
    biopython \
    scikit-learn \
    joblib \
    pandas \
    freesasa \
    matplotlib

# localcider is optional (app.py guards the import) and is the most likely
# thing to fail to build -- never let it kill the setup.
python -m pip install localcider || \
    echo "NOTE: localcider failed to install. app.py guards this import; "\
"linear hydropathy/NCPR/FCR features stay disabled. MD is unaffected."

# ------------------------------------------------------------------
banner "4. Pinning threads to $THREADS"
# ------------------------------------------------------------------
ACTIVATE_D="$CONDA_DIR/envs/$ENV_NAME/etc/conda/activate.d"
mkdir -p "$ACTIVATE_D"
cat > "$ACTIVATE_D/md_threads.sh" <<EOF
#!/bin/bash
# Written by pod_setup.sh -- applied every time the '$ENV_NAME' env activates.
export OPENMM_CPU_THREADS=$THREADS
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export CUDA_VISIBLE_DEVICES=0
EOF
chmod +x "$ACTIVATE_D/md_threads.sh"
# shellcheck disable=SC1090
source "$ACTIVATE_D/md_threads.sh"
echo "Wrote $ACTIVATE_D/md_threads.sh (OPENMM_CPU_THREADS/OMP/MKL/OPENBLAS/NUMEXPR = $THREADS)"

echo
echo "How md_refinement.py itself decides CPU thread count:"
python - <<'PYEOF'
import os
from pathlib import Path

# Mirrors md_refinement._detect_usable_cpu_count() exactly.
def detect():
    try:
        p = Path('/sys/fs/cgroup/cpu.max')
        if p.exists():
            q, per = p.read_text().split()
            if q != 'max':
                q, per = int(q), int(per)
                if per > 0 and q > 0:
                    return max(1, q // per), f"cgroup v2 cpu.max = '{q} {per}'"
    except Exception:
        pass
    try:
        qp = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        pp = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if qp.exists() and pp.exists():
            q, per = int(qp.read_text().strip()), int(pp.read_text().strip())
            if q > 0 and per > 0:
                return max(1, q // per), "cgroup v1 cfs quota"
    except Exception:
        pass
    return os.cpu_count(), "os.cpu_count() fallback (no cgroup quota found)"

n, how = detect()
print(f"  visible cores        : {os.cpu_count()}")
print(f"  md_refinement will use: {n} threads  ({how})")
if n != 13:
    print(f"  NOTE: that is {n}, not 13. md_refinement._select_best_platform() derives")
    print(f"        Threads from the cgroup quota and does NOT read OPENMM_CPU_THREADS,")
    print(f"        so this number is what the CPU platform would use. It only matters")
    print(f"        if CUDA is unavailable -- on the CUDA platform properties are empty")
    print(f"        and thread count is irrelevant. The env vars above still cap")
    print(f"        numpy/mdtraj/freesasa BLAS threads at 13 either way.")
PYEOF

# ------------------------------------------------------------------
banner "5. VERIFY: can OpenMM actually use the GPU?"
# ------------------------------------------------------------------
python -m openmm.testInstallation || true
echo
python - <<'PYEOF'
import sys
try:
    from openmm import Platform
except ImportError as e:
    print(f"FATAL: cannot import openmm ({e})")
    sys.exit(1)

names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
print(f"Registered OpenMM platforms: {names}")

if 'CUDA' not in names:
    print()
    print("FATAL: the CUDA platform is NOT registered.")
    print("md_refinement._select_best_platform() prefers CUDA > OpenCL > CPU > Reference,")
    print("so it will silently fall back to CPU and your 48 ns run will take days.")
    print()
    print("Most likely causes, in order:")
    print("  1. openmm came from pip, not conda-forge (pip wheels are CPU-only).")
    print("     Fix: conda install -c conda-forge openmm cuda-version=<driver's CUDA>")
    print("  2. libcuda.so.1 not visible inside the container (driver not mounted).")
    print("     Check: ldconfig -p | grep libcuda")
    print("  3. cuda-version pinned above the driver's supported runtime.")
    sys.exit(1)

# Prove it can actually run, not just that it's registered.
import openmm as mm, openmm.unit as unit
system = mm.System()
system.addParticle(1.0 * unit.amu)
integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
context = mm.Context(system, integrator, Platform.getPlatformByName('CUDA'))
context.setPositions([[0, 0, 0]] * 1)
context.getState(getEnergy=True)
print()
print("OK: CUDA platform registered AND a test Context ran on it.")
print(f"    OpenMM {mm.version.version}")
PYEOF

banner "DONE"
cat <<EOF
Every new SSH session, activate the env first:

    source $CONDA_DIR/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    cd $PROJECT_DIR

Then check the run is resumable before committing GPU hours to it:

    python run_ack1_rank1_50ns.py --dry-run

fpocket is NOT installed -- pocket_detector.py falls back to geometry scoring
and the MD path never calls it. Add it with 'apt-get install -y fpocket' only
if you intend to run the pocket-detection side of the app on this pod.
EOF
