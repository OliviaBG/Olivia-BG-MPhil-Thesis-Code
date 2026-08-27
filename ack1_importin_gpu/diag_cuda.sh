#!/usr/bin/env bash
# CUDA diagnosis for the ack1 pod. Run from WSL as:
#     ssh -i ~/.ssh/id_ed25519 -p 13087 root@<pod-host> 'bash -s' < diag_cuda.sh
FORGE=/opt/miniforge3
PY=$FORGE/envs/ack1/bin/python

line() { printf '\n===== %s =====\n' "$*"; }

line "driver"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader 2>&1 | head -3
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"

line "device nodes"
ls -l /dev/nvidia* 2>&1 | head -8

line "libcuda"
ldconfig -p 2>/dev/null | grep -E 'libcuda\.so|libnvidia-ml' | head -5
find / -name 'libcuda.so*' -not -path '*/proc/*' 2>/dev/null | head -5

line "openmm package build"
$FORGE/bin/conda list -n ack1 2>/dev/null | grep -iE '^(openmm|cuda|libcu)' | head -20

line "openmm platforms and plugin load errors"
$PY - <<'PY' 2>&1 | tail -30
import openmm as mm
print('openmm', mm.version.version)
print('lib path', mm.version.openmm_library_path)
print('num platforms', mm.Platform.getNumPlatforms())
for i in range(mm.Platform.getNumPlatforms()):
    print('  ', mm.Platform.getPlatform(i).getName())
errs = mm.Platform.getPluginLoadFailures()
print('plugin load failures:', len(errs))
for e in errs:
    print('   ', e)
PY

line "attempt a real CUDA context"
$PY - <<'PY' 2>&1 | tail -20
import openmm as mm, openmm.app as app
from openmm import unit
s = mm.System()
s.addParticle(1.0)
f = mm.NonbondedForce(); f.addParticle(0.0, 0.1, 0.1); s.addForce(f)
for name in ('CUDA', 'OpenCL', 'CPU'):
    try:
        p = mm.Platform.getPlatformByName(name)
    except Exception as e:
        print(f'{name}: not registered ({e})'); continue
    try:
        c = mm.Context(s, mm.VerletIntegrator(0.001*unit.picoseconds), p)
        c.setPositions([[0, 0, 0]])
        c.getState(getEnergy=True)
        print(f'{name}: WORKS')
        del c
    except Exception as e:
        print(f'{name}: FAILS -> {type(e).__name__}: {e}')
PY

line "openmm self test"
$PY -m openmm.testInstallation 2>&1 | tail -20
