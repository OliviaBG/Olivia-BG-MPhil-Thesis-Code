#!/bin/bash
#
# Source this (don't execute it) before running app.py, so remote_md_dispatch.py
# points MD refinement at your RunPod GPU pod instead of running locally:
#
#   source set_md_remote_env.sh
#   python app.py
#
# Mirrors confocal_pipeline/remote_config.py -- same pod, same key. Update
# both files together whenever the pod restarts and RunPod assigns a new
# SSH port (check the pod's "Connect" tab -> "SSH over exposed TCP" line,
# NOT the ssh.runpod.io proxy line, which doesn't support the scp transfers
# these dispatch scripts need).
#
# Switched back: this pod/key again (same one
# confocal_pipeline/remote_config.py points at).

export MD_REMOTE_HOST="<pod-host>"
export MD_REMOTE_USER="root"
export MD_REMOTE_KEY="$HOME/.ssh/id_ed25519"
export MD_REMOTE_PORT="17258"

# ASSUMED to match the previous pod's layout -- VERIFY these once you've
# confirmed whether this is the SAME disk/pod as before (just restarted, so
# md_worker/ and the conda env are still there) or a genuinely NEW pod (in
# which case see "MD files/REMOTE_MD_SETUP.md" steps 2-5 to reinstall):
#   - MD_REMOTE_WORKDIR: absolute path to the folder on the pod containing
#     remote_md_runner.py, md_refinement.py, and CRM1.pdb.
#   - MD_REMOTE_PYTHON: the conda env's python (OpenMM needs conda-forge;
#     plain python3 won't have it).
export MD_REMOTE_WORKDIR="/root/md_worker"
export MD_REMOTE_PYTHON="/root/miniforge3/envs/md/bin/python"   # <- verify this exists; update if the conda env name/location differs

echo "MD remote env set: ${MD_REMOTE_USER}@${MD_REMOTE_HOST}:${MD_REMOTE_PORT}  workdir=${MD_REMOTE_WORKDIR}"
