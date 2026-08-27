#!/bin/bash
#
# remote_md_gatekeeper.sh
#
# Forced SSH command for the "md_remote" key (see MD files/REMOTE_MD_SETUP.md).
# This script is what actually runs for EVERY connection made with that key,
# regardless of what the client asked for (that's what "command=" in
# authorized_keys does). It only allows the exact operations
# remote_md_dispatch.py needs, all confined to this directory:
#
#   - mkdir -p (this directory)
#   - upload a job_<id>_input.json file into this directory
#   - run remote_md_runner.py on a job_<id>_input.json / _output.json pair
#     in this directory
#   - download a job_<id>_output.json file from this directory
#   - delete a job_<id>_input.json / _output.json pair from this directory
#   - a health-check "is remote_md_runner.py here" test
#
# Anything else - reading other files, an interactive shell, port
# forwarding, etc. - is rejected. So even if this key is copied or stolen,
# it can only ever touch job_*.json files inside this one folder.
set -euo pipefail

# This directory, resolved absolutely, regardless of $HOME/user.
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$WORKDIR/gatekeeper.log"

cmd="${SSH_ORIGINAL_COMMAND:-}"

deny() {
    echo "$(date -Iseconds) REJECTED: ${cmd}" >> "$LOG"
    echo "gatekeeper: command not permitted for this key" >&2
    exit 1
}

# --- Exact-match cases -------------------------------------------------
case "$cmd" in
    "mkdir -p $WORKDIR")
        exec mkdir -p "$WORKDIR"
        ;;
    "test -f $WORKDIR/remote_md_runner.py && echo OK")
        exec bash -c "$cmd"
        ;;
esac

# --- Pattern-matched cases (job id = 8 lowercase hex chars) -------------
if [[ "$cmd" =~ ^scp\ -t\ ${WORKDIR}/job_[0-9a-f]{8}_input\.json$ ]]; then
    exec $cmd
elif [[ "$cmd" =~ ^scp\ -f\ ${WORKDIR}/job_[0-9a-f]{8}_output\.json$ ]]; then
    exec $cmd
elif [[ "$cmd" =~ ^rm\ -f\ ${WORKDIR}/job_[0-9a-f]{8}_input\.json\ ${WORKDIR}/job_[0-9a-f]{8}_output\.json$ ]]; then
    exec $cmd
elif [[ "$cmd" =~ ^[^[:space:]]+\ ${WORKDIR}/remote_md_runner\.py\ ${WORKDIR}/job_[0-9a-f]{8}_input\.json\ ${WORKDIR}/job_[0-9a-f]{8}_output\.json$ ]]; then
    exec $cmd
else
    deny
fi
