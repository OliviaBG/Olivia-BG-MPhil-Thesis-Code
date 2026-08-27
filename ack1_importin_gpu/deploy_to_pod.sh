#!/usr/bin/env bash
# Runs ON YOUR WINDOWS MACHINE, in Git Bash or WSL.
#
#   bash deploy_to_pod.sh
#
# Copies the pipeline and its inputs to the pod, then runs the bootstrap there
# (Miniforge, conda env, all dependencies, 13-thread cap, smoke test).
# Re-running is safe: nothing is destroyed and finished steps are skipped.
set -euo pipefail

POD_HOST="${POD_HOST:-<pod-host>}"
POD_PORT="${POD_PORT:-13087}"
POD_USER="${POD_USER:-root}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/ack1}"
THREADS="${THREADS:-13}"

# folder holding this script and the input files
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

SSH=(ssh -i "$KEY" -p "$POD_PORT" -o StrictHostKeyChecking=accept-new
     -o ServerAliveInterval=30 "${POD_USER}@${POD_HOST}")
SCP=(scp -i "$KEY" -P "$POD_PORT" -o StrictHostKeyChecking=accept-new)

FILES=(ack1_importin_gpu.py 1EJL.pdb sam_dimer_fixed.pdb kpna_seqs.fasta
       pod_bootstrap.sh run_screen.sh README_ack1_importin.md
       ack1_tether_results.txt)
# pod.sh stays local - it drives the pod from here, it is not needed on the pod

# WSL: files edited or round-tripped through Windows can carry CRLF, and a CR in a
# shebang makes the remote bootstrap fail with an unhelpful "\r: command not found".
for f in "$LOCAL_DIR"/*.sh "$LOCAL_DIR"/*.py; do
  [ -f "$f" ] || continue
  if grep -qU $'\r' "$f" 2>/dev/null; then
    sed -i 's/\r$//' "$f"
    echo "  stripped CR from $(basename "$f")"
  fi
done

if [ ! -f "$KEY" ]; then
  echo "SSH key not found at $KEY"
  echo "Set KEY=/path/to/key, or copy it into the WSL home:"
  echo "  mkdir -p ~/.ssh && chmod 700 ~/.ssh"
  echo "  cp /mnt/c/Users/<windows-username>/.ssh/id_ed25519 ~/.ssh/ && chmod 600 ~/.ssh/id_ed25519"
  exit 1
fi
keyperm="$(stat -c '%a' "$KEY" 2>/dev/null || echo unknown)"
if [ "$keyperm" != "600" ] && [ "$keyperm" != "400" ]; then
  echo "== ssh key mode is $keyperm, ssh requires 600 - fixing"
  chmod 600 "$KEY" 2>/dev/null || true
fi
case "$KEY" in
  /mnt/*) echo "WARNING: $KEY is on the Windows filesystem, where chmod has no effect."
          echo "         ssh will probably reject it. Copy it to ~/.ssh first." ;;
esac

echo "== checking local files in $LOCAL_DIR"
missing=0
for f in "${FILES[@]}"; do
  if [ -f "$LOCAL_DIR/$f" ]; then printf '  ok      %s\n' "$f"
  else printf '  MISSING %s\n' "$f"; missing=1; fi
done
if [ "$missing" -ne 0 ]; then
  echo
  echo "Run this from the folder that has those files, or set LOCAL_DIR=..."
  exit 1
fi

echo
echo "== testing the connection to ${POD_USER}@${POD_HOST}:${POD_PORT}"
"${SSH[@]}" 'echo "connected: $(hostname)"'

echo
echo "== copying to ${REMOTE_DIR}"
"${SSH[@]}" "mkdir -p '$REMOTE_DIR'"
for f in "${FILES[@]}"; do
  "${SCP[@]}" "$LOCAL_DIR/$f" "${POD_USER}@${POD_HOST}:${REMOTE_DIR}/"
  printf '  sent %s\n' "$f"
done

echo
echo "== running bootstrap on the pod (Miniforge + conda env + deps + smoke test)"
echo "   this takes roughly 5-15 minutes the first time"
echo
"${SSH[@]}" "chmod +x '$REMOTE_DIR/pod_bootstrap.sh' && \
             WORK='$REMOTE_DIR' THREADS='$THREADS' bash '$REMOTE_DIR/pod_bootstrap.sh'"

cat <<EOF

================================================================================
Done. To start the overnight run:

  bash pod.sh start        # launches it under nohup, returns immediately
  bash pod.sh status       # progress, pid, GPU utilisation
  bash pod.sh follow       # live log; Ctrl-C stops watching, not the job
  bash pod.sh analyse      # summary, part-way through is fine
  bash pod.sh fetch        # copy results back into this folder

Or log in and do it by hand:

  ssh -i $KEY -p $POD_PORT ${POD_USER}@${POD_HOST}
  setsid nohup python -u ack1_importin_gpu.py --stage screen > screen.log 2>&1 < /dev/null &
  echo $! > screen.pid
================================================================================
EOF
