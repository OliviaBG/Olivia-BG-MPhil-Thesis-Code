#!/usr/bin/env bash
# Is the finishing pod alive, and how far has it got?
#   bash check_finish.sh <host> <port>
set -u
HOST="${1:-}"; PORT="${2:-}"
[ -z "$PORT" ] && { echo "usage: bash check_finish.sh <host> <port>"; exit 1; }
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
USER_="${USER_:-root}"
DIR="${DIR:-/workspace/ack1/finish}"

ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
    "${USER_}@${HOST}" "
  echo -n 'processes: '; pgrep -cf '[b]eta_screen.py' || echo 0
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
  cd $DIR 2>/dev/null || { echo 'no working directory'; exit; }
  echo -n 'new rows this pod: '
  grep -vc '^receptor' beta_results.tsv 2>/dev/null || echo 0
  echo '--- last 8 log lines ---'
  tail -n 8 beta.log 2>/dev/null
  echo '--- any failures ---'
  grep -c '^  FAIL' beta.log 2>/dev/null || echo 0
"
