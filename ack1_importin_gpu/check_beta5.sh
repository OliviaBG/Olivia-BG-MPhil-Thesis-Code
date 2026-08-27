#!/usr/bin/env bash
# Status of all five beta pods. Read-only, safe to run any time.
#   bash check_beta5.sh
set -u
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
PODS=("<pod-host> 24072 1" "<pod-host> 23191 2" "<pod-host> 15693 3"
      "<pod-host> 16656 4" "<pod-host> 32148 5")

for p in "${PODS[@]}"; do
  set -- $p; H=$1; P=$2; S=$3
  printf '\n\033[1m--- pod%s  %s:%s ---\033[0m\n' "$S" "$H" "$P"
  ssh -i "$KEY" -p "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
      -o BatchMode=yes root@"$H" "
    # count only real python processes: the launch wrapper's command line also
    # contains the string beta_screen.py and would be miscounted by pgrep -f
    n=\$(ps -eo comm,cmd | awk '\$1 ~ /^python/ && /beta_screen.py/' | wc -l)
    echo -n \"running: \$n   \"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
    cd /workspace/ack1/beta${S} 2>/dev/null || { echo 'no working directory'; exit; }
    echo -n 'rows this pod: '
    grep -vhc '^receptor' beta_results_shard${S}of5.tsv 2>/dev/null || echo 0
    echo -n 'failures: '; grep -c 'FAIL' beta.log 2>/dev/null || echo 0
    echo 'last line:'; tail -n 2 beta.log 2>/dev/null | sed 's/^/   /'
  " 2>&1 | grep -v '^Warning: Permanently added'
done
