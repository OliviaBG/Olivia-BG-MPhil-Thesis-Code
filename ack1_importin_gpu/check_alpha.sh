#!/usr/bin/env bash
# Status of the three importin-alpha top-up pods. Read-only, safe any time.
#   bash check_alpha.sh
set -u
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
PODS=("<pod-host> 19167 1" "<pod-host> 26471 2" "<pod-host> 10569 3")

for p in "${PODS[@]}"; do
  set -- $p; H=$1; P=$2; S=$3
  printf '\n\033[1m--- pod%s  %s:%s ---\033[0m\n' "$S" "$H" "$P"
  ssh -i "$KEY" -p "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
      -o BatchMode=yes root@"$H" "
    # count real python processes only: the launch wrapper's command line also contains
    # the string ack1_importin_gpu.py and pgrep -f would count it as a second job
    n=\$(ps -eo comm,cmd | awk '\$1 ~ /^python/ && /ack1_importin_gpu.py/' | wc -l)
    echo -n \"running: \$n   \"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
    cd /workspace/ack1/top${S} 2>/dev/null || { echo 'no working directory yet'; exit; }
    echo -n 'new rows this pod: '
    grep -vhc '^paralogue' screen_results_topup${S}of3.tsv 2>/dev/null || echo 0
    echo -n 'failures: '; grep -c 'FAIL' screen.log 2>/dev/null || echo 0
    echo 'last 3 log lines:'; tail -n 3 screen.log 2>/dev/null | sed 's/^/   /'
  " 2>&1 | grep -v '^Warning: Permanently added'
done
echo
echo "target: 408 new runs (8 surfaces x 17 peptides x seeds 3,4,5)"
echo "the first 168 are the matched-pair peptides, which is what the statistics need"
