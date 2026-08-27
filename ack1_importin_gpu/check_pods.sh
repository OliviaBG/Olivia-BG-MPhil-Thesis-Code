#!/usr/bin/env bash
# Progress across all six pods. Read-only, never changes anything.
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
PODS=("<pod-host> 13087 1" "<pod-host> 33946 2" "<pod-host> 22965 3"
      "<pod-host> 10251 4" "<pod-host> 12493 5" "<pod-host> 16073 6")
TOTAL=6
done_all=0
for p in "${PODS[@]}"; do
  set -- $p; H=$1; P=$2; S=$3; D="/workspace/ack1/pod$S"
  printf '\n\033[1m--- pod %s  %s:%s ---\033[0m\n' "$S" "$H" "$P"
  ssh -i "$KEY" -p "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
      root@"$H" "cd $D 2>/dev/null || { echo 'no working directory'; exit 0; }; \
    # bracket stops pgrep matching the shell that is running this very command
    pgrep -cf '[a]ck1_importin_gpu.py' | sed 's/^/processes: /'; \
    f=screen_results_shard${S}of${TOTAL}.tsv; \
    if [ -s \$f ]; then echo -n 'rows done: '; echo \$(( \$(wc -l < \$f) - 1 )); \
    else echo 'rows done: 0'; fi; \
    tail -n 2 screen.log 2>/dev/null | tr -d '\000'; \
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>&1
done
