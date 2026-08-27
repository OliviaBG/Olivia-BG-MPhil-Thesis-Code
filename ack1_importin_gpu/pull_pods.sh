#!/usr/bin/env bash
# Pull results from all six pods, once or on a loop. Read-only on the pods.
#
#   bash pull_pods.sh once
#   bash pull_pods.sh loop every 5 minutes until you Ctrl-C
#   INTERVAL=120 bash pull_pods.sh loop
#
# Files land in results/; a dated snapshot is kept whenever the total row count moves.
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${INTERVAL:-300}"
PODS=("<pod-host> 13087 1" "<pod-host> 33946 2" "<pod-host> 22965 3"
      "<pod-host> 10251 4" "<pod-host> 12493 5" "<pod-host> 16073 6")
TOTAL=6
mkdir -p "$HERE/results" "$HERE/results/snapshots"

pull_once() {
  for p in "${PODS[@]}"; do
    set -- $p; H=$1; P=$2; S=$3
    f="screen_results_shard${S}of${TOTAL}.tsv"
    scp -i "$KEY" -P "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
        root@"$H":/workspace/ack1/pod${S}/"$f" "$HERE/results/$f" 2>/dev/null \
      && echo "  pod$S ok" || echo "  pod$S no file yet"
    scp -i "$KEY" -P "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
        root@"$H":/workspace/ack1/pod${S}/screen.log \
        "$HERE/results/screen_pod${S}.log" 2>/dev/null
  done
  n=$(cat "$HERE"/results/screen_results_shard*.tsv 2>/dev/null | grep -vc '^paralogue')
  echo "$(date +%H:%M:%S)  total rows across shards: ${n:-0}"
  echo "${n:-0}"
}

if [ "${1:-once}" = "loop" ]; then
  last=-1
  while true; do
    n=$(pull_once | tail -1)
    if [ "$n" != "$last" ]; then
      cp "$HERE"/results/screen_results_shard*.tsv "$HERE/results/snapshots/" 2>/dev/null
      for g in "$HERE"/results/snapshots/screen_results_shard*.tsv; do
        [ -f "$g" ] && mv "$g" "${g%.tsv}_$(date +%Y%m%d-%H%M%S)_${n}rows.tsv" 2>/dev/null
      done
      last=$n
    fi
    sleep "$INTERVAL"
  done
else
  pull_once >/dev/null
  n=$(cat "$HERE"/results/screen_results_shard*.tsv 2>/dev/null | grep -vc '^paralogue')
  echo "pulled; ${n:-0} rows now in results/"
fi
