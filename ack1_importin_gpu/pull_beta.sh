#!/usr/bin/env bash
# Pull beta-screen results from all six pods into results/
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PODS=("<pod-host> 13087 1" "<pod-host> 33946 2" "<pod-host> 22965 3"
      "<pod-host> 10251 4" "<pod-host> 12493 5" "<pod-host> 16073 6")
mkdir -p "$HERE/results"
for p in "${PODS[@]}"; do
  set -- $p; H=$1; P=$2; S=$3
  scp -i "$KEY" -P "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      root@"$H":/workspace/ack1/pod${S}/beta_results_shard${S}of6.tsv \
      "$HERE/results/" 2>/dev/null && echo "  pod$S ok" || echo "  pod$S nothing yet"
  scp -i "$KEY" -P "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      root@"$H":/workspace/ack1/pod${S}/beta.log "$HERE/results/beta_pod${S}.log" 2>/dev/null
done
n=$(cat "$HERE"/results/beta_results_shard*.tsv 2>/dev/null | grep -vc '^receptor')
echo "total beta rows: ${n:-0}"
# Optional: mirror the results into another directory as well (set
# MIRROR_DIR to a path). Off by default; results always land in results/.
if [ -n "${MIRROR_DIR:-}" ]; then
  mkdir -p "$MIRROR_DIR" 2>/dev/null
  cp "$HERE"/results/beta_results_shard*.tsv "$MIRROR_DIR"/ 2>/dev/null \
    && echo "mirrored to $MIRROR_DIR"
fi
