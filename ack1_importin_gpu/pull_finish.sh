#!/usr/bin/env bash
# Pull results from the single finishing pod into results/, then copy to Windows.
#   bash pull_finish.sh <host> <port>
set -u
HOST="${1:-}"; PORT="${2:-}"
[ -z "$PORT" ] && { echo "usage: bash pull_finish.sh <host> <port>"; exit 1; }
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
USER_="${USER_:-root}"
DIR="${DIR:-/workspace/ack1/finish}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Optional mirror directory; results always land in results/ regardless.
WIN="${MIRROR_DIR:-}"

mkdir -p "$HERE/results"
scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
    "${USER_}@${HOST}:${DIR}/beta_results.tsv" "$HERE/results/" 2>/dev/null \
    && echo "  beta_results.tsv ok" || echo "  nothing yet"
scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
    "${USER_}@${HOST}:${DIR}/beta.log" "$HERE/results/beta_finish.log" 2>/dev/null

n=$(cat "$HERE"/results/beta_results*.tsv 2>/dev/null | grep -vc '^receptor')
echo "total beta rows across all files: ${n:-0} of 204"
echo "--- by receptor ---"
cat "$HERE"/results/beta_results*.tsv 2>/dev/null | grep -v '^receptor' \
  | awk -F'\t' '{print $1}' | sort | uniq -c

if [ -n "$WIN" ]; then
  mkdir -p "$WIN" 2>/dev/null
  cp "$HERE"/results/beta_results*.tsv "$WIN"/ 2>/dev/null && echo "mirrored to $WIN"
fi
