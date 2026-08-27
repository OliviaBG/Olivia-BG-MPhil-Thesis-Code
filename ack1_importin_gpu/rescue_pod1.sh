#!/usr/bin/env bash
# STEP ZERO. Rescue whatever the single finishing pod has written, BEFORE anything
# kills it. Run this first and check the row count before launching the five pods.
#
#   bash rescue_pod1.sh <pod-host> 24072
#
# The file is saved as results/beta_results_rescued.tsv -- a name that collides with
# neither the six of6 shards nor the five of5 shards about to be created, and that
# still matches the beta_results*.tsv glob the screen uses to skip completed runs.
set -u
HOST="${1:-<pod-host>}"; PORT="${2:-24072}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
USER_="${USER_:-root}"
DIR="${DIR:-/workspace/ack1/finish}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/results"

echo "== rescuing from ${USER_}@${HOST}:${PORT}:${DIR}"
scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 \
    "${USER_}@${HOST}:${DIR}/beta_results.tsv" \
    "$HERE/results/beta_results_rescued.tsv" 2>/dev/null \
  && echo "   got beta_results.tsv" \
  || echo "   nothing to rescue (no completed runs yet) -- that is fine, continue"

scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 \
    "${USER_}@${HOST}:${DIR}/beta.log" "$HERE/results/beta_finish.log" 2>/dev/null

if [ -s "$HERE/results/beta_results_rescued.tsv" ]; then
  r=$(grep -vc '^receptor' "$HERE/results/beta_results_rescued.tsv")
  echo "   rescued $r completed run(s):"
  grep -v '^receptor' "$HERE/results/beta_results_rescued.tsv" \
    | awk -F'\t' '{printf "     %-8s %-26s seed%s\n", $1, $3, $6}'
else
  rm -f "$HERE/results/beta_results_rescued.tsv"
fi

echo
echo "== total completed runs now on this machine =="
cat "$HERE"/results/beta_results*.tsv 2>/dev/null | grep -v '^receptor' \
  | awk -F'\t' '{print $1"\t"$3"\t"$6}' | sort -u | wc -l
cat "$HERE"/results/beta_results*.tsv 2>/dev/null | grep -v '^receptor' \
  | awk -F'\t' '{print $1"\t"$3"\t"$6}' | sort -u | awk -F'\t' '{print $1}' \
  | uniq -c
echo "(of 204)"

# Optional: mirror the results into another directory as well (set
# MIRROR_DIR to a path). Off by default; results always land in results/.
if [ -n "${MIRROR_DIR:-}" ]; then
  mkdir -p "$MIRROR_DIR" 2>/dev/null
  cp "$HERE"/results/beta_results*.tsv "$MIRROR_DIR"/ 2>/dev/null \
    && echo "mirrored to $MIRROR_DIR"
fi
