#!/usr/bin/env bash
# Pull results from all five beta pods into results/, then copy to the Windows folder.
#   bash pull_beta5.sh
set -u
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Optional mirror directory; results always land in results/ regardless.
WIN="${MIRROR_DIR:-}"
PODS=("<pod-host> 24072 1" "<pod-host> 23191 2" "<pod-host> 15693 3"
      "<pod-host> 16656 4" "<pod-host> 32148 5")

mkdir -p "$HERE/results"
for p in "${PODS[@]}"; do
  set -- $p; H=$1; P=$2; S=$3
  scp -i "$KEY" -P "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      root@"$H":/workspace/ack1/beta${S}/beta_results_shard${S}of5.tsv \
      "$HERE/results/" 2>/dev/null && echo "  pod$S ok" || echo "  pod$S nothing yet"
  scp -i "$KEY" -P "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      root@"$H":/workspace/ack1/beta${S}/beta.log \
      "$HERE/results/beta5_pod${S}.log" 2>/dev/null
done

echo
echo "== unique completed runs =="
cat "$HERE"/results/beta_results*.tsv 2>/dev/null | grep -v '^receptor' \
  | awk -F'\t' '{print $1"\t"$3"\t"$6}' | sort -u > /tmp/_beta_uniq.$$
wc -l < /tmp/_beta_uniq.$$ | awk '{print $1" of 204"}'
awk -F'\t' '{print $1}' /tmp/_beta_uniq.$$ | sort | uniq -c
rm -f /tmp/_beta_uniq.$$

echo
echo "== scramble runs (the ones the matched-pair test needs) =="
cat "$HERE"/results/beta_results*.tsv 2>/dev/null | grep -v '^receptor' \
  | awk -F'\t' '$3 ~ /NEG_scramble/ {print $1"\t"$3"\t"$6}' | sort -u \
  | awk -F'\t' '{c[$1"  "$2]++} END {for (k in c) printf "  %-40s %d/6 seeds\n", k, c[k]}' \
  | sort

if [ -n "$WIN" ]; then
  mkdir -p "$WIN" 2>/dev/null
  cp "$HERE"/results/beta_results*.tsv "$WIN"/ 2>/dev/null && echo && echo "mirrored to $WIN"
fi
