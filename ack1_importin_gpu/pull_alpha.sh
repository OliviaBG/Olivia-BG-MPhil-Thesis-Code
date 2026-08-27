#!/usr/bin/env bash
# Pull the importin-alpha top-up results into results/ and copy to the Windows folder.
#   bash pull_alpha.sh
set -u
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIN=/mnt/c/Users/$USER/Documents/AlphaFold/ack1_importin_gpu/results
PODS=("<pod-host> 19167 1" "<pod-host> 26471 2" "<pod-host> 10569 3")

mkdir -p "$HERE/results"
for p in "${PODS[@]}"; do
  set -- $p; H=$1; P=$2; S=$3
  scp -i "$KEY" -P "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      root@"$H":/workspace/ack1/top${S}/screen_results_topup${S}of3.tsv \
      "$HERE/results/" 2>/dev/null && echo "  pod$S ok" || echo "  pod$S nothing yet"
  scp -i "$KEY" -P "$P" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      root@"$H":/workspace/ack1/top${S}/screen.log \
      "$HERE/results/alpha_topup_pod${S}.log" 2>/dev/null
done

echo
echo "== importin-alpha panel, unique runs =="
cat "$HERE"/results/screen_results_topup*.tsv "$HERE"/results/merged_results.tsv \
    "$HERE"/results/screen_results_shard*.tsv 2>/dev/null \
  | grep -v '^paralogue' | awk -F'\t' 'NF>5 {print $1"\t"$2"\t"$3"\t"$6}' | sort -u \
  > /tmp/_a_uniq.$$
wc -l < /tmp/_a_uniq.$$ | awk '{print "  "$1" of 816 (8 surfaces x 17 peptides x 6 seeds)"}'

echo
echo "== seeds per peptide, matched-pair peptides only (want 6 on every surface) =="
awk -F'\t' '$3 ~ /^(ACK1_71-73|ACK1_64-67|NEG_scramble)/ {c[$1" "$2" "$3]++}
            END {for (k in c) printf "  %-46s %d/6\n", k, c[k]}' /tmp/_a_uniq.$$ | sort
rm -f /tmp/_a_uniq.$$

mkdir -p "$WIN" 2>/dev/null
cp "$HERE"/results/screen_results_topup*.tsv "$WIN"/ 2>/dev/null \
  && echo && echo "copied to Windows folder"
