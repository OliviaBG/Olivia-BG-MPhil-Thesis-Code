#!/usr/bin/env bash
# Pull everything from the pods, merge the shards, and drop the result into the
# Windows AlphaFold folder so it can be analysed.
#
#   bash merge_and_share.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIN="${WIN:-$HERE}"

echo "== pulling from all pods"
bash "$HERE/pull_pods.sh"

echo
echo "== merging shards"
cd "$HERE/results" || exit 1
first=$(ls screen_results_shard*.tsv screen_results_known.tsv 2>/dev/null | head -1)
[ -z "$first" ] && { echo "no results found"; exit 1; }
head -1 "$first" > merged_results.tsv
for f in screen_results_shard*.tsv screen_results_known.tsv; do
  [ -f "$f" ] && tail -n +2 "$f"
done | grep -v '^paralogue' | sort -u >> merged_results.tsv
n=$(( $(wc -l < merged_results.tsv) - 1 ))
echo "   $n unique runs merged into results/merged_results.tsv"

echo
echo "== breakdown"
awk -F'\t' 'NR>1{print $1, $2}' merged_results.tsv | sort | uniq -c | sort -rn

echo
echo "== copying to the Windows folder"
mkdir -p "$WIN/results"
cp merged_results.tsv "$WIN/results/" && echo "   -> $WIN/results/merged_results.tsv"
cp screen_results_shard*.tsv "$WIN/results/" 2>/dev/null
echo
echo "Done. The merged results file is in the project folder, ready for analysis."
