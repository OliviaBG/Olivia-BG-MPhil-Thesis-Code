#!/bin/bash
#
# push_to_pod.sh
# ============================================================
# Copies everything the ACK1 rank-1 CRM1 docking MD needs from this WSL
# machine to a bare RunPod GPU pod. Run this FROM WSL, from the AlphaFold
# folder:
#
#   cd <this repository>
#   bash push_to_pod.sh
#
# Copies a targeted set (~15 MB), not the whole folder -- fpocket/,
# node_modules/, venv/, the PDB caches and the thesis figure dirs are all
# hundreds of MB and none of it is needed to run MD. Pass --full to sync
# the entire project minus those excludes instead.
#
# Idempotent: rsync only sends what changed, so re-run it freely after
# editing a script locally.

set -euo pipefail

POD_HOST="${POD_HOST:-<pod-host>}"
POD_PORT="${POD_PORT:-27676}"
POD_USER="${POD_USER:-root}"
POD_KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
POD_DIR="${POD_DIR:-/root/AlphaFold}"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_SYNC=0
[ "${1:-}" = "--full" ] && FULL_SYNC=1

SSH_OPTS="-p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=accept-new"

banner() { echo; echo "============================================================"; echo "$1"; echo "============================================================"; }

command -v rsync >/dev/null 2>&1 || { echo "rsync not found. Install it: sudo apt-get install -y rsync"; exit 1; }

banner "0. Connectivity check"
echo "Target: $POD_USER@$POD_HOST:$POD_PORT  ->  $POD_DIR"
if ! ssh $SSH_OPTS -o BatchMode=yes -o ConnectTimeout=15 "$POD_USER@$POD_HOST" 'echo "SSH OK: $(hostname)"'; then
    echo
    echo "ERROR: cannot reach the pod over SSH."
    echo "  - Is the pod running? RunPod reassigns the port on every restart."
    echo "  - Use the 'SSH over exposed TCP' line from the pod's Connect tab,"
    echo "    NOT the ssh.runpod.io proxy line (that one can't do file transfers)."
    exit 1
fi
ssh $SSH_OPTS "$POD_USER@$POD_HOST" "mkdir -p '$POD_DIR'"

# ------------------------------------------------------------------
banner "1. Hunting for the rank-1 screen states"
# ------------------------------------------------------------------
# These are the ONLY resumable point that exists for rank 1: the 20 ns run
# never saved a state of its own, and the states it did use lived on the
# now-dead pod at /root/AlphaFold/ack1_rank1_screen_states/. If a copy
# survives anywhere on this machine, it saves re-running Stage A.
FOUND_STATES=""
for guess in \
    "$SRC_DIR/ack1_rank1_screen_states" \
    "$HOME/AlphaFold/ack1_rank1_screen_states" \
    "$HOME/ack1_rank1_screen_states" ; do
    if [ -d "$guess" ] && ls "$guess"/*.xml >/dev/null 2>&1; then
        FOUND_STATES="$guess"; break
    fi
done

if [ -z "$FOUND_STATES" ]; then
    echo "Not in the usual places -- searching \$HOME and /mnt/c/Users (30s cap) ..."
    FOUND_STATES="$(timeout 30 find "$HOME" /mnt/c/Users -maxdepth 6 -type d \
        -name 'ack1_rank1_screen_states' 2>/dev/null | head -1 || true)"
fi

if [ -n "$FOUND_STATES" ]; then
    echo "FOUND: $FOUND_STATES"
    ls -la "$FOUND_STATES"
    echo "-> will be copied; the 50 ns run can resume the original screen 2."
else
    echo "NOT FOUND anywhere on this machine."
    echo "-> the original 2 ns screen states are gone with the old pod."
    echo "   Recover by re-running Stage A on the new pod (3 x 2 ns, ~12 min GPU):"
    echo "       python run_ack1_rank1_50ns.py --rebuild-screens"
    echo "   That produces a NEW screen set -- see the script's docstring for what"
    echo "   that means for comparability with the existing 20 ns result."
fi

# ------------------------------------------------------------------
banner "2. Syncing"
# ------------------------------------------------------------------
if [ "$FULL_SYNC" = "1" ]; then
    echo "--full: syncing the whole project minus the known-unneeded bulk."
    rsync -avz --progress -e "ssh $SSH_OPTS" \
        --exclude 'fpocket/' \
        --exclude 'frontend/node_modules/' \
        --exclude 'venv/' --exclude 'pdbfixer_env/' \
        --exclude '__pycache__/' \
        --exclude 'crm1_eval_pdb_cache/' --exclude 'iupred_raw_cache/' \
        --exclude 'nes_data_pipeline/nesdb_cache*/' \
        --exclude 'thesis_figures/' --exclude 'thesis_figures_preview*/' \
        --exclude 'archive_*/' --exclude 'esm_embeddings/' \
        --exclude '*.docx' --exclude '~$*' --exclude '*.xlsx' \
        --exclude 'Test code/' --exclude 'CRM1 structure/' \
        "$SRC_DIR/" "$POD_USER@$POD_HOST:$POD_DIR/"
else
    # Targeted list. Missing entries are skipped with a warning rather than
    # aborting the whole sync (rsync's default is to fail the run).
    FILE_LIST="$(mktemp)"
    trap 'rm -f "$FILE_LIST"' EXIT
    cat > "$FILE_LIST" <<'EOF'
app.py
md_refinement.py
md_job_queue.py
remote_md_dispatch.py
remote_md_runner.py
sumoylation_predictor.py
quick_helix_analysis.py
pocket_detector.py
nes_ml_predictor_improved.py
consensus_accessibility.py
nes_reference_profiles.json
requirements.txt
crm1.pdb
CRM1.pdb
crm1_reference/CRM1_Ran_only.pdb
crm1_reference/CRM1_Ran_3GJX_groove_shell_cache.json
models/
Q07912_full_pipeline_scan.json
run_ack1_rank1_50ns.py
run_ack1_rank1_continued_trajectory.py
run_ack1_rank1_best_anchor_frame.py
run_ack1_replicate_study.py
run_ack1_md_refinement.py
ack1_rank1_continued_trajectory_result.json
ack1_replicate_study_result.json
ack1_replicate_study_seed.json
pod_setup.sh
EOF

    # Drop anything that doesn't exist locally, so one missing file can't
    # abort the sync.
    PRESENT="$(mktemp)"; trap 'rm -f "$FILE_LIST" "$PRESENT"' EXIT
    while IFS= read -r entry; do
        [ -z "$entry" ] && continue
        if [ -e "$SRC_DIR/${entry%/}" ]; then
            echo "$entry" >> "$PRESENT"
        else
            echo "  skipping (not present locally): $entry"
        fi
    done < "$FILE_LIST"

    rsync -avz --progress -e "ssh $SSH_OPTS" \
        --files-from="$PRESENT" \
        --exclude '__pycache__/' \
        --exclude 'models/backup_before_*' \
        --exclude 'models/old_pre_rf/' \
        --exclude 'models/pretrain_backup_*' \
        "$SRC_DIR/" "$POD_USER@$POD_HOST:$POD_DIR/"
fi

if [ -n "$FOUND_STATES" ]; then
    echo
    echo "Copying screen states ..."
    rsync -avz --progress -e "ssh $SSH_OPTS" \
        "$FOUND_STATES/" "$POD_USER@$POD_HOST:$POD_DIR/ack1_rank1_screen_states/"
fi

# ------------------------------------------------------------------
banner "3. Verifying what landed"
# ------------------------------------------------------------------
ssh $SSH_OPTS "$POD_USER@$POD_HOST" bash -s <<EOF
cd "$POD_DIR" || exit 1
echo "Total size: \$(du -sh . | cut -f1)"
echo
for f in md_refinement.py app.py crm1_reference/CRM1_Ran_only.pdb \\
         run_ack1_rank1_50ns.py ack1_rank1_continued_trajectory_result.json \\
         pod_setup.sh ; do
    if [ -e "\$f" ]; then echo "  OK      \$f"; else echo "  MISSING \$f"; fi
done
echo
if ls ack1_rank1_screen_states/*.xml >/dev/null 2>&1; then
    echo "  OK      ack1_rank1_screen_states/ (\$(ls ack1_rank1_screen_states/*.xml | wc -l) state files)"
else
    echo "  ABSENT  ack1_rank1_screen_states/ -- use --rebuild-screens on the 50 ns run"
fi
EOF

banner "NEXT"
cat <<EOF
    ssh $POD_USER@$POD_HOST -p $POD_PORT -i $POD_KEY
    cd $POD_DIR && bash pod_setup.sh
EOF
