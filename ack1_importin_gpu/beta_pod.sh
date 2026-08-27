#!/usr/bin/env bash
# Launch the beta-family screen on ONE pod. Run once per pod.
#   bash beta_pod.sh <host> <port> <shard> <total>
HOST="$1"; PORT="$2"; SHARD="$3"; TOTAL="$4"
[ -z "$TOTAL" ] && { sed -n '2,5p' "$0"; exit 1; }
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
DIR="/workspace/ack1/pod${SHARD}"
PY=/opt/miniforge3/envs/ack1/bin/python
THREADS="${THREADS:-13}"
MD="${MD:-1000}"; SEEDS="${SEEDS:-6}"; FRAMES="${FRAMES:-25}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 root@"$HOST")
C=(scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new)

echo "== pod$SHARD  $HOST:$PORT -> $DIR"
"${S[@]}" "pkill -9 -f beta_screen.py; sleep 2; mkdir -p $DIR; echo ready"
for f in beta_screen.py ack1_importin_gpu.py 1M5N.pdb 5J3V.pdb; do
  "${C[@]}" "$HERE/$f" root@"$HOST":"$DIR"/ >/dev/null 2>&1 \
    && echo "   sent $f" || echo "   FAILED $f"
done
"${S[@]}" "cd $DIR && export OPENMM_CPU_THREADS=$THREADS ACK1_THREADS=$THREADS && \
  setsid nohup $PY -u beta_screen.py --shard $SHARD/$TOTAL --seeds $SEEDS \
      --md $MD --frames $FRAMES > beta.log 2>&1 < /dev/null & \
  sleep 8; cd $DIR && echo '--- first log lines ---' && head -n 10 beta.log"
