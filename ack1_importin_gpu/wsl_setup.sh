#!/usr/bin/env bash
# Run this ONCE inside WSL. It makes a WSL-native working copy of the pipeline under
# ~/AlphaFold, fixes line endings and permissions, and checks that ssh/scp and your key
# are usable. Nothing on the Windows side is modified.
#
#   cd <this repository>/ack1_importin_gpu
#   bash wsl_setup.sh
#
# Then work from ~/AlphaFold/ack1_importin_gpu.
set -euo pipefail

WIN_DIR="${WIN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEST="${DEST:-$HOME/AlphaFold/ack1_importin_gpu}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

grep -qi microsoft /proc/version 2>/dev/null \
  && say "WSL detected: $(grep -o 'WSL[0-9]*' /proc/version | head -1 || echo WSL)" \
  || say "not running under WSL - continuing anyway"

# ------------------------------------------------------------------- source files
if [ ! -d "$WIN_DIR" ]; then
  echo "Cannot find $WIN_DIR"
  echo "If your Windows drive is mounted elsewhere, set WIN_DIR=... and re-run."
  exit 1
fi

say "copying to $DEST"
mkdir -p "$DEST"
cp -f "$WIN_DIR"/* "$DEST"/ 2>/dev/null || true
cd "$DEST"
ls -la

# --------------------------------------------------------- line endings and modes
say "normalising line endings and permissions"
# /mnt/c round-trips can introduce CRLF; a CR in a shebang breaks the script silently
for f in *.sh *.py; do
  [ -f "$f" ] || continue
  if grep -qU $'\r' "$f" 2>/dev/null; then
    sed -i 's/\r$//' "$f"
    echo "  stripped CR: $f"
  fi
done
chmod +x ./*.sh 2>/dev/null || true
chmod 644 ./*.pdb ./*.fasta ./*.md 2>/dev/null || true
echo "  done"

# ------------------------------------------------------------------ ssh toolchain
say "ssh client"
if command -v ssh >/dev/null 2>&1 && command -v scp >/dev/null 2>&1; then
  echo "  ok: $(ssh -V 2>&1)"
else
  echo "  MISSING. Install with:"
  echo "     sudo apt update && sudo apt install -y openssh-client"
  exit 1
fi

# ----------------------------------------------------------------------- ssh key
say "ssh key: $KEY"
if [ ! -f "$KEY" ]; then
  echo "  NOT FOUND."
  case "$KEY" in
    /mnt/*) echo "  It is on the Windows filesystem, which WSL exposes as mode 0777;"
            echo "  ssh will refuse it. Copy it into the WSL home instead:"
            echo "     mkdir -p ~/.ssh && cp '$KEY' ~/.ssh/id_ed25519"
            echo "     chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519" ;;
    *)      echo "  If the key lives on the Windows side, copy it in:"
            echo "     mkdir -p ~/.ssh && chmod 700 ~/.ssh"
            echo "     cp /mnt/c/Users/<windows-username>/.ssh/id_ed25519 ~/.ssh/"
            echo "     chmod 600 ~/.ssh/id_ed25519" ;;
  esac
  exit 1
fi

perm="$(stat -c '%a' "$KEY")"
if [ "$perm" != "600" ] && [ "$perm" != "400" ]; then
  echo "  permissions are $perm; ssh requires 600. Fixing."
  chmod 700 "$(dirname "$KEY")" 2>/dev/null || true
  chmod 600 "$KEY"
  perm="$(stat -c '%a' "$KEY")"
fi
echo "  ok: mode $perm"
if [[ "$KEY" == /mnt/* ]]; then
  echo "  WARNING: the key is on /mnt/c. Windows-backed files ignore chmod, so ssh may"
  echo "  still reject it. Copy it into ~/.ssh and use that path instead."
fi

cat <<EOF

================================================================================
Working copy ready at:

    $DEST

Next:

    cd $DEST
    bash deploy_to_pod.sh

Results come back into this same folder. To copy anything to the Windows side:

    cp $DEST/screen_results.tsv "$WIN_DIR"/
================================================================================
EOF
