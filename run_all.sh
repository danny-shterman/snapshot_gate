#!/usr/bin/env bash
# Snapshot Engine Gate 0, end to end: tests -> reconstruction quality -> speed and KV benchmark.
#
#   ./run_all.sh                         # synthetic data, CPU-sized (about 30-40 min on a laptop)
#   DATA=npy LATENTS=latents/ DEVICE=cuda STEPS=20000 SIZE=32 WIDTH=64 ./run_all.sh   # real Wan latents
set -euo pipefail
cd "$(dirname "$0")"

DATA=${DATA:-synthetic}
LATENTS=${LATENTS:-}
DEVICE=${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
STEPS=${STEPS:-2000}
SIZE=${SIZE:-32}
WIDTH=${WIDTH:-24}
BATCH=${BATCH:-8}

python test_snapshot_gate.py
python test_benchmark.py

mkdir -p runs report
PATH_ARG=()
[ -n "$LATENTS" ] && PATH_ARG=(--path "$LATENTS")
python snapshot_gate.py --data "$DATA" "${PATH_ARG[@]}" --C 4 8 --size "$SIZE" --width "$WIDTH" \
    --steps "$STEPS" --batch "$BATCH" --device "$DEVICE" --log-every 0 --out runs/gate0.json

python benchmark.py --gate-results runs/gate0.json --out report --device "$DEVICE"
echo "Report: report/report.md and report/benchmark.png"
