#!/bin/bash
# Run every task config in a folder (or a single config) for several seeds.
#
# Usage:
#   bash scripts/run.sh configs/mujoco_gravity_friction            # all 32 tasks, seeds 0-4
#   bash scripts/run.sh configs/antmaze/antmaze-medium-play-3.yaml  # one task, seeds 0-4
#   SEEDS="0 1" OUT=./my_runs bash scripts/run.sh configs/adroit
#
# Runs are sequential; on a cluster, submit one job per (config, seed) instead.
set -euo pipefail

TARGET=${1:?"usage: bash scripts/run.sh <config.yaml | config folder>"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
OUT=${OUT:-./runs}

if [[ -d "$TARGET" ]]; then
  CONFIGS=$(ls "$TARGET"/*.yaml)
else
  CONFIGS=$TARGET
fi

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    echo "=== $cfg | seed $seed"
    python train.py --config "$cfg" --seed "$seed" --dir "$OUT"
  done
done
