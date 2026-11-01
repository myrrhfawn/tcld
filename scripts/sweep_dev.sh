#!/usr/bin/env bash
# Dev sweep on the laptop: all method variants and baselines on the given cameras.
# Usage: bash scripts/sweep_dev.sh <dataset: virat|mot> "<cameras>" [iters] ["exps"] ["seeds"]
set -u
DS=${1:-virat}; CAMS=${2:-"0102"}; ITERS=${3:-300}; SEEDS=${5:-"0"}
EXPS=${4:-"tcld_ema_w1 tcld_ema_kl tcld_dev tcld_sym_w1 tcld_ema_w1_noanchor tcld_ema_w1_nogate base_coord base_sameframe_kl base_st"}
LOG=${TCLD_RUNS:-runs}/logs/sweep_${DS}_$(date +%Y%m%d_%H%M).log; mkdir -p "$(dirname "$LOG")"
echo "logging to $LOG"
exec > >(tee -a "$LOG") 2>&1
for seed in $SEEDS; do for exp in $EXPS; do
  for cam in $CAMS; do
    if [ -f "${TCLD_RUNS:-runs}/$exp/$DS/result_${cam}_seed${seed}.json" ]; then echo "skip $exp $DS $cam seed$seed (result exists)"; continue; fi
    echo "=== $exp $DS $cam seed$seed $(date +%H:%M)"
    timeout -s KILL "${TCLD_CELL_TIMEOUT:-5400}" python scripts/run_adapt.py --exp configs/exp/$exp.yml --profile laptop --dataset $DS --cameras $cam --seed $seed --iters $ITERS --notes "sweep" 2>&1 \
      | grep --line-buffered -E '^\[.*\] (pairs|eval|base|diag it (100|200|300))|Error|Traceback|non-finite'
  done
done; done
