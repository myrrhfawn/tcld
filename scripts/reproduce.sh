#!/usr/bin/env bash
# Reproduce every number of the paper tables: run inside the container, e.g.
#   make docker-run CMD="bash scripts/reproduce.sh all"
# Stages (any subset, in order): zeroshot method baselines enc xrec L virat all
#   method    : TCLD and its control (same recipe, tcl_weight 0) on the original, night and jpeg settings, seeds 0 1 2
#   baselines : Tent, ActMAD, self-training, Mean Teacher, Mean Teacher + TCLD, coordinate / moment / same-frame
#               consistency, same three settings, seeds 0 1 2
#   enc       : the *_enc variants (trainable encoder) of everything above
#   xrec      : the cross-recording protocol (adapt on the unlabeled test recording, evaluate on the labeled one)
#   L         : D-FINE-L method and control, four settings, seed 0, 1500 iterations (same number of pairs as S at 750)
#   virat     : VIRAT Ground 2.0, 11 scenes, method and control, seeds 0 1 2
# Every cell writes runs/<exp>/<dataset>/result_<camera>_seed<seed>.json and is skipped when that file exists,
# so the script can be re-run after an interruption. Expect ~15 min per S cell and ~25 min per L cell on a 6 GB GPU.
# Afterwards: python scripts/audit_protocol.py && python scripts/stats_method_vs_control.py --preset xrec5 &&
#             python scripts/stats_baselines.py && python scripts/abs_table.py
set -u
cd "$(dirname "$0")/.."
export TCLD_NO_CKPT=${TCLD_NO_CKPT:-1}   # do not keep adapted checkpoints (several GB per sweep); unset to keep them

STAGES=${*:-all}
C6="MOT17-02 MOT17-04 MOT17-09 MOT20-01 MOT20-03 MOT20-05"          # original recordings
C7="MOT17-02 MOT17-04 MOT17-09 MOT20-01 MOT20-02 MOT20-03 MOT20-05" # night / jpeg shifts
V11="0000 0001 0002 0100 0101 0102 0400 0401 0500 0502 0503"
PAIRS="MOT17-04:MOT17-03 MOT20-01:MOT20-07 MOT20-03:MOT20-04 MOT20-05:MOT20-04 MOT17-02:MOT17-01"  # eval:adapt
METHOD=tf_a03_w5_n32_iou08_dense; CTRL=ctrl_a03_dense
BASE="tent_a03_dense actmad_a03_dense st_a03_dense mt_a03_dense mttcl_w10_a03_dense \
coord_a03_w5_n32_iou08_dense moments_a03_w5_n32_iou08_dense sameframe_a03_w5_n32_iou08_dense"
XBASE="tent_a03_dense actmad_a03_dense mt_a03_dense mttcl_w10_a03_dense \
coord_a03_w5_n32_iou08_dense moments_a03_w5_n32_iou08_dense sameframe_a03_w5_n32_iou08_dense"

has() { [[ " $STAGES " == *" all "* || " $STAGES " == *" $1 "* ]]; }
sweep() { bash scripts/sweep_dev.sh "$1" "$2" "$3" "$4" "$5"; }   # dataset cams iters exps seeds

# cross-recording cell: adaptation sequence differs from the evaluation sequence
xcell() {  # exp seed iters
  local exp=$1 seed=$2 iters=$3 p cam src
  for p in $PAIRS; do
    cam=${p%%:*}; src=${p##*:}
    [ -f "runs/$exp/mot/result_${cam}_seed${seed}.json" ] && { echo "skip $exp $cam seed$seed"; continue; }
    echo "=== $exp mot $cam (adapt on $src) seed$seed $(date +%H:%M)"
    python scripts/run_adapt.py --exp configs/exp/$exp.yml --profile laptop --dataset mot --cameras $cam \
        --adapt-seq $src --seed $seed --iters $iters --notes "reproduce"
  done
}

if has zeroshot; then
  python scripts/run_eval.py --dataset mot --profile laptop --model dfine_s --cameras $C7 --frame-stride 3
  python scripts/run_eval.py --dataset virat --profile laptop --model dfine_s --cameras $V11 --frame-stride 30
fi
if has method; then
  sweep mot "$C6" 750 "$METHOD $CTRL" "0 1 2"
  for s in night jpeg; do sweep mot "$C7" 750 "${METHOD}_$s ${CTRL}_$s" "0 1 2"; done
fi
if has baselines; then
  sweep mot "$C6" 750 "$BASE" "0 1 2"
  for s in night jpeg; do sweep mot "$C7" 750 "$(for e in $BASE; do echo -n "${e}_$s "; done)" "0 1 2"; done
fi
if has enc; then
  sweep mot "$C6" 750 "${METHOD}_enc ${CTRL}_enc $(for e in $BASE; do echo -n "${e}_enc "; done)" "0 1 2"
  for s in night jpeg; do
    sweep mot "$C7" 750 "${METHOD}_${s}_enc ${CTRL}_${s}_enc $(for e in $BASE; do echo -n "${e}_${s}_enc "; done)" "0 1 2"
  done
fi
if has xrec; then
  for seed in 0 1 2; do
    for e in ${METHOD}_xrec ${CTRL}_xrec ${METHOD}_enc_xrec ${CTRL}_enc_xrec; do xcell $e $seed 750; done
    for e in $XBASE; do xcell ${e}_xrec $seed 750; xcell ${e}_enc_xrec $seed 750; done
  done
fi
if has L; then
  sweep mot "$C6" 1500 "tf_a03_w5_n16_iou08_dense_L ${CTRL}_L" "0"
  for s in night jpeg; do sweep mot "$C7" 1500 "tf_a03_w5_n16_iou08_dense_${s}_L ${CTRL}_${s}_L" "0"; done
  for e in tf_a03_w5_n16_iou08_dense_xrec_L ${CTRL}_xrec_L; do xcell $e 0 1500; done
fi
if has virat; then
  sweep virat "$V11" 750 "${METHOD}_virat ${CTRL}_virat" "0 1 2"
fi
echo "reproduce done $(date +%H:%M)"
