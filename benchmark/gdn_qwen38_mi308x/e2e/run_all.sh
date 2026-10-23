#!/usr/bin/env bash
# 3 rounds x 4 variants in parallel on 4 TP2 GPU pairs; GPU pair assignment rotates per round.
# GPU utilisation of all 8 GPUs is sampled every 10 s (other tenants share these GPUs).
G=/root/workspace/gdn_optimization
OUTROOT=${OUTROOT:-$G/results/e2e_final}
VARIANTS=(base sgl_tuned aiter_pf aiter_all)
PAIRS=(0,1 2,3 4,5 6,7)
mkdir -p $OUTROOT
( while true; do echo "$(date +%s) $(rocm-smi --showuse | grep 'GPU use' | awk '{printf "%s ", $NF}')"; sleep 10; done ) >> $OUTROOT/gpu_use.log 2>&1 &
MON=$!
for round in ${ROUNDS:-1 2 3}; do
  # start a round only after all 8 GPUs read 0% for 3 consecutive samples (no other tenant)
  idle=0; while (( idle < 3 )); do
    if [[ -z $(rocm-smi --showuse | grep 'GPU use' | awk '$NF != 0') ]]; then idle=$((idle+1)); else idle=0; fi; sleep 10
  done
  echo "round $round start $(date +%s)" >> $OUTROOT/rounds.log
  for i in 0 1 2 3; do
    v=${VARIANTS[$i]}; p=${PAIRS[$(( (i + round - 1) % 4 ))]}
    echo "round $round $v $p" >> $OUTROOT/rounds.log
    PROFILE=$([[ $round == 1 ]] && echo 1 || echo 0) RESULTS_ROOT=$OUTROOT DIVERSITY=1 $G/e2e/run_variant.sh $v $p $((7200 + i)) $round > $OUTROOT/${v}_round$round.out 2>&1 &
  done
  wait $(jobs -p | grep -v $MON) 2>/dev/null
  while [[ $(jobs -rp | grep -vc $MON) -gt 0 ]]; do sleep 5; done
  echo "round $round end $(date +%s)" >> $OUTROOT/rounds.log
done
kill $MON
echo ALL DONE
