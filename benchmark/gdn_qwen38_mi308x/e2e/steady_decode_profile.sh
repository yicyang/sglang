#!/usr/bin/env bash
# Steady-state decode profile: 31 long requests (mamba pool caps running reqs at 31),
# profile 20 decode steps once every request is decoding.
# usage: steady_decode_profile.sh <variant> <gpus> <port>
set -uo pipefail
V=$1 GPUS=$2 PORT=$3
G=/root/workspace/gdn_optimization
OUT=$G/results/e2e/$V/round0
WORKLOADS=none HOLD_SECONDS=600 $G/e2e/run_variant.sh $V $GPUS $PORT 0 > $G/results/e2e/steady_$V.out 2>&1 &
for i in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && grep -q "server up" $G/results/e2e/steady_$V.out && break; sleep 10; done
python3 -m sglang.bench_serving --backend sglang --model /models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16 \
  --dataset-name random --host localhost --port $PORT --num-prompts 31 --random-input 2000 --random-output 1500 \
  --random-range-ratio 1.0 --warmup-requests 2 --max-concurrency 31 > $OUT/steady_bench.log 2>&1 &
sleep 60
curl -s localhost:$PORT/start_profile -H 'Content-Type: application/json' \
  -d "{\"output_dir\": \"$OUT/profile_steady\", \"num_steps\": 20, \"activities\": [\"CPU\", \"GPU\"]}"
echo
wait %2 2>/dev/null
sleep 5
echo "steady profile done: $V"
