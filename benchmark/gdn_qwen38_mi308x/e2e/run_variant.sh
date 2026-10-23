#!/usr/bin/env bash
# End-to-end run of one GDN backend variant on one TP2 GPU pair.
# usage: run_variant.sh <variant> <gpus e.g. 0,1> <port> <round>
#   variants:
#     base        sglang branch head (unmodified), GDN triton (current default)
# All variants share one aiter checkout (installed commit 4752a408 + GDN-only changes).
#     sgl_tuned   sglang with gfx942-tuned FLA triton configs, GDN triton
#     aiter_pf    tuned sglang, --linear-attn-prefill-backend aiter (decode: tuned triton)
#     aiter_all   tuned sglang, --linear-attn-backend aiter (prefill aiter opt_vk + decode flydsl)
#     aiter_pf_flyk5  as aiter_pf with SGLANG_GDN_AITER_K5=flydsl
# env: WORKLOADS (default "A B C D"), GSM8K=1, PROFILE=1
set -uo pipefail
VARIANT=$1 GPUS=$2 PORT=$3 ROUND=${4:-0}
G=/root/workspace/gdn_optimization
OUT=${RESULTS_ROOT:-$G/results/e2e}/$VARIANT/round$ROUND
mkdir -p "$OUT"

case $VARIANT in
  base)      SGL=$G/sglang_base/python; AIT=$G/aiter_fork; EXTRA=() ;;
  sgl_tuned) SGL=$G/sglang/python; AIT=$G/aiter_fork; EXTRA=() ;;
  aiter_pf)  SGL=$G/sglang/python; AIT=$G/aiter_fork; EXTRA=(--linear-attn-prefill-backend aiter) ;;
  aiter_pf_flyk5) SGL=$G/sglang/python; AIT=$G/aiter_fork; EXTRA=(--linear-attn-prefill-backend aiter); export SGLANG_GDN_AITER_K5=flydsl ;;
  aiter_all) SGL=$G/sglang/python; AIT=$G/aiter_fork; EXTRA=(--linear-attn-backend aiter) ;;
  # diagnostic (needs the hooks in e2e/gdn_aiter_with_diag_hooks.py.txt): aiter backend, triton extend
  aiter_pf_diag) SGL=$G/sglang/python; AIT=$G/aiter_fork; EXTRA=(--linear-attn-prefill-backend aiter); export GDN_DIAG_TRITON_EXTEND=1 ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac
export PYTHONPATH=${AIT:+$AIT:}$SGL
# The validated PyHIP wheel (pyhip.ops.gr_read / QSA) is not installed here; both are
# GDN-independent and disabled identically for every variant.
export SGLANG_GR_READ_FLYDSL=0 SGLANG_USE_PYHIP_QSA=0
export HIP_VISIBLE_DEVICES=$GPUS
export PORT TP_SIZE=2
export LOG_FILE=$OUT/server.log
export SGLANG_TORCH_PROFILER_DIR=$OUT/profile

bash $G/sglang/scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh "${EXTRA[@]}" > /dev/null 2>&1 &
SERVER_PID=$!
cleanup() { pkill -P $SERVER_PID 2>/dev/null; kill $SERVER_PID 2>/dev/null; sleep 5; pkill -9 -f "port $PORT" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 180); do
  curl -sf localhost:$PORT/health > /dev/null && break
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server died"; tail -50 $OUT/server.log; exit 1; fi
  sleep 10
done
curl -sf localhost:$PORT/health > /dev/null || { echo "server not healthy"; exit 1; }
python -c "import sglang, aiter; print('sglang', sglang.__file__); print('aiter', aiter.__file__)" > $OUT/versions.txt 2>/dev/null
echo "server up: $VARIANT on $GPUS:$PORT"

bench() {  # name in out conc nprompts
  python3 -m sglang.bench_serving --backend sglang --model /models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16 \
    --dataset-name random --host localhost --port $PORT --num-prompts $5 --random-input $2 --random-output $3 \
    --random-range-ratio 1.0 --warmup-requests 10 --max-concurrency $4 --seed $((1000 + ROUND)) \
    --output-file $OUT/bench_$1.jsonl > $OUT/bench_$1.log 2>&1
  grep -E "Mean TTFT|Mean TPOT|Output token throughput|Median TTFT|Median TPOT" $OUT/bench_$1.log | sed "s/^/  [$1] /"
}
for w in ${WORKLOADS:-A B C D}; do
  case $w in
    A) bench A 12000 350 1 32 ;;   # script default: long prompt, single user
    B) bench B 12000 1 1 16 ;;     # prefill only (TTFT)
    C) bench C 4000 350 16 64 ;;   # mixed, mid concurrency
    D) bench D 2000 500 32 128 ;;  # decode heavy at bs 32
    none) ;;
  esac
done

if [[ ${GSM8K:-0} == 1 ]]; then
  python3 $G/bench/gsm8k_eval.py --data $G/e2e/gsm8k_test.jsonl --num-questions 1314 \
    --parallel 64 --port $PORT --out $OUT/gsm8k_preds.jsonl > $OUT/gsm8k.log 2>&1
  grep -E "Accuracy|Invalid" $OUT/gsm8k.log | sed "s/^/  [gsm8k] /"
fi

if [[ ${PROFILE:-0} == 1 ]]; then
  for spec in "prefill 12000 5 4" "decode 2000 64 32"; do
    set -- $spec
    rm -rf $OUT/profile; mkdir -p $OUT/profile
    SGLANG_TORCH_PROFILER_DIR=$OUT/profile python3 -m sglang.bench_serving --backend sglang \
      --model /models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16 --dataset-name random --host localhost --port $PORT \
      --num-prompts $4 --random-input $2 --random-output $3 --random-range-ratio 1.0 --warmup-requests 4 \
      --max-concurrency $4 --profile > $OUT/profile_$1.log 2>&1
    sleep 20
    mv $OUT/profile $OUT/profile_$1
  done
fi
if [[ ${DIVERSITY:-0} == 1 ]]; then  # guards against numerically degenerate (repetitive) outputs
  python3 $G/bench/random_prompt_outputs.py $PORT $OUT/random_prompt_outputs.json > $OUT/diversity.log 2>&1
  sed "s/^/  [diversity] /" $OUT/diversity.log
fi

[[ -n ${HOLD_SECONDS:-} ]] && sleep $HOLD_SECONDS
echo "done: $VARIANT"
