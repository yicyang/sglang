# Sequence-parallel hyper-connection residual for Qwen4-Exp TP prefill

Qwen4-Exp (Qwen3.8-Flash-Next) keeps a 4-lane hyper-connection residual stream. Between the TP-sharded
blocks, three per-token ops run on it: `hc_combine` (inject the block output), `hc_norm`
(grouped Gemma RMSNorm, variance per token and per 2560-wide lane) and the low-rank `mix` that builds the
next block's input. With the usual layout every TP rank repeats them on all tokens; in a 12k-token
prefill step that is ~165 ms per rank (31% of the step at TP4, 22% at TP2).

With `SGLANG_QWEN4_SEQ_PARALLEL=1` each rank keeps only its 1/TP slice of the token rows of the residual
stream during prefill:

```
partial block output [T, 2560]
  -> reduce-scatter            (this rank's T/TP rows, summed)
  -> hc_combine, hc_norm, mix  (on T/TP rows)
  -> all-gather                (mixed input, [T, 2560])
  -> attention / GDN / MoE     (unchanged, on all T rows; MoE skips its own all-reduce)
```

The reduce-scatter plus all-gather (aiter custom kernels) move the same bytes as the all-reduce they
replace, and AG(RS(x)) is bit-identical to AR(x). The residual is sharded after the last PLE layer
(PLE's short conv mixes neighbouring tokens). At the end the final mixer runs on the local rows and only
its [T, 2560] output is gathered; the [T, 4 x 2560] residual (HC state) is gathered only when the forward
captures hidden states (speculative decoding, `return_hidden_states`).

| Knob | Default | Effect |
|---|---|---|
| `SGLANG_QWEN4_SEQ_PARALLEL` | 0 | enable the sequence-parallel residual for prefill |
| `SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS` | 2048 | smaller batches keep the all-reduce |

The path covers eager EXTEND / MIXED / TARGET_VERIFY forwards with at least
`SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS` tokens. Decode, CUDA-graph-captured batches (including captured
target verify), draft-model forwards, DP attention and EP all-to-all keep the original path.

## Scripts

| Script | What it measures |
|---|---|
| `bench_rs_ag.py` | aiter all-reduce vs reduce-scatter + all-gather, and their bit-identity |

## Results (8x MI308X, 12000 in / 350 out, `scripts/run_pure_text_bench.sh`)

12k-token prefill step (mean of 4): TP4 542.0 -> 434.9 ms, TP2 748.7 -> 691.8 ms. Serving req/s, two
interleaved rounds (TP4 and TP2 run one after the other):

| conc | TP4 base | TP4 SP | TP2 base | TP2 SP |
|---|---|---|---|---|
| 1 | 0.342 / 0.339 | 0.349 / 0.353 | 0.302 / 0.302 | 0.310 / 0.309 |
| 4 | 0.776 / 0.769 | 0.821 / 0.836 | 0.594 / 0.597 | 0.619 / 0.616 |
| 8 | 1.003 / 1.012 | 1.087 / 1.122 | 0.798 / 0.792 | 0.834 / 0.826 |
| 16 | 1.211 / 1.205 | 1.359 / 1.373 | 0.930 / 0.931 | 0.991 / 0.993 |
| 32 | 1.353 / 1.347 | 1.541 / 1.553 | 1.008 / 1.006 | 1.079 / 1.081 |

Greedy outputs of a long-context probe are bit-identical to the baseline; the long-context check is
48/48 and gsm8k stays within run-to-run noise.

## Reduce-scatter over SDMA (ROCm)

`SGLANG_QWEN4_SP_SDMA_RS=1` moves the sequence-parallel reduce-scatters to the SDMA copy engines
(`layers/sdma_reduce_scatter.py`). Each rank computes the projection (o_proj / GDN out_proj) in row
chunks, the blocks owned by other ranks first and its own block last, and pushes each finished chunk to
its owner with an SDMA copy plus a flag; the owner waits on the flags on its compute stream right before
it sums and combines each chunk. The MoE output is reduce-scattered the same way after the experts.
With `fp8`, pushed rows are float8 e4m3fnuz with one bf16 scale per 32 values (the local rows stay bf16).
The all-gather stays on aiter.

| Knob | Default | Effect |
|---|---|---|
| `SGLANG_QWEN4_SP_SDMA_RS` | 0 | reduce-scatter over SDMA (needs `SGLANG_QWEN4_SEQ_PARALLEL=1`) |
| `SGLANG_QWEN4_SP_SDMA_RS_TRANSPORT` | bf16 | `bf16` (exact rows) or `fp8` (lossy, ~half the bytes) |
| `SGLANG_QWEN4_SP_SDMA_RS_CHUNKS` | 2 | row chunks per owned block; measured best: 2 at TP2, 1 at TP4 |

12k-token prefill step on top of sequence parallelism (mean of 4): TP2 691.8 -> 656.5 (bf16) /
646.7 ms (fp8), TP4 434.9 -> 433.5 / 421.9 ms. Serving req/s vs sequence parallelism alone, two rounds
each: TP2 0.0~+4.1% (bf16) and +0.5~+4.7% (fp8), growing with concurrency; TP4 -2.9%~+1.6% (no gain,
and TTFT at c4/c8 is 5-7% worse), so it is only recommended at TP2.

| Script | What it measures |
|---|---|
| `bench_sdma_rs.py` | `SdmaReduceScatter` vs an fp32 reference (bf16 / fp8, uneven T), and GEMM + RS + combine vs aiter |
