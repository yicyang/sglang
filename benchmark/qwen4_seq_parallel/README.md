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
(PLE's short conv mixes neighbouring tokens) and gathered back before the final mixer.

| Knob | Default | Effect |
|---|---|---|
| `SGLANG_QWEN4_SEQ_PARALLEL` | 0 | enable the sequence-parallel residual for prefill |
| `SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS` | 2048 | smaller batches keep the all-reduce |

Decode, CUDA graph capture, speculative modes, DP attention and EP all-to-all keep the original path.

## Scripts

| Script | What it measures |
|---|---|
| `bench_rs_ag.py` | aiter all-reduce vs reduce-scatter + all-gather, and their bit-identity |

## Results (8x MI308X, 12000 in / 350 out, `scripts/run_pure_text_bench.sh`)

12k-token prefill step: TP4 534 -> 436 ms, TP2 760 -> 685 ms. Serving req/s, two interleaved rounds:

| conc | TP4 base | TP4 SP | TP2 base | TP2 SP |
|---|---|---|---|---|
| 1 | 0.334 / 0.339 | 0.346 / 0.349 | 0.297 / 0.302 | 0.307 / 0.306 |
| 4 | 0.775 / 0.764 | 0.831 / 0.814 | 0.586 / 0.588 | 0.615 / 0.610 |
| 8 | 1.011 / 1.001 | 1.120 / 1.113 | 0.786 / 0.780 | 0.830 / 0.823 |
| 16 | 1.216 / 1.209 | 1.387 / 1.370 | 0.925 / 0.926 | 0.987 / 0.992 |
| 32 | 1.343 / 1.338 | 1.570 / 1.559 | 0.999 / 1.000 | 1.070 / 1.078 |

Greedy outputs of a long-context probe are bit-identical to the baseline; gsm8k and the probe
(48/48) are unchanged.
