# SDMA projection pipeline for TP2 (ROCm)

Opt-in prefill path for Qwen4-Exp (Qwen3.8-Flash-Next) at TP2 on MI300-class GPUs:
the attention-side output projection (`o_proj` / GDN `out_proj`) is computed in
row chunks, and each rank pulls the peer's chunks over the SDMA copy engine while
the next chunk's GEMM runs. The local add completes the all-reduce, so the
attention-side `cross_device_reduce` disappears from prefill.

```bash
SGLANG_QWEN4_SDMA_PROJ_PIPELINE=1 \
SGLANG_QWEN4_SDMA_PROJ_PIPELINE_CHUNKS=4 \
SGLANG_QWEN4_SDMA_PROJ_PIPELINE_MIN_TOKENS=4096 \
  bash scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
```

Only batches of at least `MIN_TOKENS` rows take the pipeline; decode, CUDA graph
capture and other TP sizes keep the original all-reduce.

## Scripts

| Script | What it measures |
|---|---|
| `bench_sdma_overlap.py` | Peer copy concurrent with GEMM. SDMA overlaps fully; with `HSA_ENABLE_SDMA=0` (blit kernels) it does not. |
| `bench_proj_pipeline.py` | FP8 o_proj GEMM + aiter all-reduce vs `SdmaTp2Pipeline`, plus a numerical check. |

## Results (8x MI308X, TP2)

`bench_sdma_overlap.py`, 29 MiB pull vs 6000x2560x6656 bf16 GEMMs:

| transport | pull | overlap with GEMM |
|---|---|---|
| SDMA | 0.743 ms (41 GB/s) | 1.0 |
| blit kernel (`HSA_ENABLE_SDMA=0`) | 0.766 ms | 0.31 |

`bench_proj_pipeline.py`, 12000x3072 -> 2560 FP8:

| | ms |
|---|---|
| GEMM + aiter all-reduce | 1.849 |
| pipeline, 4 chunks | 1.580 |
| pipeline, 8 chunks | 1.564 |

End to end (12000 in / 350 out, `bench_serving`, 3 interleaved rounds per arm):
+1.8% req/s at concurrency 16 and +1.7% at 32; gsm8k unchanged.
