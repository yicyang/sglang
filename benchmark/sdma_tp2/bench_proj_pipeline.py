"""TP2 projection + all-reduce: aiter custom AR vs the SDMA chunk pipeline.

Runs the o_proj shape of Qwen3.8-Flash-Next (12000 x 3072 -> 2560, FP8) on two
GPUs, times GEMM + aiter all-reduce against `SdmaTp2Pipeline` for several chunk
counts, and checks the pipelined result against the aiter one.

    HIP_VISIBLE_DEVICES=0,1 python benchmark/sdma_tp2/bench_proj_pipeline.py
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.layers.sdma_tp_pipeline import SdmaTp2Pipeline


def _bench(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    elapsed = torch.tensor([start.elapsed_time(end) / iters])
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def _worker(rank, args):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(args.port))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    custom_ar = CustomAllreduce(dist.group.WORLD, device)
    x = (
        torch.randn(
            args.tokens,
            args.k,
            device=device,
            generator=torch.Generator(device=device).manual_seed(rank),
        )
        * 0.1
    ).to(torch.float8_e4m3fnuz)
    w = (
        (
            torch.randn(
                args.n,
                args.k,
                device=device,
                generator=torch.Generator(device=device).manual_seed(7),
            )
            * 0.1
        )
        .to(torch.float8_e4m3fnuz)
        .t()
    )
    scale = torch.tensor(1.0, device=device)

    def project(rows):
        return torch._scaled_mm(
            rows, w, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16
        )

    def baseline():
        return custom_ar.all_reduce(project(x))

    pipeline = SdmaTp2Pipeline(
        capacity_tokens=args.tokens,
        hidden_size=args.n,
        dtype=torch.bfloat16,
        device=device,
        cpu_group=dist.group.WORLD,
    )
    reference = baseline().float()
    t_gemm = _bench(lambda: project(x), args.iters)
    t_base = _bench(baseline, args.iters)
    if rank == 0:
        print(f"tokens={args.tokens} K={args.k} N={args.n}")
        print(f"GEMM {t_gemm:.3f} ms | GEMM + aiter all-reduce {t_base:.3f} ms")
        print(
            f"{'chunks':>6} {'pipelined ms':>12} {'saved ms':>9} {'max|diff|/max|ref|':>19}"
        )
    for chunks in args.chunks:
        t_pipe = _bench(
            lambda: pipeline.run(project=project, x=x, chunks=chunks), args.iters
        )
        out = pipeline.run(project=project, x=x, chunks=chunks).float()
        torch.cuda.synchronize()
        err = ((out - reference).abs().max() / reference.abs().max()).item()
        if rank == 0:
            print(f"{chunks:>6} {t_pipe:12.3f} {t_base - t_pipe:9.3f} {err:19.2e}")
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=12000)
    parser.add_argument("--k", type=int, default=3072)
    parser.add_argument("--n", type=int, default=2560)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--port", type=int, default=29581)
    mp.spawn(_worker, args=(parser.parse_args(),), nprocs=2)


if __name__ == "__main__":
    main()
