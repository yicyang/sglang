"""Experimental Qwen4 TP2 sequence-parallel prefill MoE through one persistent "megamoe" kernel.

For long prefill forwards (T >= SGLANG_QWEN4_MEGAMOE_MIN_TOKENS) the MoE block's
gate -> topk -> fused-shared-expert append stays unchanged, and the experts + top-k sum +
bf16 SDMA reduce-scatter run inside one FlyDSL kernel (x quant, routing sort, gate-up,
down, top-k sum, and an in-kernel SDMA put of the peer's rows). The result is handed to
`_combine_reduce_scattered` as the usual PendingReduceScatter. Everything else, and every
forward that does not qualify, keeps the original path.

The kernel lives in an aiter worktree (not in the installed aiter). It is loaded from
SGLANG_QWEN4_MEGAMOE_KERNEL_DIR as an alias package next to the installed
`aiter.ops.flydsl.kernels`, so the installed fused_moe used by the original path is untouched.
"""

import functools
import importlib.util
import logging
import os
import sys
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_ALIAS = "aiter.ops.flydsl.kernels._megamoe_wt"
_ENABLED = os.environ.get("SGLANG_QWEN4_MEGAMOE", "0") == "1"
_MIN_TOKENS = int(os.environ.get("SGLANG_QWEN4_MEGAMOE_MIN_TOKENS", "32768"))
_KERNEL_DIR = os.environ.get(
    "SGLANG_QWEN4_MEGAMOE_KERNEL_DIR",
    "/root/workspace/aiter-moe-opt2/aiter/ops/flydsl/kernels/moe_gemm_2stage_gfx942",
)
_CTL_INTS = 65536
_HSTRIDE, _HSSTRIDE = 384, 32  # fp8 h row / per-route scale stride (whole cache lines)

_warned = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("Qwen4 megamoe: %s", msg)


def enabled() -> bool:
    return _ENABLED


@functools.lru_cache(maxsize=1)
def _kernel_module():
    """mega_prefill from the worktree, as an alias package beside the installed kernels."""
    import aiter.ops.flydsl.kernels  # noqa: F401  (parent package for relative imports)

    if _ALIAS not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _ALIAS, os.path.join(_KERNEL_DIR, "__init__.py"), submodule_search_locations=[_KERNEL_DIR]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[_ALIAS] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(_ALIAS + ".mega_prefill")


class _MegaMoeTP2:
    """One instance per process (layers run one after another and share the buffers)."""

    def __init__(self, *, hidden, inter, topk, experts, max_tokens, device, fp8_push, rec_bytes):
        mp = _kernel_module()
        from aiter.ops.flydsl.fused_moe_gfx942 import _activation_scalars

        self.H, self.I, self.topk, self.E, self.cap_t = hidden, inter, topk, experts, max_tokens
        self.launch = mp.build_mega_prefill_v3(
            model_dim=hidden, inter_dim=inter, topk=topk, experts=experts, push=True,
            fuse_xq=True, fuse_sort=True, peer_first=True, dyn=True, push_fp8=fp8_push,
        )
        self.fp8_push = fp8_push
        # fp8 RS records of the peer rows, written by the kernel and pushed from here
        self.recs = (torch.empty(-(-max_tokens // 2) * rec_bytes, dtype=torch.int8, device=device)
                     if fp8_push else None)
        self.sc = _activation_scalars("silu", 1.0, 1.0, None)
        T = max_tokens
        cap = T * topk + 2 * experts * 64
        fp8 = torch.float8_e4m3fnuz
        z = dict(device=device)
        self.sid = torch.empty(cap, dtype=torch.int32, **z)
        self.sw = torch.empty(cap, dtype=torch.float32, **z)
        self.seid = torch.empty(cap // 64, dtype=torch.int32, **z)
        self.nv = torch.zeros(2, dtype=torch.int32, **z)
        self.loc = torch.empty(T * topk, dtype=torch.int32, **z)
        self.xq = torch.empty(T, hidden, dtype=fp8, **z)
        self.xs = torch.empty(T, 1, dtype=torch.float32, **z)
        self.h = torch.empty(T * topk, _HSTRIDE, dtype=fp8, **z)
        self.hs = torch.empty(T * topk, _HSSTRIDE, dtype=torch.float32, **z)
        self.hbf = torch.empty(T, topk, inter, dtype=torch.bfloat16, **z)
        self.g2 = torch.empty(cap // 64 * 64, hidden, dtype=torch.bfloat16, **z)
        self.out = torch.empty(T, hidden, dtype=torch.bfloat16, **z)
        self.ctl = torch.zeros(_CTL_INTS, dtype=torch.int32, **z)
        self.calls = 0

    def run(self, x, topk_ids, topk_w, w13, s13, w2, s2, rs, plan):
        from aiter.ops.flydsl.fused_moe_gfx942 import _launch

        from sglang.srt.layers.sdma_reduce_scatter import PendingReduceScatter

        T, L = plan.num_tokens, plan.local_rows
        rank, peer = plan.rank, 1 - plan.rank
        base_own, base_peer = rank * L, peer * L
        hub = rs._mori
        slot = rs._slot
        rs._slot ^= 1
        dst = rs._peer_recv[peer] + (slot * (rs._world - 1) + 0) * rs._local_capacity * rs._row_bytes
        loc = self.loc[: T * self.topk]
        out = self.out[:T]
        p64 = torch.zeros(16, dtype=torch.int64, pin_memory=True)
        p64[3] = hub._dev_comm.ptr
        p64[4] = dst
        p64[5] = self.recs.data_ptr() if self.fp8_push else out[base_peer:].data_ptr()
        p64[8:16] = torch.tensor([
            loc[base_peer * self.topk:].data_ptr(), out[base_peer:].data_ptr(),
            loc[base_own * self.topk:].data_ptr(), out[base_own:].data_ptr(),
            x.data_ptr(), topk_ids.data_ptr(), topk_w.data_ptr(), loc.data_ptr(),
        ], dtype=torch.int64)
        par = p64.view(torch.int32)
        par[0], par[1], par[2], par[3], par[4] = 0, L, L, peer, 0
        par[12], par[13] = base_own, L
        par_dev = par.to(x.device, non_blocking=True)
        self.ctl.zero_()
        received_before = hub.received
        hub.received += 1
        _launch(self.launch, self.xq[:T], w13, self.sid, self.sw, self.seid, self.nv, s13,
                self.xs[:T], T, *self.sc, self.h, self.hs, w2, s2, self.g2, self.ctl, self.hbf,
                par_dev)
        self.calls += 1
        pend = PendingReduceScatter(
            carrier=out, local_chunks=[out[base_own:base_own + L]],
            recv_records=rs._recv[slot].view(1, -1).view(torch.int8), rec_bytes=rs._rec_bytes,
            fp8=rs._fp8, bounds=[0, L],
            wait_landed=functools.partial(rs._wait_landed, received_before=received_before),
        )
        carrier = pend.carrier
        carrier._sglang_sp_pending = pend
        return carrier


_runner: Optional[_MegaMoeTP2] = None


def try_run(mlp, hidden_states: torch.Tensor, plan, rs) -> Optional[torch.Tensor]:
    """The SP MoE output carrier through the megamoe kernel, or None to use the original path."""
    global _runner
    if not _ENABLED or plan is None:
        return None
    T = plan.num_tokens
    if T < _MIN_TOKENS:
        return None
    reasons = []
    if plan.world_size != 2:
        reasons.append("TP != 2")
    if rs is None:
        reasons.append("no SDMA reduce-scatter")
    if T % 2 or plan.local_rows * 2 != T:
        reasons.append("odd token count")
    if not mlp.enable_shared_expert_fusion:
        reasons.append("shared expert not fused")
    if hidden_states.dtype != torch.bfloat16 or hidden_states.shape[0] != T:
        reasons.append("input not [T, H] bf16")
    if reasons:
        _warn_once("skip:" + ",".join(reasons), f"T={T} uses the original path ({', '.join(reasons)})")
        return None
    experts = mlp.experts
    params = dict(experts.named_parameters(recurse=False))
    if "w13_weight_scale" not in params or "w2_weight_scale" not in params:
        _warn_once("scales", (
            f"no per-channel weight scales on {type(experts).__name__} "
            f"(quant_method={type(experts.quant_method).__name__}); params={sorted(params)}; "
            f"scale attrs={sorted(k for k in vars(experts) if 'scale' in k)}; original path"))
        return None
    w13, w2 = experts.w13_weight, experts.w2_weight
    E, I = w2.shape[0], w2.shape[-1]
    if _runner is None:
        cap = max(T, int(os.environ.get("SGLANG_QWEN4_MEGAMOE_MAX_TOKENS", "49152")))
        _runner = _MegaMoeTP2(hidden=hidden_states.shape[1], inter=I, topk=mlp._mega_top_k,
                              experts=E, max_tokens=cap, device=hidden_states.device,
                              fp8_push=rs._fp8, rec_bytes=rs._rec_bytes)
        logger.info("Qwen4 megamoe: enabled for T >= %d (H=%d, I=%d, E=%d, topk=%d, cap=%d)",
                    _MIN_TOKENS, hidden_states.shape[1], I, E, mlp._mega_top_k, cap)
    if _runner.fp8_push != rs._fp8:
        _warn_once("transport", "RS transport changed after the megamoe runner was built; original path")
        return None
    if T > _runner.cap_t:
        _warn_once("cap", f"T={T} > buffer capacity {_runner.cap_t}; original path")
        return None
    router_logits, _ = mlp.gate(hidden_states)
    topk_output = mlp.topk(hidden_states, router_logits)
    topk_output = mlp._append_shared_to_topk_output(topk_output, hidden_states)
    ids = topk_output.topk_ids.to(torch.int32).contiguous()
    tw = topk_output.topk_weights.to(torch.float32).contiguous()
    if ids.shape[1] != _runner.topk:
        _warn_once("topk", f"topk {ids.shape[1]} != {_runner.topk}; original path")
        return None
    out = _runner.run(hidden_states.contiguous(), ids, tw, w13, params["w13_weight_scale"], w2,
                      params["w2_weight_scale"], rs, plan)
    if _runner.calls == 1:
        logger.info("Qwen4 megamoe: first megamoe forward, T=%d", T)
    return out
