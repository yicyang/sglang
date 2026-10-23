"""AITER-based GDN kernels for ROCm.

Prefill wraps ``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk``,
which consumes sglang's native layouts directly: token-major q/k/v
``[1, T, H, D]``, fp32 g/beta ``[1, T, HV]`` and the ``[slots, HV, V, K]`` state
pool. The prepare stage (cumsum + KKT + solve_tril + w/u) runs as one FlyDSL
kernel and the hidden-state recurrence (K5) runs as HIP or FlyDSL.

Packed decode uses ``aiter.ops.flydsl.flydsl_gdr_decode`` on strided views of
the packed ``mixed_qkv`` and the same state pool (no copies). Unpacked decode
and target verify stay on the inherited Triton kernels.
"""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

logger = logging.getLogger(__name__)


class AiterGDNKernel(TritonGDNKernel):
    """Prefill via aiter opt_vk, packed decode via aiter FlyDSL."""

    def __init__(self):
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode
        from aiter.ops.prefill_batch_metadata import (
            build_gated_delta_rule_prefill_metadata,
        )
        from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
            chunk_gated_delta_rule_opt_vk,
        )

        k5_backend = envs.SGLANG_GDN_AITER_K5.get()
        if k5_backend not in ("flydsl", "hip"):
            raise ValueError(
                f"SGLANG_GDN_AITER_K5 must be 'flydsl' or 'hip', got {k5_backend!r}"
            )
        self._opt_vk = chunk_gated_delta_rule_opt_vk
        self._gdr_decode = flydsl_gdr_decode
        self._build_metadata = build_gated_delta_rule_prefill_metadata
        # FlyDSL mfma16 K5 cannot index the state pool, so the touched slots are
        # gathered / scattered around the call; HIP K5 updates the pool in place.
        self._gather_state = k5_backend == "flydsl"
        self._use_prepare_flydsl = self._prepare_flydsl_is_stable()
        # The schedule is shared by every GDN layer of a forward pass; aiter binds
        # it to the exact query_start_loc tensor object it was built from.
        self._meta_key = None
        self._meta = None

    def _prepare_flydsl_is_stable(self) -> bool:
        """FlyDSL prepare before aiter 987203ba5 (ROCm/aiter#4952) diverges on
        correlated keys, which real GDN layers produce; random keys do not trigger it."""
        gen = torch.Generator(device="cuda").manual_seed(0)
        T, HG, H, D = 256, 1, 2, 128
        base = torch.randn(1, 1, HG, D, device="cuda", generator=gen)
        noise = torch.randn(1, T, HG, D, device="cuda", generator=gen)
        k = (0.9 * base + 0.1 * noise).bfloat16()
        q = torch.randn(1, T, HG, D, device="cuda", generator=gen).bfloat16()
        v = torch.randn(1, T, H, D, device="cuda", generator=gen).bfloat16()
        g = -0.1 * torch.rand(1, T, H, device="cuda", generator=gen)
        beta = torch.full((1, T, H), 0.99, device="cuda")
        cu = torch.tensor([0, T], device="cuda", dtype=torch.int32)
        outs = []
        for use_prepare_flydsl in (False, True):
            o, _ = self._opt_vk(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=torch.zeros(1, H, D, D, device="cuda"),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
                use_prepare_flydsl=use_prepare_flydsl,
                state_dtype=torch.float32,
                prefill_metadata=self._build_metadata([T], cu_seqlens=cu),
            )
            outs.append(o.float())
        err = ((outs[1] - outs[0]).abs().max() / outs[0].abs().max()).item()
        stable = err < 5e-2
        if not stable:
            logger.warning(
                "aiter FlyDSL GDN prepare disagrees with its Triton prepare "
                "(rel err %.2e); falling back to Triton prepare. Use an aiter with "
                "ROCm/aiter#4952 for the faster path.",
                err,
            )
        return stable

    def _prefill_metadata(self, query_start_loc, seq_lens_cpu):
        if seq_lens_cpu is None:
            seq_lens_cpu = query_start_loc.diff().tolist()
        seq_lens = tuple(seq_lens_cpu)
        key = self._meta_key
        if key is None or key[0] is not query_start_loc or key[1] != seq_lens:
            self._meta = self._build_metadata(
                list(seq_lens), cu_seqlens=query_start_loc
            )
            self._meta_key = (query_start_loc, seq_lens)
        return self._meta

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_cpu=None,
        **kwargs,
    ) -> tuple:
        if kwargs.get("num_state_checkpoints"):
            raise NotImplementedError(
                "aiter GDN prefill does not support state checkpoints"
            )
        metadata = self._prefill_metadata(query_start_loc, seq_lens_cpu)
        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
            use_prepare_flydsl=self._use_prepare_flydsl,
            state_dtype=ssm_states.dtype,
            prefill_metadata=metadata,
        )
        if self._gather_state:
            idx = cache_indices.long()
            o, final_state = self._opt_vk(
                initial_state=ssm_states.index_select(0, idx),
                use_chunk_flydsl=True,
                **common,
            )
            ssm_states.index_copy_(0, idx, final_state.to(ssm_states.dtype))
        else:
            o, _ = self._opt_vk(
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                inplace_final_state=True,
                use_chunk_hip=True,
                **common,
            )
        return o, None, None

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs.get("replayssm_d") is not None:
            return super().packed_decode(
                mixed_qkv,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                **kwargs,
            )
        batch = mixed_qkv.shape[0]
        head_k_dim = ssm_states.shape[-1]
        qk_dim = (mixed_qkv.shape[1] - num_v_heads * head_v_dim) // 2
        num_k_heads = qk_dim // head_k_dim
        assert scale == head_k_dim**-0.5, "flydsl_gdr_decode fixes scale to K^-0.5"
        q, k, v = torch.split(
            mixed_qkv, [qk_dim, qk_dim, num_v_heads * head_v_dim], dim=-1
        )
        out = mixed_qkv.new_empty(batch, 1, num_v_heads, head_v_dim)
        # need_shuffle_state=False: the kernel reads the [slots, HV, V, K] pool as is.
        self._gdr_decode(
            query=q.view(batch, 1, num_k_heads, head_k_dim),
            key=k.view(batch, 1, num_k_heads, head_k_dim),
            value=v.view(batch, 1, num_v_heads, head_v_dim),
            a=a.view(batch, 1, num_v_heads),
            b=b.view(batch, 1, num_v_heads),
            dt_bias=dt_bias,
            A_log=A_log,
            indices=cache_indices.to(torch.int32),
            state=ssm_states,
            out=out,
            use_qk_l2norm=True,
            need_shuffle_state=False,
        )
        return out.transpose(0, 1)
