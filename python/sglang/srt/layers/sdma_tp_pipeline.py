"""TP2 row-parallel projection + all-reduce, pipelined over row chunks on ROCm.

Each rank writes projection row chunks into its IPC-shared send buffer and
bumps a flag in the peer's memory; the peer's comm stream waits on that flag
(hipStreamWaitValue32) and pulls the chunk with an SDMA copy, so the transfer
runs on the copy engine while the next chunk's GEMM uses the CUs. The local add
finishes the all-reduce.

Flags carry monotonically increasing sequence numbers, so they are never
reset. Send buffers are double-buffered: a rank only reuses a slot after it
observed the peer's next-call flags, which the peer writes after its pulls of
the previous call completed.
"""

import ctypes
from typing import Callable, Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

_HIP_IPC_LAZY_PEER = 0x1
# hipMemcpyDeviceToDeviceNoCU: force the SDMA engine instead of a blit kernel.
_HIP_MEMCPY_D2D_NO_CU = 1024
_HIP_STREAM_WAIT_GEQ = 0x0
_MAX_CHUNKS = 64


class _HipIpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def _load_hip():
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_HipIpcHandle), ctypes.c_void_p]
    hip.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _HipIpcHandle,
        ctypes.c_uint,
    ]
    hip.hipMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    hip.hipMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    hip.hipStreamWaitValue32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint,
        ctypes.c_uint32,
    ]
    return hip


@triton.jit
def _signal_peer(flag_ptr, value):
    # System-scope release: the chunk written by the preceding kernel is visible
    # to the peer's SDMA read once the peer observes the flag.
    tl.atomic_xchg(flag_ptr, value, sem="release", scope="sys")


class _RawPtr:
    """Pointer argument for Triton when there is no torch tensor to pass."""

    def __init__(self, ptr: int, dtype: torch.dtype):
        self._ptr = ptr
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self._ptr


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with hipError {rc}")


class SdmaTp2Pipeline:
    def __init__(
        self,
        *,
        capacity_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cpu_group: dist.ProcessGroup,
    ):
        self._hip = _load_hip()
        self._capacity = capacity_tokens
        self._row_bytes = hidden_size * torch.empty(0, dtype=dtype).element_size()
        self._send = torch.empty(
            2, capacity_tokens, hidden_size, dtype=dtype, device=device
        )
        self._recv = torch.empty(
            capacity_tokens, hidden_size, dtype=dtype, device=device
        )
        self._flags = torch.zeros(_MAX_CHUNKS, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)

        rank = dist.get_rank(cpu_group)
        exported = [None, None]
        dist.all_gather_object(
            exported,
            (self._export(self._send), self._export(self._flags)),
            group=cpu_group,
        )
        peer_send, peer_flags = exported[1 - rank]
        self._peer_send = self._import(peer_send)
        self._peer_flags = self._import(peer_flags)
        self._comm = torch.cuda.Stream(device)
        self._seq = 0
        self._slot = 0

    def fits(self, num_tokens: int) -> bool:
        return num_tokens <= self._capacity

    def run(
        self,
        *,
        project: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        chunks: int,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        comp = torch.cuda.current_stream()
        self._comm.wait_stream(comp)
        send = self._send[self._slot]
        peer_send = self._peer_send + self._slot * self._capacity * self._row_bytes
        bounds = [num_tokens * k // chunks for k in range(chunks + 1)]
        for k in range(chunks):
            r0, r1 = bounds[k], bounds[k + 1]
            send[r0:r1].copy_(project(x[r0:r1]))
            self._seq += 1
            _signal_peer[(1,)](
                _RawPtr(self._peer_flags + 4 * k, torch.int32), self._seq
            )
            self._pull(
                dst=self._recv[r0:r1],
                src=peer_send + r0 * self._row_bytes,
                nbytes=(r1 - r0) * self._row_bytes,
                flag=self._flags[k],
            )
        comp.wait_stream(self._comm)
        self._slot ^= 1
        return torch.add(send[:num_tokens], self._recv[:num_tokens])

    def _pull(
        self, *, dst: torch.Tensor, src: int, nbytes: int, flag: torch.Tensor
    ) -> None:
        stream = self._comm.cuda_stream
        _check(
            self._hip.hipStreamWaitValue32(
                stream, flag.data_ptr(), self._seq, _HIP_STREAM_WAIT_GEQ, 0xFFFFFFFF
            ),
            "hipStreamWaitValue32",
        )
        _check(
            self._hip.hipMemcpyAsync(
                dst.data_ptr(), src, nbytes, _HIP_MEMCPY_D2D_NO_CU, stream
            ),
            "hipMemcpyAsync",
        )

    def _export(self, t: torch.Tensor):
        # The caching allocator hands out sub-ranges; IPC handles cover whole allocations.
        base, size = ctypes.c_void_p(), ctypes.c_size_t()
        _check(
            self._hip.hipMemGetAddressRange(
                ctypes.byref(base), ctypes.byref(size), t.data_ptr()
            ),
            "hipMemGetAddressRange",
        )
        handle = _HipIpcHandle()
        _check(
            self._hip.hipIpcGetMemHandle(ctypes.byref(handle), base),
            "hipIpcGetMemHandle",
        )
        return ctypes.string_at(ctypes.addressof(handle), 64), t.data_ptr() - base.value

    def _import(self, exported) -> int:
        raw, offset = exported
        handle = _HipIpcHandle()
        ctypes.memmove(ctypes.addressof(handle), raw, 64)
        ptr = ctypes.c_void_p()
        _check(
            self._hip.hipIpcOpenMemHandle(
                ctypes.byref(ptr), handle, _HIP_IPC_LAZY_PEER
            ),
            "hipIpcOpenMemHandle",
        )
        return ptr.value + offset


_PIPELINE: Optional[SdmaTp2Pipeline] = None


def get_sdma_tp2_pipeline(
    *,
    capacity_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    cpu_group: dist.ProcessGroup,
) -> SdmaTp2Pipeline:
    """Process-wide pipeline; the first call is collective over the TP group."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SdmaTp2Pipeline(
            capacity_tokens=capacity_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=torch.device("cuda", torch.cuda.current_device()),
            cpu_group=cpu_group,
        )
    return _PIPELINE
