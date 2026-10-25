"""File-backed Engram tables for 2× Spark UMA.

vLLM's stock ParallelEngramEmbedding pins this rank's hash heads
(~47 GiB × 2 layers) with cudaHostAlloc. On GB10 that is the same pool as
GPU memory, so layer 14's torch.empty dies after layer 1 succeeds.

This replacement keeps dummy 0-size parameters (so load_weights does not
expect a 95 GiB copy), opens the native safetensors via row_store, and
gathers rows through cudaLaunchHostFunc so CUDA graphs still work.
"""
from __future__ import annotations

import ctypes as C
import glob
import json
import logging
import os
from pathlib import Path
import struct
import threading

import torch
from torch import nn

P, U = C.c_void_p, C.c_uint64
_LIB_CANDIDATES = (
    Path("/opt/dsv41/librow_store.so"),
    Path(__file__).resolve().parent / "librow_store.so",
)

logger = logging.getLogger("dsv41.engram_file")
_INSTALLED = False
_STORES: list[tuple[object, int]] = []
_STATS_STARTED = False


class Work(C.Structure):
    _fields_ = [
        ("store", P),
        ("ids", P),
        ("weights", P),
        ("scales", P),
        ("count", U),
    ]


def _lib_path() -> Path:
    for path in _LIB_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "librow_store.so missing (rebuild dsv41-flash-exl3:local)"
    )


def _load_lib():
    lib = C.CDLL(str(_lib_path()))
    lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
    lib.row_store_open.restype = P
    lib.row_store_range.argtypes = [P, U, U]
    lib.row_store_stats.argtypes = [P, C.POINTER(U)]
    lib.row_store_attach_packed.argtypes = [P, C.c_char_p, U]
    lib.row_store_attach_packed.restype = C.c_int
    lib.row_store_lookup.argtypes = [P]
    return lib


def _load_cudart():
    paths = []
    paths += glob.glob("/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so*")
    paths += glob.glob("/usr/local/cuda*/targets/x86_64-linux/lib/libcudart.so*")
    paths += glob.glob("/usr/local/cuda/lib64/libcudart.so*")
    paths += glob.glob(
        "/usr/local/lib/python*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so*"
    )
    seen, errors = set(), []
    for path in paths:
        if path in seen or path.endswith(".a") or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            lib = C.CDLL(path)
            lib.cudaLaunchHostFunc.argtypes = [P, P, P]
            lib.cudaLaunchHostFunc.restype = C.c_int
            return lib
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    raise RuntimeError(
        "libcudart.so not found for Engram host callbacks. Tried: "
        + "; ".join(errors or paths or ["<none>"])
    )


_lib = None
_cuda = None


def _ensure_native():
    global _lib, _cuda
    if _lib is None:
        _lib = _load_lib()
    if _cuda is None:
        _cuda = _load_cudart()
    return _lib, _cuda


def _stats(store):
    lib, _ = _ensure_native()
    out = (U * 9)()
    lib.row_store_stats(store, out)
    hits, misses, reads, cache_bytes, slots, scale_bytes, ways, threads, packed = out
    return dict(
        hits=hits,
        misses=misses,
        reads=reads,
        cache_bytes=cache_bytes,
        slots=slots,
        scale_bytes=scale_bytes,
        ways=ways,
        threads=threads,
        packed=bool(packed),
    )


def _report_loop():
    period = float(os.getenv("DSV41_STATS_SECONDS", "60"))
    if period <= 0:
        return
    previous: dict[int, dict] = {}
    while True:
        threading.Event().wait(period)
        for store, layer_id in list(_STORES):
            now = _stats(store)
            was = previous.get(layer_id)
            previous[layer_id] = now
            if was is None:
                continue
            hits = now["hits"] - was["hits"]
            misses = now["misses"] - was["misses"]
            total = hits + misses
            if not total:
                continue
            logger.info(
                "Engram layer=%s lookups=%s hit_rate=%.1f%% reads=%s "
                "cache=%.1fGiB packed=%s",
                layer_id,
                total,
                100.0 * hits / total,
                now["reads"] - was["reads"],
                now["cache_bytes"] / 2**30,
                now["packed"],
            )


def _register_store(store, layer_id: int) -> None:
    global _STATS_STARTED
    _STORES.append((store, layer_id))
    if not _STATS_STARTED:
        _STATS_STARTED = True
        threading.Thread(target=_report_loop, daemon=True, name="engram-stats").start()


def _table_dir() -> Path:
    env = os.environ.get("DSV41_SOURCE") or os.environ.get("ENGRAM_MOUNT") or ""
    if env:
        return Path(env)
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config().model_config.hf_config
        path = getattr(cfg, "engram_table_dir", None) or ""
        if path:
            return Path(path)
    except Exception:
        pass
    return Path("/engram-src")


def _layer_id_from_embeddings(num_embeddings: int, table_dir: Path) -> int:
    cfg = json.loads((table_dir / "config.json").read_text())
    text = cfg.get("text_config") or cfg
    ids = [int(x) for x in text["engram_layer_ids"]]
    rows = [int(x) for x in text["engram_num_embeddings"]]
    for layer_id, n in zip(ids, rows, strict=True):
        if n == int(num_embeddings):
            return layer_id
    raise RuntimeError(
        f"no Engram layer with {num_embeddings} rows in {table_dir}/config.json"
    )


def _tensor_span(root: Path, layer_id: int):
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"layers.{layer_id}.engram.embed."
    name = index[prefix + "weight"]
    if index[prefix + "scale"] != name:
        raise RuntimeError("Engram weight and scale must share a shard")
    path = root / name
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    weight, scale = header[prefix + "weight"], header[prefix + "scale"]
    base = 8 + length
    return (
        path,
        int(weight["shape"][0]),
        base + int(weight["data_offsets"][0]),
        base + int(scale["data_offsets"][0]),
    )


def _skip_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    del param, loaded_weight


def _make_init(module):
    def init(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
    ) -> None:
        del cpu_offload
        nn.Module.__init__(self)
        from vllm.config import get_current_vllm_config
        from vllm.distributed import (
            get_engram_dp_size,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.utils import set_weight_attrs

        lib, _ = _ensure_native()
        tp_size = get_tensor_model_parallel_world_size()
        dp_size = get_engram_dp_size()
        tp_rank = get_tensor_model_parallel_rank()
        if not head_sizes or any(size <= 0 for size in head_sizes):
            raise ValueError("Engram head_sizes must be positive")
        self.num_embeddings = int(num_embeddings)
        self.dim = int(dim)
        self.block_size = int(block_size)
        self.n_hash_cols = len(head_sizes)
        num_shards = tp_size * dp_size
        self.part_n_hash_cols = (self.n_hash_cols + num_shards - 1) // num_shards
        if (num_shards - 1) * self.part_n_hash_cols >= self.n_hash_cols:
            raise RuntimeError(
                f"Engram sharding leaves ranks without hash heads: "
                f"{self.n_hash_cols} heads over TP={tp_size} x DP={dp_size}"
            )
        self.head_start = module.engram_head_shard_rank() * self.part_n_hash_cols
        head_end = self.head_start + self.part_n_hash_cols
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.cpu_offload = True
        self._views = None
        self._view_src = None
        self._num_sms = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        ).multi_processor_count

        # Dummy params so named_parameters() has the mapped names, but load
        # never materializes 47 GiB. The file backend is the source of truth.
        self.weight = nn.Parameter(
            torch.empty(0, dim, dtype=torch.float8_e4m3fn, device="cpu"),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(0, dim // block_size, dtype=torch.uint8, device="cpu"),
            requires_grad=False,
        )
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {
                    "weight_loader": _skip_loader,
                    "engram_vocab_start": self.vocab_start_idx,
                },
            )

        table_dir = _table_dir()
        layer_id = _layer_id_from_embeddings(self.num_embeddings, table_dir)
        path, rows, woff, soff = _tensor_span(table_dir, layer_id)
        ranks_per_host = max(
            1, tp_size // max(1, int(os.environ.get("NNODES", "1")))
        )
        budget = int(float(os.getenv("DSV41_CACHE_GIB", "8")) * 2**30) // (
            2 * ranks_per_host
        )
        store = lib.row_store_open(
            str(path).encode(), rows, woff, soff, budget
        )
        if not store:
            raise RuntimeError(f"Could not open Engram backing shard: {path}")
        lib.row_store_range(store, self.vocab_start_idx, self.vocab_end_idx)
        packed_dir = os.environ.get("DSV41_PACKED_DIR", "")
        packed = False
        if packed_dir:
            shard = (
                Path(packed_dir)
                / f"engram-l{layer_id}-r{tp_rank}of{tp_size}.bin"
            )
            if shard.is_file():
                packed = bool(
                    lib.row_store_attach_packed(
                        store, str(shard).encode(), layer_id
                    )
                )
        self._store = store
        self._layer_id = layer_id
        self._works = {}
        _register_store(store, layer_id)
        stats = _stats(store)
        vllm_config = get_current_vllm_config()
        max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        self._max_rows = max(1, max_tokens * self.dp_size * self.part_n_hash_cols)
        cap = 1 << (self._max_rows - 1).bit_length()
        device = torch.device(
            f"cuda:{torch.accelerator.current_device_index()}"
        )
        # Model init runs under torch.device("cuda"); pin_memory needs CPU.
        self._ids = torch.empty(
            cap, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._host_w = torch.empty(
            (cap, self.dim),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self._host_s = torch.empty(
            (cap, self.dim // self.block_size),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self._dev_w = torch.empty(
            (cap, self.dim), dtype=torch.uint8, device=device
        )
        self._dev_s = torch.empty(
            (cap, self.dim // self.block_size),
            dtype=torch.uint8,
            device=device,
        )
        logger.info(
            "File-backed Engram layer=%s rank=%s/%s rows=[%s,%s) "
            "cache=%.1fGiB scales=%.1fGiB packed=%s (not pinned 47GiB UVA)",
            layer_id,
            tp_rank,
            tp_size,
            self.vocab_start_idx,
            self.vocab_end_idx,
            stats["cache_bytes"] / 2**30,
            stats["scale_bytes"] / 2**30,
            packed or stats["packed"],
        )
        print(
            f"File-backed Engram layer={layer_id} rank={tp_rank}/{tp_size} "
            f"rows=[{self.vocab_start_idx},{self.vocab_end_idx}) "
            f"cache={stats['cache_bytes']/2**30:.1f}GiB "
            f"packed={packed or stats['packed']}",
            flush=True,
        )

    return init


def _dequant_kernel():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _dequant_rows_kernel(
        weight,
        scales,
        out,
        num_rows,
        DIM: tl.constexpr,
        QUANT_BLOCK: tl.constexpr,
        BLOCK_R: tl.constexpr,
        GRID: tl.constexpr,
    ):
        cols = tl.arange(0, DIM)
        scale_cols = cols // QUANT_BLOCK
        for base in tl.range(
            tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R
        ):
            rows = base + tl.arange(0, BLOCK_R)
            valid = rows < num_rows
            values = tl.load(
                weight + rows[:, None] * DIM + cols[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            scale = tl.load(
                scales
                + rows[:, None] * (DIM // QUANT_BLOCK)
                + scale_cols[None, :],
                mask=valid[:, None],
                other=0,
            )
            scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            tl.store(
                out + rows[:, None] * DIM + cols[None, :],
                (values.to(tl.float32) * scale).to(tl.bfloat16),
                mask=valid[:, None],
            )

    return _dequant_rows_kernel


_DEQUANT = None


def _make_lookup(module):
    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        # `module` is a closure variable: `del module` here would make it a
        # local and raise UnboundLocalError on the first lookup.
        del background
        global _DEQUANT
        lib, cuda = _ensure_native()
        tokens = int(indices.shape[0])
        local_heads = int(self.part_n_hash_cols)
        rows = tokens * local_heads
        if not rows:
            return
        head_end = min(self.head_start + local_heads, indices.shape[1])
        local_ids = indices[:, self.head_start : head_end]
        if local_ids.shape[1] < local_heads:
            pad = local_ids.new_full(
                (tokens, local_heads - local_ids.shape[1]), -1
            )
            local_ids = torch.cat((local_ids, pad), dim=1)
        flat = local_ids.reshape(-1)

        if rows > int(self._ids.numel()):
            raise RuntimeError(
                f"Engram lookup rows={rows} exceeds staging {int(self._ids.numel())}"
            )
        work_key = rows
        if work_key not in self._works:
            self._works[work_key] = Work(
                self._store,
                self._ids.data_ptr(),
                self._host_w.data_ptr(),
                self._host_s.data_ptr(),
                rows,
            )
        work = self._works[work_key]
        self._ids[:rows].copy_(flat, non_blocking=True)
        error = cuda.cudaLaunchHostFunc(
            torch.cuda.current_stream().cuda_stream,
            C.cast(lib.row_store_lookup, P),
            C.addressof(work),
        )
        if error:
            raise RuntimeError(f"CUDA Engram host callback failed: {error}")
        self._dev_w[:rows].copy_(self._host_w[:rows], non_blocking=True)
        self._dev_s[:rows].copy_(self._host_s[:rows], non_blocking=True)
        if _DEQUANT is None:
            _DEQUANT = _dequant_kernel()
        from vllm.triton_utils import triton

        packed_out = out.reshape(rows, self.dim)
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, max(1, self._num_sms))
        # _dev_w holds the raw fp8 e4m3 row bytes (uint8 staging). Hand the
        # kernel an fp8 VIEW so its .to(tl.float32) decodes the fp8 encoding;
        # loading the uint8 buffer directly converts the byte values 0..255
        # as integers (every embedding ~30x too large and all positive), which
        # is what made every boot up to 2026-09-12 emit garbage tokens.
        _DEQUANT[(grid,)](
            self._dev_w.view(torch.float8_e4m3fn),
            self._dev_s,
            packed_out,
            rows,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
        )

    return lookup


def _make_storage():
    def _storage(self):
        raise RuntimeError(
            "file-backed Engram has no UVA table; lookup() gathers rows from NVMe/NFS"
        )

    return _storage


def install(module=None) -> None:
    """Replace ParallelEngramEmbedding init/lookup. Safe to call more than once."""
    global _INSTALLED
    if module is None:
        from vllm.models.deepseek_v4_1.common import engram as module
    if getattr(module, "_dsv41_engram_file", False):
        _INSTALLED = True
        return
    module.ParallelEngramEmbedding.__init__ = _make_init(module)
    module.ParallelEngramEmbedding.lookup = _make_lookup(module)
    module.ParallelEngramEmbedding._storage = _make_storage()
    module._dsv41_engram_file = True
    _INSTALLED = True
    logger.info("installed file-backed Engram (row_store, no 47GiB pin)")


if __name__ == "__main__":
    install()
    print("engram_file_backend: installed")
