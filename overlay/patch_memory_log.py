#!/usr/bin/env python3
"""Log unified-memory usage around each vLLM worker boot phase (GB10 UMA).

The 2026-09-11 boot went silent after "Loading safetensors checkpoint shards
39/39" and the host wedged two hours later. Nothing in vLLM's default logging
says where the memory went between "weights loaded" and "KV cache allocated".
This footer wraps the V1 Worker's phase methods and prints one line before and
after each:

  [dsv41-mem] after load_model: MemAvailable=18.9GiB cuda_free=... torch_alloc=... torch_reserved=... rss=...

MemAvailable is the number that matters on a Spark (cudaMalloc commits host
memory immediately). Grep `dsv41-mem` in logs/head.log / worker.log.
"""
from __future__ import annotations

from pathlib import Path
import sys

MARK = "dsv41-memory-log"

FOOTER = '''
# [dsv41-memory-log]
def _dsv41_install_memory_log() -> None:
    import functools
    import os
    import time

    def _gib(n):
        return f"{n / 2**30:.2f}GiB"

    def _snapshot():
        parts = []
        try:
            import psutil

            vm = psutil.virtual_memory()
            parts.append(f"MemAvailable={_gib(vm.available)}")
            parts.append(f"MemFree={_gib(vm.free)}")
            parts.append(f"rss={_gib(psutil.Process(os.getpid()).memory_info().rss)}")
        except Exception as exc:  # pragma: no cover
            parts.append(f"psutil_err={exc!r}")
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                parts.append(f"cuda_free={_gib(free)}/{_gib(total)}")
                parts.append(f"torch_alloc={_gib(torch.cuda.memory_allocated())}")
                parts.append(f"torch_reserved={_gib(torch.cuda.memory_reserved())}")
                parts.append(f"torch_peak={_gib(torch.cuda.max_memory_allocated())}")
        except Exception as exc:  # pragma: no cover
            parts.append(f"torch_err={exc!r}")
        return " ".join(parts)

    def _say(msg):
        line = f"[dsv41-mem] {msg}"
        print(line, flush=True)
        try:
            logger.info(line)
        except Exception:
            pass

    def _mem_free_gib():
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemFree:"):
                    return int(line.split()[1]) / 2**20
        except Exception:
            pass
        return -1.0

    def _drop_page_cache(self):
        """The mmap load leaves ~99 GiB of shard pages in the page cache. On GB10
        the GPU driver allocates from MemFree and does not reclaim page cache
        (NVRM NV_ERR_NO_MEMORY with MemFree ~1 GiB and MemAvailable ~9 GiB,
        2026-09-12 boot 11), so give the clean pages back right after the load."""
        if os.environ.get("DSV41_DROP_PAGE_CACHE", "1") == "0":
            return
        import glob

        paths = []
        try:
            paths.append(str(self.vllm_config.model_config.model))
        except Exception:
            pass
        before = _mem_free_gib()
        n = 0
        for d in paths:
            for f in glob.glob(os.path.join(d, "*.safetensors")):
                try:
                    fd = os.open(f, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                    n += 1
                except OSError as exc:
                    _say(f"fadvise {f}: {exc!r}")
        _say(f"dropped page cache of {n} shard files: MemFree {before:.2f}GiB -> {_mem_free_gib():.2f}GiB")

    def _pretune_exl3(self):
        """Autotune every EXL3 GEMM shape before CUDA-graph capture (see
        pretune_exl3_shapes in the EXL3 overlay). DSV41_EXL3_PRETUNE=0 skips."""
        if os.environ.get("DSV41_EXL3_PRETUNE", "1") == "0":
            return
        from vllm.model_executor.layers.quantization import exl3 as _exl3

        fn = getattr(_exl3, "pretune_exl3_shapes", None)
        if fn is None:
            _say("EXL3 pretune: overlay has no pretune_exl3_shapes; skipping")
            return
        runner = getattr(self, "model_runner", None)
        mods = [getattr(runner, "model", None)]
        spec = getattr(runner, "speculator", None)
        mods.append(getattr(spec, "model", None))
        sizes = tuple(
            int(s) for s in os.environ.get("DSV41_EXL3_PRETUNE_SIZES", "1,2,4,8,16").split(",") if s
        )
        t0 = time.perf_counter()
        shapes, launches = fn(mods, sizes=sizes)
        _say(
            f"EXL3 pretune: {shapes} distinct shapes x {len(sizes)} row buckets "
            f"= {launches} launches in {time.perf_counter() - t0:.1f}s"
        )

    def _wrap(name):
        orig = getattr(Worker, name, None)
        if orig is None or getattr(orig, "_dsv41_mem_wrapped", False):
            return

        @functools.wraps(orig)
        def wrapped(self, *args, **kwargs):
            if name == "compile_or_warm_up_model":
                try:
                    _drop_page_cache(self)
                except Exception as exc:
                    _say(f"drop page cache failed: {exc!r}")
                try:
                    _pretune_exl3(self)
                except Exception as exc:  # never break the boot
                    _say(f"EXL3 pretune failed: {exc!r}")
            _say(f"before {name}: {_snapshot()}")
            t0 = time.perf_counter()
            try:
                return orig(self, *args, **kwargs)
            finally:
                _say(f"after {name} ({time.perf_counter() - t0:.0f}s): {_snapshot()}")

        wrapped._dsv41_mem_wrapped = True
        setattr(Worker, name, wrapped)

    def _install_prefill_empty_cache():
        """Give a long prefill's transient memory back after every chunk (the
        native DS4.1 recipe's prefill_empty_cache hook, for the V1 worker).

        Every prefill chunk of a DeepSeek-V4.1 sequence scores its queries
        against the whole prefix, so the per-chunk indexer/attention transients
        grow with the prefix and PyTorch's caching allocator cannot reuse the
        previous chunk's slightly smaller blocks: reserved memory grows with the
        square of the prompt. On a Spark that is host memory (a 100k prefill
        took the head from 4.9 GiB free to the 1.5 GiB guard, 2026-09-12).
        After any step whose longest sequence is at least
        DSV41_PREFILL_EMPTY_CACHE_TOKENS (default 8192) and which is a prefill
        step (>64 scheduled tokens for some request), release the unused cached
        blocks so the peak is one chunk's live set. 0 disables.

        Releasing after every chunk has a price on a Spark: the next chunk must
        cudaMalloc (commit and zero host pages for) its whole transient set
        again, which cost ~20 % of prefill throughput at 50k tokens. So the
        release is adaptive: it happens only when the node's MemAvailable after
        the step is below DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB (default 2.5;
        0 = release after every qualifying step, the 2026-09-13 behaviour).
        While there is headroom the allocator keeps its blocks and reuses them
        for the next chunk; as the prefix (and the per-chunk transient set)
        grows and headroom shrinks, the hook falls back to per-chunk release,
        so the memguard floor is protected exactly as before."""
        threshold = int(os.environ.get("DSV41_PREFILL_EMPTY_CACHE_TOKENS", "8192") or 0)
        memavail_gib = float(os.environ.get("DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB", "2.5") or 0)
        # Release once when a long prefill hands over to decode (default on): the
        # transient set a long prefill leaves mapped (up to ~2 GiB) squeezes the
        # OS page cache that the file-backed Engram rows are read through, and
        # decode right after a 100k-180k prefill ran at 11-13 tok/s instead of
        # 22 on the boots where the allocator had just been emptied (2026-09-13).
        end_release = os.environ.get("DSV41_PREFILL_END_EMPTY_CACHE", "0") not in ("0", "", "false")
        orig = getattr(Worker, "execute_model", None)
        if threshold <= 0 or orig is None or getattr(orig, "_dsv41_empty_cache", False):
            return
        state = {"calls": 0, "skipped": 0, "in_prefill": False, "end_calls": 0}

        # DSV41_MEM_SOURCE=cuda (discrete GPUs, e.g. RTX PRO 6000): the headroom
        # that matters is free VRAM, not host MemAvailable. meminfo = GB10 UMA.
        mem_source = os.environ.get("DSV41_MEM_SOURCE", "meminfo").strip().lower()

        def _memavail_gib():
            if mem_source == "cuda":
                try:
                    import torch

                    return torch.cuda.mem_get_info()[0] / (1024 * 1024 * 1024)
                except Exception:
                    return 0.0
            try:
                with open("/proc/meminfo") as fh:
                    for line in fh:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1]) / (1024 * 1024)
            except OSError:
                pass
            return 0.0

        @functools.wraps(orig)
        def execute_model(self, scheduler_output, *args, **kwargs):
            out = orig(self, scheduler_output, *args, **kwargs)
            try:
                nst = getattr(scheduler_output, "num_scheduled_tokens", None)
                if nst and max(nst.values()) <= 64 and state["in_prefill"]:
                    # First decode-shaped step after a long prefill: give the
                    # prefill's transient set back before the decode loop runs.
                    state["in_prefill"] = False
                    if end_release:
                        import torch

                        torch.cuda.empty_cache()
                        state["end_calls"] += 1
                        if state["end_calls"] in (1, 2, 10):
                            _say(
                                f"prefill-end empty_cache #{state['end_calls']} "
                                f"(MemAvailable {_memavail_gib():.2f} GiB after)"
                            )
                if nst and max(nst.values()) > 64:
                    longest = 0
                    for req in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
                        longest = max(
                            longest,
                            int(getattr(req, "num_computed_tokens", 0) or 0)
                            + int(nst.get(getattr(req, "req_id", ""), 0)),
                        )
                    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
                    if cached is not None:
                        for rid, nc in zip(cached.req_ids, cached.num_computed_tokens):
                            longest = max(longest, int(nc) + int(nst.get(rid, 0)))
                    if longest >= threshold:
                        import torch

                        state["in_prefill"] = True
                        avail = _memavail_gib() if memavail_gib > 0 else 0.0
                        if memavail_gib > 0 and avail >= memavail_gib:
                            state["skipped"] += 1
                            if state["skipped"] in (1, 100, 1000):
                                _say(
                                    f"prefill empty_cache skipped #{state['skipped']} "
                                    f"(longest seq {longest}, MemAvailable {avail:.2f} GiB >= {memavail_gib} GiB)"
                                )
                        else:
                            torch.cuda.empty_cache()
                            state["calls"] += 1
                            if state["calls"] in (1, 10, 100):
                                _say(
                                    f"prefill empty_cache #{state['calls']} (longest seq {longest}, "
                                    f"MemAvailable {avail:.2f} GiB): {_snapshot()}"
                                )
            except Exception as exc:  # never break a step
                if state["calls"] == 0:
                    _say(f"prefill empty_cache hook error: {exc!r}")
                    state["calls"] = -1
            return out

        execute_model._dsv41_empty_cache = True
        Worker.execute_model = execute_model
        _say(
            f"prefill empty_cache hook armed (threshold {threshold} tokens, "
            f"release when {'cuda free' if mem_source == 'cuda' else 'MemAvailable'} < {memavail_gib} GiB{' [always]' if memavail_gib <= 0 else ''}, "
            f"prefill-end release {'on' if end_release else 'off'})"
        )

    _install_prefill_empty_cache()

    for _name in (
        "init_device",
        "load_model",
        "determine_available_memory",
        "initialize_from_config",
        "compile_or_warm_up_model",
    ):
        _wrap(_name)

    # Ticks: one line every DSV41_MEMORY_LOG_TICK seconds while load_model runs,
    # so a drain inside the load phase is visible with its rate.
    import threading

    tick = float(os.environ.get("DSV41_MEMORY_LOG_TICK", "10"))
    orig_load = Worker.load_model

    @functools.wraps(orig_load)
    def load_with_ticks(self, *args, **kwargs):
        stop = threading.Event()

        def loop():
            t0 = time.perf_counter()
            while not stop.wait(tick):
                _say(f"tick load_model +{time.perf_counter() - t0:.0f}s: {_snapshot()}")

        th = threading.Thread(target=loop, daemon=True, name="dsv41-memlog")
        if tick > 0:
            th.start()
        try:
            return orig_load(self, *args, **kwargs)
        finally:
            stop.set()
            try:
                _drop_page_cache(self)
            except Exception as exc:  # never break the load
                _say(f"drop page cache failed: {exc!r}")

    load_with_ticks._dsv41_mem_wrapped = True
    Worker.load_model = load_with_ticks

    # Inner phases of load_model: weights, post-load processing, DSpark draft.
    def _wrap_attr(mod_name, owner_attr, fn_name, label):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            owner = getattr(mod, owner_attr) if owner_attr else mod
            fn = getattr(owner, fn_name)
        except Exception as exc:
            _say(f"memory log: cannot wrap {label}: {exc!r}")
            return
        if getattr(fn, "_dsv41_mem_wrapped", False):
            return

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            _say(f"before {label}: {_snapshot()}")
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _say(f"after {label} ({time.perf_counter() - t0:.0f}s): {_snapshot()}")

        wrapped._dsv41_mem_wrapped = True
        setattr(owner, fn_name, wrapped)

    _wrap_attr("vllm.model_executor.model_loader.default_loader", "DefaultModelLoader", "load_weights", "load_weights")
    # base_loader imported the function by name; rebind it there.
    _wrap_attr("vllm.model_executor.model_loader.base_loader", None, "process_weights_after_loading", "process_weights_after_loading")
    _wrap_attr("vllm.v1.worker.gpu.spec_decode.dspark.speculator", "DSparkSpeculator", "load_model", "dspark_draft.load_model")


if os.environ.get("DSV41_MEMORY_LOG", "1") != "0":
    _dsv41_install_memory_log()
'''


def apply(text: str) -> tuple[str, str]:
    if MARK in text:
        return text, "skipped"
    if "class Worker(" not in text:
        return text, "missing:class Worker"
    if "\nimport os\n" not in text and not text.startswith("import os\n"):
        # gpu_worker.py imports os already in every vLLM we have seen; keep the
        # footer self-sufficient anyway.
        text = text.replace("\nimport gc\n", "\nimport gc\nimport os\n", 1) if "\nimport gc\n" in text else "import os\n" + text
    if not text.endswith("\n"):
        text += "\n"
    return text + FOOTER, "applied"


def main() -> int:
    try:
        import vllm

        root = Path(vllm.__file__).resolve().parent
    except Exception as exc:
        print(f"WARN: vllm not importable; skip memory log patch ({exc})", file=sys.stderr)
        return 0
    path = root / "v1/worker/gpu_worker.py"
    if not path.is_file():
        print(f"WARN: {path} missing", file=sys.stderr)
        return 1
    out, status = apply(path.read_text())
    if status == "applied":
        path.write_text(out)
        print(f"memory log: applied ({path})")
        return 0
    if status == "skipped":
        print("memory log: already applied")
        return 0
    print(f"WARN: memory log: {status}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
